import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

from vsrtom.dataset.metadata_generation import generate_metadata
from vsrtom.dataset.qa_generation import can_see, VisibilityState, generate_all_qa
from vsrtom.dataset.filtering_and_sampling import (
    extract_answer_patterns, sample_for_level_task, generate_balanced_dataset,
    create_target_samples, export_sampled_dataset,
)
from vsrtom.dataset.shuffle import shuffle_dataset_options
from vsrtom.evaluate import aggregate
from vsrtom.methods import predict
from vsrtom.scoring import parse_vsrtom_response
from vsrtom.tasks import build_prompt, image_path
from examples.make_smoke_data import record


class FakeBackend:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate(self, prompt, image=None):
        self.calls.append((prompt, image))
        return next(self.answers)


class CoreTests(unittest.TestCase):
    def test_deper_text_only_second_round_and_last_answer(self):
        first = '<type>allocentric</type><cogmap>[{"object":"ball","position":[2,3]}]</cogmap><answer>A</answer>'
        backend = FakeBackend([first, '<answer>B</answer>'])
        result = predict(backend, record(), Path('smoke.png'), method='deper')
        self.assertEqual(len(backend.calls), 2)
        self.assertIsNotNone(backend.calls[0][1])
        self.assertIsNone(backend.calls[1][1])
        second_prompt = backend.calls[1][0]
        self.assertIn('Vis(A,B) && Vis(A,C) && Vis(B,C)', second_prompt)
        self.assertIn('[Cognitive Map]<cogmap>', second_prompt)
        self.assertIn('[Question]', second_prompt)
        self.assertNotIn('Man orientation:', second_prompt)
        self.assertEqual(parse_vsrtom_response(result.response), 'B')

    def test_baseline_and_deper_fallback_do_not_call_second_round(self):
        for method, response in [('default', '<cogmap>[]</cogmap><answer>A</answer>'),
                                 ('deper', '<type>egocentric</type><answer>B</answer>'),
                                 ('deper', '</cogmap><cogmap><answer>B</answer>')]:
            backend = FakeBackend([response])
            prediction = predict(backend, record(), Path('smoke.png'), method=method)
            self.assertEqual(len(backend.calls), 1)
            self.assertIsNone(prediction.round2)

    def test_direction_protocol_and_missing_tool(self):
        doc = record()
        self.assertNotIn('Man orientation:', build_prompt(doc, method='default'))
        self.assertIn('Man orientation: [-1, 0]', build_prompt(doc, method='deper'))
        del doc['vision_tool']
        with self.assertRaises(ValueError):
            build_prompt(doc, method='deper')
        # The source Real task supplies available directions to default too.
        self.assertIn('The orientation of A person', build_prompt(record(True), 'vsrtom_real', 'default'))
        self.assertIn('The orientation of A person', build_prompt(record(True), 'vsrtom_real', 'deper'))

    def test_visibility_geometry(self):
        self.assertEqual(can_see((0, 0, 0), (1, 0, 0), (3, 0, 0)), VisibilityState.VISIBLE)
        self.assertEqual(can_see((0, 0, 0), (1, 0, 0), (-3, 0, 0)), VisibilityState.NOT_VISIBLE_BY_BEHIND)
        self.assertEqual(can_see((0, 0, 0), (1, 0, 0), (0, 4, 0)), VisibilityState.AMBIGUOUS)
        self.assertEqual(can_see((-5, 0, 0), (1, 0, 0), (3, 0, 0),
                                [{'position': [0, 0, 0]}]), VisibilityState.NOT_VISIBLE_BY_OBS)

    def test_exclusion_and_answer_balance(self):
        data = []
        for index in range(48):
            data.append({'sample_id': index + 10, 'answers': list(('A' if index % 2 else 'B') * 9),
                         'not_visible_analysis': {'man_cant_sees_balls_by_behind': False,
                                                 'man_cant_sees_balls_by_obs': False,
                                                 'woman_cant_sees_balls_by_behind': False,
                                                 'woman_cant_sees_balls_by_obs': False}})
        excluded = copy.deepcopy(data[0]); excluded['sample_id'] = 999
        excluded['answers'][8] = 'EXCLUDED_AMBIGUOUS'; data.append(excluded)
        with contextlib.redirect_stdout(io.StringIO()):
            patterns = extract_answer_patterns(data)
            random.seed(42)
            selected, _ = sample_for_level_task('0-order', 'visibility', patterns, 12)
        self.assertNotIn(999, [index for index, _ in patterns])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(sum(data[index - 10]['answers'][0] == 'A' for index in selected), 6)

    def test_generation_export_shuffle_preserves_answer_content(self):
        args = SimpleNamespace(random_seed=42, max_attempts_per_sample=1000)
        for count in range(4):
            for suffix in ['with_obstacles', 'without_obstacles']:
                setattr(args, f'num_samples{count}_{suffix}', 500)
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = generate_metadata(args)
            self.assertEqual(metadata, generate_metadata(args))
            qa = generate_all_qa(metadata)
            selected, stats = generate_balanced_dataset(qa, create_target_samples([[8] * 3] * 3), seed=42, verbose=False)
            with tempfile.TemporaryDirectory() as temp:
                export_sampled_dataset(qa, selected, stats, temp)
                source = Path(temp) / 'VSRToM_Bench.jsonl'
                target = Path(temp) / 'shuffled.jsonl'
                shuffle_dataset_options(str(source), str(target), seed=42)
                before = [json.loads(line) for line in source.read_text().splitlines()]
                after = [json.loads(line) for line in target.read_text().splitlines()]
        self.assertEqual(len(before), 72)
        self.assertEqual(len(after), 72)
        self.assertEqual(len({(r['level'], r['task']) for r in after}), 9)
        lookup = {(r['sample_id'], r['level'], r['task']): r for r in before}
        for item in after:
            old = lookup[(item['sample_id'], item['level'], item['task'])]
            task = item['task']
            self.assertEqual(old['answer_options'][task][old['answer']], item['answer_options'][task][item['answer']])

    def test_image_path_rejects_private_and_escaping_locations(self):
        with tempfile.TemporaryDirectory() as temp:
            for value in ['../secret.png', str(Path(temp).resolve() / 'absolute.png'), 'https://invalid/image.png']:
                with self.assertRaises(ValueError):
                    image_path({'image': value}, temp)

    def test_aggregation_reports_micro_and_macro(self):
        result = aggregate([{'category': 'a', 'correct': True}, {'category': 'a', 'correct': False},
                            {'category': 'b', 'correct': True}])
        self.assertAlmostEqual(result['accuracy'], 2/3)
        self.assertEqual(result['macro_accuracy'], .75)


if __name__ == '__main__':
    unittest.main()
