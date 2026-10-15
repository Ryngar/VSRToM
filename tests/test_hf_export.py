from io import BytesIO
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_hf_dataset import normalize, portable_png
from examples.make_smoke_data import record
from vsrtom.tasks import build_prompt


class ExportTests(unittest.TestCase):
    def test_real_aliases_and_crop_removal(self):
        source = record(True)
        source.pop('sample_id'); source.pop('image')
        source['image_id'] = 12
        source['order'] = source.pop('level')
        source['task_type'] = source.pop('task')
        self.assertIn('[Question]', build_prompt(source, 'vsrtom_real'))
        with tempfile.TemporaryDirectory() as temp:
            source['persons']['person_0'] = {'crop_path': str(Path(temp) / 'crop.png'),
                                             'orientation': {'azimuth': 90}}
            exported = normalize(source, 'real', 0)
        self.assertNotIn('crop_path', exported['annotation_json'])
        self.assertEqual(exported['level'], '0-order')
        self.assertEqual(exported['task'], 'visibility')
        self.assertEqual(json.loads(exported['annotation_json'])['persons']['person_0']['orientation']['azimuth'], 90)

    def test_person_hint_order_matches_original_prompts(self):
        source = record(True)
        source.pop('sample_id'); source.pop('image')
        source['image_id'] = 1
        source['order'] = source.pop('level')
        source['task_type'] = source.pop('task')
        source['persons'] = {
            'person wearing yellow': {'orientation': {'azimuth': 90}},
            'person wearing blue': {'orientation': {'azimuth': 180}},
        }
        restored = json.loads(normalize(source, 'real', 0)['annotation_json'])
        for method in ['default', 'deper']:
            self.assertEqual(build_prompt(source, 'vsrtom_real', method),
                             build_prompt(restored, 'vsrtom_real', method))

    def test_unreviewed_fields_are_rejected(self):
        source = record()
        source.pop('image')
        source['private_notes'] = 'not for publication'
        with self.assertRaises(ValueError):
            normalize(source, 'sim', 0)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow not installed')
    def test_png_preserves_rgb_pixels_and_removes_metadata(self):
        from PIL import Image, PngImagePlugin
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.png'
            image = Image.new('RGB', (3, 3), (31, 83, 112))
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('local_note', str(path))
            exif = Image.Exif(); exif[270] = 'temporary metadata'
            image.save(path, pnginfo=metadata, exif=exif)
            encoded = portable_png(str(path))
            portable_png.cache_clear()
            with Image.open(BytesIO(encoded)) as restored:
                self.assertEqual(restored.tobytes(), image.tobytes())
                self.assertFalse(restored.getexif())
                self.assertNotIn('local_note', restored.info)


if __name__ == '__main__':
    unittest.main()
