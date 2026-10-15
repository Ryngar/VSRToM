"""VSRToM-Sim and VSRToM-Real prompt construction.

Adapted from the VSRToM tasks in lmms-eval; see NOTICE.
"""
import json
import math
from pathlib import Path

PROMPT_DIR = Path(__file__).with_name("prompts")
LEVELS = ("0-order", "1st-order", "2nd-order")
TASKS = ("visibility", "count", "layout")


def load_prompts(task):
    if task not in ("vsrtom", "vsrtom_real"):
        raise ValueError("Unknown task")
    return json.loads((PROMPT_DIR / f"{task}.json").read_text(encoding="utf-8"))


def options_for(doc, task):
    if task == "vsrtom":
        candidates = [v for v in doc.get("answer_options", {}).values() if isinstance(v, dict)]
        if len(candidates) != 1:
            raise ValueError("Sim records must contain exactly one answer_options category")
        options = candidates[0]
    else:
        options = {k: v for k, v in doc.get("options", {}).items() if isinstance(v, dict)}
    if not options or not set(options) <= set("ABCD"):
        raise ValueError("Expected multiple-choice options labeled A-D")
    return options


def validate_doc(doc, task):
    if doc.get("level", doc.get("order")) not in LEVELS or doc.get("task", doc.get("task_type")) not in TASKS:
        raise ValueError("Unknown reasoning level or task category")
    if not isinstance(doc.get("question"), str) or not doc["question"].strip():
        raise ValueError("Missing question")
    options = options_for(doc, task)
    if str(doc.get("answer", "")).strip().upper() not in options:
        raise ValueError("Correct answer must be one of the option labels")
    for value in options.values():
        if "text" not in value or "description" not in value:
            raise ValueError("Options need text and description fields")


def question_text(doc, task):
    options = options_for(doc, task)
    lines = []
    for label, value in sorted(options.items()):
        line = f"{label}. {value['text']} ({value['description']})"
        # Sim preserves the original presence check; Real omits null use_case.
        if "use_case" in value and (task == "vsrtom" or value["use_case"] is not None):
            line += f" ({value['use_case']})"
        lines.append(line)
    return f"\n[Question]\n{doc['question'].strip()}\n[Options]\n" + "\n".join(lines) + "\n"


def orientation_vector(azimuth, real=False):
    if not isinstance(azimuth, (int, float)) or not math.isfinite(azimuth) or not 0 <= azimuth <= 360:
        raise ValueError("Orientation azimuth must be a finite angle in [0, 360]")
    angles = list(range(0, 361, 45 if real else 90))
    angle = min(angles, key=lambda x: abs(x - azimuth)) % 360
    vectors = {0: "[0, 1]", 45: "[-1, 1]", 90: "[-1, 0]", 135: "[-1, -1]",
               180: "[0, -1]", 225: "[1, -1]", 270: "[1, 0]", 315: "[1, 1]"}
    return vectors[angle]


def orientation_hints(doc, task, method):
    if task == "vsrtom":
        if method == "default":
            return ""
        tool = doc.get("vision_tool", {})
        lines = []
        for name in ("man", "woman"):
            result = tool.get(f"{name}_orientation")
            if not isinstance(result, dict) or result.get("azimuth") is None:
                raise ValueError("Sim DePeR requires tool-estimated man/woman azimuths in vision_tool; "
                                 "run orientation preprocessing first (ground-truth metadata is not a substitute)")
            lines.append(f"{name.capitalize()} orientation: {orientation_vector(result['azimuth'])}")
        return "\n".join(lines) + "\n"
    # Original Real task appends available person directions for BOTH methods.
    lines = []
    persons = doc.get("persons", {})
    if not isinstance(persons, dict):
        raise ValueError("persons must be an object")
    for description, person in persons.items():
        if description.startswith("person_") or not isinstance(person, dict):
            continue
        orientation = person.get("orientation")
        if isinstance(orientation, dict) and orientation.get("azimuth") is not None:
            vector = orientation_vector(orientation["azimuth"], real=True)
            lines.append(f"The orientation of {description.capitalize()} is: {vector}")
    return "\n".join(lines) + ("\n" if lines else "")


def build_prompt(doc, task="vsrtom", method="default"):
    validate_doc(doc, task)
    if method not in ("default", "deper"):
        raise ValueError("Only default and deper are included")
    settings = load_prompts(task)[method]
    return settings["pre_prompt"] + question_text(doc, task) + settings["post_prompt"] + orientation_hints(doc, task, method)


def image_path(doc, image_dir, task="vsrtom"):
    """Accept relative image names only; never use paths embedded in private dumps."""
    relative = doc.get("image")
    if relative is None:
        sample_id = str(doc.get("image_id" if task == "vsrtom_real" else "sample_id", ""))
        if not sample_id or not all(c.isalnum() or c in "_-" for c in sample_id):
            raise ValueError("Set a relative image filename or a simple sample_id")
        relative = f"{sample_id}.jpg" if task == "vsrtom_real" else f"{sample_id}.png"
    if not isinstance(relative, str):
        raise ValueError("image must be a relative filename string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative or "\\" in relative:
        raise ValueError("image must be a safe relative path")
    root = Path(image_dir).resolve()
    result = (root / path).resolve()
    if root not in result.parents:
        raise ValueError("Image resolves outside the configured image directory")
    if not result.is_file():
        raise FileNotFoundError(f"Missing image: {path.name}")
    return result
