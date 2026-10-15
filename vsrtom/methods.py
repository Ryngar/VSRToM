"""Single-pass baseline and DePeR map-conditioned second pass."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tasks import build_prompt


@dataclass
class Prediction:
    response: str
    round1: str
    round2: str | None = None


def extract_cogmap(response):
    """Match the original gate: first ordered, case-sensitive cogmap tag pair.

    Keep the original slice (opening tag included, closing tag excluded) to
    preserve the exact second-stage input used by the source implementation.
    """
    start = response.find("<cogmap>")
    end = response.find("</cogmap>")
    return response[start:end] if start >= 0 and end > start else None


def round2_prompt(doc, task, cogmap):
    settings = json.loads((Path(__file__).with_name("prompts") / "round2.json").read_text())
    first_prompt = build_prompt(doc, task, "deper")
    question = first_prompt[first_prompt.index("[Question]"):first_prompt.index("[Hints]")]
    return (f"{settings['prompt']}\n[Cognitive Map]{cogmap}\n\n{question}"
            f"\n\n[Hints]\n{settings['camera_description']}{settings['person_hints']}\n")


def predict(backend, doc, image, task="vsrtom", method="default"):
    first = backend.generate(build_prompt(doc, task, method), image=image)
    cogmap = extract_cogmap(first) if method == "deper" else None
    if cogmap is None:
        return Prediction(response=first, round1=first)
    # This is a fresh text-only request, not a continuation containing the image.
    second = backend.generate(round2_prompt(doc, task, cogmap), image=None)
    return Prediction(response=f"[Round1]{first}\n\n[Round2]{second}", round1=first, round2=second)
