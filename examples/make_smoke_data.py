"""Create artificial I/O fixtures, not paper data or a performance benchmark."""
import argparse
import json
from pathlib import Path
import struct
import zlib


def png(width=8, height=8):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = b''.join(b'\x00' + bytes((220, 225, 230)) * width for _ in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def record(real=False):
    options = {"A": {"text": "Yes", "description": "At least one basketball is visible"},
               "B": {"text": "No", "description": "No basketballs are visible"}}
    doc = {"sample_id": "smoke", "image": "smoke.png", "level": "0-order", "task": "visibility",
           "question": "From the camera's perspective, is any basketball visible?", "answer": "B"}
    if real:
        doc["options"] = options
        doc["persons"] = {"a person wearing a blue shirt": {"orientation": {"azimuth": 90}}}
    else:
        doc["answer_options"] = {"visibility": options}
        doc["vision_tool"] = {"man_orientation": {"azimuth": 90}, "woman_orientation": {"azimuth": 180}}
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke.png").write_bytes(png())
    for name, real in [("sim", False), ("real", True)]:
        (args.output_dir / f"{name}.jsonl").write_text(json.dumps(record(real)) + "\n")
    print("Created artificial smoke fixtures; do not use them to measure accuracy.")


if __name__ == "__main__":
    main()
