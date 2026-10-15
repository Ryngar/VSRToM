"""Optional Grounding DINO + Orient Anything preprocessing for synthetic scenes.

Supply the external Orient Anything checkout/checkpoint explicitly; neither is
vendored. Run this in a separate environment from Qwen evaluation if needed.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def initialize_models(args):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor
    external = args.orient_repo.resolve()
    if not (external / "vision_tower.py").is_file() or not (external / "inference.py").is_file():
        raise ValueError("--orient-repo must contain the Orient Anything implementation")
    sys.path.insert(0, str(external))
    from vision_tower import DINOv2_MLP
    from inference import get_3angle
    global ORIENTATION_INFERENCE
    ORIENTATION_INFERENCE = get_3angle
    device = args.device
    processor = AutoProcessor.from_pretrained(args.detector)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(device).eval()
    orient_model = DINOv2_MLP(dino_mode='large', in_dim=1024,
                             out_dim=360+180+180+2, evaluate=True,
                             mask_dino=False, frozen_back=False)
    orient_model.load_state_dict(torch.load(args.orient_checkpoint, map_location='cpu', weights_only=True))
    orient_model = orient_model.to(device).eval()
    preprocess = AutoImageProcessor.from_pretrained(args.dino_processor)
    return processor, detector, orient_model, preprocess, device


def detect_and_crop(image, text_labels, processor, model, device):
    """Detect objects and return cropped images with their info"""
    import torch
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    result = results[0]
    
    detected_objects = {}
    
    for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
        box = box.cpu().tolist()
        
        # Proper coordinate handling
        x1 = math.floor(box[0])
        y1 = math.floor(box[1])
        x2 = math.ceil(box[2])
        y2 = math.ceil(box[3])
        
        # Crop the image
        cropped_image = image.crop((x1, y1, x2, y2))
        
        detected_objects[label] = {
            'image': cropped_image,
            'box': [x1, y1, x2, y2],
            'confidence': score.item()
        }
    
    return detected_objects

def get_orientation(cropped_image, orient_model, val_preprocess, device):
    """Get 3D orientation angles for a cropped image"""
    angles = ORIENTATION_INFERENCE(cropped_image, orient_model, val_preprocess, device)
    azimuth, polar, rotation, confidence = map(float, angles)
    return {"azimuth": azimuth, "polar": polar, "rotation": rotation, "confidence": confidence}



def main():
    from PIL import Image
    from vsrtom.tasks import image_path
    from vsrtom.evaluate import read_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--orient-repo", type=Path, required=True)
    parser.add_argument("--orient-checkpoint", type=Path, required=True)
    parser.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--dino-processor", default="facebook/dinov2-large")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be a new file, distinct from the input")
    processor, detector, model, preprocess, device = initialize_models(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    missing = 0
    with args.output.open("x", encoding="utf-8") as stream:
        for index, doc in enumerate(read_records(args.input)):
            path = image_path(doc, args.images)
            if path not in cache:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                detected = detect_and_crop(image, [["man", "woman"]], processor, detector, device)
                tool = {}
                for name in ("man", "woman"):
                    info = detected.get(name)
                    tool[f"{name}_orientation"] = get_orientation(info["image"], model, preprocess, device) if info else None
                    tool[f"{name}_detection_confidence"] = info["confidence"] if info else None
                    missing += info is None
                cache[path] = tool
            doc["vision_tool"] = cache[path]
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"Processed record {index + 1}", flush=True)
    print(f"Missing detections across unique images: {missing}; inspect before DePeR evaluation.")


if __name__ == "__main__":
    main()
