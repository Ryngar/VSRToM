"""Qwen2.5-VL backend adapted from the lmms-eval VSRToM implementation.

The standalone adapter removes harness-specific batching, unrelated methods,
API clients and private configuration. See NOTICE for provenance.
"""


class Qwen25VL:
    def __init__(self, model, device_map="auto", dtype="bfloat16", max_new_tokens=1024,
                 min_pixels=200704, max_pixels=1605632, attn_implementation=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("Unsupported dtype")
        if max_new_tokens < 1 or min_pixels < 1 or max_pixels < min_pixels:
            raise ValueError("Invalid token/pixel limits")
        kwargs = {"device_map": device_map,
                  "torch_dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model, min_pixels=min_pixels, max_pixels=max_pixels)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt, image=None):
        import torch
        from qwen_vl_utils import process_vision_info
        content = []
        if image is not None:
            from PIL import Image
            # Decode locally; do not send images to any external API.
            with Image.open(image) as source:
                content.append({"type": "image", "image": source.convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=False, num_beams=1, use_cache=True,
                                         eos_token_id=self.processor.tokenizer.eos_token_id,
                                         pad_token_id=self.processor.tokenizer.pad_token_id)
        trimmed = output[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
