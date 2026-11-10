"""VLM wrappers. Every model exposes generate(messages) -> str.

messages: [{"role": "user"|"assistant", "content": [{"type": "image", "path": ...} |
                                                     {"type": "text", "text": ...}]}]
"""
from PIL import Image

MOCK_ANSWER = '{"present": "yes", "confidence": 1.0, "evidence": "mock", "action": "slow"}'


class ScriptedModel:
    """Returns canned answers in order, then a fixed answer. For tests and dry runs without a GPU."""

    def __init__(self, answers=()):
        self.answers = list(answers)

    def generate(self, messages):
        return self.answers.pop(0) if self.answers else MOCK_ANSWER


class HFModel:
    """Any Hugging Face image-text-to-text chat model, e.g. Qwen/Qwen2.5-VL-3B-Instruct or
    llava-hf/llava-onevision-qwen2-0.5b-ov-hf."""

    def __init__(self, model_id, load_4bit=False, max_side=1024, max_new_tokens=128):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # fp16 on GPU (T4 has no fast bf16); fp32 on CPU, where fp16 is slow or unsupported
        kwargs = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                  "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_side = max_side  # downscale 1600x900 frames so multi-frame history fits in VRAM (lower it if OOM)
        self.max_new_tokens = max_new_tokens

    def generate(self, messages):
        images, chat = [], []
        for m in messages:
            content = []
            for c in m["content"]:
                if c["type"] == "image":
                    img = Image.open(c["path"]).convert("RGB")
                    img.thumbnail((self.max_side, self.max_side))
                    images.append(img)
                    content.append({"type": "image"})
                else:
                    content.append(c)
            chat.append({"role": m["role"], "content": content})
        text = self.processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]


def load_model(name, load_4bit=False, max_side=1024):
    return ScriptedModel() if name == "mock" else HFModel(name, load_4bit, max_side)
