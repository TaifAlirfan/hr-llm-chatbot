from dataclasses import dataclass
from typing import List, Dict, Optional

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from src.config import MODEL_MODE, GROQ_API_KEY, GROQ_MODEL, LOCAL_MODEL_ID


# =========================
# Data structure
# =========================
@dataclass
class LLMResponse:
    text: str


# =========================
# Groq (API – OpenAI compatible)
# =========================
def groq_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> LLMResponse:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in .env")

    from openai import OpenAI

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )

    return LLMResponse(text=response.choices[0].message.content.strip())


# =========================
# Local model (HF, CPU-safe)
# =========================
_LOCAL_TOKENIZER = None
_LOCAL_MODEL = None
_LOCAL_KIND = None  # "seq2seq" or "causal"


def _detect_model_kind(model_id: str) -> str:
    """
    Detect whether the model is Seq2Seq (T5/Flan)
    or CausalLM (Qwen, TinyLlama, Falcon).
    """
    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "is_encoder_decoder", False):
        return "seq2seq"
    return "causal"


def _load_local_model():
    global _LOCAL_TOKENIZER, _LOCAL_MODEL, _LOCAL_KIND

    if _LOCAL_TOKENIZER is not None and _LOCAL_MODEL is not None:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL, _LOCAL_KIND

    _LOCAL_KIND = _detect_model_kind(LOCAL_MODEL_ID)

    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)

    if _LOCAL_KIND == "seq2seq":
        _LOCAL_MODEL = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
    else:
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID)

    return _LOCAL_TOKENIZER, _LOCAL_MODEL, _LOCAL_KIND


def local_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> LLMResponse:
    tokenizer, model, kind = _load_local_model()

    # Build a simple instruction-style prompt
    prompt_parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")

    prompt = "\n".join(prompt_parts).strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    if kind == "seq2seq":
        # Flan / T5 style
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()
        return LLMResponse(text=text)

    # CausalLM (Qwen, TinyLlama, Falcon)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=(temperature > 0),
        temperature=max(0.1, float(temperature)),
        pad_token_id=tokenizer.pad_token_id,
    )

    # Only decode newly generated tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
    ).strip()

    return LLMResponse(text=text)


# =========================
# Router (API vs Local)
# =========================
def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    mode: Optional[str] = None,
) -> LLMResponse:
    selected_mode = (mode or MODEL_MODE).strip().lower()

    if selected_mode == "local":
        return local_chat(messages, temperature=temperature)

    return groq_chat(messages, temperature=temperature)
