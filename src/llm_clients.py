from dataclasses import dataclass
from typing import List, Dict, Optional
import streamlit as st


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from src.config import MODEL_MODE, GROQ_API_KEY, GROQ_MODEL, LOCAL_MODEL_ID, HF_TOKEN


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
# Local model (HF, cached for Streamlit)
# =========================

_LOCAL_KIND = None  # just for reference


def _detect_model_kind(model_id: str) -> str:
    cfg = AutoConfig.from_pretrained(model_id, token=HF_TOKEN if HF_TOKEN else None)
    if getattr(cfg, "is_encoder_decoder", False):
        return "seq2seq"
    return "causal"


@st.cache_resource(show_spinner=True)
def _cached_local_model(model_id: str):
    """
    Load once per Streamlit session (fast reruns).
    """
    kind = _detect_model_kind(model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    if kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            token=HF_TOKEN if HF_TOKEN else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN if HF_TOKEN else None,
        )

    return tokenizer, model, kind


def _load_local_model():
    return _cached_local_model(LOCAL_MODEL_ID)

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

    # CausalLM (Gemma, Qwen, TinyLlama, Falcon)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=(temperature > 0),
        temperature=max(0.1, float(temperature)),
        pad_token_id=tokenizer.pad_token_id,
    )

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
