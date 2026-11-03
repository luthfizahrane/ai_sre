"""Pure helpers for model response handling (no network dependencies)."""
import json
import re
from typing import Any


def message_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    raise ValueError("LLM response does not contain text")


def parse_llm_json(message: Any) -> dict:
    raw = message_to_text(message).strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?\s*```$", "", raw).strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("LLM response must be a JSON object")
    return obj
