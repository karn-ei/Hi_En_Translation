from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class EIAuthConfig:
    base_url: str
    auth_token: str
    product_id: int
    feature_id: int
    token_ttl: int = 3600

    token_endpoint: str = "/api/v1/token"
    generate_endpoint: str = "/api/v1/multimode-to-text?stream=false"

    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 80


class EITranslationClient:
    def __init__(self, cfg: EIAuthConfig, timeout_s: int = 45) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s

        self._token: Optional[str] = None
        self._expires_at_epoch: float = 0.0

        self.token_calls = 0
        self.generate_calls = 0

        self.cache_hits = 0
        self.ai_calls = 0

        self.last_http_debug: Dict[str, Any] = {}

    def reset_counters(self) -> None:
        self.token_calls = 0
        self.generate_calls = 0
        self.cache_hits = 0
        self.ai_calls = 0
        self.last_http_debug = {}

    def _get_custom_api_token(self) -> str:
        now = time.time()

        if self._token and now < (self._expires_at_epoch - 300):
            return self._token

        url = self.cfg.base_url.rstrip("/") + self.cfg.token_endpoint
        headers = {
            "Authorization": f"Bearer {self.cfg.auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "product_id": self.cfg.product_id,
            "ttl": self.cfg.token_ttl,
        }

        self.last_http_debug = {"stage": "token", "url": url, "payload": payload, "status": None}

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        self.token_calls += 1
        self.last_http_debug["status"] = r.status_code

        r.raise_for_status()
        data = r.json()

        token = data.get("token") or data.get("access_token") or data.get("data", {}).get("token")
        if not token:
            raise RuntimeError(f"Token response missing token field. Keys: {list(data.keys())}")

        self._token = token

        expires_in = data.get("expires_in") or data.get("data", {}).get("expires_in")
        if expires_in:
            self._expires_at_epoch = now + float(expires_in)
        else:
            self._expires_at_epoch = now + float(self.cfg.token_ttl)

        return self._token

    @staticmethod
    def _build_translation_prompt(instructions: str, hindi_text: str) -> str:
        instructions = (instructions or "").strip()
        return (
            "Task: Translate Hindi (Devanagari) to English.\n"
            "Rules: Output ONLY the English translation text. No explanation. No quotes.\n"
            "Keep vocabulary simple for school students.\n"
            + (f"Additional instructions:\n{instructions}\n" if instructions else "")
            + "\nHindi text:\n"
            + hindi_text
            + "\n\nEnglish translation:"
        )

    @staticmethod
    def _extract_text_from_any(obj: Any) -> Optional[str]:
        """
        EI seems to be returning an envelope: {"status","message","data":...}
        This helper tries to pull a text response from many plausible shapes.
        """
        if obj is None:
            return None

        # direct string
        if isinstance(obj, str):
            s = obj.strip()
            return s if s else None

        # bytes
        if isinstance(obj, (bytes, bytearray)):
            try:
                s = obj.decode("utf-8", errors="ignore").strip()
                return s if s else None
            except Exception:
                return None

        # list: try first element, then scan
        if isinstance(obj, list):
            for item in obj:
                t = EITranslationClient._extract_text_from_any(item)
                if t:
                    return t
            return None

        # dict: check common keys
        if isinstance(obj, dict):
            for k in ("text", "output", "generated_text", "completion", "result"):
                if isinstance(obj.get(k), str) and obj[k].strip():
                    return obj[k].strip()

            # OpenAI-ish: choices
            if isinstance(obj.get("choices"), list) and obj["choices"]:
                c0 = obj["choices"][0]
                if isinstance(c0, dict):
                    if isinstance(c0.get("text"), str) and c0["text"].strip():
                        return c0["text"].strip()
                    # chat-ish
                    msg = c0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
                        return msg["content"].strip()

            # Anthropic-ish: content: [{type:"text", text:"..."}]
            if isinstance(obj.get("content"), list):
                for item in obj["content"]:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        if item["text"].strip():
                            return item["text"].strip()

            # nested: data.text / data.output / etc
            if "data" in obj:
                t = EITranslationClient._extract_text_from_any(obj.get("data"))
                if t:
                    return t

        return None

    def translate_hi_to_en(self, hindi_text: str, prompt: str) -> str:
        combined_prompt = self._build_translation_prompt(prompt, hindi_text)
        api_token = self._get_custom_api_token()

        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_endpoint
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "X-Product-ID": str(self.cfg.product_id),
            "X-Feature-ID": str(self.cfg.feature_id),
        }

        payload: Dict[str, Any] = {
            "content_type": "text",
            "data": str(hindi_text),
            "prompt": combined_prompt,
            "model": self.cfg.model_name,
            "model_temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_tokens),
        }

        self.last_http_debug = {
            "stage": "generate",
            "url": url,
            "payload_preview": {
                "content_type": payload["content_type"],
                "data_preview": payload["data"][:50] + ("..." if len(payload["data"]) > 50 else ""),
                "prompt_preview": combined_prompt[:200] + ("..." if len(combined_prompt) > 200 else ""),
                "model": payload["model"],
                "model_temperature": payload["model_temperature"],
                "max_tokens": payload["max_tokens"],
            },
            "headers_preview": {
                "X-Product-ID": str(self.cfg.product_id),
                "X-Feature-ID": str(self.cfg.feature_id),
            },
            "status": None,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        self.generate_calls += 1
        self.last_http_debug["status"] = r.status_code

        if r.status_code >= 400:
            try:
                self.last_http_debug["error_json"] = r.json()
            except Exception:
                self.last_http_debug["error_text"] = r.text[:2000]
            r.raise_for_status()

        resp_json = r.json()

        # ✅ NEW: store a small success preview for debugging without leaking big payloads
        try:
            self.last_http_debug["success_preview"] = {
                "top_keys": list(resp_json.keys()) if isinstance(resp_json, dict) else str(type(resp_json)),
                "status": resp_json.get("status") if isinstance(resp_json, dict) else None,
                "message": resp_json.get("message") if isinstance(resp_json, dict) else None,
                "data_type": type(resp_json.get("data")).__name__ if isinstance(resp_json, dict) and "data" in resp_json else None,
            }
        except Exception:
            pass

        # EI envelope: {"status","message","data":...}
        # But also handle non-envelope shapes just in case.
        text = self._extract_text_from_any(resp_json)
        if text:
            self.ai_calls += 1
            return text

        # If nothing extracted, raise with helpful context
        raise RuntimeError(
            f"Unexpected EI generate response shape. Keys: "
            f"{list(resp_json.keys()) if isinstance(resp_json, dict) else type(resp_json)}"
        )