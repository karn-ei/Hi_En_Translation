from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests


@dataclass
class EIAuthConfig:
    base_url: str
    auth_token: str          # Bearer token used ONLY to fetch custom api token
    product_id: int
    feature_id: int
    token_ttl: int = 3600

    token_endpoint: str = "/api/v1/token"
    generate_endpoint: str = "/api/v1/multimode-to-text?stream=false"

    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 80


class EITranslationClient:
    """
    Implements:
    1) POST /api/v1/token with bearer AUTH_TOKEN -> returns custom api token
    2) POST /api/v1/multimode-to-text?stream=false with that token -> returns generated text

    IMPORTANT CHANGE:
    - We now embed translation instructions directly into 'prompt', because your test shows
      the service/model is behaving as if it only sees the raw text ("सौ") and ignores 'data'.
    - We still send 'data' (for compatibility), but we don't depend on it.
    """

    def __init__(self, cfg: EIAuthConfig, timeout_s: int = 45) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s

        self._token: Optional[str] = None
        self._expires_at_epoch: float = 0.0

        # Counters
        self.token_calls = 0
        self.generate_calls = 0

        # These are updated by translator_core for run stats
        self.cache_hits = 0
        self.ai_calls = 0

        # Latest HTTP debug blob (safe: does not include tokens)
        self.last_http_debug: Dict[str, Any] = {}

    def reset_counters(self) -> None:
        self.token_calls = 0
        self.generate_calls = 0
        self.cache_hits = 0
        self.ai_calls = 0
        self.last_http_debug = {}

    def _get_custom_api_token(self) -> str:
        now = time.time()

        # 5-minute buffer
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

        try:
            data = r.json()
        except Exception:
            data = {"_raw_text": r.text}

        self.last_http_debug["response"] = data
        r.raise_for_status()

        token = data.get("token") or data.get("access_token")
        if not token:
            raise RuntimeError(f"/api/v1/token did not return token. Keys: {list(data.keys())}")

        self._token = token
        self._expires_at_epoch = now + float(self.cfg.token_ttl)
        return token

    @staticmethod
    def _extract_text(resp_json: Dict[str, Any]) -> str:
        candidates = [
            resp_json.get("response"),
            resp_json.get("text"),
            resp_json.get("content"),
            (resp_json.get("data") or {}).get("response") if isinstance(resp_json.get("data"), dict) else None,
            (resp_json.get("data") or {}).get("text_response") if isinstance(resp_json.get("data"), dict) else None,
            resp_json.get("data"),
        ]
        for c in candidates:
            if isinstance(c, str) and c.strip():
                return c.strip()

        if isinstance(resp_json.get("data"), dict):
            for k in ("response", "text_response", "text", "content"):
                v = resp_json["data"].get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        raise RuntimeError(f"Could not extract text from response. Keys: {list(resp_json.keys())}")

    @staticmethod
    def _build_translation_prompt(instructions: str, hindi_text: str) -> str:
        """
        Minimal, strong instruction format that prevents 'question unclear' replies.
        Keeps token usage low while being explicit.

        We DO NOT send the whole Excel cell, only the Hindi block, but we include
        short instructions so the model knows what to do.
        """
        instructions = (instructions or "").strip()

        # If user prompt file is long, it can be; but we keep a strong wrapper.
        # The wrapper is what forces correct behavior.
        return (
            "Task: Translate Hindi (Devanagari) to English.\n"
            "Rules: Output ONLY the English translation text. No explanation. No quotes.\n"
            "Keep vocabulary simple for school students.\n"
            + (f"Additional instructions:\n{instructions}\n" if instructions else "")
            + "\nHindi text:\n"
            + hindi_text
            + "\n\nEnglish translation:"
        )

    def translate_hi_to_en(self, hindi_text: str, prompt: str) -> str:
        token = self._get_custom_api_token()

        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_endpoint
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Product-ID": str(self.cfg.product_id),
            "X-Feature-ID": str(self.cfg.feature_id),
            "Content-Type": "application/json",
        }

        # IMPORTANT: Put instructions into prompt (not just data)
        combined_prompt = self._build_translation_prompt(prompt, hindi_text)

        body = {
            "content_type": "text",
            "prompt": combined_prompt,
            # keep data too (harmless if ignored; useful if honored)
            "data": prompt,
            "model_name": self.cfg.model_name,
            "model_temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        # Store debug without tokens
        self.last_http_debug = {
            "stage": "generate",
            "url": url,
            "headers": {
                "X-Product-ID": str(self.cfg.product_id),
                "X-Feature-ID": str(self.cfg.feature_id),
                "Content-Type": "application/json",
            },
            "body_preview": {
                "content_type": body["content_type"],
                # don’t dump full prompt into UI; just a short preview
                "prompt_preview": combined_prompt[:200] + ("..." if len(combined_prompt) > 200 else ""),
                "model_name": body["model_name"],
                "model_temperature": body["model_temperature"],
                "max_tokens": body["max_tokens"],
            },
            "status": None,
        }

        r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
        self.generate_calls += 1
        self.last_http_debug["status"] = r.status_code

        try:
            data = r.json()
        except Exception:
            data = {"_raw_text": r.text}

        self.last_http_debug["response"] = data
        r.raise_for_status()

        text = self._extract_text(data)
        return text.strip().strip('"').strip("'").strip()
