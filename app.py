from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from openpyxl import load_workbook

from llm_clients import EIAuthConfig, EITranslationClient
from translator_core import process_workbook_inplace


# ---------------------------
# Local dev: optional .env
# ---------------------------
# Streamlit Community Cloud will NOT use .env.
# Locally, if .env exists, load it for convenience.
if Path(".env").exists():
    load_dotenv(".env", override=True)


def load_default_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_setting(name: str, default: str = "") -> str:
    """
    Read config from Streamlit Secrets first (Cloud + local secrets.toml),
    then fall back to environment variables (including .env if loaded).
    """
    # 1) Streamlit secrets (TOML / Cloud UI)
    try:
        if name in st.secrets:
            v = st.secrets[name]
            return str(v).strip()
    except Exception:
        # st.secrets may throw if no secrets are configured locally.
        pass

    # 2) Environment variables
    return os.getenv(name, default).strip()


def setting_int(name: str, default: int) -> int:
    raw = get_setting(name, "")
    try:
        return int(raw) if raw else default
    except Exception:
        return default


def setting_float(name: str, default: float) -> float:
    raw = get_setting(name, "")
    try:
        return float(raw) if raw else default
    except Exception:
        return default


def build_ei_client_from_settings() -> Optional[EITranslationClient]:
    base_url = get_setting("EI_BASE_URL", "")
    bearer = get_setting("EI_AUTH_TOKEN", "")
    if not base_url or not bearer:
        return None

    cfg = EIAuthConfig(
        base_url=base_url,
        auth_token=bearer,
        product_id=setting_int("EI_PRODUCT_ID", 4),
        feature_id=setting_int("EI_FEATURE_ID", 1),
        token_ttl=setting_int("EI_TOKEN_TTL", 3600),
        token_endpoint=get_setting("EI_TOKEN_ENDPOINT", "/api/v1/token") or "/api/v1/token",
        generate_endpoint=get_setting("EI_GENERATE_ENDPOINT", "/api/v1/multimode-to-text?stream=false")
        or "/api/v1/multimode-to-text?stream=false",
        model_name=get_setting("EI_MODEL_NAME", "gpt-4o") or "gpt-4o",
        temperature=setting_float("EI_TEMPERATURE", 0.0),
        max_tokens=setting_int("EI_MAX_TOKENS", 120),
    )

    timeout_s = setting_int("EI_TIMEOUT_S", 60)
    return EITranslationClient(cfg, timeout_s=timeout_s)


def format_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Excel Hindi Block Translator", layout="wide")

st.title("Excel Hindi to English Translator")
st.write("Upload an .xlsx, Test connection and Run translation.\n")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

default_prompt_path = os.path.join("prompts", "translate_prompt.txt")
default_prompt = load_default_prompt(default_prompt_path)

if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = default_prompt

left, right = st.columns([2, 1])
with left:
    st.session_state.prompt_text = st.text_area(
        "Translation prompt (editable for this session only)",
        value=st.session_state.prompt_text,
        height=180,
    )

with right:
    if st.button("Reset prompt to file default"):
        st.session_state.prompt_text = default_prompt
        st.rerun()

use_ai = st.checkbox("Use AI translation", value=True)
use_class_guidance = st.checkbox("Use 'Class' column for grade-level English", value=True)

dictionary_override: Dict[str, str] = {"सौ": "hundred"}
client = build_ei_client_from_settings() if use_ai else None


# ---------------------------
# Connection test
# ---------------------------
st.subheader("Connection test")
if st.button('Test EI Connection (translate "सौ")'):
    if not use_ai:
        st.error("AI is disabled.")
    elif client is None:
        st.error(
            "EI client is not configured.\n\n"
            "On Streamlit Community Cloud: set EI_* keys in App → Settings → Secrets.\n"
            "Locally: set .streamlit/secrets.toml or a .env with EI_* vars."
        )
    else:
        try:
            out = client.translate_hi_to_en("सौ", prompt=st.session_state.prompt_text)
            st.success("✅ EI call succeeded")
            st.code(out)
        except Exception as e:
            st.error("❌ EI call failed")
            st.code(format_exception(e))
            if getattr(client, "last_http_debug", None):
                st.json(client.last_http_debug)

st.divider()


# ---------------------------
# Run translation
# ---------------------------
if uploaded is not None:
    if st.button("Run translation"):
        wb = load_workbook(filename=BytesIO(uploaded.read()))
        if client is not None:
            client.reset_counters()

        progress_bar = st.progress(0)
        status_line = st.empty()
        latest_box = st.empty()
        debug_box = st.empty()

        start_time = time.time()
        ui_state = {"last_ui_update": 0.0}

        def progress_cb(info: Dict[str, Any]) -> None:
            now = time.time()
            if now - ui_state["last_ui_update"] < 0.15 and info.get("event") not in ("changed", "candidates"):
                return
            ui_state["last_ui_update"] = now

            if info.get("event") == "candidates":
                cols = info.get("candidate_columns", [])
                last_row = info.get("effective_last_row", None)
                latest_box.markdown(
                    "**Detected end of data**: "
                    + (f"`row {last_row}`" if last_row else "`unknown`")
                    + "  \n**Columns containing Hindi**:  \n"
                    + (", ".join([f"`{c}`" for c in cols]) if cols else "_None_")
                )
                return

            done = info.get("cells_done", 0)
            total = info.get("cells_total", 1) or 1
            pct = min(1.0, max(0.0, done / total))
            progress_bar.progress(pct)

            elapsed = now - start_time
            status_line.write(
                f"Processed {done:,}/{total:,} candidate cells ({pct*100:.1f}%). "
                f"Elapsed: {elapsed:.1f}s. "
                f"AI calls: {info.get('ai_calls', 0)} | Cache hits: {info.get('cache_hits', 0)}"
            )

            if info.get("event") == "changed":
                def clip(s: str, n: int = 280) -> str:
                    s = s.replace("\n", "\\n")
                    return s if len(s) <= n else s[:n] + "…"

                latest_box.markdown(
                    f"**Latest change**  \n"
                    f"- Row: `{info.get('row')}`  \n"
                    f"- Column: `{info.get('col_name')}`  \n"
                    f"- Before: `{clip(info.get('before',''))}`  \n"
                    f"- After: `{clip(info.get('after',''))}`"
                )

            if info.get("event") == "error":
                debug_box.error("Error during translation")
                debug_box.code(info.get("error", "Unknown error"))
                if info.get("http_debug"):
                    debug_box.json(info["http_debug"])

        try:
            with st.spinner("Processing (live updates below)..."):
                process_workbook_inplace(
                    wb,
                    base_prompt=st.session_state.prompt_text,
                    dictionary_override=dictionary_override,
                    client=client,
                    use_class_guidance=use_class_guidance,
                    progress_cb=progress_cb,
                    update_every_cells=300,
                )

            out = BytesIO()
            wb.save(out)
            out.seek(0)

            st.success("Done.")
            if client:
                st.write(f"Token calls: {client.token_calls}")
                st.write(f"Generate calls: {client.generate_calls}")
                st.write(f"AI calls: {client.ai_calls}")
                st.write(f"Cache hits: {client.cache_hits}")

            st.download_button(
                "Download translated Excel",
                data=out.getvalue(),
                file_name=f"translated_{uploaded.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error("Run failed.")
            st.code(format_exception(e))
            if client and getattr(client, "last_http_debug", None):
                st.json(client.last_http_debug)
