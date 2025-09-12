# app.py
"""
Streamlit chat app (Ollama-friendly) with dynamic model list + Docling parsing

- Queries endpoint for available models and populates the selector
- "Refresh models" button to re-query on demand
- Clears chat when model changes
- Blocks chat until selected model is warmed/loaded
- Emoji UI, sliders, system prompt
- Docling converter (PDF/DOCX -> Markdown/Text) with preview + insert to chat
- Streaming with spinner that stops on first token
- No duplicate/greyed messages (rerun after saving reply)
- Image uploader that attaches base64 (data URI) directly into the chat
"""

from __future__ import annotations

from typing import List, TypedDict, Literal, cast, Tuple, Dict, Any, Optional
import io
import json
import mimetypes
import base64
import hashlib

import requests
import streamlit as st
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

# =========================
# CONFIG
# =========================
APP_TITLE = "💬 Chat Playground"
# Built-in fallback list if /v1/models isn't available on your gateway
FALLBACK_MODEL_OPTIONS: List[str] = [
    "gemma3:12b",
    "gpt-oss:20b",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]
DEFAULT_MODEL = FALLBACK_MODEL_OPTIONS[0]
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use concise markdown."

# Docling server (adjust if needed)
DOCLING_BASE_URL = "http://127.0.0.1:5001"
DOCLING_ENDPOINT = f"{DOCLING_BASE_URL}/v1/convert/file"

# =========================
# MESSAGE TYPE
# =========================
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


# =========================
# CLIENT
# =========================
@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    """Initialize the OpenAI-compatible client (e.g., an Ollama gateway) from secrets."""
    api_key = st.secrets["openai"]["api_key"]
    gpt_host = st.secrets["openai"]["host"]
    return OpenAI(base_url=gpt_host, api_key=api_key)


client = get_client()


# =========================
# DYNAMIC MODEL LIST
# =========================
@st.cache_data(show_spinner=False, ttl=30)
def fetch_available_models() -> List[str]:
    """
    Query the endpoint for available models.
    Returns sorted model IDs. Falls back to a built-in list on error.
    """
    try:
        resp = client.models.list()
        ids = [m.id for m in getattr(resp, "data", []) if getattr(m, "id", None)]
        ids = sorted(set(ids))
        return ids if ids else FALLBACK_MODEL_OPTIONS
    except Exception:
        return FALLBACK_MODEL_OPTIONS


def refresh_models_cache() -> None:
    """Clear the cached model list so the next call re-queries the endpoint."""
    fetch_available_models.clear()  # type: ignore[attr-defined]


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # holds List[Message]
if "openai_model" not in st.session_state:
    st.session_state.openai_model = DEFAULT_MODEL
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "warming" not in st.session_state:
    st.session_state.warming = False

# Docling parse cache (latest results)
if "docling_result" not in st.session_state:
    st.session_state.docling_result = {"md": None, "text": None, "raw": None}
# Track whether current Docling MD was inserted
if "docling_md_hash" not in st.session_state:
    st.session_state.docling_md_hash = None
if "docling_md_inserted" not in st.session_state:
    st.session_state.docling_md_inserted = False

# Track last processed image (avoid duplicate insertions on rerun)
if "last_img_digest" not in st.session_state:
    st.session_state.last_img_digest = None


# =========================
# WARM-UP HELPER
# =========================
def warm_up_model(model: str) -> None:
    """Force-load the model into memory with a tiny one-shot call."""
    client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Say 'ready'"}],
        max_tokens=1,
        temperature=0,
        stream=False,
    )


# =========================
# DOCLING HELPERS
# =========================
def guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


def docling_convert_file(
    uploaded,  # st.uploaded_file_manager.UploadedFile
    *,
    ocr_engine: str = "easyocr",
    pdf_backend: str = "dlparse_v2",
    from_formats: Tuple[str, ...] = ("pdf", "docx"),
    force_ocr: bool = False,
    image_export_mode: str = "embedded",
    ocr_langs: Tuple[str, ...] = ("en",),
    table_mode: str = "fast",
    abort_on_error: bool = False,
    to_formats: Tuple[str, ...] = ("md", "text"),
    do_ocr: bool = True,
) -> Dict[str, Any]:
    """
    POST multipart/form-data to Docling's /v1/convert/file.
    Mirrors the provided curl. Returns parsed JSON.
    """
    file_bytes = uploaded.read()
    filename = uploaded.name
    mime = uploaded.type or guess_mime(filename)

    data: List[Tuple[str, str]] = [
        ("ocr_engine", ocr_engine),
        ("pdf_backend", pdf_backend),
        ("force_ocr", "true" if force_ocr else "false"),
        ("image_export_mode", image_export_mode),
        ("table_mode", table_mode),
        ("abort_on_error", "true" if abort_on_error else "false"),
        ("do_ocr", "true" if do_ocr else "false"),
    ]
    for fr in from_formats:
        data.append(("from_formats", fr))
    for fmt in to_formats:
        data.append(("to_formats", fmt))
    for lang in ocr_langs:
        data.append(("ocr_lang", lang))

    files = {"files": (filename, io.BytesIO(file_bytes), mime)}
    headers = {"accept": "application/json"}

    resp = requests.post(
        DOCLING_ENDPOINT, data=data, files=files, headers=headers, timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def pick_docling_outputs(
    payload: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract Markdown/Text from Docling JSON. Tries a few common shapes:
    - {"document": {"md_content": "...", "text_content": "..."}}
    - {"md": "...", "text": "..."}
    - {"results": [{"format":"md","content":"..."}, ...]}
    - {"md":{"content":"..."}, "text":{"content":"..."}}
    Returns (md, text).
    """
    md: Optional[str] = None
    txt: Optional[str] = None

    if isinstance(payload.get("document"), dict):
        if "md_content" in payload["document"]:
            md = payload["document"]["md_content"]
        if "text_content" in payload["document"]:
            txt = payload["document"]["text_content"]

    if not (md or txt):
        seq = payload.get("results") or payload.get("outputs") or payload.get("data")
        if isinstance(seq, list):
            for item in seq:
                if not isinstance(item, dict):
                    continue
                fmt = (item.get("format") or item.get("type") or "").lower()
                content = item.get("content")
                if isinstance(content, str):
                    if fmt == "md" and md is None:
                        md = content
                    elif fmt in ("text", "txt") and txt is None:
                        txt = content

    if not (md or txt):
        maybe_md = payload.get("md") or payload.get("markdown")
        if isinstance(maybe_md, dict) and isinstance(maybe_md.get("content"), str):
            md = maybe_md["content"]
        maybe_txt = payload.get("text")
        if isinstance(maybe_txt, dict) and isinstance(maybe_txt.get("content"), str):
            txt = maybe_txt["content"]

    return md, txt


# =========================
# SIDEBAR
# =========================
def sidebar_controls() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")

        # Dynamic model list with refresh
        cols = st.columns([3, 1])
        with cols[1]:
            if st.button("🔄", help="Refresh models"):
                refresh_models_cache()
                st.rerun()
        with cols[0]:
            available_models = fetch_available_models()

        # Ensure the selected model is present; otherwise switch to the first available
        if st.session_state.openai_model not in available_models:
            st.session_state.openai_model = available_models[0]
            st.session_state.messages = []
            st.session_state.model_ready = False
            st.session_state.warming = False
            st.session_state.loaded_model = None

        prev_model = st.session_state.openai_model
        model = st.selectbox(
            "🤖 Model",
            options=available_models,
            index=available_models.index(prev_model),
        )

        # If model changed, clear chat and reset warm-up state
        if model != st.session_state.openai_model:
            st.session_state.openai_model = model
            st.session_state.messages = []  # Clear chat on model change
            st.session_state.model_ready = False
            st.session_state.warming = False
            st.session_state.loaded_model = None
            st.rerun()

        temperature = st.slider("🔥 Temperature", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.7, 0.05)
        max_tokens = st.number_input(
            "📝 Max tokens (0 = default)", min_value=0, value=0, step=50
        )

        system_prompt = st.text_area(
            "📜 System prompt", value=DEFAULT_SYSTEM_PROMPT, height=100
        )

        st.divider()
        st.subheader("📂 Docling: parse PDF/DOCX")
        up = st.file_uploader(
            "Choose file…", type=["pdf", "docx"], accept_multiple_files=False
        )

        with st.expander("Docling options"):
            do_ocr = st.checkbox("Run OCR", value=True)
            ocr_engine = st.selectbox("OCR engine", ["easyocr"], index=0)
            ocr_langs_raw = st.text_input("OCR languages (comma-separated)", value="en")
            table_mode = st.selectbox("Table mode", ["fast", "accurate"], index=0)
            pdf_backend = st.selectbox("PDF backend", ["dlparse_v2"], index=0)
            image_export_mode = st.selectbox("Image export", ["embedded"], index=0)
            to_md = st.checkbox("Output Markdown", value=True)
            to_txt = st.checkbox("Output Text", value=True)

        if st.button(
            "🧾 Convert with Docling", use_container_width=True, disabled=up is None
        ):
            if up is None:
                st.warning("Please select a file first.")
            else:
                with st.spinner("Converting with Docling…"):
                    try:
                        langs = tuple(
                            [s.strip() for s in ocr_langs_raw.split(",") if s.strip()]
                        ) or ("en",)
                        to_formats = tuple(
                            [fmt for fmt, on in (("md", to_md), ("text", to_txt)) if on]
                        ) or ("md", "text")
                        res = docling_convert_file(
                            up,
                            ocr_engine=ocr_engine,
                            pdf_backend=pdf_backend,
                            from_formats=("pdf", "docx"),
                            force_ocr=False,
                            image_export_mode=image_export_mode,
                            ocr_langs=langs,
                            table_mode=table_mode,
                            abort_on_error=False,
                            to_formats=to_formats,
                            do_ocr=do_ocr,
                        )
                        md_out, txt_out = pick_docling_outputs(res)
                        st.session_state.docling_result = {
                            "md": md_out,
                            "text": txt_out,
                            "raw": res,
                        }
                        # reset one-shot button state for this specific markdown
                        cur_hash = hash(md_out) if md_out else None
                        st.session_state.docling_md_hash = cur_hash
                        st.session_state.docling_md_inserted = False
                        st.success("✅ Parsed with Docling!")
                    except requests.HTTPError as e:
                        st.error(
                            f"❌ Docling HTTP error: {e.response.status_code} {e.response.text}"
                        )
                    except requests.ConnectionError:
                        st.error(
                            f"❌ Could not reach Docling at {DOCLING_ENDPOINT}. Is the server running?"
                        )
                    except Exception as e:
                        st.error(f"❌ Docling error: {e}")

        # ---------- Image -> chat (base64) ----------
        st.divider()
        st.subheader("🖼️ Add image to chat")
        img_up = st.file_uploader(
            "Upload image…",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=False,
            key="img_uploader",
        )

        # As soon as a file is chosen, convert to base64, attach to chat, rerun.
        if img_up is not None:
            img_bytes = img_up.read()
            digest = hashlib.md5(
                img_bytes + img_up.name.encode() + str(len(img_bytes)).encode()
            ).hexdigest()

            if st.session_state.last_img_digest != digest:
                mime = img_up.type or guess_mime(img_up.name)
                b64 = base64.b64encode(img_bytes).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"

                content = (
                    f"🖼️ **Image:** `{img_up.name}` ({mime}, {len(img_bytes)} bytes)\n\n"
                    f"![{img_up.name}]({data_uri})"
                )

                st.session_state.messages.append({"role": "user", "content": content})
                st.session_state.last_img_digest = digest
                st.success("✅ Image attached to chat")
                st.rerun()

        # Clear chat button
        st.divider()
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "model": st.session_state.openai_model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "system_prompt": system_prompt.strip(),
    }


# =========================
# RENDER HISTORY
# =========================
def render_history(messages: List[Message]) -> None:
    st.title(APP_TITLE)

    # Show Docling panel (if any results)
    md = st.session_state.docling_result.get("md")
    raw = st.session_state.docling_result.get("raw")

    if md or raw:
        with st.expander("🧾 Docling result", expanded=False):
            st.subheader("Markdown")
            if md:
                st.markdown(md)
            else:
                st.caption("No Markdown output.")

            with st.expander("Raw JSON"):
                st.code(json.dumps(raw, ensure_ascii=False, indent=2), language="json")

        # One-shot "Insert Markdown" button logic
        if md:
            cur_hash = hash(md)
            # If the parsed MD changed since last time, re-enable the button
            if st.session_state.docling_md_hash != cur_hash:
                st.session_state.docling_md_hash = cur_hash
                st.session_state.docling_md_inserted = False

            if not st.session_state.docling_md_inserted:
                if st.button("➕ Insert Markdown into chat", key="insert_md"):
                    st.session_state.messages.append({"role": "user", "content": md})
                    st.session_state.docling_md_inserted = True
                    # Optional tiny flash; remove if you want zero UI message
                    # st.toast("Inserted Markdown into chat")
                    st.rerun()  # hides the button immediately

    # Chat history
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


import re


def strip_data_uris_from_markdown(text: str) -> str:
    """
    Remove inline base64 data URIs from markdown before sending to the model.
    Preserves a readable placeholder so the model knows an image was attached.
    """
    # Replace markdown images that use data URIs
    pattern_md = r"!\[([^\]]*)\]\((data:[^)]+)\)"
    text = re.sub(
        pattern_md, r"![\1](attachment:base64-omitted)", text, flags=re.IGNORECASE
    )

    # Also scrub any stray data URIs that aren't in image markdown
    pattern_raw = r"data:[a-z0-9.+/-]+;base64,[A-Za-z0-9+/=\s]+"
    text = re.sub(pattern_raw, "[base64 omitted]", text, flags=re.IGNORECASE)

    return text


# =========================
# BUILD PAYLOAD
# =========================
def build_payload(messages: List[Message], system_prompt: str) -> List[Message]:
    payload: List[Message] = []
    if system_prompt:
        payload.append({"role": "system", "content": system_prompt})

    for m in messages:
        sanitized = strip_data_uris_from_markdown(m["content"])
        payload.append({"role": m["role"], "content": sanitized})

    return payload


# =========================
# STREAM REPLY
# =========================
def stream_assistant_reply(
    *,
    model: str,
    messages: List[Message],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    with st.chat_message("assistant"):
        thinking_ph = st.empty()
        thinking_ph.markdown("🤔 *Thinking…*")

        try:
            sdk_messages = cast(List[ChatCompletionMessageParam], messages)

            if max_tokens > 0:
                stream = client.chat.completions.create(
                    model=model,
                    messages=sdk_messages,
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            else:
                stream = client.chat.completions.create(
                    model=model,
                    messages=sdk_messages,
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                )

            first_token = [True]
            parts: List[str] = []

            def token_generator():
                for chunk in stream:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta and getattr(delta, "content", None):
                        if first_token[0]:
                            thinking_ph.empty()  # stop spinner at first token
                            first_token[0] = False
                        text = delta.content
                        parts.append(text)
                        yield text

            final_text = st.write_stream(token_generator())
            return final_text if isinstance(final_text, str) else "".join(parts)

        except OpenAIError as e:
            thinking_ph.empty()
            st.error(f"❌ Model error: {e}")
            return ""
        except Exception as e:
            thinking_ph.empty()
            st.error(f"💥 Unexpected error: {e}")
            return ""


# =========================
# MAIN + WARM-UP GATE
# =========================
def main() -> None:
    settings = sidebar_controls()

    needs_warmup = (not st.session_state.model_ready) or (
        st.session_state.loaded_model != settings["model"]
    )

    if needs_warmup and not st.session_state.warming:
        st.session_state.warming = True

    if st.session_state.warming:
        st.title(APP_TITLE)
        with st.status("📦 Loading model into memory…", expanded=False) as status:
            try:
                warm_up_model(settings["model"])
                status.update(label="✅ Model loaded and ready!", state="complete")
                st.session_state.model_ready = True
                st.session_state.loaded_model = settings["model"]
                st.session_state.warming = False
            except OpenAIError as e:
                st.session_state.model_ready = False
                st.session_state.warming = False
                status.update(label="⚠️ Failed to load model", state="error")
                err_txt = str(e).lower()
                if "not found" in err_txt or "try pulling" in err_txt:
                    st.error(
                        f"❌ The model **{settings['model']}** is not available locally.\n\n"
                        f"Run this in your terminal first:\n\n"
                        f"```bash\nollama pull {settings['model']}\n```"
                    )
                return
            except Exception as e:
                st.session_state.model_ready = False
                st.session_state.warming = False
                status.update(label="⚠️ Failed to load model", state="error")
                st.error(f"Unexpected error while loading '{settings['model']}': {e}")
                return

        # Rerun after closing status on success
        st.rerun()
        return

    # Model ready: show history + chat input
    render_history(st.session_state.messages)

    user_text = st.chat_input("💭 Type your message…")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    payload = build_payload(st.session_state.messages, settings["system_prompt"])
    reply_text = stream_assistant_reply(
        model=settings["model"],
        messages=payload,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_tokens=settings["max_tokens"],
    )

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    st.rerun()


if __name__ == "__main__":
    main()
