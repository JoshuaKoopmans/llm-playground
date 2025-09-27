# app.py
"""
Streamlit chat app (Ollama-friendly)

- Queries endpoint for available models and populates the selector
- "Refresh models" button to re-query on demand
- Clears chat when model changes
- Blocks chat until selected model is warmed/loaded
- Emoji UI, sliders, system prompt
- Streaming with spinner that stops on first token
- No duplicate/greyed messages (rerun after saving reply)
- Image uploader that attaches base64 (data URI) directly into the chat
  ➜ uses structured content [{"type":"text",...}, {"type":"image_url",...}]
"""

from __future__ import annotations

from typing import List, TypedDict, Literal, cast, Any, Union
import mimetypes
import base64
import hashlib
import re

import streamlit as st
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

# =========================
# CONFIG
# =========================
APP_TITLE = "💬 Chat Playground"
# Built-in fallback list if /v1/models isn't available on your gateway
FALLBACK_MODEL_OPTIONS: List[str] = ["gemma3:12b", "gemma3:27b"]
DEFAULT_MODEL = FALLBACK_MODEL_OPTIONS[0]
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use concise markdown."

# =========================
# MESSAGE TYPE
# =========================
Role = Literal["system", "user", "assistant"]


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlPart(TypedDict):
    type: Literal["image_url"]
    image_url: str


ContentPart = Union[TextPart, ImageUrlPart]
Content = Union[str, List[ContentPart]]


class Message(TypedDict):
    # Required keys so Pylance is happy when accessing m["role"] / m["content"]
    role: Role
    content: Content  # can be a plain string or a list of content parts


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
        return FALLBACK_MODEL_OPTIONS  # return ids if ids else FALLBACK_MODEL_OPTIONS
    except Exception:
        return FALLBACK_MODEL_OPTIONS


def refresh_models_cache() -> None:
    """Clear the cached model list so the next call re-queries the endpoint."""
    fetch_available_models.clear()  # type: ignore[attr-defined]


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    # No type annotation on attribute assignment (avoids Pylance reportInvalidTypeForm)
    st.session_state.messages = []  # holds List[Message]
if "openai_model" not in st.session_state:
    st.session_state.openai_model = DEFAULT_MODEL
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "warming" not in st.session_state:
    st.session_state.warming = False

# Track last processed image (avoid duplicate insertions on rerun)
if "last_img_digest" not in st.session_state:
    st.session_state.last_img_digest = None


# =========================
# HELPERS
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


def guess_mime(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or "application/octet-stream"


# =========================
# SIDEBAR
# =========================
def sidebar_controls() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")

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

        # ---------- Image -> chat (base64, structured content) ----------
        st.divider()
        st.subheader("🖼️ Add image to chat")
        img_up = st.file_uploader(
            "Upload image…",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=False,
            key="img_uploader",
        )

        # As soon as a file is chosen, convert to base64, attach to chat (structured parts), rerun.
        if img_up is not None:
            img_bytes = img_up.read()
            digest = hashlib.md5(
                img_bytes + img_up.name.encode() + str(len(img_bytes)).encode()
            ).hexdigest()

            if st.session_state.last_img_digest != digest:
                mime = img_up.type or guess_mime(img_up.name)
                b64 = base64.b64encode(img_bytes).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"

                # Structured message (you can add a text part if you like)
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": data_uri},
                        ],
                    }
                )
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

    # Chat history
    for m in messages:
        with st.chat_message(m["role"]):
            c = m["content"]
            # If content is structured parts, render text and images nicely
            if isinstance(c, list):
                text_chunks: List[str] = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_chunks.append(str(part.get("text", "")))
                if text_chunks:
                    st.markdown("\n\n".join(text_chunks))
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = str(part.get("image_url", ""))
                        st.markdown(f"![uploaded image]({url})")
            else:
                # Plain string content
                st.markdown(str(c))


# =========================
# SANITIZER
# =========================
def strip_data_uris_from_markdown(text: Any) -> Any:
    """
    Remove inline base64 data URIs from markdown strings before sending to the model.
    If content is structured (list), return as-is.
    """
    if not isinstance(text, str):
        return text

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
        c = m["content"]
        sanitized = strip_data_uris_from_markdown(c)
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
