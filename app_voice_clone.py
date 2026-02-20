# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# Voice Clone Mode - Clone any voice from a reference audio
import os
import spaces
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login
from qwen_tts import Qwen3TTSModel

# HF_TOKEN = os.environ.get('HF_TOKEN')
# login(token=HF_TOKEN)

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]

# Language choices
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


# ============================================================================
# MODEL LOADING - Base (Voice Clone) models - both sizes
# ============================================================================
print("Loading Base models for Voice Clone...")

print("Loading Base 0.6B model...")
base_model_0_6b = Qwen3TTSModel.from_pretrained(
    get_model_path("Base", "0.6B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    # token=HF_TOKEN,
    attn_implementation="kernels-community/flash-attn3",
)

print("Loading Base 1.7B model...")
base_model_1_7b = Qwen3TTSModel.from_pretrained(
    get_model_path("Base", "1.7B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    # token=HF_TOKEN,
    attn_implementation="kernels-community/flash-attn3",
)

print("Base models loaded successfully!")

# Model lookup dictionary for easy access
BASE_MODELS = {
    "0.6B": base_model_0_6b,
    "1.7B": base_model_1_7b,
}


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


@spaces.GPU(duration=60)
def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        tts = BASE_MODELS[model_size]
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Voice Clone") as demo:
        gr.Markdown(
            """
# Qwen3-TTS - Voice Clone (Base)
Clone any voice from a reference audio sample.

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
"""
        )

        gr.Markdown("### Clone Voice from Reference Audio")
        with gr.Row():
            with gr.Column(scale=2):
                clone_ref_audio = gr.Audio(
                    label="Reference Audio (Upload a voice sample to clone)",
                    type="numpy",
                )
                clone_ref_text = gr.Textbox(
                    label="Reference Text (Transcript of the reference audio)",
                    lines=2,
                    placeholder="Enter the exact text spoken in the reference audio...",
                )
                clone_xvector = gr.Checkbox(
                    label="Use x-vector only (No reference text needed, but lower quality)",
                    value=False,
                )

            with gr.Column(scale=2):
                clone_target_text = gr.Textbox(
                    label="Target Text (Text to synthesize with cloned voice)",
                    lines=4,
                    placeholder="Enter the text you want the cloned voice to speak...",
                )
                with gr.Row():
                    clone_language = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGES,
                        value="Auto",
                        interactive=True,
                    )
                    clone_model_size = gr.Dropdown(
                        label="Model Size",
                        choices=MODEL_SIZES,
                        value="1.7B",
                        interactive=True,
                    )
                clone_btn = gr.Button("Clone & Generate", variant="primary")

        with gr.Row():
            clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
            clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

        clone_btn.click(
            generate_voice_clone,
            inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
            outputs=[clone_audio_out, clone_status],
        )

        gr.Markdown(
            """
---
**Note**: This demo uses HuggingFace Spaces Zero GPU. Each generation has a time limit.
For longer texts, please split them into smaller segments.
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
