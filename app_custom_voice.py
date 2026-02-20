# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# TTS (CustomVoice) Mode - Generate speech with predefined speakers
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

# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


# ============================================================================
# MODEL LOADING - CustomVoice models - both sizes
# ============================================================================
print("Loading CustomVoice models...")

print("Loading CustomVoice 0.6B model...")
custom_voice_model_0_6b = Qwen3TTSModel.from_pretrained(
    get_model_path("CustomVoice", "0.6B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    # token=HF_TOKEN,
    attn_implementation="kernels-community/flash-attn3",
)

print("Loading CustomVoice 1.7B model...")
custom_voice_model_1_7b = Qwen3TTSModel.from_pretrained(
    get_model_path("CustomVoice", "1.7B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    # token=HF_TOKEN,
    attn_implementation="kernels-community/flash-attn3",
)

print("CustomVoice models loaded successfully!")

# Model lookup dictionary for easy access
CUSTOM_VOICE_MODELS = {
    "0.6B": custom_voice_model_0_6b,
    "1.7B": custom_voice_model_1_7b,
}


@spaces.GPU(duration=60)
def generate_custom_voice(text, language, speaker, instruct, model_size, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        tts = CUSTOM_VOICE_MODELS[model_size]
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Generation completed successfully!"
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

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS CustomVoice") as demo:
        gr.Markdown(
            """
# Qwen3-TTS - TTS (CustomVoice)
Generate speech with predefined speakers and optional style instructions.

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
"""
        )

        gr.Markdown("### Text-to-Speech with Predefined Speakers")
        with gr.Row():
            with gr.Column(scale=2):
                tts_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=4,
                    placeholder="Enter the text you want to convert to speech...",
                    value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities."
                )
                with gr.Row():
                    tts_language = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGES,
                        value="English",
                        interactive=True,
                    )
                    tts_speaker = gr.Dropdown(
                        label="Speaker",
                        choices=SPEAKERS,
                        value="Ryan",
                        interactive=True,
                    )
                with gr.Row():
                    tts_instruct = gr.Textbox(
                        label="Style Instruction (Optional)",
                        lines=2,
                        placeholder="e.g., Speak in a cheerful and energetic tone",
                    )
                    tts_model_size = gr.Dropdown(
                        label="Model Size",
                        choices=MODEL_SIZES,
                        value="1.7B",
                        interactive=True,
                    )
                tts_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column(scale=2):
                tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

        tts_btn.click(
            generate_custom_voice,
            inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
            outputs=[tts_audio_out, tts_status],
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
