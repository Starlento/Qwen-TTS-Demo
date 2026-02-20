# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# Voice Design Mode - Create custom voices using natural language descriptions
import os
import spaces
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download, login
from qwen_tts import Qwen3TTSModel

# HF_TOKEN = os.environ.get('HF_TOKEN')
# login(token=HF_TOKEN)

# Language choices for Voice Design
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


# ============================================================================
# MODEL LOADING - Voice Design model only (1.7B)
# ============================================================================
print("Loading VoiceDesign 1.7B model...")
voice_design_model = Qwen3TTSModel.from_pretrained(
    get_model_path("VoiceDesign", "1.7B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    # token=HF_TOKEN,
    attn_implementation="kernels-community/flash-attn3",
)
print("VoiceDesign model loaded successfully!")


@spaces.GPU(duration=60)
def generate_voice_design(text, language, voice_description, progress=gr.Progress(track_tqdm=True)):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        wavs, sr = voice_design_model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice design generation completed successfully!"
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

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Voice Design") as demo:
        gr.Markdown(
            """
# Qwen3-TTS - Voice Design
Create custom voices using natural language descriptions.

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
"""
        )

        gr.Markdown("### Create Custom Voice with Natural Language")
        with gr.Row():
            with gr.Column(scale=2):
                design_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=4,
                    placeholder="Enter the text you want to convert to speech...",
                    value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                )
                design_language = gr.Dropdown(
                    label="Language",
                    choices=LANGUAGES,
                    value="Auto",
                    interactive=True,
                )
                design_instruct = gr.Textbox(
                    label="Voice Description",
                    lines=3,
                    placeholder="Describe the voice characteristics you want...",
                    value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                )
                design_btn = gr.Button("Generate with Custom Voice", variant="primary")

            with gr.Column(scale=2):
                design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                design_status = gr.Textbox(label="Status", lines=2, interactive=False)

        design_btn.click(
            generate_voice_design,
            inputs=[design_text, design_language, design_instruct],
            outputs=[design_audio_out, design_status],
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
