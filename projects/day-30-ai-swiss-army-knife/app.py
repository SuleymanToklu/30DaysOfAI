import gradio as gr
from transformers import pipeline
import torch
from diffusers import LDMTextToImagePipeline
from PIL import Image

print("Loading all models, this will take a moment...")
image_generator_pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
caption_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
transcriber_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
print("All models loaded successfully!")

def generate_image(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a text prompt.")
    with torch.no_grad():
        image = image_generator_pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

def generate_caption(image):
    if image is None:
        raise gr.Error("Please upload an image.")
    result = caption_pipe(image)
    return result[0]['generated_text']

def transcribe_speech(audio):
    if audio is None:
        return "Please record or upload an audio file."
    result = transcriber_pipe(audio)
    return result["text"]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), title="AI Swiss Army Knife") as demo:
    gr.Markdown("# 🇨🇭 AI Swiss Army Knife: The Ultimate AI Toolkit")
    gr.Markdown("A single app combining the power of a creative artist, a wise storyteller, and a diligent scribe. Built by Süleyman Toklu.")

    with gr.Tabs():
        with gr.TabItem("🎨 Görüntü Üretici", id=0):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input_1 = gr.Textbox(label="Prompt", lines=4, placeholder="A high-tech robot reading a book in a cozy library, digital art...")
                    submit_btn_1 = gr.Button("Generate Image", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["A cinematic shot of a an astronaut on a futuristic motorcycle on Mars"],
                            ["A raccoon wearing a suit and reading a newspaper on a park bench, photorealistic"],
                            ["A beautiful watercolor painting of a quiet, misty lake at dawn"]
                        ],
                        inputs=prompt_input_1
                    )
                with gr.Column(scale=1):
                    image_output_1 = gr.Image(label="Generated Image")

        with gr.TabItem("🖼️ Resimden Hikaye", id=1):
            with gr.Row():
                with gr.Column():
                    image_input_2 = gr.Image(type="pil", label="Upload an Image")
                    submit_btn_2 = gr.Button("Generate Story", variant="primary")
                with gr.Column():
                    text_output_2 = gr.Textbox(label="The AI's Story")

        with gr.TabItem("🎙️ Sesten Metne", id=2):
            audio_input_3 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or Upload")
            submit_btn_3 = gr.Button("Transcribe", variant="primary")
            text_output_3 = gr.Textbox(label="Transcribed Text")
            
            gr.Examples(
                examples=[
                    ["ornek1.wav"],
                    ["ornek2.wav"]
                ],
                inputs=audio_input_3
            )
    
    submit_btn_1.click(fn=generate_image, inputs=prompt_input_1, outputs=image_output_1, api_name="generate_image")
    submit_btn_2.click(fn=generate_caption, inputs=image_input_2, outputs=text_output_2, api_name="generate_caption")
    submit_btn_3.click(fn=transcribe_speech, inputs=audio_input_3, outputs=text_output_3, api_name="transcribe_speech")

if __name__ == "__main__":
    demo.launch()