import gradio as gr
import torch
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

image_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-large",
    device=device
)

tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

speaker_embeddings = torch.zeros((1, 512)).to(device)


def image_to_story(image):
    if image is None:
        raise gr.Error("Lütfen bir resim yükleyin.")

    caption_result = image_captioner(image)
    generated_text = caption_result[0]['generated_text']

    inputs = tts_processor(text=generated_text, return_tensors="pt").to(device)
    speech_spectrogram = tts_model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings
    )
    
    waveform = vocoder(speech_spectrogram).detach().cpu().numpy()
    
    sampling_rate = 16000
    
    return generated_text, (sampling_rate, waveform)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️🔊 Görüntüden Sesli Hikaye")
    gr.Markdown("Bir resim yükleyin ve yapay zekanın hem resmi anlatmasını hem de anlattığı hikayeyi size okumasını dinleyin.")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Görüntü Yükle")
        
        with gr.Column():
            text_output = gr.Textbox(label="Oluşturulan Hikaye (Metin)", interactive=False)
            audio_output = gr.Audio(label="Oluşturulan Sesli Hikaye")

    generate_button = gr.Button("Hikayeyi Oluştur ve Seslendir", variant="primary")
    
    generate_button.click(
        fn=image_to_story,
        inputs=[image_input],
        outputs=[text_output, audio_output]
    )
    
    gr.Examples(
        examples=[
            ["astronaut.png"], 
            ["cat.png"]
        ],
        inputs=image_input,
        label="Örnek Resimler"
    )

demo.launch()
