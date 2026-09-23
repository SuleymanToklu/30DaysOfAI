import gradio as gr
from transformers import pipeline
from PIL import Image


try:
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    model_loaded = True
    loading_error = None
except Exception as e:
    print(f"Error loading pipeline: {e}")
    caption_pipeline = None
    model_loaded = False
    loading_error = str(e) 

def generate_caption(image):
    """
    Takes an image and returns a generated caption.
    """
    if not model_loaded:
        raise gr.Error(f"Model yüklenemedi. Lütfen daha sonra tekrar deneyin veya sunucu yöneticisine başvurun. Hata: {loading_error}")
    
    if image is None:
        raise gr.Error("Lütfen bir resim yükleyin.")
    
    try:
        result = caption_pipeline(image)
        caption = result[0]['generated_text']
        return caption
    except Exception as e:
        return f"Açıklama oluşturulurken bir hata oluştu: {e}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Yapay Zeka Anlatıcı: Resimden Hikaye Üretici")
    gr.Markdown("Bir resim yükleyin ve yapay zekanın bu resmi kelimelerle nasıl anlattığını görün. Model yüklenemezse lütfen bekleyin veya daha sonra tekrar deneyin.")
    
    if not model_loaded:
        gr.Warning(f"**UYARI:** Yapay zeka modeli yüklenirken bir hata oluştu. Uygulama çalışmayabilir. Hata: {loading_error}")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Görüntü Yükle")
        caption_output = gr.Textbox(label="Yapay Zekanın Oluşturduğu Hikaye", interactive=False)

    generate_button = gr.Button("Resmi Anlat!", variant="primary")
    
    generate_button.click(
        fn=generate_caption,
        inputs=[image_input],
        outputs=[caption_output]
    )

    gr.Examples(
        examples=[
            ["cat.png"],
            ["astronaut.png"],
            ["isparta.png"] 
        ],
        inputs=image_input,
        label="Örnek Resimleri Deneyin"
    )

if __name__ == "__main__":
    demo.launch()