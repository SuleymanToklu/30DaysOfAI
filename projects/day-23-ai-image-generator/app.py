import gradio as gr
import torch
from diffusers import DiffusionPipeline

# Modelimizi yüklüyoruz. 'torch_dtype' yerine güncel olan 'dtype' kullanıldı.
pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256", dtype=torch.float32)

# Fonksiyon, bir metin ve bir sayı (num_steps) olmak üzere iki argüman alıyor.
def generate_image(prompt, num_steps):
    print(f"Generating image for prompt: '{prompt}' with {num_steps} steps.")
    # Kullanıcıdan gelen num_steps değerini integer'a çevirip kullanıyoruz
    image = pipe(prompt, num_inference_steps=int(num_steps)).images[0]
    return image

# Gradio Arayüzü
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Görselleştirmek istediğiniz metni girin:",
            placeholder="Örn: A robot astronaut playing a guitar"
        ),
        gr.Slider(
            label="Kalite/Hız Adımları (Steps)",
            minimum=10,
            maximum=50,
            step=1,
            value=25 # Varsayılan değer
        )
    ],
    outputs=gr.Image(label="Oluşturulan Görsel"),
    title="🚀 Hızlı Yapay Zeka Görüntü Üretici (Optimize)",
    description="Adım sayısını (Steps) düşürerek görsel oluşturma hızını artırabilirsiniz. Düşük adım sayısı = Hızlı ama düşük kalite.",
    
    # Her örnek artık hem bir metin hem de bir sayı içeriyor.
    examples=[
        ["A cinematic shot of a Corgi wearing a superhero cape", 20],
        ["A watercolor painting of a cozy library with a fireplace", 30],
        ["An abstract logo for a tech company named 'QuantumLeap'", 25]
    ]
)

# Uygulamayı başlat
iface.launch()