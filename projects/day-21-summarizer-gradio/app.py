import gradio as gr
from transformers import pipeline

# Load the summarization model once when the app starts.
# This prevents reloading the model on every single function call.
summarizer = pipeline("summarization", model="t5-small")

def summarize_text(text_to_summarize, min_length, max_length):
    """
    Summarizes the input text using the pre-loaded T5 model.
    """
    # Ensure inputs are correctly typed for the pipeline.
    min_len = int(min_length)
    max_len = int(max_length)

    # Perform summarization. truncation=True handles texts longer than the model's max input size.
    summary = summarizer(text_to_summarize, min_length=min_len, max_length=max_len, truncation=True)
    
    # The pipeline returns a list of dictionaries, we need to extract the summary text.
    return summary[0]['summary_text']

# Pre-defined example text to make testing easier.
example_text = """
Yapay zeka (AI), genellikle insan zekasını taklit eden ve problem çözme, öğrenme, algılama gibi görevleri yerine getirebilen sistemlerin veya makinelerin oluşturulmasıyla ilgilenen bir bilgisayar bilimi dalıdır. 
Son yıllarda, özellikle derin öğrenme ve büyük veri setlerinin yükselişiyle birlikte yapay zeka alanında devrim niteliğinde gelişmeler yaşanmıştır. 
Bu teknolojiler, doğal dil işleme (NLP), bilgisayarlı görü ve otonom sürüş gibi birçok alanda pratik uygulamalara dönüşmüştür. 
AI'nın amacı, sadece mevcut görevleri otomatikleştirmek değil, aynı zamanda insan yeteneklerini aşan yeni çözümler ve keşifler yapmaktır.
"""

# Define the Gradio interface.
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🚀 Akıllı Metin Özetleyici
        Bu uygulama, [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) kütüphanesindeki `t5-small` modelini kullanarak girdiğiniz uzun metinleri özetler.
        Aşağıdaki metin kutusuna bir makale, rapor veya herhangi bir uzun metin yapıştırın ve sonucu görün!
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(lines=15, label="Özetlenecek Metin", placeholder="Buraya uzun bir metin yapıştırın...")
            summarize_button = gr.Button("Özeti Oluştur", variant="primary")
        
        with gr.Column(scale=1):
            min_length_slider = gr.Slider(minimum=20, maximum=100, value=30, step=5, label="Özetin Minimum Uzunluğu")
            max_length_slider = gr.Slider(minimum=50, maximum=250, value=80, step=10, label="Özetin Maksimum Uzunluğu")
            text_output = gr.Textbox(label="Oluşturulan Özet", lines=10, interactive=False)

    gr.Examples(
        examples=[[example_text, 30, 80]],
        inputs=[text_input, min_length_slider, max_length_slider]
    )

    # Connect the button click event to the summarization function.
    summarize_button.click(
        fn=summarize_text, 
        inputs=[text_input, min_length_slider, max_length_slider], 
        outputs=text_output
    )

# Launch the web interface.
iface.launch()