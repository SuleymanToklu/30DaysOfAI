import gradio as gr
from analysis_engine import analyze_sentence, find_closest_words
from visualizations import plot_attention_heatmaps, plot_embedding_space

def process_and_visualize(model_name, sentence, layer_index):
    """
    Main controller function that runs the analysis and generates all plots and insights.
    """
    progress = gr.Progress(track_tqdm=True)
    progress(0, desc="Model analiz ediliyor...")

    analysis_data = analyze_sentence(model_name, sentence)

    if analysis_data is None:
        return None, None, "Lütfen bir cümle girip model seçin."

    progress(0.5, desc="Grafikler ve analizler oluşturuluyor...")
    
    # Ensure slider value is an integer for indexing.
    layer_index = int(layer_index)
    
    attention_fig = plot_attention_heatmaps(
        analysis_data["attention"],
        analysis_data["tokens"],
        layer_index
    )
    
    embedding_fig = plot_embedding_space(
        analysis_data["hidden_states"],
        analysis_data["tokens"]
    )

    # Generate the dynamic insight text.
    last_layer_embeddings = analysis_data["hidden_states"][-1].squeeze(0).detach().numpy()
    insight_text = find_closest_words(last_layer_embeddings, analysis_data["tokens"])

    return attention_fig, embedding_fig, insight_text

# Define the UI layout and components using Gradio Blocks.
with gr.Blocks(theme=gr.themes.Soft(), title="Transformer Gözlemcisi") as demo:
    gr.Markdown("# 🧠 Transformer Gözlemcisi")
    gr.Markdown(
        """
        **Hoş geldiniz!** Bu araç, bir cümlenin bir yapay zeka modeli tarafından nasıl "anlaşıldığını" görselleştirir. 
        Aşağıya bir cümle yazıp modeli seçerek analizi başlatın ve sekmelerdeki grafikleri inceleyerek modelin zihnindeki yolculuğa tanık olun.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ayarlar")
            model_dd = gr.Dropdown(
                label="Model Seçin",
                choices=["dbmdz/bert-base-turkish-cased", "bert-base-uncased", "distilbert-base-uncased"],
                value="dbmdz/bert-base-turkish-cased"
            )
            sentence_tb = gr.Textbox(
                label="Analiz Edilecek Cümle",
                value="Isparta, gülleri ve gölleri ile meşhur bir şehirdir.",
                lines=3
            )
            analyze_btn = gr.Button("Analizi Başlat", variant="primary")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Dikkat Desenleri"):
                    gr.Markdown(
                        """
                        **Bu grafik ne anlama geliyor?**
                        Bu ısı haritaları, modelin her bir kelime için cümlenin geri kalanındaki diğer kelimelere ne kadar "dikkat ettiğini" gösterir.
                        Parlak kareler, o iki kelime arasında güçlü bir anlamsal bağ kurulduğunu ifade eder. Her "dikkat kafası" farklı türde ilişkilere odaklanabilir.
                        """
                    )
                    layer_slider = gr.Slider(
                        label="İncelenecek Katman", minimum=0, maximum=11, step=1, value=5
                    )
                    attention_plot = gr.Plot()

                with gr.TabItem("Kelime Gömülmeleri Uzayı"):
                    gr.Markdown(
                        """
                        **Bu grafik ne anlama geliyor?**
                        Bu grafik, cümledeki kelime vektörlerinin 2 boyuta indirgenmiş halidir. 
                        Birbirine yakın duran kelimeler, modelin o kelimeleri anlamsal olarak benzer veya ilişkili gördüğü anlamına gelir.
                        """
                    )
                    embedding_insight_md = gr.Markdown()
                    embedding_plot = gr.Plot()
    
    # Connect the button click event to the controller function.
    analyze_btn.click(
        fn=process_and_visualize,
        inputs=[model_dd, sentence_tb, layer_slider],
        outputs=[attention_plot, embedding_plot, embedding_insight_md]
    )

# Launch the Gradio app.
if __name__ == "__main__":
    demo.launch()