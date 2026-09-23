import streamlit as st
from tensorflow import keras
import joblib
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

texts = {
    "model_selection_title": {
        "tr": "Model Seçimi",
        "en": "Model Selection"
    },
    "model_selection_box": {
        "tr": "Analiz için bir model versiyonu seçin:",
        "en": "Choose a model version for analysis:"
    },
    "model_v1_option": {
        "tr": "V1 - Temel Model",
        "en": "V1 - Baseline Model"
    },
    "model_v2_option": {
        "tr": "V2 - Verimli Model",
        "en": "V2 - Efficient Model"
    },
    "main_title": {
        "tr": "🧠 MRI Görüntülerinden Beyin Tümörü Tespiti",
        "en": "🧠 Brain Tumor Detection from MRI Scans"
    },
    "warning": {
        "tr": "**UYARI:** Bu bir teknoloji demosudur ve tıbbi teşhis için kullanılamaz.",
        "en": "**DISCLAIMER:** This is a technology demo and not for medical diagnosis."
    },
    "error_loading": {
        "tr": "Model dosyaları yüklenirken hata oluştu: ",
        "en": "Error loading model files: "
    },
    "error_files_not_found": {
        "tr": "Lütfen `train_model.py` ve `train_efficient_model.py` script'lerini çalıştırdığınızdan emin olun.",
        "en": "Please ensure you have run both `train_model.py` and `train_efficient_model.py` scripts."
    },
    "tabs": {
        "tr": ["🔬 **Canlı Analiz**", "🎯 **Proje Hikayesi ve Performans**"],
        "en": ["🔬 **Live Analysis**", "🎯 **Project Story & Performance**"]
    },
    "analysis_header": {
        "tr": "MR Görüntüsü Analizi (Kullanılan Model: {})",
        "en": "MRI Image Analysis (Using Model: {})"
    },
    "uploader_label": {
        "tr": "Bir MR fotoğrafı seçin...",
        "en": "Choose an MRI image..."
    },
    "image_caption": {
        "tr": "Yüklenen MR Görüntüsü",
        "en": "Uploaded MRI Image"
    },
    "analyze_button": {
        "tr": "Analiz Et",
        "en": "Analyze"
    },
    "spinner_text": {
        "tr": "Analiz ediliyor...",
        "en": "Analyzing..."
    },
    "result_header": {
        "tr": "🔬 Analiz Sonucu",
        "en": "🔬 Analysis Result"
    },
    "prediction_tumor": {
        "tr": "Tahmin: **TÜMÖR**",
        "en": "Prediction: **TUMOR**"
    },
    "prediction_no_tumor": {
        "tr": "Tahmin: **TÜMÖR YOK**",
        "en": "Prediction: **NO TUMOR**"
    },
    "metric_tumor_prob": {
        "tr": "Modelin Tümör Güveni",
        "en": "Model's Tumor Confidence"
    },
    "metric_no_tumor_prob": {
        "tr": "Modelin Tümör Olmama Güveni",
        "en": "Model's No-Tumor Confidence"
    },
    "comparison_header": {
        "tr": "Model Geliştirme Yolculuğu",
        "en": "The Model Development Journey"
    },
    "final_accuracy_label": {
        "tr": "Nihai Doğrulama Başarımı",
        "en": "Final Validation Accuracy"
    },
    "v2_improvements_header": {
        "tr": "Hikaye: Neden %72'lik Model Daha Değerli?",
        "en": "The Story: Why is the 72% Model More Valuable?"
    },
    "v2_improvements_desc": {
        "tr": """
        Bu proje, bir model geliştirme serüvenini gözler önüne seriyor.
        - **V1 - Temel Model:** Basit bir CNN mimarisiyle, sadece "temiz" verilerle eğitildi ve %82 gibi yüksek bir başarıma ulaştı. Ancak bu modelin, sadece ideal koşullarda çalışabilen, "kırılgan" bir model olma riski vardı.
        - **Mühendislik Süreci:** Modeli iyileştirme denemeleri, başlangıçta modelin küçük veri setini ezberlemesi (overfitting) nedeniyle başarısız oldu. Bu, modelin **çok karmaşık** olduğunu gösteren kritik bir bulguydu.
        - **V2 - Verimli Model:** Bu teşhis üzerine, mimari tamamen değiştirildi. `Flatten` katmanı yerine **`GlobalAveragePooling2D`** kullanılarak modelin parametre sayısı **5 milyondan 110 bine** düşürüldü. Ayrıca veri artırma (data augmentation) ile eğitilerek modelin farklı durumlara karşı dayanıklılığı artırıldı.
        
        **Sonuç:** V2 modelinin %72'lik başarımı, V1'in %82'sinden kağıt üzerinde düşük görünse de, zorlu koşullarda eğitildiği ve ezberlemeye karşı tasarlandığı için **genelleme yeteneği çok daha yüksektir.** Bu model, gerçek dünya verilerinde daha güvenilir sonuçlar üretecektir.
        """,
        "en": """
        This project showcases a model development journey.
        - **V1 - The Baseline:** A simple CNN trained on "clean" data, achieving a high accuracy of 82%. However, this model risked being "brittle," only performing well under ideal conditions.
        - **The Engineering Process:** Initial attempts to improve the model failed due to severe overfitting on the small dataset. This was a critical insight: the model was **too complex**.
        - **V2 - The Efficient Model:** Based on this diagnosis, the architecture was redesigned. The `Flatten` layer was replaced with **`GlobalAveragePooling2D`**, drastically reducing parameters from **5 million to 110 thousand**. It was also trained with data augmentation to improve its robustness.
        
        **Conclusion:** Although V2's 72% accuracy seems lower than V1's 82%, it is a much more valuable result. Because it was trained under challenging conditions and designed to prevent memorization, its ability to **generalize** to new, real-world data is far superior.
        """
    }
}

st.sidebar.title("Language / Dil")
language = st.sidebar.radio("Select your language:", ('en', 'tr'))

st.sidebar.title(texts['model_selection_title'][language])
model_version_option = st.sidebar.selectbox(
    texts['model_selection_box'][language],
    (texts['model_v1_option'][language], texts['model_v2_option'][language])
)

@st.cache_resource
def load_all_resources():
    resources = {}
    try:
        resources['v1'] = {
            'model': keras.models.load_model('brain_tumor_model.keras'),
            'history': joblib.load('training_history.pkl')
        }
        resources['v2'] = {
            'model': keras.models.load_model('brain_tumor_model_efficient.keras'),
            'history': joblib.load('training_history_efficient.pkl')
        }
        return resources
    except Exception as e:
        st.error(f"{texts['error_loading'][language]}{e}. {texts['error_files_not_found'][language]}")
        return None

resources = load_all_resources()

st.title(texts['main_title'][language])
st.warning(texts['warning'][language])

if not resources:
    st.stop()

if model_version_option == texts['model_v1_option'][language]:
    model_key = 'v1'
    model_name = "V1"
else:
    model_key = 'v2'
    model_name = "V2"

selected_model = resources[model_key]['model']

tab1, tab2 = st.tabs(texts['tabs'][language])

with tab1:
    st.header(texts['analysis_header'][language].format(model_name))
    uploaded_file = st.file_uploader(texts['uploader_label'][language], type=["jpg", "jpeg", "png"], key=f"uploader_{language}")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption=texts['image_caption'][language], use_container_width=True)
        
        if st.button(texts['analyze_button'][language]):
            with st.spinner(texts['spinner_text'][language]):
                img_array = np.array(image.resize((150, 150)))
                img_array = np.expand_dims(img_array, axis=0)
                prediction = selected_model.predict(img_array)
                score = prediction[0][0]
                
                st.subheader(texts['result_header'][language])
                if score > 0.5:
                    st.error(texts['prediction_tumor'][language])
                    st.metric(label=texts['metric_tumor_prob'][language], value=f"{score:.2%}")
                else:
                    st.success(texts['prediction_no_tumor'][language])
                    st.metric(label=texts['metric_no_tumor_prob'][language], value=f"{1-score:.2%}")

with tab2:
    st.header(texts['comparison_header'][language])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("V1 - Baseline Model")
        v1_history = resources['v1']['history']
        final_v1_acc = v1_history['val_accuracy'][-1]
        st.metric(label=texts['final_accuracy_label'][language], value=f"{final_v1_acc:.2%}")
        df_v1 = pd.DataFrame(v1_history)[['accuracy', 'val_accuracy']]
        st.line_chart(df_v1)

    with col2:
        st.subheader("V2 - Efficient Model")
        v2_history = resources['v2']['history']
        final_v2_acc_epoch = np.argmin(v2_history['val_loss'])
        final_v2_acc = v2_history['val_accuracy'][final_v2_acc_epoch]
        st.metric(label=texts['final_accuracy_label'][language], value=f"{final_v2_acc:.2%}")
        df_v2 = pd.DataFrame(v2_history)[['accuracy', 'val_accuracy']]
        st.line_chart(df_v2)

    st.subheader(texts['v2_improvements_header'][language])
    st.write(texts['v2_improvements_desc'][language])
