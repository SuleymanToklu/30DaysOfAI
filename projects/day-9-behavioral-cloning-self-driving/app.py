import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import joblib

TEXT_CONTENT = {
    'TR': {
        'page_title': "Yol Nesnesi Tanıma",
        'page_icon': "🛣️",
        'main_title': "🛣️ Görüntüdeki Yol Nesnesini Tanıma",
        'model_error': "Model dosyaları bulunamadı. Lütfen önce `train_model.py`'yi çalıştırın.",
        'tab1_name': "🧠 Nesne Tahmini",
        'tab2_name': "🎯 Proje Detayları",
        'tab1_header': "Canlı Analiz",
        'tab1_desc': "Bir yol fotoğrafı yükleyerek, modelin bu fotoğraftaki ana nesnenin ne olduğunu (araba, yaya vb.) tahmin etmesini sağlayın.",
        'uploader_label': "Bir fotoğraf seçin...",
        'image_caption': "Yüklenen Fotoğraf",
        'button_label': "Nesneyi Tanı",
        'spinner_text': "Fotoğraf analiz ediliyor...",
        'result_header': "🔮 Tahmin Sonucu",
        'metric_label_object': "Tanımlanan Nesne",
        'metric_label_confidence': "Modelin Güven Skoru",
        'info_high_confidence': "Model bu tahmininden oldukça emin görünüyor.",
        'info_medium_confidence': "Modelin tahmini orta düzeyde bir güvene sahip.",
        'info_low_confidence': "Model bu nesneyi tanımakta zorlandı veya emin değil.",
        'tab2_header': "Projenin Amacı ve Teknik Detaylar",
        'tab2_desc': """
        Bu projenin amacı, bir arabanın ön kamera görüntülerindeki çeşitli nesneleri (arabalar, yayalar, trafik işaretleri vb.) tanıyabilen bir derin öğrenme modeli oluşturmaktır.
        
        - **Model:** `MobileNetV2` mimarisi üzerine kurulu bir Evrişimli Sinir Ağı (CNN) kullanılmıştır. Bu model, Transfer Learning tekniği ile 'imagenet' üzerinde eğitilmiş ağırlıkları kullanarak daha hızlı ve etkili öğrenir.
        - **Veri Ön İşleme:** Her bir görüntü, modelin giriş boyutuna (`224x224`) uygun hale getirilmiş ve renk kanalları düzenlenmiştir.
        - **Problem Türü:** Bu bir **çok sınıflı sınıflandırma (multi-class classification)** problemidir. Model, görüntüdeki nesnenin önceden tanımlanmış kategorilerden hangisine ait olduğunu tahmin eder.
        """
    },
    'EN': {
        'page_title': "Road Object Recognition",
        'page_icon': "🛣️",
        'main_title': "🛣️ Road Object Recognition in Images",
        'model_error': "Model files not found. Please run `train_model.py` first.",
        'tab1_name': "🧠 Object Prediction",
        'tab2_name': "🎯 Project Details",
        'tab1_header': "Live Analysis",
        'tab1_desc': "Upload a road image and let the model predict the main object in it (e.g., car, pedestrian, etc.).",
        'uploader_label': "Choose an image...",
        'image_caption': "Uploaded Image",
        'button_label': "Recognize Object",
        'spinner_text': "Analyzing image...",
        'result_header': "🔮 Prediction Result",
        'metric_label_object': "Recognized Object",
        'metric_label_confidence': "Model Confidence Score",
        'info_high_confidence': "The model seems quite confident in this prediction.",
        'info_medium_confidence': "The model has a medium level of confidence in its prediction.",
        'info_low_confidence': "The model struggled to recognize this object or is not confident.",
        'tab2_header': "Project Goal and Technical Details",
        'tab2_desc': """
        The goal of this project is to create a deep learning model capable of recognizing various objects in front-camera images (such as cars, pedestrians, traffic signs, etc.).
        
        - **Model:** A Convolutional Neural Network (CNN) based on the `MobileNetV2` architecture was used. This model utilizes Transfer Learning with pre-trained weights from 'imagenet' for faster and more effective learning.
        - **Image Preprocessing:** Each image is resized to fit the model's input dimensions (`224x224`), and its color channels are adjusted accordingly.
        - **Problem Type:** This is a **multi-class classification** problem. The model predicts which of the predefined categories the object in the image belongs to.
        """
    }
}

st.set_page_config(page_title="Yol Nesnesi Tanıma", page_icon="🛣️", layout="wide")

st.sidebar.title("Language / Dil")
lang = st.sidebar.radio("Choose Language", ('TR', 'EN'), label_visibility="collapsed")
TEXT = TEXT_CONTENT[lang]

st.title(TEXT['main_title'])

@st.cache_resource
def load_model_and_encoder():
    try:
        model = keras.models.load_model('road_object_model.keras')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"{TEXT['model_error']}: {e}")
        return None, None

model, encoder = load_model_and_encoder()

if not model or not encoder:
    st.stop()

tab1, tab2 = st.tabs([TEXT['tab1_name'], TEXT['tab2_name']])

with tab1:
    st.header(TEXT['tab1_header'])
    st.write(TEXT['tab1_desc'])

    uploaded_file = st.file_uploader(TEXT['uploader_label'], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=TEXT['image_caption'], use_column_width=True, width=300)
        
        if st.button(TEXT['button_label']):
            with st.spinner(TEXT['spinner_text']):
                img_array = np.array(image)
                img_resized = cv2.resize(img_array, (224, 224))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_expanded = np.expand_dims(img_rgb, axis=0)
                
                prediction = model.predict(img_expanded)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = encoder.inverse_transform([predicted_class_index])[0]
                confidence = np.max(prediction) * 100

                st.subheader(TEXT['result_header'])
                st.metric(label=TEXT['metric_label_object'], value=str(predicted_class_name).capitalize())
                st.metric(label=TEXT['metric_label_confidence'], value=f"{confidence:.2f}%")

                if confidence > 80:
                    st.success(TEXT['info_high_confidence'])
                elif confidence > 50:
                    st.warning(TEXT['info_medium_confidence'])
                else:
                    st.error(TEXT['info_low_confidence'])

with tab2:
    st.header(TEXT['tab2_header'])
    st.markdown(TEXT['tab2_desc'])