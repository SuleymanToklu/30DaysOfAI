import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import json

st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🖼️",
    layout="wide" 
)

TEXTS = {
    'tr': {
        'title': "🖼️ Yapay Zeka Görüntü Tanıma",
        'header': "Bir resim yükleyin, yapay zeka içinde ne olduğunu tahmin etsin!",
        'sidebar_title': "Ayarlar",
        'lang_select': "Dil / Language",
        'about_title': "Bu Uygulama Hakkında",
        'about_text': """
        Bu uygulama, PyTorch ile eğitilmiş **MobileNetV2** modelini kullanarak bir resimdeki nesneleri tanır.
        
        **Neden MobileNetV2?**
        - **Hızlı ve Hafif:** Kısıtlı kaynaklara sahip platformlarda (Streamlit'in ücretsiz sunucuları gibi) bile verimli bir şekilde çalışmak üzere tasarlanmıştır.
        - **Yüksek Başarı:** Milyonlarca resim içeren ImageNet veri setiyle eğitildiği için 1000 farklı kategorideki nesneyi yüksek doğrulukla tanıyabilir.
        
        Bu proje, büyük yapay zeka modellerinin pratik uygulamalarda nasıl kullanılabileceğinin harika bir örneğidir.
        """,
        'uploader_label': "Tanımlamak için bir resim seçin...",
        'uploaded_image_caption': "Yüklenen Resim",
        'spinner_analyzing': "Resim analiz ediliyor...",
        'predictions_header': "🧠 Yapay Zeka Tahminleri:",
        'top_prediction': "En Yüksek Tahmin",
        'confidence': "Güven",
        'all_predictions_expander': "Tüm Tahminleri Gör"
    },
    'en': {
        'title': "🖼️ AI Image Recognition",
        'header': "Upload an image and let the AI guess what's inside!",
        'sidebar_title': "Settings",
        'lang_select': "Language / Dil",
        'about_title': "About This App",
        'about_text': """
        This application recognizes objects in an image using the **MobileNetV2** model, pre-trained with PyTorch.
        
        **Why MobileNetV2?**
        - **Fast and Lightweight:** It is designed to run efficiently even on platforms with limited resources, like Streamlit's free tier.
        - **Highly Accurate:** Trained on the ImageNet dataset with millions of images, it can recognize 1000 different categories with high accuracy.
        
        This project is a great example of how large AI models can be used in practical applications.
        """,
        'uploader_label': "Choose an image to identify...",
        'uploaded_image_caption': "Uploaded Image",
        'spinner_analyzing': "Analyzing image...",
        'predictions_header': "🧠 AI Predictions:",
        'top_prediction': "Top Prediction",
        'confidence': "Confidence",
        'all_predictions_expander': "See All Predictions"
    }
}

@st.cache_resource
def load_model_and_classes():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval() # Set the model to evaluation mode

    with open('imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
    
    idx_to_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    return model, idx_to_labels

def transform_image(image):
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformation(image).unsqueeze(0)

st.sidebar.title(TEXTS['tr']['sidebar_title']) # Title is always Turkish for consistency
lang_choice = st.sidebar.radio(TEXTS['tr']['lang_select'], ["Türkçe", "English"])
lang_code = 'tr' if lang_choice == 'Türkçe' else 'en'

st.sidebar.markdown("---")
st.sidebar.subheader(TEXTS[lang_code]['about_title'])
st.sidebar.info(TEXTS[lang_code]['about_text'])


st.title(TEXTS[lang_code]['title'])
st.markdown(TEXTS[lang_code]['header'])

model, class_labels = load_model_and_classes()

uploaded_file = st.file_uploader(
    TEXTS[lang_code]['uploader_label'], 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=TEXTS[lang_code]['uploaded_image_caption'], use_column_width=True)

    with col2:
        with st.spinner(TEXTS[lang_code]['spinner_analyzing']):
            tensor = transform_image(image)
            outputs = model(tensor)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_5_prob, top_5_catid = torch.topk(probabilities, 5)

            st.header(TEXTS[lang_code]['predictions_header'])
            
            top_prediction_label = class_labels[top_5_catid[0][0]].replace("_", " ").title()
            top_prediction_prob = top_5_prob[0][0].item()
            st.success(f"**{TEXTS[lang_code]['top_prediction']}: {top_prediction_label}** ({TEXTS[lang_code]['confidence']}: {top_prediction_prob:.2%})")

            with st.expander(TEXTS[lang_code]['all_predictions_expander']):
                for i in range(top_5_prob.size(1)):
                    prob = top_5_prob[0][i].item()
                    label = class_labels[top_5_catid[0][i]].replace("_", " ").title()
                    st.write(f"{i+1}. {label} - {prob:.2%}")
