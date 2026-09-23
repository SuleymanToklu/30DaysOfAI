import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO 

TEXT_CONTENT = {
    'TR': {
        'page_title': "YOLO ile Nesne Tespiti",
        'page_icon': "📸",
        'main_title': "📸 YOLOv8 ile Gerçek Zamanlı Nesne Tespiti",
        'description': "Bir resim yükleyin veya kameranızı kullanarak canlı nesne tespiti yapın.",
        'tab1_name': "🖼️ Resimden Tespit",
        'tab2_name': "🎯 Proje Detayları",
        'tab1_header': "Bir Görüntü Yükleyin",
        'uploader_label': "Bir resim dosyası seçin",
        'image_caption': "Yüklenen Resim",
        'detection_inprogress': "Tespit ediliyor...",
        'detection_result_caption': "Tespit Edilen Nesneler",
        'results_header': "Tespit Edilen Nesneler ve Güven Skorları:",
        'tab2_header': "Projenin Amacı ve Teknik Detaylar",
        'tab2_desc': """
        Bu projenin amacı, modern bir derin öğrenme modeli olan **YOLOv8**'i kullanarak, bir görüntüdeki nesneleri gerçek zamanlı olarak tespit eden bir sistem oluşturmaktır. Bu teknoloji, otonom araçlar, güvenlik kameraları, perakende analitiği ve daha birçok alanda kullanılmaktadır.
        
        - **Model:** `YOLOv8n`. Bu, Ultralytics tarafından geliştirilen, hızlı ve oldukça isabetli bir nesne tespit modelidir. Model, COCO veri seti üzerinde 80 farklı nesne sınıfı (insan, araba, kedi, köpek, şişe vb.) için önceden eğitilmiştir.
        - **Yöntem:** Bu projede model **eğitilmemiştir**. Bunun yerine, **önceden eğitilmiş (pre-trained)** bir modelin gücünden faydalanılmıştır. Bu, bir AI mühendisinin en önemli yeteneklerinden biridir: Sıfırdan başlamak yerine, mevcut en iyi araçları kendi problemine hızla entegre etmek.
        - **Nesne Tespiti vs. Sınıflandırma:** Önceki projelerimiz bir resmin tamamına tek bir etiket veriyordu (örn: "bu bir kedi"). Nesne tespiti ise bir adım ileri giderek, resimdeki her bir nesnenin **hem ne olduğunu (sınıfını) hem de nerede olduğunu (sınır kutusunu)** bulur.
        """
    },
    'EN': {
        'page_title': "Object Detection with YOLO",
        'page_icon': "📸",
        'main_title': "📸 Real-Time Object Detection with YOLOv8",
        'description': "Upload an image or use your camera for live object detection.",
        'tab1_name': "🖼️ Detection from Image",
        'tab2_name': "🎯 Project Details",
        'tab1_header': "Upload an Image",
        'uploader_label': "Choose an image file",
        'image_caption': "Uploaded Image",
        'detection_inprogress': "Detecting...",
        'detection_result_caption': "Detected Objects",
        'results_header': "Detected Objects and Confidence Scores:",
        'tab2_header': "Project Goal and Technical Details",
        'tab2_desc': """
        The goal of this project is to create a system that performs real-time object detection in an image using the modern deep learning model **YOLOv8**. This technology is used in various fields such as autonomous vehicles, security cameras, and retail analytics.
        
        - **Model:** `YOLOv8n`. This is a fast and highly accurate object detection model developed by Ultralytics, easily accessible. The model is pre-trained on the COCO dataset for 80 different object classes (person, car, cat, dog, bottle, etc.).
        - **Method:** The model was **not trained** in this project. Instead, the power of a **pre-trained** model was leveraged. This is a key skill for an AI engineer: rapidly integrating the best existing tools to solve a problem instead of starting from scratch.
        - **Object Detection vs. Classification:** Our previous projects assigned a single label to an entire image (e.g., "this is a cat"). Object detection goes a step further by identifying **both what each object is (its class) and where it is (its bounding box)** in the image.
        """
    }
}


st.sidebar.title("Language / Dil")
lang = st.sidebar.radio("Choose Language", ('TR', 'EN'), label_visibility="collapsed")
TEXT = TEXT_CONTENT[lang]

st.set_page_config(page_title=TEXT['page_title'], page_icon=TEXT['page_icon'], layout="wide")

@st.cache_resource
def load_model():
    """Loads the YOLOv8 model from the ultralytics library."""
    model = YOLO('yolov8n.pt') 
    return model

model = load_model()

st.title(TEXT['main_title'])
st.write(TEXT['description'])

tab1, tab2 = st.tabs([TEXT['tab1_name'], TEXT['tab2_name']])

with tab1:
    st.header(TEXT['tab1_header'])
    uploaded_file = st.file_uploader(TEXT['uploader_label'], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=TEXT['image_caption'], use_column_width=True)
        
        st.write("")
        with st.spinner(TEXT['detection_inprogress']):
            results = model(image)

            annotated_image = results[0].plot()
            
            annotated_image_rgb = annotated_image[..., ::-1]

        st.image(annotated_image_rgb, caption=TEXT['detection_result_caption'], use_column_width=True)
        
        st.subheader(TEXT['results_header'])
     
        detections = results[0].boxes.data
        if detections.shape[0] > 0:
            df = pd.DataFrame(detections.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
            df['name'] = df['class'].apply(lambda x: model.names[int(x)])
            st.dataframe(df[['name', 'confidence']])
        else:
            st.write("Herhangi bir nesne tespit edilemedi.")
            
with tab2:
    st.header(TEXT['tab2_header'])
    st.write(TEXT['tab2_desc'])
