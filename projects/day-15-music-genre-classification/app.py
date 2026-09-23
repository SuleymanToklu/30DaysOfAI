import streamlit as st
import librosa
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd

# Page Settings 
st.set_page_config(page_title="Müzik Türü Sınıflandırma", page_icon="🎵", layout="wide")

# Custom CSS for Modern UI 
page_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    
    #root > div:nth-child(1) > div > div > div > div > section[data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4F8BF9;
        color: #4F8BF9;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #FFFFFF;
        color: #FFFFFF;
        background-color: #4F8BF9;
    }
    .stButton>button:focus {
        box-shadow: none !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1e2a38, #273444);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .result-card h3 {
        color: #FFFFFF;
        font-size: 24px;
        margin-bottom: 0.5rem;
    }
    .result-card .genre {
        font-size: 42px;
        font-weight: bold;
        color: #4F8BF9;
        margin-bottom: 1.5rem;
    }
    .result-card .confidence {
        font-size: 20px;
        color: #a9c1e1;
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        background-color: #1a1a2e;
        border-radius: 15px;
        border: 2px dashed #4F8BF9;
        text-align: center;
        color: #FFFFFF;
    }
    .empty-state svg {
        width: 60px;
        height: 60px;
        margin-bottom: 1rem;
    }
    
    .loader {
        border: 5px solid #2e2e3e;
        border-top: 5px solid #4F8BF9;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid #262730;
        color: #FAFAFA;
    }
    .footer a {
        color: #4F8BF9;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Language
TEXTS = {
    "tr": {
        "page_title": "Ses Analizi ile Müzik Türü Sınıflandırma",
        "tab_predict": "🧠 Tür Tahmini",
        "tab_details": "🎯 Proje Detayları",
        "header": "Canlı Müzik Analizi",
        "subheader": "Bir .wav ses dosyası yükleyerek müzik türünü tahmin edin.",
        "uploader_label": "Bir .wav dosyası seçin",
        "button_label": "Müzik Türünü Tahmin Et",
        "result_header": "🔮 Analiz Sonucu",
        "predicted_genre_label": "Bu parçanın türü",
        "confidence_label": "Modelin Güven Oranı",
        "error_loading": "Gerekli dosyalar bulunamadı. Lütfen önce `process_and_train.py`'yi çalıştırın.",
        "error_processing": "Ses dosyası işlenirken hata oluştu: ",
        "lang_selector": "Dil / Language",
        "details_header": "Projenin Amacı ve Geliştirme Süreci",
        "details_content": """
        ### Temel Amaç
        Bu projenin temel amacı, bir müzik parçasının 30 saniyelik ses sinyalini analiz ederek türünü (örn: Blues, Rock, Caz) otomatik olarak sınıflandıran bir yapay zeka modeli oluşturmaktır. Bu teknoloji, müzik platformlarında otomatik etiketleme ve tavsiye sistemlerinin temelini oluşturur.
        - **Model:** Ses özellikleri gibi karmaşık ve yüksek boyutlu verileri sınıflandırmada etkili olan `Destek Vektör Makineleri (SVC)` algoritması kullanılmıştır.
        - **Özellik Mühendisliği:** Modelin sesi "anlayabilmesi" için, **`librosa`** kütüphanesi ile her ses dosyasından `MFCC`, `Spectral Centroid`, `Zero Crossing Rate` gibi o sesin tınısını, ritmini ve perdesini temsil eden sayısal özellikler çıkarılmıştır.
        ---
        ### Proje Geliştirmeleri ve Analiz (Sürüm 2.0)
        #### Model Performansı Üzerine Notlar
        Uygulama testleri sırasında, modelin güven skorlarının bazen düşük olabildiği gözlemlenmiştir. Bu bir hata değil, projenin mevcut kapsamı için beklenen bir durumdur. Sebepleri şunlardır:
        - **Sınırlı Veri Seti:** GTZAN veri seti (10 tür, 1000 şarkı), derinlemesine bir öğrenme için görece küçüktür.
        - **Basit Özellik Çıkarımı:** Şarkının 30 saniyelik bir kesitindeki ses özelliklerinin *ortalamasının* alınması, müziğin zamana bağlı dinamiklerini (örn: bir nakaratın yükselişi) basitleştirerek bilgi kaybına neden olur.
        - **Model Seçimi:** SVC iyi bir başlangıç olsa da, bu tür karmaşık ses problemlerinde genellikle Konvolüsyonel Sinir Ağları (CNN) gibi derin öğrenme modelleri daha yüksek başarımlar elde eder.
        #### Arayüz ve Kullanıcı Deneyimi Geliştirmeleri
        Projenin işlevselliğini ve sunumunu iyileştirmek amacıyla aşağıdaki geliştirmeler yapılmıştır:
        - **Modern Arayüz (CSS & HTML):** Streamlit'in standart bileşenlerinin dışına çıkılarak, özel CSS stilleri ve HTML yapıları ile daha akıcı ve estetik bir kullanıcı deneyimi hedeflenmiştir. Sonuçların gösterildiği kart tasarımı bu geliştirmeye bir örnektir.
        - **Çoklu Dil Desteği (TR/EN):** Uygulamanın daha geniş bir kitleye hitap etmesi ve uluslararası standartlarda bir proje olması için dinamik bir dil seçim özelliği eklenmiştir.
        """,
        "empty_state_header": "Analize Başlamaya Hazır",
        "empty_state_text": "Lütfen bir .wav dosyası yükleyerek müzik türünü keşfedin.",
        "chart_title": "Tüm Türler için Olasılık Dağılımı",
        "chart_xaxis": "Güven Skoru (%)",
        "chart_yaxis": "Müzik Türü"
    },
    "en": {
        "page_title": "Music Genre Classification with Audio Analysis",
        "tab_predict": "🧠 Genre Prediction",
        "tab_details": "🎯 Project Details",
        "header": "Live Music Analysis",
        "subheader": "Predict the music genre by uploading a .wav audio file.",
        "uploader_label": "Choose a .wav file",
        "button_label": "Predict Music Genre",
        "result_header": "🔮 Analysis Result",
        "predicted_genre_label": "The genre of this track is",
        "confidence_label": "Model Confidence Score",
        "error_loading": "Required model files not found. Please run `process_and_train.py` first.",
        "error_processing": "Error processing the audio file: ",
        "lang_selector": "Language / Dil",
        "details_header": "Project Goal and Development Process",
        "details_content": """
        ### Core Objective
        The primary goal of this project is to create an AI model that automatically classifies the genre (e.g., Blues, Rock, Jazz) of a music track by analyzing a 30-second audio signal. This technology forms the basis for automatic tagging and recommendation systems on music platforms.
        - **Model:** A `Support Vector Classifier (SVC)` algorithm was used, which is effective for classifying complex, high-dimensional data like audio features.
        - **Feature Engineering:** To enable the model to "understand" the audio, numerical features representing the sound's timbre, rhythm, and pitch, such as `MFCC`, `Spectral Centroid`, and `Zero Crossing Rate`, were extracted from each audio file using the **`librosa`** library.
        ---
        ### Project Enhancements and Analysis (Version 2.0)
        #### Notes on Model Performance
        During application testing, it was observed that the model's confidence scores can sometimes be low. This is not an error but an expected outcome for the current scope of the project. The reasons include:
        - **Limited Dataset:** The GTZAN dataset (10 genres, 1000 songs) is relatively small for in-depth learning.
        - **Simple Feature Extraction:** Taking the *average* of audio features over a 30-second clip simplifies the track's time-based dynamics (e.g., the rise of a chorus), leading to information loss.
        - **Model Choice:** While SVC is a good starting point, deep learning models like Convolutional Neural Networks (CNNs) typically achieve higher accuracy on complex audio problems like this.
        #### Interface and User Experience Improvements
        The following enhancements were made to improve the project's functionality and presentation:
        - **Modern UI (CSS & HTML):** To move beyond Streamlit's standard components, a more fluid and aesthetic user experience was targeted with custom CSS styles and HTML structures. The card design used to display results is an example of this enhancement.
        - **Multi-Language Support (TR/EN):** A dynamic language selection feature was added to help the application appeal to a wider audience and meet international project standards.
        """,
        "empty_state_header": "Ready for Analysis",
        "empty_state_text": "Please upload a .wav file to discover its music genre.",
        "chart_title": "Probability Distribution for All Genres",
        "chart_xaxis": "Confidence Score (%)",
        "chart_yaxis": "Music Genre"
    }
}


# Helper Functions 
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError:
        return None, None, None

def extract_features_from_upload(uploaded_file):
    try:
        y, sr = librosa.load(uploaded_file, mono=True, duration=30)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_means = [np.mean(e) for e in mfcc]
        features = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr] + mfcc_means
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(T["error_processing"] + str(e))
        return None

def create_probability_chart(probabilities, labels, lang_dict):
    df = pd.DataFrame({
        'Genre': labels,
        'Probability': probabilities * 100
    }).sort_values(by='Probability', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Probability'],
        y=df['Genre'],
        orientation='h',
        marker=dict(
            color=df['Probability'],
            colorscale='Blues',
            showscale=False
        ),
        text=df['Probability'].apply(lambda x: f'{x:.2f}%'),
        textposition='inside',
        insidetextanchor='middle'
    ))

    fig.update_layout(
        title_text=lang_dict["chart_title"],
        xaxis_title=lang_dict["chart_xaxis"],
        yaxis_title=lang_dict["chart_yaxis"],
        yaxis=dict(autorange="reversed"),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', family='Poppins, sans-serif')
    )
    return fig

#  Sidebar 
st.sidebar.image("https://www.gstatic.com/lamda/images/gemini_wordmark_2023_white_1x.png", width=120)
st.sidebar.title("Ayarlar / Settings")
selected_lang = st.sidebar.selectbox(TEXTS["tr"]["lang_selector"], ("Türkçe (TR)", "English (EN)"))
lang = "tr" if selected_lang == "Türkçe (TR)" else "en"
T = TEXTS[lang]

#  Load Model and Resources 
model, scaler, encoder = load_resources()

#  Main Application 
st.title(f"🎵 {T['page_title']}")

if not all([model, scaler, encoder]):
    st.error(T["error_loading"])
    st.stop()

tab1, tab2 = st.tabs([T["tab_predict"], T["tab_details"]])

with tab1:
    st.header(T["header"])
    st.write(T["subheader"])

    uploaded_file = st.file_uploader(T["uploader_label"], type=["wav"], label_visibility="collapsed")

    if uploaded_file is None:
        empty_state_html = f"""
        <div class="empty-state">
            <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" class="bi bi-music-note-beamed" viewBox="0 0 16 16">
              <path d="M6 13c0 1.105-1.12 2-2.5 2S1 14.105 1 13c0-1.104 1.12-2 2.5-2s2.5.896 2.5 2zm9-2c0 1.105-1.12 2-2.5 2s-2.5-.895-2.5-2 1.12-2 2.5-2 2.5.895 2.5 2z"/>
              <path fill-rule="evenodd" d="M14 11V2h1v9h-1zM6 3v10H5V3h1z"/>
              <path d="M5 2.905a1 1 0 0 1 .9-.995l8-.8a1 1 0 0 1 1.1.995V3L5 4V2.905z"/>
            </svg>
            <h4>{T['empty_state_header']}</h4>
            <p>{T['empty_state_text']}</p>
        </div>
        """
        st.markdown(empty_state_html, unsafe_allow_html=True)
    else:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button(T["button_label"]):
            loader_placeholder = st.empty()
            loader_html = '<div class="loader-container" style="text-align: center;"><div class="loader"></div></div>'
            loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
            
            features = extract_features_from_upload(uploaded_file)
            loader_placeholder.empty() 
            
            if features is not None:
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)
                prediction_proba = model.predict_proba(scaled_features)
                
                predicted_genre = encoder.inverse_transform(prediction)[0]
                confidence = np.max(prediction_proba) * 100

                result_html = f"""
                <div class="result-card">
                    <h3>{T['predicted_genre_label']}</h3>
                    <p class="genre">{predicted_genre.capitalize()}</p>
                    <p class="confidence">{T['confidence_label']}: {confidence:.2f}%</p>
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)

                all_probabilities = prediction_proba[0]
                all_genres = encoder.classes_
                prob_fig = create_probability_chart(all_probabilities, all_genres, T)
                st.plotly_chart(prob_fig, use_container_width=True)


with tab2:
    st.header(T["details_header"])
    st.markdown(T["details_content"], unsafe_allow_html=True)

#  Footer 
footer_html = """
    <div class="footer">
      <p>Geliştiren: Süleyman Toklu | 
        <a href="https://github.com/SuleymanToklu" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/suleyman-toklu10/recent-activity/all/" target="_blank">LinkedIn</a>
      </p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)