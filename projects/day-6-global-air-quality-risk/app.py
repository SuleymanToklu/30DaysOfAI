import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import date, timedelta

LANGUAGES = {
    "tr": {
        "page_title": "Hava Kalitesi Riski",
        "error_message": "Gerekli model dosyaları bulunamadı. Lütfen önce `train_model.py`'yi çalıştırıp dosyaları GitHub'a gönderdiğinizden emin olun.",
        "api_error": "Veri alınamadı. Lütfen farklı bir tarih veya şehir deneyin. API geçmiş verileri sınırlı olabilir.",
        "api_success": "Veriler başarıyla alındı!",
        "tab1_title": "🔮 **Otonom Risk Tahmini**",
        "tab2_title": "🎯 **Proje Detayları**",
        "main_title": "🌍 Global Hava Kalitesi ve Sağlık Riski Tahmincisi",
        "tab1_header": "Geçmişe Yönelik Otomatik Risk Tahmini",
        "tab1_write": "Bir ülke, şehir ve tarih seçerek, o güne ait gerçek hava kalitesi verileriyle sağlık riskini otonom olarak tahmin edin.",
        "form_country_label": 'Ülke',
        "form_city_label": 'Şehir',
        "form_date_label": 'Tarih',
        "form_submit_button": 'Riski Tahmin Et',
        "prediction_subheader": '🔮 Tahmin Sonucu',
        "data_subheader": "📊 Ölçülen Kirletici Değerleri",
        "map_subheader": "🗺️ Seçilen Konum",
        "high_risk_text": "YÜKSEK RİSK",
        "high_risk_warning": "Hava kalitesi sağlıksız. Dışarıdaki aktiviteleri sınırlamak ve gerekirse maske kullanmak önerilir.",
        "moderate_risk_text": "ORTA RİSK",
        "moderate_risk_info": "Hava kalitesi hassas gruplar için riskli olabilir. Solunum rahatsızlığı olanlar dikkatli olmalıdır.",
        "low_risk_text": "DÜŞÜK RİSK",
        "low_risk_info": "Hava kalitesi iyi. Dışarıdaki aktiviteler için bir risk bulunmamaktadır.",
        "tab2_header": "Projenin Amacı ve Teknik Detaylar",
        "tab2_write": """
        ### 🎯 Projenin Amacı
        Bu projenin temel amacı, seçilen bir şehir ve tarih için geçmişe dönük hava kalitesi verilerini kullanarak bir sağlık riski tahmini sunmaktır. Uygulama, ham verileri anlamlı bir risk seviyesine (`Düşük`, `Orta`, `Yüksek`) dönüştürerek kullanıcıların belirli bir gündeki hava kalitesi koşullarını kolayca anlamasını sağlar.

        ### 🤖 Makine Öğrenmesi Modeli
        - **Model Tipi:** `RandomForestClassifier` (Rastgele Orman Sınıflandırıcısı)
        - **Neden Bu Model?:** Rastgele Orman, birden çok karar ağacı oluşturarak yüksek doğruluk ve aşırı öğrenmeye (overfitting) karşı direnç sunan güçlü bir sınıflandırma algoritmasıdır. Bu proje gibi kategorik sonuçları (düşük, orta, yüksek risk) tahmin etme görevleri için oldukça uygundur.
        - **Eğitim Verisi:** Model, Kaggle'dan alınan [Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset) kullanılarak eğitilmiştir.
        - **Hedef Değişken (`Target`):** Model, `Risk_Level` adında sentetik olarak oluşturulmuş bir hedefi tahmin etmek üzere eğitilmiştir. Bu değişken, veri setindeki genel `AQI Value` (Hava Kalitesi İndeksi) üzerinden türetilmiştir (0-50: Düşük, 51-100: Orta, 100+: Yüksek).

        ### ⚙️ Teknoloji ve Kütüphaneler
        - **Python:** Projenin ana programlama dili.
        - **Streamlit:** Hızlı ve interaktif web uygulamaları oluşturmak için kullanılan arayüz kütüphanesi.
        - **Pandas:** Veri işleme ve analizi için temel kütüphane.
        - **Scikit-learn:** Makine öğrenmesi modelini (RandomForest) oluşturmak, eğitmek ve değerlendirmek için kullanıldı.
        - **Joblib:** Eğitilmiş modelin ve diğer Python nesnelerinin dosyaya kaydedilip tekrar yüklenmesini sağladı.
        - **Requests:** Open-Meteo API'sine bağlanarak canlı ve geçmişe dönük hava kalitesi verilerini çekmek için kullanıldı.

        ### ✨ Öne Çıkan Özellikler
        - **Otonom Veri Çekme:** Manuel veri girişi yerine, Open-Meteo API'si üzerinden gerçek zamanlı hava kalitesi verilerini otomatik olarak alır.
        - **İnteraktif Arayüz:** Kullanıcıların ülke, şehir ve tarih seçmesine olanak tanıyan dinamik ve kullanıcı dostu bir arayüz.
        - **Görsel Geri Bildirim:** Sonuçlar, interaktif bir harita ve ölçülen değerleri gösteren metrik kartlarla zenginleştirilmiştir.
        - **Çoklu Dil Desteği:** İngilizce ve Türkçe dil seçenekleri mevcuttur.
        """
    },
    "en": {
        "page_title": "Air Quality Risk",
        "error_message": "Required model files not found. Please make sure you have run `train_model.py` first and pushed the files to GitHub.",
        "api_error": "Could not retrieve data. Please try a different date or city. The API may have limited historical data.",
        "api_success": "Data successfully retrieved!",
        "tab1_title": "🔮 **Autonomous Risk Forecast**",
        "tab2_title": "🎯 **Project Details**",
        "main_title": "🌍 Global Air Quality & Health Risk Forecaster",
        "tab1_header": "Automated Historical Risk Prediction",
        "tab1_write": "Select a country, city, and date to autonomously predict the health risk using real air quality data from that day.",
        "form_country_label": 'Country',
        "form_city_label": 'City',
        "form_date_label": 'Date',
        "form_submit_button": 'Predict Risk',
        "prediction_subheader": '🔮 Prediction Result',
        "data_subheader": "📊 Measured Pollutant Values",
        "map_subheader": "🗺️ Selected Location",
        "high_risk_text": "HIGH RISK",
        "high_risk_warning": "Air quality is unhealthy. It is recommended to limit outdoor activities and use a mask if necessary.",
        "moderate_risk_text": "MODERATE RISK",
        "moderate_risk_info": "Air quality may be risky for sensitive groups. People with respiratory conditions should be cautious.",
        "low_risk_text": "LOW RISK",
        "low_risk_info": "Air quality is good. There is no risk for outdoor activities.",
        "tab2_header": "Project Goal and Technical Details",
        "tab2_write": """
        ### 🎯 Project Goal
        The primary goal of this project is to provide a health risk prediction based on historical air quality data for a selected city and date. The application transforms raw data into a meaningful risk level (`Low`, `Moderate`, `High`), allowing users to easily understand the air quality conditions on a specific day.

        ### 🤖 Machine Learning Model
        - **Model Type:** `RandomForestClassifier`
        - **Why This Model?:** Random Forest is a powerful classification algorithm that builds multiple decision trees to offer high accuracy and resistance to overfitting. It is well-suited for tasks like this project, which involve predicting categorical outcomes (low, moderate, high risk).
        - **Training Data:** The model was trained using the [Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset) from Kaggle.
        - **Target Variable:** The model was trained to predict a synthetically created target called `Risk_Level`. This variable was derived from the overall `AQI Value` in the dataset (0-50: Low, 51-100: Moderate, 100+: High).

        ### ⚙️ Technology and Libraries
        - **Python:** The main programming language for the project.
        - **Streamlit:** The framework used to build the fast and interactive web application interface.
        - **Pandas:** The fundamental library for data manipulation and analysis.
        - **Scikit-learn:** Used to build, train, and evaluate the machine learning model (RandomForest).
        - **Joblib:** Enabled saving the trained model and other Python objects to disk for later use.
        - **Requests:** Used to connect to the Open-Meteo API and fetch live and historical air quality data.

        ### ✨ Key Features
        - **Autonomous Data Fetching:** Automatically retrieves real air quality data via the Open-Meteo API instead of requiring manual input.
        - **Interactive Interface:** A dynamic and user-friendly interface that allows users to select a country, city, and date.
        - **Visual Feedback:** Results are enriched with an interactive map and metric cards displaying the measured values.
        - **Multi-Language Support:** The application is available in both English and Turkish.
        """
    }
}

CITIES_DATA = {
    "Turkey": {
        "Istanbul": {"lat": 41.0082, "lon": 28.9784},
        "Ankara": {"lat": 39.9334, "lon": 32.8663},
        "Izmir": {"lat": 38.4237, "lon": 27.1428}
    },
    "United States": {
        "New York": {"lat": 40.7128, "lon": -74.0060},
        "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
        "Chicago": {"lat": 41.8781, "lon": -87.6298}
    },
    "France": {
        "Paris": {"lat": 48.8566, "lon": 2.3522}
    },
    "United Kingdom": {
        "London": {"lat": 51.5074, "lon": -0.1278}
    },
    "Japan": {
        "Tokyo": {"lat": 35.6895, "lon": 139.6917}
    },
    "China": {
        "Beijing": {"lat": 39.9042, "lon": 116.4074}
    },
    "Australia": {
        "Sydney": {"lat": -33.8688, "lon": 151.2093}
    },
    "Germany": {
        "Berlin": {"lat": 52.5200, "lon": 13.4050}
    }
}


if 'language' not in st.session_state:
    st.session_state.language = 'tr'

texts = LANGUAGES[st.session_state.language]

st.set_page_config(page_title=texts["page_title"], page_icon="🌍", layout="wide")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_features = joblib.load('model_features.pkl')
        city_encoder = joblib.load('city_encoder.pkl')
        return model, model_features, city_encoder
    except FileNotFoundError:
        return None, None, None

def get_air_quality_data(lat, lon, query_date):
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": query_date,
        "end_date": query_date,
        "hourly": "pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
        "domains": "cams_global" 
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' not in data or not data['hourly']['time']:
            return None
            
        df = pd.DataFrame(data['hourly'])
        daily_avg = df.mean(numeric_only=True).to_dict()
        return daily_avg
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return None

model, model_features, city_encoder = load_resources()

_, lang_col = st.columns([0.85, 0.15])
with lang_col:
    st.selectbox(
        label="Dil / Language",
        options=['tr', 'en'],
        format_func=lambda x: "Türkçe" if x == 'tr' else "English",
        key='language'
    )

st.title(texts["main_title"])

if not all([model, model_features, city_encoder]):
    st.error(texts["error_message"])
    st.stop()

tab1, tab2 = st.tabs([texts["tab1_title"], texts["tab2_title"]])

with tab1:
    st.header(texts["tab1_header"])
    st.write(texts["tab1_write"])
    
    col1, col2 = st.columns(2)
    with col1:
        country_options = sorted(CITIES_DATA.keys())
        selected_country = st.selectbox(texts["form_country_label"], options=country_options)
    
    city_options = sorted(CITIES_DATA[selected_country].keys())

    with st.form(key='prediction_form'):
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            selected_city = st.selectbox(texts["form_city_label"], options=city_options)
        with form_col2:
            selected_date = st.date_input(
                texts["form_date_label"],
                value=date.today() - timedelta(days=7),
                min_value=date(2022, 1, 1),
                max_value=date.today() - timedelta(days=2)
            )
        
        submit_button = st.form_submit_button(label=texts["form_submit_button"])

    if submit_button:
        coords = CITIES_DATA[selected_country][selected_city]
        date_str = selected_date.strftime("%Y-%m-%d")
        
        with st.spinner(f"{selected_city} için {date_str} tarihli veriler alınıyor..."):
            air_data = get_air_quality_data(coords['lat'], coords['lon'], date_str)

        if air_data:
            st.toast(texts["api_success"], icon='✅')
            
            map_data = pd.DataFrame({'lat': [coords['lat']], 'lon': [coords['lon']]})
            st.subheader(texts["map_subheader"])
            st.map(map_data, zoom=10)

            if selected_city in city_encoder.classes_:
                city_encoded = city_encoder.transform([selected_city])[0]
            else:
                city_encoded = 0

            input_dict = {
                'CO AQI Value': air_data.get('carbon_monoxide', 0) / 100,
                'Ozone AQI Value': air_data.get('ozone', 0),
                'NO2 AQI Value': air_data.get('nitrogen_dioxide', 0),
                'PM2.5 AQI Value': air_data.get('pm2_5', 0),
                'City_Encoded': city_encoded
            }
            
            input_df = pd.DataFrame([input_dict])[model_features]
            
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader(texts["prediction_subheader"])
            risk_level = prediction[0]
            
            if risk_level == 2:
                st.error(f"**{texts['high_risk_text']}** ({prediction_proba[0][2]:.0%})")
                st.warning(texts["high_risk_warning"])
            elif risk_level == 1:
                st.warning(f"**{texts['moderate_risk_text']}** ({prediction_proba[0][1]:.0%})")
                st.info(texts["moderate_risk_info"])
            else:
                st.success(f"**{texts['low_risk_text']}** ({prediction_proba[0][0]:.0%})")
                st.info(texts["low_risk_info"])

            st.subheader(texts["data_subheader"])
            metric_cols = st.columns(4)
            metric_cols[0].metric(label="PM2.5 (μg/m³)", value=f"{air_data.get('pm2_5', 0):.2f}")
            metric_cols[1].metric(label="Ozon (μg/m³)", value=f"{air_data.get('ozone', 0):.2f}")
            metric_cols[2].metric(label="NO₂ (μg/m³)", value=f"{air_data.get('nitrogen_dioxide', 0):.2f}")
            metric_cols[3].metric(label="CO (mg/m³)", value=f"{air_data.get('carbon_monoxide', 0)/1000:.2f}")

        else:
            st.error(texts["api_error"])

with tab2:
    st.header(texts["tab2_header"])
    st.markdown(texts["tab2_write"], unsafe_allow_html=True)
