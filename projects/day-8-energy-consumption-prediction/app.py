import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

st.set_page_config(page_title="Enerji Tüketimi Tahmini", page_icon="💡", layout="wide")

selected_language = st.sidebar.selectbox("Language / Dil", ["English", "Türkçe"])
lang_code = 'tr' if selected_language == 'Türkçe' else 'en'

translations = {
    "en": {
        "page_title": "💡 Smart Appliance Energy Consumption Forecaster",
        "tab_predictor": "🧠 Model Comparison",
        "tab_performance": "📊 Model Performance",
        "tab_models_explained": "📖 About the Models",
        "tab_details": "🎯 Project Details",
        "predictor_header": "Compare Model Predictions on Random Data",
        "predictor_desc": "This tool generates random sensor and time data, then predicts total energy consumption using multiple different regression models. The results are visualized below to compare their performance.",
        "button_generate": "Generate New Data & Predict",
        "results_header": "🔮 Prediction Results",
        "prediction_chart_desc": "The bar chart below visualizes the energy consumption (Wh) prediction made by each model for a single, randomly generated data scenario. This is useful for seeing how different models react to an instantaneous data point.",
        "results_chart_title": "Model Prediction Comparison",
        "results_chart_model": "Model",
        "results_chart_prediction": "Predicted Consumption (Wh)",
        "performance_header": "Model Performance Metrics on Test Data",
        "performance_desc": "The table below shows the performance of each model on a held-out test dataset (20% of the original data). This provides a more reliable measure of how well the models generalize to new, unseen data.",
        "performance_table_interpretation": "**Interpretation:** Generally, models with a high R² (close to `~1.0`) and low MAE/RMSE values are considered more successful. These metrics indicate the models' ability to generalize on the test data. By examining the table, you can determine which model is most suitable for this problem.",
        "mae_desc": "**MAE (Mean Absolute Error):** The average absolute difference between the predicted and actual values. Lower is better. A MAE of 50 means the model's predictions are, on average, off by 50 Wh.",
        "rmse_desc": "**RMSE (Root Mean Squared Error):** Similar to MAE, but penalizes larger errors more heavily. It's a good indicator of the model's sensitivity to outliers. Lower is better.",
        "r2_desc": "**R² (R-squared):** Represents the proportion of the variance in the energy consumption that is predictable from the sensor and time features. A value of 0.9 means the model can explain 90% of the data's variability. Closer to 1 is better.",
        "feature_importance_header": "Feature Importance",
        "feature_importance_desc": "This chart shows which features (sensors, time data) the model considers most important for making predictions. This helps to understand what drives energy consumption.",
        "feature_importance_select": "Select a model to view its feature importances:",
        "feature_importance_chart_title": "{model_name} - Feature Importance",
        "feature_importance_chart_note": "**Note:** The length of the horizontal bars indicates the degree of importance of the feature for the model. Longer bars indicate that the model gives more weight to that feature in its predictions.",
        "feature_axis_title": "Feature",
        "importance_axis_title": "Importance",
        "models_explained_header": "Understanding the Models",
        "models": {
            "linear_regression": {"title": "Linear, Ridge & Lasso Regression", "description": "...", "performance": "..."},
            "svr": {"title": "Support Vector Regressor (SVR)", "description": "...", "performance": "..."},
            "random_forest": {"title": "Random Forest Regressor", "description": "...", "performance": "..."},
            "boosting": {"title": "XGBoost & LightGBM Regressors", "description": "...", "performance": "..."}
        },
        "project_details_header": "Project Goal and Technical Details",
        "project_details_text": "...",
        "error_loading": "Could not load necessary model files. Please run `train_model.py` first to generate all `.pkl` files."
    },
    "tr": {
        "page_title": "💡 Akıllı Cihaz Enerji Tüketimi Tahmincisi",
        "tab_predictor": "🧠 Model Karşılaştırması",
        "tab_performance": "📊 Model Başarımı",
        "tab_models_explained": "📖 Modelleri Tanıyalım",
        "tab_details": "🎯 Proje Detayları",
        "predictor_header": "Rastgele Verilerle Model Tahminlerini Karşılaştır",
        "predictor_desc": "Bu araç, rastgele sensör ve zaman verileri oluşturur, ardından birden çok farklı regresyon modeli kullanarak toplam enerji tüketimini tahmin eder. Sonuçlar, performanslarını karşılaştırmak için aşağıda görselleştirilmiştir.",
        "button_generate": "Yeni Veri Üret & Tahmin Et",
        "results_header": "🔮 Tahmin Sonuçları",
        "prediction_chart_desc": "Aşağıdaki çubuk grafik, üretilen tek bir rastgele veri senaryosu için her bir modelin yaptığı enerji tüketimi (Wh) tahminini görselleştirir. Bu, modellerin anlık bir veriye nasıl farklı tepkiler verdiğini görmek için kullanışlıdır.",
        "results_chart_title": "Model Tahmin Karşılaştırması",
        "results_chart_model": "Model",
        "results_chart_prediction": "Tahmini Tüketim (Wh)",
        "performance_header": "Test Verisi Üzerindeki Model Başarım Metrikleri",
        "performance_desc": "Aşağıdaki tablo, her bir modelin orijinal verinin %20'si olan ve daha önce görmediği test verisi üzerindeki performansını göstermektedir. Bu, modellerin yeni verilere ne kadar iyi genelleme yapabildiğinin güvenilir bir ölçüsüdür.",
        "performance_table_interpretation": "**Yorumlama:** Genel olarak, yüksek R² (`~1.0`'e yakın) ve düşük MAE/RMSE değerlerine sahip modeller daha başarılı kabul edilir. Bu metrikler, modellerin test verisindeki genelleme yeteneğini gösterir. Tabloyu inceleyerek hangi modelin bu problem için daha uygun olduğunu belirleyebilirsiniz.",
        "mae_desc": "**MAE (Ortalama Mutlak Hata):** Tahmin edilen ve gerçek değerler arasındaki ortalama mutlak fark. Düşük olması daha iyidir. 50 MAE, modelin tahminlerinin ortalama olarak 50 Wh saptığı anlamına gelir.",
        "rmse_desc": "**RMSE (Kök Ortalama Kare Hata):** MAE'ye benzer, ancak büyük hataları daha ağır şekilde cezalandırır. Modelin aykırı değerlere olan hassasiyetinin iyi bir göstergesidir. Düşük olması daha iyidir.",
        "r2_desc": "**R² (Belirlilik Katsayısı):** Enerji tüketimindeki değişkenliğin ne kadarının sensör ve zaman verileriyle açıklanabildiğini gösterir. 0.9 değeri, modelin verideki değişkenliğin %90'ını açıklayabildiği anlamına gelir. 1'e ne kadar yakınsa o kadar iyidir.",
        "feature_importance_header": "Özellik Önemi",
        "feature_importance_desc": "Bu grafik, modelin tahmin yaparken hangi özellikleri (sensörler, zaman verileri) en önemli olarak gördüğünü gösterir. Bu, enerji tüketimini neyin tetiklediğini anlamaya yardımcı olur.",
        "feature_importance_select": "Özellik önemini görmek için bir model seçin:",
        "feature_importance_chart_title": "{model_name} - Özellik Önemi",
        "feature_importance_chart_note": "**Not:** Yatay barların uzunluğu, özelliğin model için önem derecesini gösterir. Daha uzun barlar, modelin tahminlerinde o özelliğe daha fazla ağırlık verdiğini belirtir.",
        "feature_axis_title": "Özellik",
        "importance_axis_title": "Önem Derecesi",
        "models_explained_header": "Kullanılan Modelleri Anlamak",
        "models": {
            "linear_regression": {"title": "Doğrusal (Linear), Ridge ve Lasso Regresyon", "description": "...", "performance": "..."},
            "svr": {"title": "Destek Vektör Regresyonu (SVR)", "description": "...", "performance": "..."},
            "random_forest": {"title": "Rastgele Orman (Random Forest) Regresyonu", "description": "...", "performance": "..."},
            "boosting": {"title": "XGBoost & LightGBM Regresyonları", "description": "...", "performance": "..."}
        },
        "project_details_header": "Projenin Amacı ve Teknik Detaylar",
        "project_details_text": "...",
        "error_loading": "Gerekli model dosyaları yüklenemedi. Lütfen önce `train_model.py` betiğini çalıştırarak tüm `.pkl` dosyalarını oluşturduğunuzdan emin olun."
    }
}
# Uzun metinleri daha temiz bir şekilde yönetmek için ayrı bir atama yapalım
translations['en']['models']['linear_regression']['description'] = """
**What are they?** These are the simplest forms of regression models. They try to find a linear relationship (a straight line) between the input features (like temperature) and the target variable (energy consumption).
- **Linear Regression** finds the best-fitting line.
- **Ridge & Lasso** are advanced versions that add a penalty for complexity, which helps prevent overfitting and makes the model more robust.
"""
translations['en']['models']['linear_regression']['performance'] = """
**Performance in This Project (R² ≈ 0.17):** Very Poor.
An R² score of 0.17 means these models could only explain about 17% of the variance in energy consumption. This poor performance is a strong indicator that the relationship between the sensor data and energy usage is **not linear**. The complex interactions (e.g., how time of day combined with temperature affects usage) cannot be captured by a simple straight line, causing these models to fail.
"""
translations['en']['models']['svr']['description'] = """
**What is it?** SVR works by fitting a line that stays within a certain error margin (a "street") of as many data points as possible. Its main strength is its ability to use "kernels" to transform the data, allowing it to find complex, non-linear relationships that linear models can't.
"""
translations['en']['models']['svr']['performance'] = """
**Performance in This Project (R² ≈ -0.09):** Extremely Poor.
A negative R² score means the model performed worse than simply guessing the average energy consumption for every data point. This is a catastrophic failure, and the primary reason is **lack of data preprocessing and hyperparameter tuning**. SVR is highly sensitive to the scale of input features and its internal parameters (`C`, `gamma`). Without scaling the data and finding the right parameters, it can produce completely meaningless results, as seen here. This is a perfect example of why preprocessing is crucial.
"""
translations['en']['models']['random_forest']['description'] = """
**What is it?** Random Forest is an "ensemble" model. It builds hundreds of individual "decision trees" and then averages their predictions. Each tree is trained on a random sample of the data and features, which makes the overall model very robust and less prone to overfitting. It's excellent at capturing complex, non-linear interactions.
"""
translations['en']['models']['random_forest']['performance'] = """
**Performance in This Project (R² ≈ 0.56):** The Best Performer.
Random Forest achieved the highest R² score, successfully explaining about 56% of the data's variance. Its success demonstrates that the data contains complex, non-linear patterns and feature interactions. It can learn specific rules (e.g., "if the hour is late AND the lights are on, consumption is high") that are impossible for linear models to capture. This result confirms that a tree-based ensemble approach is well-suited for this problem.
"""
translations['en']['models']['boosting']['description'] = """
**What are they?** These are both powerful, advanced "gradient boosting" models. They also build decision trees, but they do it sequentially. Each new tree is specifically built to correct the errors made by the previous ones. This focused, iterative improvement process makes them some of the most powerful models for tabular data. LightGBM is a modern optimization of this idea, often being much faster than XGBoost.
"""
translations['en']['models']['boosting']['performance'] = """
**Performance in This Project (R² ≈ 0.52 & 0.47):** Very Good.
Both models performed strongly, coming in just behind Random Forest. This is expected, as boosting algorithms are industry-standard for high performance. Their ability to correct their own mistakes over iterations allows them to create highly accurate and robust prediction models. The small performance difference between them and Random Forest is likely due to using default hyperparameters. With proper tuning, they could potentially become the top performers.
"""
translations['en']['project_details_text'] = """
        This project aims to predict the energy consumption of appliances in a home using time-series data from various IoT sensors. The goal is not only to make accurate predictions but also to compare the effectiveness of different regression algorithms on this specific dataset.

        **Methodology:**
        1.  **Data Preparation:** The dataset was loaded and enhanced with time-based features (`hour`, `day_of_week`, `month`) derived from the timestamp column.
        2.  **Train-Test Split:** To ensure a realistic evaluation of the models, the data was split into a training set (80%) and a testing set (20%).
        3.  **Model Training:** Seven different regression models were trained on the training data: Linear Regression, Ridge, Lasso, SVR, Random Forest, XGBoost, and LightGBM.
        4.  **Performance Evaluation:** Each model's performance was rigorously evaluated on the unseen test data using standard metrics: MAE, RMSE, and R².
        5.  **Interactive Application:** A multi-language Streamlit application was developed to visualize and compare model predictions on new random data, display performance metrics, and analyze the feature importances of tree-based models.

        **Tech Stack:** Python, Streamlit, Pandas, Scikit-learn, XGBoost, LightGBM, Altair.
        """
translations['tr']['models']['linear_regression']['description'] = translations['en']['models']['linear_regression']['description'].replace("What are they?", "Nedir?").replace("These are the simplest forms of regression models.", "Bunlar regresyon modellerinin en temel formlarıdır.").replace("They try to find a linear relationship (a straight line) between the input features (like temperature) and the target variable (energy consumption).", "Girdi özellikleri (örn: sıcaklık) ile hedef değişken (enerji tüketimi) arasında doğrusal bir ilişki (düz bir çizgi) bulmaya çalışırlar.").replace("finds the best-fitting line.", "en uygun çizgiyi bulur.").replace("are advanced versions that add a penalty for complexity, which helps prevent overfitting and makes the model more robust.", "ise bu modelin karmaşıklığa bir ceza eklenmiş gelişmiş versiyonlarıdır. Bu, ezberlemeyi (overfitting) önlemeye yardımcı olur ve modeli daha güvenilir kılar.")
translations['tr']['models']['linear_regression']['performance'] = translations['en']['models']['linear_regression']['performance'].replace("Performance in This Project", "Bu Projedeki Performansı").replace("Very Poor", "Çok Zayıf").replace("An R² score of 0.17 means these models could only explain about 17% of the variance in energy consumption.", "0.17'lik bir R² skoru, bu modellerin enerji tüketimindeki değişkenliğin sadece yaklaşık %17'sini açıklayabildiği anlamına gelir.").replace("This poor performance is a strong indicator that the relationship between the sensor data and energy usage is", "Bu zayıf performans, sensör verileri ile enerji kullanımı arasındaki ilişkinin").replace("The complex interactions (e.g., how time of day combined with temperature affects usage) cannot be captured by a simple straight line, causing these models to fail.", "Basit bir düz çizgi, verideki karmaşık etkileşimleri (örneğin, günün saati ile sıcaklığın birleşiminin tüketimi nasıl etkilediği gibi) yakalayamaz, bu yüzden bu modeller başarısız olmuştur.")
translations['tr']['models']['svr']['description'] = translations['en']['models']['svr']['description'].replace("What is it?", "Nedir?").replace("SVR works by fitting a line that stays within a certain error margin (a \"street\") of as many data points as possible.", "SVR, mümkün olduğunca çok veri noktasına belirli bir hata payı (\"sokak\" olarak adlandırılan bir aralık) içinde kalan en uygun çizgiyi bulmaya çalışır.").replace("Its main strength is its ability to use \"kernels\" to transform the data, allowing it to find complex, non-linear relationships that linear models can't.", "Asıl gücü, veriyi dönüştürmek için \"çekirdek\" (kernel) adı verilen fonksiyonları kullanarak doğrusal modellerin bulamadığı karmaşık ve doğrusal olmayan ilişkileri bulabilmesidir.")
translations['tr']['models']['svr']['performance'] = translations['en']['models']['svr']['performance'].replace("Performance in This Project", "Bu Projedeki Performansı").replace("Extremely Poor", "Aşırı Kötü").replace("A negative R² score means the model performed worse than simply guessing the average energy consumption for every data point.", "Negatif bir R² skoru, modelin her veri noktası için sadece ortalama enerji tüketimini tahmin etmekten bile daha kötü performans gösterdiği anlamına gelir.").replace("This is a catastrophic failure, and the primary reason is", "Bu feci sonucun temel nedeni").replace("SVR is highly sensitive to the scale of input features and its internal parameters (`C`, `gamma`). Without scaling the data and finding the right parameters, it can produce completely meaningless results, as seen here. This is a perfect example of why preprocessing is crucial.", "SVR, girdi özelliklerinin ölçeğine ve kendi iç parametrelerine (`C`, `gamma`) karşı aşırı hassastır. Veriyi ölçeklendirmeden ve doğru parametreleri bulmadan, burada görüldüğü gibi tamamen anlamsız sonuçlar üretebilir. Bu durum, ön işlemenin neden hayati olduğunu gösteren mükemmel bir örnektir.")
translations['tr']['models']['random_forest']['description'] = translations['en']['models']['random_forest']['description'].replace("What is it?", "Nedir?").replace("Random Forest is an \"ensemble\" model.", "Rastgele Orman, bir \"topluluk\" (ensemble) modelidir.").replace("It builds hundreds of individual \"decision trees\" and then averages their predictions.", "Yüzlerce bireysel \"karar ağacı\" oluşturur ve ardından bu ağaçların tahminlerinin ortalamasını alır.").replace("Each tree is trained on a random sample of the data and features, which makes the overall model very robust and less prone to overfitting.", "Her ağaç, verinin ve özelliklerin rastgele bir alt kümesi üzerinde eğitilir. Bu, genel modeli çok güvenilir kılar ve ezberleme riskini azaltır.").replace("It's excellent at capturing complex, non-linear interactions.", "Karmaşık ve doğrusal olmayan etkileşimleri yakalamada mükemmeldir.")
translations['tr']['models']['random_forest']['performance'] = translations['en']['models']['random_forest']['performance'].replace("Performance in This Project", "Bu Projedeki Performansı").replace("The Best Performer", "En Başarılı Model").replace("Random Forest achieved the highest R² score, successfully explaining about 56% of the data's variance.", "En yüksek R² skorunu Rastgele Orman elde ederek verideki değişkenliğin yaklaşık %56'sını başarıyla açıklamıştır.").replace("Its success demonstrates that the data contains complex, non-linear patterns and feature interactions.", "Bu başarı, verinin karmaşık, doğrusal olmayan desenler ve özellik etkileşimleri içerdiğini göstermektedir.").replace("It can learn specific rules (e.g., \"if the hour is late AND the lights are on, consumption is high\") that are impossible for linear models to capture.", "Model, doğrusal modellerin yakalayamayacağı belirli kuralları (örneğin, \"eğer saat akşam ise VE ışıklar açıksa, tüketim yüksektir\" gibi) öğrenebilir.").replace("This result confirms that a tree-based ensemble approach is well-suited for this problem.", "Bu sonuç, ağaç tabanlı bir topluluk yaklaşımının bu problem için çok uygun olduğunu doğrulamaktadır.")
translations['tr']['models']['boosting']['description'] = translations['en']['models']['boosting']['description'].replace("What are they?", "Nedir?").replace("These are both powerful, advanced \"gradient boosting\" models.", "İkisi de güçlü ve gelişmiş \"gradyan artırma\" (gradient boosting) modelleridir.").replace("They also build decision trees, but they do it sequentially.", "Onlar da karar ağaçları oluşturur, ancak bunu sıralı bir şekilde yaparlar.").replace("Each new tree is specifically built to correct the errors made by the previous ones.", "Her yeni ağaç, özellikle bir önceki ağaçların yaptığı hataları düzeltmek için oluşturulur.").replace("This focused, iterative improvement process makes them some of the most powerful models for tabular data.", "Bu odaklanmış, yinelemeli iyileştirme süreci, onları yapısal (tabular) veriler için en güçlü modellerden biri yapar.").replace("LightGBM is a modern optimization of this idea, often being much faster than XGBoost.", "LightGBM, bu fikrin modern bir optimizasyonudur ve genellikle XGBoost'tan çok daha hızlıdır.")
translations['tr']['models']['boosting']['performance'] = translations['en']['models']['boosting']['performance'].replace("Performance in This Project", "Bu Projedeki Performansı").replace("Very Good", "Çok İyi").replace("Both models performed strongly, coming in just behind Random Forest.", "Her iki model de güçlü bir performans sergileyerek Rastgele Orman'ın hemen arkasında yer aldı.").replace("This is expected, as boosting algorithms are industry-standard for high performance.", "Bu beklenen bir durumdur, çünkü artırma algoritmaları yüksek performans için endüstri standardıdır.").replace("Their ability to correct their own mistakes over iterations allows them to create highly accurate and robust prediction models.", "Kendi hatalarını tekrar tekrar düzeltebilme yetenekleri, son derece isabetli ve güvenilir tahmin modelleri oluşturmalarını sağlar.").replace("The small performance difference between them and Random Forest is likely due to using default hyperparameters.", "Rastgele Orman ile aralarındaki küçük performans farkı, muhtemelen varsayılan hiperparametrelerin kullanılmasından kaynaklanmaktadır.").replace("With proper tuning, they could potentially become the top performers.", "Doğru bir optimizasyon ile potansiyel olarak en iyi performansı gösterebilirler.")
translations['tr']['project_details_text'] = translations['en']['project_details_text'].replace("This project aims to predict the energy consumption of appliances in a home using time-series data from various IoT sensors.", "Bu proje, bir evdeki çeşitli IoT sensörlerinden gelen zaman serisi verilerini kullanarak cihazların enerji tüketimini tahmin etmeyi amaçlamaktadır.").replace("The goal is not only to make accurate predictions but also to compare the effectiveness of different regression algorithms on this specific dataset.", "Hedef, yalnızca doğru tahminler yapmak değil, aynı zamanda bu veri setinde farklı regresyon algoritmalarının etkinliğini karşılaştırmaktır.").replace("Methodology", "Metodoloji").replace("Data Preparation", "Veri Hazırlığı").replace("The dataset was loaded and enhanced with time-based features (`hour`, `day_of_week`, `month`) derived from the timestamp column.", "Veri seti yüklendi ve zaman damgası sütunundan `saat`, `haftanın günü`, `ay` gibi zamana dayalı yeni özellikler türetilerek zenginleştirildi.").replace("Train-Test Split", "Train-Test Ayrımı").replace("To ensure a realistic evaluation of the models, the data was split into a training set (80%) and a testing set (20%).", "Modellerin gerçekçi bir şekilde değerlendirilmesini sağlamak için veri, bir eğitim seti (%80) ve bir test setine (%20) ayrıldı.").replace("Model Training", "Model Eğitimi").replace("Seven different regression models were trained on the training data: Linear Regression, Ridge, Lasso, SVR, Random Forest, XGBoost, and LightGBM.", "Eğitim verisi üzerinde yedi farklı regresyon modeli eğitildi: Doğrusal Regresyon, Ridge, Lasso, SVR, Rastgele Orman, XGBoost ve LightGBM.").replace("Performance Evaluation", "Performans Değerlendirmesi").replace("Each model's performance was rigorously evaluated on the unseen test data using standard metrics: MAE, RMSE, and R².", "Her modelin performansı, daha önce görmediği test verileri üzerinde MAE, RMSE ve R² gibi standart metrikler kullanılarak titizlikle değerlendirildi.").replace("Interactive Application", "İnteraktif Uygulama").replace("A multi-language Streamlit application was developed to visualize and compare model predictions on new random data, display performance metrics, and analyze the feature importances of tree-based models.", "Yeni rastgele veriler üzerindeki model tahminlerini görselleştirmek, performans metriklerini görüntülemek ve ağaç tabanlı modellerin özellik önemini analiz etmek için çok dilli bir Streamlit uygulaması geliştirildi.").replace("Tech Stack", "Kullanılan Teknolojiler")

T = translations[lang_code]
T_MODELS = T["models"]

@st.cache_resource
def load_resources():
    try:
        model_names = ["Linear_Regression", "Ridge", "Lasso", "SVR", "Random_Forest", "XGBoost", "LightGBM"]
        models = {name.replace("_", " "): joblib.load(f'model_{name}.pkl') for name in model_names}
        model_features = joblib.load('model_features.pkl')
        model_metrics = joblib.load('model_metrics.pkl')
        return models, model_features, model_metrics
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all models and metrics are generated by running `train_model.py`.")
        return None, None, None

models, model_features, model_metrics = load_resources()

st.title(T["page_title"])

if not all([models, model_features, model_metrics]):
    st.error(T["error_loading"])
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([T["tab_predictor"], T["tab_performance"], T["tab_models_explained"], T["tab_details"]])

with tab1:
    st.header(T["predictor_header"])
    st.write(T["predictor_desc"])

    if st.button(T["button_generate"]):
        input_dict = {'lights': np.random.randint(0, 50), 'T1': np.random.uniform(19, 27), 'RH_1': np.random.uniform(33, 53), 'T2': np.random.uniform(16, 28), 'RH_2': np.random.uniform(20, 50), 'T3': np.random.uniform(20, 28), 'RH_3': np.random.uniform(35, 50), 'T4': np.random.uniform(18, 27), 'RH_4': np.random.uniform(30, 50), 'T5': np.random.uniform(18, 26), 'RH_5': np.random.uniform(40, 55), 'T6': np.random.uniform(-5, 25), 'RH_6': np.random.uniform(1, 100), 'T7': np.random.uniform(18, 26), 'RH_7': np.random.uniform(28, 45), 'T8': np.random.uniform(20, 27), 'RH_8': np.random.uniform(35, 55), 'T9': np.random.uniform(18, 24), 'RH_9': np.random.uniform(38, 50), 'T_out': np.random.uniform(-6, 29), 'Press_mm_hg': np.random.uniform(730, 770), 'RH_out': np.random.uniform(24, 100), 'Windspeed': np.random.uniform(0, 14), 'Visibility': np.random.uniform(1, 65), 'Tdewpoint': np.random.uniform(-15, 20), 'hour': np.random.randint(0, 24), 'day_of_week': np.random.randint(0, 7), 'month': np.random.randint(1, 13)}
        input_df = pd.DataFrame([input_dict])[model_features]
        predictions = {name: max(0, int(model.predict(input_df)[0])) for name, model in models.items()}
        st.subheader(T["results_header"])
        st.markdown(T["prediction_chart_desc"])
        pred_df = pd.DataFrame(list(predictions.items()), columns=[T["results_chart_model"], T["results_chart_prediction"]])
        chart = alt.Chart(pred_df).mark_bar().encode(x=alt.X(T["results_chart_model"], sort=None, title=T["results_chart_model"]), y=alt.Y(T["results_chart_prediction"], title=T["results_chart_prediction"]), color=alt.Color(T["results_chart_model"], legend=None), tooltip=[T["results_chart_model"], T["results_chart_prediction"]]).properties(title=T["results_chart_title"])
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.header(T["feature_importance_header"])
    st.write(T["feature_importance_desc"])
    
    importance_models = ["Random Forest", "XGBoost", "LightGBM"]
    selected_model_for_importance = st.selectbox(T["feature_importance_select"], options=importance_models)

    if selected_model_for_importance:
        model = models[selected_model_for_importance]
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': model_features, 'importance': importances}).sort_values('importance', ascending=False).head(15)
        
        imp_chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X('importance:Q', title=T["importance_axis_title"]),
            y=alt.Y('feature:N', sort='-x', title=T["feature_axis_title"])
        ).properties(
            title=T["feature_importance_chart_title"].format(model_name=selected_model_for_importance)
        )
        st.altair_chart(imp_chart, use_container_width=True)
        st.markdown(T["feature_importance_chart_note"])

with tab2:
    st.header(T["performance_header"])
    st.write(T["performance_desc"])
    
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.rename_axis('Model').reset_index()
    
    st.dataframe(metrics_df.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}"
    }), use_container_width=True)
    
    st.markdown(T["performance_table_interpretation"])
    st.markdown("---")
    st.info(T["mae_desc"])
    st.info(T["rmse_desc"])
    st.info(T["r2_desc"])

with tab3:
    st.header(T["models_explained_header"])

    with st.expander(T_MODELS["linear_regression"]["title"]):
        st.markdown(T_MODELS["linear_regression"]["description"])
        st.markdown(T_MODELS["linear_regression"]["performance"])

    with st.expander(T_MODELS["svr"]["title"]):
        st.markdown(T_MODELS["svr"]["description"])
        st.markdown(T_MODELS["svr"]["performance"])

    with st.expander(T_MODELS["random_forest"]["title"]):
        st.markdown(T_MODELS["random_forest"]["description"])
        st.markdown(T_MODELS["random_forest"]["performance"])

    with st.expander(T_MODELS["boosting"]["title"]):
        st.markdown(T_MODELS["boosting"]["description"])
        st.markdown(T_MODELS["boosting"]["performance"])

with tab4:
    st.header(T["project_details_header"])
    st.markdown(T["project_details_text"])