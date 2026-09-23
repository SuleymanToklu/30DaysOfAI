import gradio as gr
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import os
import shap
from train_model import train_and_save # Import the training function

# Use a non-interactive backend for Matplotlib to prevent display issues on servers
matplotlib.use('Agg')

# --- Check for Model Files and Train if Necessary ---
# This part ensures the app can self-initialize on a new server
MODEL_FILE = 'heart_disease_model.pkl'
if not os.path.exists(MODEL_FILE):
    train_and_save()

# --- Load Pre-trained Objects ---
try:
    model = joblib.load('heart_disease_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError as e:
    print(f"FATAL ERROR: Model files not found even after training attempt: {e}")
    exit()

# --- Language Dictionary for UI Text ---
# This dictionary holds all text elements for easy translation
LANG_TEXT = {
    'tr': {
        'title': "🩺 Kalp Hastalığı Risk Tahmini (XAI ile)",
        'description': "Bu uygulama, tıbbi verilerinize dayanarak kalp hastalığı riskini tahmin eder ve her bir tahminin nedenlerini SHAP grafiği ile açıklar.",
        'lang_label': "Dil",
        'age': {"label": "Yaş", "info": "Yıl olarak yaşınız."},
        'sex': {"label": "Cinsiyet", "info": "0: Kadın, 1: Erkek"},
        'cp': {"label": "Göğüs Ağrısı Tipi (cp)", "info": "1: Tipik Anjina, 2: Atipik Anjina, 3: Anjina Olmayan Ağrı, 4: Asemptomatik"},
        'trestbps': {"label": "Dinlenme Kan Basıncı (trestbps)", "info": "Hastaneye girişteki kan basıncı (mm Hg)."},
        'chol': {"label": "Serum Kolesterolü (chol)", "info": "mg/dl cinsinden serum kolesterolü."},
        'fbs': {"label": "Açlık Kan Şekeri > 120 mg/dl (fbs)", "info": "1: Evet, 0: Hayır"},
        'restecg': {"label": "Dinlenme EKG Sonuçları (restecg)", "info": "0: Normal, 1: ST-T dalga anormalliği, 2: Olası veya kesin sol ventrikül hipertrofisi"},
        'thalach': {"label": "Ulaşılan Maksimum Kalp Atış Hızı (thalach)", "info": "Egzersiz sırasında ulaşılan en yüksek kalp atış hızı."},
        'exang': {"label": "Egzersize Bağlı Anjina (exang)", "info": "Egzersiz sonrası göğüs ağrısı oldu mu? 1: Evet, 0: Hayır"},
        'oldpeak': {"label": "ST Depresyonu (oldpeak)", "info": "Egzersizin dinlenmeye göre neden olduğu ST çökmesi."},
        'slope': {"label": "ST Segment Eğimi (slope)", "info": "Zirve egzersiz ST segmentinin eğimi. 1: Yükselen, 2: Düz, 3: Alçalan"},
        'ca': {"label": "Renkli Ana Damar Sayısı (ca)", "info": "Floroskopi ile boyanan ana damar sayısı (0-3)."},
        'thal': {"label": "Talasemi Durumu (thal)", "info": "3: Normal, 6: Sabit Defekt, 7: Tersinir Defekt"},
        'submit_button': "Tahmin Et",
        'output_label': "Tahmin Sonucu",
        'output_plot': "Tahmin Açıklaması (SHAP Şelale Grafiği)",
        'no_disease': "Kalp Hastalığı Yok",
        'has_disease': "Kalp Hastalığı Riski Var"
    },
    'en': {
        'title': "🩺 Heart Disease Risk Prediction with XAI",
        'description': "This app predicts heart disease risk based on your medical data and explains the reasons for each prediction with a SHAP plot.",
        'lang_label': "Language",
        'age': {"label": "Age", "info": "Your age in years."},
        'sex': {"label": "Sex", "info": "0: Female, 1: Male"},
        'cp': {"label": "Chest Pain Type (cp)", "info": "1: Typical Angina, 2: Atypical Angina, 3: Non-anginal Pain, 4: Asymptomatic"},
        'trestbps': {"label": "Resting Blood Pressure (trestbps)", "info": "Blood pressure on admission to the hospital (mm Hg)."},
        'chol': {"label": "Serum Cholestoral (chol)", "info": "Serum cholestoral in mg/dl."},
        'fbs': {"label": "Fasting Blood Sugar > 120 mg/dl (fbs)", "info": "1: True, 0: False"},
        'restecg': {"label": "Resting ECG Results (restecg)", "info": "0: Normal, 1: Having ST-T wave abnormality, 2: Showing probable or definite left ventricular hypertrophy"},
        'thalach': {"label": "Maximum Heart Rate Achieved (thalach)", "info": "Highest heart rate achieved during exercise."},
        'exang': {"label": "Exercise Induced Angina (exang)", "info": "Did exercise lead to chest pain? 1: Yes, 0: No"},
        'oldpeak': {"label": "ST Depression (oldpeak)", "info": "ST depression induced by exercise relative to rest."},
        'slope': {"label": "Slope of the peak exercise ST segment (slope)", "info": "1: Upsloping, 2: Flat, 3: Downsloping"},
        'ca': {"label": "Number of major vessels colored (ca)", "info": "Number of major vessels (0-3) colored by flourosopy."},
        'thal': {"label": "Thalassemia Status (thal)", "info": "3: Normal, 6: Fixed Defect, 7: Reversable Defect"},
        'submit_button': "Predict",
        'output_label': "Prediction Result",
        'output_plot': "Prediction Explanation (SHAP Waterfall Plot)",
        'no_disease': "No Heart Disease",
        'has_disease': "Risk of Heart Disease"
    }
}


# --- Prediction and Explanation Function (Core Logic) ---
def predict_and_explain(lang_code, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=feature_names
    )
    
    prediction_proba = model.predict_proba(input_data)[0]
    
    lang = 'tr' if lang_code == 'TR' else 'en'
    confidences = {
        LANG_TEXT[lang]['no_disease']: float(prediction_proba[0]),
        LANG_TEXT[lang]['has_disease']: float(prediction_proba[1])
    }
    
    # NEW, MORE ROBUST PLOTTING LOGIC
    # Create a SHAP Explanation object, which is the modern way to handle SHAP values.
    explanation = explainer(input_data)

    # We want to explain the prediction for the "Heart Disease" class (which is class 1).
    # We select the values for the first (and only) sample [0] and for the class 1 [:, 1].
    shap_values_for_class_1 = explanation[0, :, 1]

    # Create the waterfall plot. This function is more stable than force_plot.
    # It handles figure creation internally when show=False is passed.
    shap.waterfall_plot(shap_values_for_class_1, show=False)
    
    # Get the current matplotlib figure object that the waterfall_plot just drew on.
    fig = plt.gcf()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    return confidences, fig

# --- UI Language Update Function ---
def update_language(lang_code):
    lang = 'tr' if lang_code == 'TR' else 'en'
    
    # Create a list of gr.update() objects for each component
    updates = [
        gr.update(label=LANG_TEXT[lang]['age']['label'], info=LANG_TEXT[lang]['age']['info']),
        gr.update(label=LANG_TEXT[lang]['sex']['label'], info=LANG_TEXT[lang]['sex']['info']),
        gr.update(label=LANG_TEXT[lang]['cp']['label'], info=LANG_TEXT[lang]['cp']['info']),
        gr.update(label=LANG_TEXT[lang]['trestbps']['label'], info=LANG_TEXT[lang]['trestbps']['info']),
        gr.update(label=LANG_TEXT[lang]['chol']['label'], info=LANG_TEXT[lang]['chol']['info']),
        gr.update(label=LANG_TEXT[lang]['fbs']['label'], info=LANG_TEXT[lang]['fbs']['info']),
        gr.update(label=LANG_TEXT[lang]['restecg']['label'], info=LANG_TEXT[lang]['restecg']['info']),
        gr.update(label=LANG_TEXT[lang]['thalach']['label'], info=LANG_TEXT[lang]['thalach']['info']),
        gr.update(label=LANG_TEXT[lang]['exang']['label'], info=LANG_TEXT[lang]['exang']['info']),
        gr.update(label=LANG_TEXT[lang]['oldpeak']['label'], info=LANG_TEXT[lang]['oldpeak']['info']),
        gr.update(label=LANG_TEXT[lang]['slope']['label'], info=LANG_TEXT[lang]['slope']['info']),
        gr.update(label=LANG_TEXT[lang]['ca']['label'], info=LANG_TEXT[lang]['ca']['info']),
        gr.update(label=LANG_TEXT[lang]['thal']['label'], info=LANG_TEXT[lang]['thal']['info']),
        gr.update(value=LANG_TEXT[lang]['submit_button']),
        gr.update(label=LANG_TEXT[lang]['output_label']),
        gr.update(label=LANG_TEXT[lang]['output_plot']),
        gr.update(value=LANG_TEXT[lang]['title']),
        gr.update(value=LANG_TEXT[lang]['description'])
    ]
    return updates

# --- Gradio UI with gr.Blocks for dynamic updates ---
with gr.Blocks() as app:
    # First, define all components
    title = gr.Markdown(LANG_TEXT['tr']['title'])
    description = gr.Markdown(LANG_TEXT['tr']['description'])
    
    with gr.Row():
        lang_selector = gr.Radio(['TR', 'EN'], value='TR', label="Dil / Language")

    with gr.Row():
        with gr.Column(scale=1):
            age = gr.Slider(minimum=20, maximum=80, value=52, **LANG_TEXT['tr']['age'])
            sex = gr.Radio([0, 1], value=1, **LANG_TEXT['tr']['sex'])
            cp = gr.Radio([1, 2, 3, 4], value=4, **LANG_TEXT['tr']['cp'])
            trestbps = gr.Slider(minimum=90, maximum=200, value=120, **LANG_TEXT['tr']['trestbps'])
            chol = gr.Slider(minimum=120, maximum=570, value=240, **LANG_TEXT['tr']['chol'])
            fbs = gr.Radio([0, 1], value=0, **LANG_TEXT['tr']['fbs'])
            restecg = gr.Radio([0, 1, 2], value=0, **LANG_TEXT['tr']['restecg'])
        
        with gr.Column(scale=1):
            thalach = gr.Slider(minimum=70, maximum=210, value=150, **LANG_TEXT['tr']['thalach'])
            exang = gr.Radio([0, 1], value=0, **LANG_TEXT['tr']['exang'])
            oldpeak = gr.Slider(minimum=0.0, maximum=6.2, step=0.1, value=1.0, **LANG_TEXT['tr']['oldpeak'])
            slope = gr.Radio([1, 2, 3], value=2, **LANG_TEXT['tr']['slope'])
            ca = gr.Radio([0, 1, 2, 3], value=0, **LANG_TEXT['tr']['ca'])
            thal = gr.Radio([3, 6, 7], value=3, **LANG_TEXT['tr']['thal'])
    
    submit_button = gr.Button(LANG_TEXT['tr']['submit_button'], variant="primary")
    
    with gr.Row():
        output_label = gr.Label(num_top_classes=2, label=LANG_TEXT['tr']['output_label'])
        output_plot = gr.Plot(label=LANG_TEXT['tr']['output_plot'])
    
    # Define component lists for easier event handling
    input_components = [lang_selector, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    output_components = [output_label, output_plot]
    ui_components_to_update = [
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, 
        oldpeak, slope, ca, thal, submit_button, output_label, 
        output_plot, title, description
    ]

    # --- Event Handlers ---
    # When the language selector changes, update all UI component labels
    lang_selector.change(
        fn=update_language,
        inputs=lang_selector,
        outputs=ui_components_to_update
    )
    
    # When the submit button is clicked, run the prediction
    submit_button.click(
        fn=predict_and_explain,
        inputs=input_components,
        outputs=output_components
    )

if __name__ == "__main__":
    app.launch()

