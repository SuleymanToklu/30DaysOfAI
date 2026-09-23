import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
try:
    df = pd.read_csv('day-1/zomato.csv')
except FileNotFoundError:
    st.error("The file 'zomato.csv' was not found. Please ensure it is in the correct directory.")
    st.stop()
# Set page configuration
st.set_page_config(layout="wide")

# --- LOAD RESOURCES ---

@st.cache_resource
def load_model():
    """Loads the pre-trained restaurant success model."""
    try:
        model = joblib.load('day-1/restaurant_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'restaurant_model.pkl' not found.")
        return None

@st.cache_data
def load_data():
    """Loads and caches the zomato dataset from the local repository."""
    try:
        df = pd.read_csv('day-1/zomato.csv')
        df.dropna(subset=['location', 'rest_type', 'cuisines', 'listed_in(city)', 'listed_in(type)'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Dataset file 'zomato.csv' not found. Make sure it's in the 'day-1' directory.")
        return None

    
@st.cache_data
def load_model_columns():
    """Loads the list of columns the model was trained on."""
    try:
        # Provide the full path from the root of the repository
        return joblib.load('day-1/model_columns.pkl')
    except FileNotFoundError:
        st.error("Model columns file 'model_columns.pkl' not found.")
        return None

# Load all necessary files
model = load_model()
df_original = load_data()
model_columns = load_model_columns()

# --- PAGE CONTENT ---

st.title('📈 Restoran Başarı Skoru Tahmincisi')
st.write("Lütfen soldaki menüden restoranın özelliklerini girerek potansiyel başarı puanını tahmin edin.")

# --- SIDEBAR UI ---

st.sidebar.header('Restoran Özellikleri')

if df_original is not None:
    # Use original, correct column names from the CSV for the selectbox options
    location = st.sidebar.selectbox('Konum (Location)', options=sorted(df_original['location'].unique()))
    rest_type = st.sidebar.selectbox('Restoran Tipi', options=sorted(df_original['rest_type'].unique()))
    cuisines = st.sidebar.selectbox('Mutfak Türü', options=sorted(df_original['cuisines'].unique()))
    city = st.sidebar.selectbox('Bölge (City)', options=sorted(df_original['listed_in(city)'].unique()))
    type_ = st.sidebar.selectbox('Servis Tipi', options=sorted(df_original['listed_in(type)'].unique()))
    cost_for_two = st.sidebar.number_input('İki Kişilik Ortalama Maliyet (INR)', min_value=100, max_value=6000, value=700, step=50)
    votes = st.sidebar.number_input('Toplam Oy Sayısı (Tahmini)', min_value=0, max_value=17000, value=500, step=10)
    online_order = st.sidebar.radio('Online Sipariş Var mı?', ('Yes', 'No'))
    book_table = st.sidebar.radio('Masa Rezervasyonu Var mı?', ('Yes', 'No'))
else:
    st.sidebar.error("Veri seti yüklenemediği için özellikler seçilemiyor.")

# --- PREDICTION LOGIC ---

if st.sidebar.button('Başarıyı Tahmin Et!'):
    # Check if all necessary files were loaded
    if all(v is not None for v in [model, df_original, model_columns]):
        
        # 1. Create a dictionary with user inputs, using the original column names
        input_dict = {
            'online_order': online_order, 'book_table': book_table, 'votes': votes, 'location': location,
            'rest_type': rest_type, 'cuisines': cuisines, 'cost_for_two': cost_for_two,
            'listed_in(type)': type_,
            'listed_in(city)': city
        }
        
        # 2. Preprocess the input data
        st.write("---")
        st.subheader("İşlem Adımları:")
        with st.spinner('Veri ön işleniyor ve model çalıştırılıyor...'):
            # Create a single-row DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Map Yes/No to 1/0
            input_df['online_order'] = input_df['online_order'].map({'Yes': 1, 'No': 0})
            input_df['book_table'] = input_df['book_table'].map({'Yes': 1, 'No': 0})
            
            # This will hold the final, processed data for the model
            model_input = pd.DataFrame()
            
            # Add non-categorical features first
            model_input['online_order'] = input_df['online_order']
            model_input['book_table'] = input_df['book_table']
            model_input['votes'] = input_df['votes']
            model_input['cost_for_two'] = input_df['cost_for_two']
            
            # Encode categorical features
            for column in ['location', 'rest_type', 'cuisines', 'listed_in(type)', 'listed_in(city)']:
                le = LabelEncoder()
                le.fit(df_original[column].astype(str).unique())
                
                # The name of the column after renaming in the notebook
                renamed_col = column
                if column == 'listed_in(type)':
                    renamed_col = 'type'
                elif column == 'listed_in(city)':
                    renamed_col = 'city'
                
                # Transform the user input and add to the model_input DataFrame
                model_input[renamed_col] = le.transform(input_df[column].astype(str))
            
            st.write("✅ Ön işleme tamamlandı.")
            
            # CRITICAL FIX: Reorder the columns to match the exact order the model was trained on
            model_input_reordered = model_input[model_columns]
            
            # 3. Make prediction
            prediction = model.predict(model_input_reordered)
            st.write("✅ Tahmin yapıldı.")
        
        # 4. Display the result
        st.subheader('📈 Tahmin Sonucu')
        st.metric(label="Tahmini Restoran Puanı (5 üzerinden)", value=f"{prediction[0]:.2f}")
        
        if prediction[0] >= 4.1:
            st.success('Bu restoranın başarılı olma potansiyeli YÜKSEK! Yatırım yapılabilir. 🥳')
            st.balloons()
        elif prediction[0] >= 3.5:
            st.warning('Bu restoran ortalama bir potansiyele sahip. Dikkatli bir değerlendirme gerekir. 🤔')
        else:
            st.error('Bu restoranın başarılı olma potansiyeli DÜŞÜK görünüyor. Riskli bir yatırım olabilir. 😥')
    else:
        st.error("Model veya veri dosyaları yüklenemedi. Lütfen önce notebook'u çalıştırdığından emin ol.")