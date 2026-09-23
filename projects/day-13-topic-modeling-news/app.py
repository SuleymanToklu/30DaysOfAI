import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Konu Keşfi: Örümcek Ağındaki Kelimeler",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 
# "Örümcek Ağı" ve "Yapay Zeka" temasına uygun karanlık ve fütüristik bir stil
CUSTOM_CSS = """
<style>
    /* Ana arkaplan ve metin renkleri */
    .stApp {
        background-color: #0a0a1a;
        color: #e0e0e0;
    }
    
    /* Başlık stilleri */
    h1, h2, h3 {
        color: #00bfff; /* Deep Sky Blue */
        text-shadow: 2px 2px 4px #000000;
    }

    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1a2e;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #a0a0a0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00bfff;
        color: white;
        font-weight: bold;
    }
    
    /* Konu kartları için özel stil */
    .topic-card {
        background-color: #1a1a2e;
        border: 1px solid #00bfff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0, 191, 255, 0.2);
        transition: 0.3s;
        height: 100%;
    }
    .topic-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0, 191, 255, 0.4);
    }
    .topic-title {
        font-size: 1.5em;
        color: #ffffff;
        border-bottom: 2px solid #00bfff;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .topic-words {
        font-size: 1.1em;
        color: #c0c0c0;
        line-height: 1.6;
    }
    
    /* Expander stilleri */
    .st-expander {
        border: 1px solid #333;
        border-radius: 5px;
    }
    .st-expander header {
        font-size: 1.2em;
        font-weight: bold;
        color: #00bfff;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Kaynakları Yükleme 
@st.cache_resource
def load_resources():
    """LDA modelini ve vektörleyiciyi diskten yükler."""
    try:
        lda_model = joblib.load('lda_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return lda_model, vectorizer
    except FileNotFoundError:
        return None, None

lda_model, vectorizer = load_resources()

# Ana Başlık ve Giriş 
st.title("🕸️ Konu Keşfi: Haber Başlıkları Ağındaki Gizli Desenler")
st.markdown("""
Bu proje, bir **Gözetimsiz Öğrenme** modeli olan **LDA (Latent Dirichlet Allocation)** kullanarak binlerce haber başlığının ardındaki gizli "konuları" ortaya çıkarıyor. 
Tıpkı bir örümceğin ağını örerek farklı noktaları birleştirmesi gibi, bu yapay zeka da kelimeler arasındaki görünmez bağlantıları bularak anlamlı konu grupları oluşturuyor.
""")

# Hata Kontrolü 
if lda_model is None or vectorizer is None:
    st.error(
        "Gerekli model dosyaları (.pkl) bulunamadı. "
        "Lütfen önce `python process_data.py` komutunu çalıştırdığınızdan emin olun."
    )
    st.stop()

# Sekmeli İçerik Yapısı 
tab1, tab2 = st.tabs(["**🕷️ KEŞFEDİLEN KONU AĞLARI**", "**🧠 MODELİN BEYNİ: DERİNLEMESİNE ANALİZ**"])

#  Helper Function for Topic Card 
def create_topic_card(topic_num, words):
    """HTML ve CSS kullanarak şık bir konu kartı oluşturur."""
    words_html = f"<span class='topic-words'>{', '.join(words)}</span>"
    card_html = f"""
    <div class="topic-card">
        <h3 class="topic-title">Konu Ağı #{topic_num}</h3>
        {words_html}
    </div>
    """
    return card_html

with tab1:
    st.header("Yapay Zekanın Ördüğü 10 Anlam Ağı")
    st.write(
        "LDA modeli, 20,000 haber başlığını analiz ederek aşağıdaki 10 gizli konuyu ve bu konuları en iyi tanımlayan "
        "anahtar kelimeleri tamamen kendi başına, hiçbir insan müdahalesi olmadan buldu. Her bir kart, kelimelerin bir araya gelerek oluşturduğu bir 'anlam ağını' temsil ediyor."
    )
    st.markdown("---")

    feature_names = vectorizer.get_feature_names_out()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        with columns[topic_idx % 5]:
            card = create_topic_card(topic_idx + 1, top_words)
            st.markdown(card, unsafe_allow_html=True)


with tab2:
    st.header("Projenin ve Modelin Ardındaki Mantık")

    st.markdown("""
    Bu bölümde, projenin "nasıl" çalıştığını, Gözetimsiz Öğrenme'nin ne anlama geldiğini ve LDA modelinin kelime yığınlarından nasıl anlamlı konular çıkardığını keşfedeceğiz.
    """)

    with st.expander("📖 Projenin Hikayesi: Kaos İçindeki Düzen Arayışı"):
        st.markdown("""
        **Problem:** Her gün milyonlarca haber üretiliyor. Bu devasa bilgi okyanusunda hangi konuların popüler olduğunu, hangi olayların birbiriyle ilişkili olduğunu manuel olarak takip etmek imkansız. Elimizde sadece etiketlenmemiş binlerce başlık var.
        
        **Amaç:** Bu kaotik veri yığınına bir düzen getirmek. Başlıkları, içerdikleri ortak anlamlara göre otomatik olarak gruplayan bir sistem kurmak. Yani, "kaosun içindeki gizli düzeni" bulmak. İşte Gözetimsiz Öğrenme burada devreye giriyor.
        """)

    with st.expander("🤔 Gözetimsiz Öğrenme Nedir? (Kütüphaneci Analojisi)"):
        st.markdown("""
        Gözetimsiz Öğrenme'yi anlamanın en kolay yolu **kütüphaneci analojisidir**:
        
        Imagine a library receives thousands of books with no titles, no covers, and no category labels. A supervised learning librarian would need someone to read each book and tell them "this is a science fiction book," "this is a history book."
        
        An **unsupervised** learning librarian, on the other hand, is given no such help. They must figure out the categories on their own. How?
        
        1.  They start looking for patterns. They notice some books frequently use words like "galaxy," "spaceship," and "alien." They put these into a pile.
        2.  They find another group of books that mention "king," "battle," and "castle" a lot. That's another pile.
        3.  A third pile contains words like "economy," "market," and "shares."
        
        At the end of the day, the librarian has created piles (topics) without ever being told what the topics were. They *inferred* the topics from the words inside the books.
        
        **Bizim projemiz de tam olarak bunu yapıyor: Haber başlıkları kitaplar, LDA modeli ise bu zeki kütüphaneci.**
        """)

    with st.expander("🔮 LDA Modelinin Sırrı: Bir 'Tarif' Olarak Konular"):
        st.markdown(r"""
        **Latent Dirichlet Allocation (LDA)**, ilk bakışta karmaşık gelse de temelinde basit ve güçlü bir fikre dayanır:
        
        > 1.  Her **belge (haber başlığı)**, birden çok **konunun bir karışımıdır**.
        > 2.  Her **konu**, belirli **kelimelerin bir karışımıdır** (olasılık dağılımıdır).
        
        Bunu bir **smoothie tarifi** gibi düşünebiliriz:
        
        -   **Belge (Smoothie):** "Çilekli Muzlu Yaz Rüyası" smoothiemiz, `%60 Çilek Lezzeti`, `%30 Muz Lezzeti` ve `%10 Vanilya Esintisi`'nin bir karışımı olabilir.
        -   **Konu (Lezzet):** "Çilek Lezzeti" konusu ise `%50 'çilek'`, `%20 'şeker'`, `%15 'taze'`, `%10 'kırmızı'` gibi kelimelerin bir karışımından oluşur.
        
        LDA'nın yaptığı sihir, bize hem smoothielerin içindeki lezzet oranlarını ($P(\text{konu } | \text{ belge})$) hem de her bir lezzeti oluşturan kelime tariflerini ($P(\text{kelime } | \text{ konu})$) aynı anda sunmasıdır. Bunu, kelimelerin belgeler arasında nasıl dağıldığını istatistiksel olarak analiz ederek yapar.
        """)

    with st.expander("⚙️ Adım Adım Bizim Pipeline'ımız"):
        st.markdown("""
        Peki bu teoriyi pratiğe nasıl döktük? `process_data.py` script'i şu adımları izledi:
        
        1.  **Veri Yükleme:** `abcnews-date-text.csv` dosyasından 20,000 adet rastgele haber başlığı seçtik.
        2.  **Metin Temizleme ve Vektörleştirme (`CountVectorizer`):**
            -   Metinleri sayılara dönüştürmemiz gerekiyordu. `CountVectorizer` her bir başlığı aldı.
            -   'the', 'a', 'is' gibi yaygın İngilizce kelimeleri (`stop_words`) attı.
            -   Her kelimenin her başlıkta kaç kez geçtiğini sayarak dev bir matris oluşturdu. Bu matris, modelimizin "okuyabildiği" formattır.
        3.  **Model Eğitimi (`LatentDirichletAllocation`):**
            -   Bu sayısal matrisi LDA modeline verdik ve ona "Bu verinin içinde 10 tane gizli konu bul" dedik.
            -   Model, kelimelerin hangi konularda ve konuların hangi belgelerde bir araya gelme olasılığının en yüksek olduğunu hesaplayarak "konu ağlarını" ördü.
        4.  **Modeli Kaydetme (`joblib.dump`):**
            -   Eğittiğimiz bu zeki "kütüphaneciyi" (LDA modeli) ve kelime sözlüğünü (Vectorizer) daha sonra bu uygulamada kullanmak üzere `.pkl` dosyaları olarak kaydettik.
        """)