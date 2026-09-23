import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Restoran Başarı Tahmincisi - Ana Sayfa",
    page_icon="🍽️",
    layout="wide"
)

st.title("🚀 Restoran Başarı Skoru Tahmincisi")
st.markdown("---")

st.header("📖 Projenin Amacı")
st.write("""
Bu proje, **30 Günde 30 AI Projesi Maratonu**'nun ilk adımıdır. 
Amacı, bir restoranın çeşitli özelliklerine dayanarak, o restoranın Zomato'daki derecelendirme puanını tahmin eden bir makine öğrenmesi modeli geliştirmektir. 
Bu uygulama, geliştirilen bu modelin interaktif bir arayüzle sunulmasını sağlar.
""")

st.header("📊 Veri Seti Hakkında")
st.info("""
Kullanılan veri seti, Hindistan'ın teknoloji merkezi olarak bilinen **Bangalore** şehrindeki 50,000'den fazla restorana ait halka açık Zomato verilerini içermektedir. 
Bu veri seti, her bir restoran için konum, mutfak türü, ortalama maliyet ve müşteri oyları gibi zengin bilgiler barındırır.
""")

st.header("⚙️ Uygulama Nasıl Kullanılır?")
st.write("""
Uygulamanın tahmin yeteneklerini test etmek için lütfen sol taraftaki menüden **'Tahmin Uygulaması'** sayfasına geçiş yapın. 
Bu sayfada, yeni bir restoranın özelliklerini girerek modelimizin bu restoran için potansiyel başarı puanını nasıl tahmin ettiğini görebilirsiniz.
""")

st.subheader("Modelin Kullandığı Temel Özellikler:")
st.markdown("""
- **Konum (Location):** Restoranın Bangalore'daki konumu.
- **Mutfak Türü (Cuisines):** Restoranın sunduğu mutfak çeşitleri.
- **İki Kişilik Ortalama Maliyet (Cost for Two):** Restoranın fiyat seviyesi.
- **Online Sipariş ve Masa Rezervasyonu:** Restoranın sunduğu hizmetler.
- **Toplam Oy Sayısı (Votes):** Restoranın popülerliğinin bir göstergesi.
""")

st.markdown("---")
st.write("Süleyman Toklu tarafından geliştirilmiştir.")