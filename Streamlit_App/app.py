import streamlit as st
import pandas as pd 
import numpy as np
import joblib
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns




def main():
    
    st.sidebar.title('Streamlit ile ML Uygulaması')
    selected_page = st.sidebar.selectbox('Sayfa Seçiniz..',["-","Tahmin Yap","İstatislik Görüntüle","Hakkında"])

    if selected_page == "-":
        image = Image.open('../Media/logo.png')
        st.image(image, use_column_width=True)
        st.title('Streamlit Uygulamasına Hoşgeldiniz 👋')

        st.markdown(
            """
            Bu proje makine öğrenmesi uygulamalarının web ortamında streamlit
            kullanılarak yayınlanmasına örnek olarak geliştirilmiştir. Bir e-ticaret sitesi üzerinden 1824 adet telefon verileri çekilmiş
            ve incelenmiştir. Bu veriler kullanılarak makine öğrenmesi modelleri eğitilmiş ve projeye dahil edilmiştir.
            
            """)
        st.info("Tahmin yapmak,istatistlikleri görüntülemek ve proje hakkında daha fazla bilgi edinmek için sol tarafta bulunan menüyü kullanınız.")
        


    if selected_page == "Tahmin Yap":
        predict()
    
    if selected_page == "İstatislik Görüntüle":
        eda() 
    
    if selected_page == "Hakkında":
        about() 
    

def about():
    st.title('Geliştirici Bilgileri')
    st.subheader('Web Sayfası : [Hamdi DAMAR](http://hamdidamar.com/)')
    st.subheader('Github Adresi : [Hamdi DAMAR](https://github.com/hamdidamar/)')
    st.subheader('Linkedin Adresi : [Hamdi DAMAR](https://www.linkedin.com/in/hamdi-damar-bb6a8b159/)')
    st.subheader('Medium Adresi : [Hamdi DAMAR](https://medium.com/@hamdidamar)')
    st.subheader('Mail Adresi : hamdi.damar@hotmail.com')

def eda():
    st.title('İstatistlikler')

    data = pd.read_csv("../data/phones-to-presentation.csv")

    st.header("Bütün Veriler")
    st.dataframe(data)

    plt.figure(figsize=(16,16))
    plt.subplot(2,1,1)
    sns.countplot(x='Marka',data = data,order = data['Marka'].value_counts().index)
    plt.xticks(rotation = 90)
    plt.xlabel("Marka Adı")
    plt.ylabel("Ürün Sayısı")
    st.header("Ürün Sayısına Göre Markaların Sıralaması")
    st.pyplot(fig=plt)

    plt.figure(figsize=(16,16))
    plt.subplot(2,1,1)
    sns.countplot(x='Isletim_Sistemi',data = data,order = data['Isletim_Sistemi'].value_counts().index)
    plt.xlabel("İşletim Sistemi")
    plt.ylabel("Ürün Sayısı")
    plt.xticks(rotation = 90)
    

    st.header("Ürün Sayısına Göre İşletim Sistemleri")
    st.pyplot(fig=plt)
    



def predict():

    # Markalar ve Modellerin yüklenmesi
    markalar = load_data()
    

    # Kullanıcı arayüzü ve değer alma
    st.title('Merhaba, *Streamlit!* 👨‍💻')
    selected_brand = marka_index(markalar,st.selectbox('Marka Seçiniz..',markalar))
    

    selected_os = isletim_sistemi(st.radio("İşletim Sistemi",('iOS','Android','Android 10','Android 10 (Q)')))
    

    selected_memory = st.number_input('Dahili Hafıza',min_value=8,max_value=512)
    st.write("Dahili Hafıza :"+str(selected_memory)+" GB")

    selected_front_cam = st.slider("Ön Kamera Çözünürlüğü",min_value=0,max_value=48)
    st.write("Ön Kamera Çözünürlüğü :"+str(selected_front_cam)+" MP")

    selected_back_cam = st.slider("Arka Kamera Çözünürlüğü",min_value=0,max_value=108)
    st.write("Arka Kamera Çözünürlüğü :"+str(selected_back_cam)+" MP")

    selected_ram = st.number_input('Bellek Kapasitesi',min_value=1,max_value=12)
    st.write("Bellek Kapasitesi :"+str(selected_ram)+" GB")

    selected_battery = st.slider("Batarya Kapasitesi",min_value=1500,max_value=7000,step=500)
    st.write("Batarya Kapasitesi :"+str(selected_battery)+" mAh")

    selected_model = st.selectbox('Tahmin Modeli Seçiniz..',["Decision Tree","Multiple Linear","Random Forest"])


    prediction_value = create_prediction_value(selected_brand,selected_os,selected_memory,selected_front_cam,selected_back_cam,selected_ram,selected_battery)
    prediction_model = load_models(selected_model)


    if st.button("Tahmin Yap"):
            result = predict_models(prediction_model,prediction_value)
            if result != None:
                st.success('Tahmin Başarılı')
                st.balloons()
                st.write("Tahmin Edilen Fiyat: "+ result + "TL")
            else:
                st.error('Tahmin yaparken hata meydana geldi..!')




    


def load_data():
    markalar = pd.read_csv("../data/markalar.csv")
    return markalar

def load_models(modelName):
    if modelName == "Decision Tree":
        dt_model = joblib.load("../Models/phones_decision_tree_model.pkl")
        return dt_model

    elif modelName == "Multiple Linear":
        mlinear_model = joblib.load("../Models/phones_multiple_linear_model.pkl")
        return mlinear_model

    elif modelName == "Random Forest":  
        rf_model = joblib.load("../Models/phones_random_forest_model.pkl")
        return rf_model

    else:
        st.write("Model yüklenirken hata meydana geldi..!")
        return 0


def marka_index(markalar,marka):
    index = int(markalar[markalar["Markalar"]==marka].index.values)
    return index


def isletim_sistemi(isletim_sistemi):
    if isletim_sistemi == "iOS":
        return 1
    else:
        return 0


def create_prediction_value(marka,isletim_sistemi,dahili_hafiza,on_kamera,arka_kamera,bellek_kapasitesi,batarya_kapasitesi):
    res = pd.DataFrame(data = 
            {'Marka':[marka],'Isletim_Sistemi':[isletim_sistemi],'Dahili_Hafiza':[dahili_hafiza],
             'On_Kamera_Cozunurlugu':[on_kamera],'Arka_Kamera_Cozunurlugu':[arka_kamera],
              'Bellek_Kapasitesi':[bellek_kapasitesi],'Batarya_Kapasitesi':[batarya_kapasitesi]})
    return res


def predict_models(model,res):
    result = str(int(model.predict(res))).strip('[]')
    return result



if __name__ == "__main__":
    main()