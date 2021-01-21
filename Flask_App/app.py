from flask import Flask, render_template,request,url_for
import pandas as pd
import joblib



# create the object of Flask
app = Flask(__name__)

markalar = pd.read_csv("static/data/markalar.csv")

list_marka = markalar.Markalar.values.tolist()



# creating our routes
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():

    
    if request.method == 'GET':
        return render_template('predict.html',list_marka = list_marka,len = len(list_marka))

    if request.method == 'POST':
        marka = request.form['marka']
        isletim_sistemi = request.form['options']
        dahili_hafiza = request.form['dahili_hafiza']
        on_kamera = request.form['on_kamera']
        arka_kamera = request.form['arka_kamera']
        bellek = request.form['bellek']
        batarya = request.form['batarya']

        model = request.form['models']


        res = pd.DataFrame(data = 
            {'Marka':[marka],'Isletim_Sistemi':[isletim_sistemi],'Dahili_Hafiza':[dahili_hafiza],
             'On_Kamera_Cozunurlugu':[on_kamera],'Arka_Kamera_Cozunurlugu':[arka_kamera],
              'Bellek_Kapasitesi':[bellek],'Batarya_Kapasitesi':[batarya]}) 
        
        if model == "1":
            dt = joblib.load("static/models/phones_decision_tree_model.pkl")
            sonuc = str(int(dt.predict(res))).strip('[]')

        if model == "2":
            ml = joblib.load("static/models/phones_multiple_linear_model.pkl")
            sonuc = str(int(ml.predict(res))).strip('[]')

        if model == "3":
            rf = joblib.load("static/models/phones_random_forest_model.pkl")
            sonuc = str(int(rf.predict(res))).strip('[]')



        if marka == None:
            return render_template('predict.html', sonuc='Lütfen Doğru Değer Gönderin')
        else:
            return render_template('predict.html', sonuc = sonuc , marka=marka,list_marka = list_marka,len = len(list_marka))


@app.route('/statistics')
def statistics():
    return render_template('statistics.html')


@app.route('/about')
def about():
    return render_template('about.html')


# run flask app
if __name__ == "__main__":
    app.run(debug=True)
