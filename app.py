from flask import Flask, render_template, request, session
import pandas as pd
import pickle
import warnings

app = Flask(__name__)
app.secret_key = 'dasjdisaIAHSDASO123812312DA@#*@09'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Mendapatkan data dari formulir
    gadget_hiburan = float(request.form['gadget_hiburan'])
    gadget_studi = float(request.form['gadget_studi'])
    durasi_belajar = float(request.form['durasi_belajar'])
    jumlah_aktivitas_sosial = float(request.form['jumlah_aktivitas_sosial'])
    IPK = float(request.form['IPK'])

    # Memuat model SVM dari file pickle
    with open('nb_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)

    # Membuat prediksi
    input_data = [[gadget_hiburan, gadget_studi, durasi_belajar, jumlah_aktivitas_sosial, IPK]]
    prediction = svm_model.predict(input_data)

    # Mengirim hasil ke result.html
    return render_template('result.html', gadget_hiburan=gadget_hiburan, gadget_studi=gadget_studi,
                           durasi_belajar=durasi_belajar, jumlah_aktivitas_sosial=jumlah_aktivitas_sosial,
                           IPK=IPK, prediction=prediction[0])


@app.route('/result')
def result():
    # Retrieve values from query parameters
    gadget_hiburan = request.args.get('gadget_hiburan', None)
    gadget_studi = request.args.get('gadget_studi', None)
    durasi_belajar = request.args.get('durasi_belajar', None)
    jumlah_aktivitas_sosial = request.args.get('jumlah_aktivitas_sosial', None)
    IPK = request.args.get('IPK', None)
    prediction = request.args.get('prediction', None)

    # Render the result template with the retrieved values
    return render_template('result.html', gadget_hiburan=gadget_hiburan, gadget_studi=gadget_studi,
                           durasi_belajar=durasi_belajar, jumlah_aktivitas_sosial=jumlah_aktivitas_sosial,
                           IPK=IPK, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
