import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Load dataset
dataset_path = "data_analisis_gadget.xlsx"
data = pd.read_excel(dataset_path)

# Pisahkan fitur dan label
X = data[['gadget_hiburan', 'gadget_studi', 'durasi_belajar', 'jumlah_aktivitas_sosial','IPK']]
y = data['output']

# Memisahkan data menjadi set pelatihan (80%) dan set pengujian (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1, random_state=62)  # Ganti kernel dan parameter sesuai kebutuhan
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f'Akurasi SVM: {svm_accuracy * 100:.2f}%')

# Menyimpan model ke file menggunakan pickle
model_filename = 'svm_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(svm_model, model_file)

# Memuat kembali model dari file
with open(model_filename, 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

# Menggunakan model yang telah dimuat
loaded_svm_pred = loaded_svm_model.predict(X_test)
loaded_svm_accuracy = accuracy_score(y_test, loaded_svm_pred)
print(f'Akurasi NB (setelah memuat): {loaded_svm_accuracy * 100:.2f}%')