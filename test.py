import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
dataset_path = "data_analisis_gadget.xlsx"
data = pd.read_excel(dataset_path)

# Pisahkan fitur dan label
X = data[['gadget_hiburan', 'gadget_studi', 'durasi_belajar', 'jumlah_aktivitas_sosial','IPK']]
y = data['output']

# Memisahkan data menjadi set pelatihan (80%) dan set pengujian (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1, random_state=42)  # Ganti kernel dan parameter sesuai kebutuhan

# Inisialisasi list untuk menyimpan akurasi
train_accuracies = []
test_accuracies = []

# Pelatihan model
svm_model.fit(X_train, y_train)

# Evaluasi pada set pelatihan
train_pred = svm_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
train_accuracies.append(train_accuracy)

# Evaluasi pada set pengujian
test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_accuracies.append(test_accuracy)

print(f'Akurasi SVM pada set pelatihan: {train_accuracy * 100:.2f}%')
print(f'Akurasi SVM pada set pengujian: {test_accuracy * 100:.2f}%')

# # Visualisasi kurva pembelajaran
# epochs = range(1, len(train_accuracies) + 1)
# plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, test_accuracies, 'b', label='Testing accuracy')
# plt.title('Training and Testing Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
