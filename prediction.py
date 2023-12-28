import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
dataset_path = "data_analisis_gadget.xlsx"
data = pd.read_excel(dataset_path)

# Pisahkan fitur dan label
X = data[['gadget_hiburan', 'gadget_studi', 'durasi_belajar', 'jumlah_aktivitas_sosial', 'IPK']]
y = data['output']

# Memisahkan data menjadi set pelatihan (80%) dan set pengujian (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy KNN: {accuracy_knn * 100:.2f}%')

# Decision Tree dengan pembatasan kedalaman pohon
dt_model = DecisionTreeClassifier(max_depth=5, random_state=62)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f'Akurasi Decision Tree: {dt_accuracy * 100:.2f}%')

# Random Forest dengan pembatasan kedalaman pohon dan jumlah pohon
rf_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=62)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Akurasi Random Forest: {rf_accuracy * 100:.2f}%')

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
print(f'Akurasi Naive Bayes: {nb_accuracy * 100:.2f}%')

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', C=1, random_state=62)  # Ganti kernel dan parameter sesuai kebutuhan
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f'Akurasi SVM: {svm_accuracy * 100:.2f}%')