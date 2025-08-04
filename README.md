# Sistem Prediksi Churn Pelanggan Telco

---

## Gambaran Umum Proyek

Proyek ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi pelanggan mana yang kemungkinan besar akan berhenti menggunakan layanan (churn) di sebuah perusahaan telekomunikasi (telco). Dengan mengidentifikasi pelanggan berisiko tinggi lebih awal, perusahaan dapat mengambil tindakan proaktif (misalnya, menawarkan diskon, dukungan khusus, atau promosi) untuk mempertahankan mereka, sehingga mengurangi tingkat churn dan meningkatkan retensi pelanggan.

Proyek ini menggunakan metode **Supervised Learning** untuk masalah **Klasifikasi**.

---

## Dataset

Dataset yang digunakan adalah **Telco Customer Churn** dari Kaggle. Dataset ini berisi informasi demografi pelanggan, layanan yang digunakan, informasi kontrak, tagihan, dan status churn mereka.

* **Sumber**: [Link ke Dataset Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Jumlah Baris Awal**: 7043
* **Jumlah Kolom Awal**: 21

---

## Alur Kerja Data Science (End-to-End)

Proyek ini mengikuti alur kerja standar dalam proyek Machine Learning:

### 1. Pengumpulan & Pemahaman Data (Data Acquisition & Understanding)
* Memuat data dari file CSV.
* Menganalisis statistik deskriptif dan tipe data.
* Mengidentifikasi kolom `TotalCharges` sebagai `object` yang seharusnya numerik dan memiliki nilai kosong (spasi).

### 2. Pra-pemrosesan Data (Data Preprocessing)
* **Penanganan `TotalCharges`**: Mengubah tipe data `TotalCharges` dari `object` ke `float` dan menangani 11 nilai kosong (dikonversi ke NaN) dengan menghapus baris yang relevan.
* **Penghapusan Kolom Tidak Relevan**: Menghapus kolom `customerID`.
* **Encoding Variabel Kategorikal Biner**: Mengubah kolom biner (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) dari string menjadi nilai numerik (0 dan 1).
* **One-Hot Encoding**: Menerapkan One-Hot Encoding pada kolom kategorikal nominal lainnya (misalnya `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`) untuk mengubahnya menjadi format numerik yang dapat diproses model.

### 3. Eksplorasi Data Lanjutan (Exploratory Data Analysis - EDA)
* **Distribusi Target (`Churn`)**: Menganalisis proporsi pelanggan yang churn dan tidak churn. Ditemukan bahwa dataset memiliki **ketidakseimbangan kelas** (sekitar [Sebutkan Persentase Churn Anda]% pelanggan churn).
* **Analisis Korelasi**: Memvisualisasikan matriks korelasi untuk memahami hubungan antar fitur dan hubungan fitur dengan variabel target (`Churn`). Mengidentifikasi fitur-fitur yang paling berkorelasi positif/negatif dengan churn.
    * **Fitur berkorelasi positif kuat dengan churn**: [Sebutkan 2-3 fitur utama dari analisis korelasi Anda, contoh: `MonthlyCharges`, `InternetService_Fiber optic`, `PaperlessBilling_Yes`].
    * **Fitur berkorelasi negatif kuat dengan churn**: [Sebutkan 2-3 fitur utama dari analisis korelasi Anda, contoh: `tenure`, `Contract_Two year`, `OnlineSecurity_Yes`].

### 4. Pembagian Data & Penanganan Ketidakseimbangan Kelas
* Membagi dataset menjadi fitur (`X`) dan target (`y`).
* Membagi data menjadi **training set** (80%) dan **test set** (20%) menggunakan `train_test_split` dengan `stratify` untuk menjaga proporsi kelas.
* Menerapkan **SMOTE (Synthetic Minority Over-sampling Technique)** pada **training set** untuk mengatasi ketidakseimbangan kelas, sehingga jumlah sampel kelas `Churn` dan `Non-Churn` menjadi seimbang.

### 5. Pembangunan & Pelatihan Model
* Melatih tiga model klasifikasi populer:
    * **Logistic Regression**
    * **Random Forest Classifier**
    * **XGBoost Classifier**
* Semua model dilatih pada data training yang sudah di-resample (setelah SMOTE).

### 6. Evaluasi Model
* Model dievaluasi pada **test set** (data asli yang belum di-resample) menggunakan metrik-metrik berikut:
    * **Accuracy**
    * **Precision**
    * **Recall**
    * **F1-Score**
    * **ROC-AUC**
    * **Classification Report**
    * **Confusion Matrix**

* **Ringkasan Hasil Performa:**
    | Model                  | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |
    | :--------------------- | :------- | :-------- | :------- | :------- | :------- |
    | Logistic Regression    | [Nilai Anda] | [Nilai Anda]  | **[Nilai Anda]** | [Nilai Anda] | **[Nilai Anda]** |
    | Random Forest          | **[Nilai Anda]** | **[Nilai Anda]** | [Nilai Anda] | [Nilai Anda] | [Nilai Anda] |
    | XGBoost                | [Nilai Anda] | [Nilai Anda]  | [Nilai Anda] | [Nilai Anda] | [Nilai Anda] |

* **Kesimpulan Evaluasi**:
    Berdasarkan hasil, **Logistic Regression** dipilih sebagai model terbaik untuk proyek ini. Meskipun Random Forest memiliki akurasi dan presisi sedikit lebih tinggi, Logistic Regression menunjukkan **Recall dan ROC-AUC yang jauh lebih unggul**. Dalam konteks prediksi churn, Recall tinggi sangat krusial karena kita ingin mengidentifikasi sebanyak mungkin pelanggan yang akan churn untuk intervensi proaktif, bahkan jika itu berarti menerima beberapa *false positives*. ROC-AUC yang tinggi juga menunjukkan kemampuan model yang baik dalam membedakan antara pelanggan yang churn dan tidak churn.

### 7. Interpretasi Model (Logistic Regression)
* Menganalisis koefisien dari model Logistic Regression untuk mengidentifikasi fitur-fitur yang paling memengaruhi kemungkinan churn.

* **Fitur Paling Berpengaruh (Dampak Negatif - Mengurangi Churn):**
    * `PhoneService`
    * `InternetService_Fiber optic`
    * `Contract_Two year`
    * `OnlineSecurity_Yes`
    * `TechSupport_Yes`
    * [Sebutkan fitur relevan lainnya dari output Anda]

* **Fitur Paling Berpengaruh (Dampak Positif - Meningkatkan Churn):**
    * [Contoh: Meskipun koefisiennya negatif, jika dibandingkan dengan kategori dasar One-Hot Encoding, bisa jadi fitur ini secara implisit meningkatkan churn. Dari output Anda, `PaymentMethod_Electronic check` adalah satu-satunya dengan koefisien negatif paling kecil/mendekati nol, yang bisa diartikan relatif "kurang mengurangi churn" dibanding metode lain, atau bahkan meningkatkan dibandingkan base. Perhatikan ini baik-baik.]

* **Wawasan Bisnis Utama**:
    * Pelanggan dengan **kontrak jangka panjang (satu atau dua tahun)** cenderung sangat loyal.
    * Layanan tambahan seperti **keamanan online, dukungan teknis, backup online, dan perlindungan perangkat** secara signifikan mengurangi kemungkinan churn.
    * [Tambahkan wawasan lain berdasarkan interpretasi fitur Anda, misalnya tentang metode pembayaran, atau layanan internet].

---

## Bagaimana Menjalankan Proyek Ini?

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/customer-churn-prediction.git](https://github.com/YourGitHubUsername/customer-churn-prediction.git)
    cd customer-churn-prediction
    ```
2.  **Unduh Dataset:**
    Unduh file `WA_Fn-UseC_-Telco-Customer-Churn.csv` dari [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
3.  **Tempatkan Dataset:**
    Unggah file CSV ini ke Google Drive Anda dan sesuaikan `file_path` di notebook Colab Anda agar menunjuk ke lokasinya.
4.  **Buka di Google Colab:**
    Buka file `Customer_Churn_Prediction.ipynb` di Google Colab. Pastikan Anda sudah *mount* Google Drive Anda di Colab.
5.  **Jalankan Sel-sel Kode:**
    Jalankan setiap sel kode secara berurutan untuk mereplikasi analisis dan pembangunan model.

---

## Teknologi yang Digunakan

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**
* **XGBoost**
* **Google Colab**

---

## Kontributor

* [Nama Lengkap Anda] ([Link Profil GitHub Anda](https://github.com/YourGitHubUsername)) - (Anda bisa menambahkan sedikit deskripsi peran Anda)

---
