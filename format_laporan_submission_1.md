# Laporan Proyek Machine Learning - Muhamad Nur Rohman

## Domain Proyek

Pemanfaatan pendekatan berbasis data (data-driven) dan kecerdasan buatan (Artificial Intelligence/AI) semakin berkembang dalam dunia medis, terutama untuk mendukung proses diagnosis, termasuk prediksi risiko penyakit jantung. Deteksi dini penyakit jantung memiliki potensi besar dalam menyelamatkan nyawa pasien, namun ketersediaan fasilitas diagnostik yang canggih masih belum merata, khususnya di daerah atau fasilitas kesehatan dengan sumber daya terbatas.

Dalam konteks tersebut, pendekatan berbasis Machine Learning menawarkan solusi yang efektif untuk mendeteksi gejala awal penyakit jantung. Sistem prediksi berbasis Machine Learning dapat mengidentifikasi pasien pada tahap awal, sehingga tidak hanya meningkatkan peluang keberhasilan pengobatan, tetapi juga membantu menekan biaya perawatan jangka panjang.

Pendekatan ini sangat relevan dalam kerangka sistem kesehatan nasional maupun asuransi kesehatan swasta, di mana efisiensi biaya dan aksesibilitas layanan kesehatan menjadi fokus utama.

## Business Understanding

Setelah memahami urgensi dan konteks masalah yang dihadapi, tahap selanjutnya adalah merumuskan permasalahan secara sistematis. Langkah ini bertujuan untuk mendefinisikan pertanyaan inti secara spesifik, menetapkan sasaran utama yang hendak dicapai, serta merancang pendekatan solusi berbasis data yang efektif dan realistis untuk diterapkan dalam konteks medis.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Bagaimana cara memprediksi risiko seseorang mengidap penyakit jantung berdasarkan data klinis yang tersedia?
- Faktor apa yang memiliki pengaruh paling signifikan dalam meningkatkan potensi seseorang mengalami penyakit jantung?
- Bagaimana membangun model prediktif yang mampu mendeteksi risiko penyakit jantung secara otomatis?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model Machine Learning yang akurat untuk melakukan prediksi risiko secara otomatis guna mendukung keputusan medis secara cepat dan tepat.
- Mengidentifikasi variabel yang paling berkontribusi terhadap peningkatan risiko penyakit jantung, seperti tekanan darah, kadar kolesterol, dan parameter kesehatan lainnya.
- Mengklasifikasikan individu ke dalam kelompok risiko tinggi dan rendah terhadap penyakit jantung berdasarkan atribut klinis dasar.

    ### Solution statements
- Mengembangkan dan membandingkan sejumlah model klasifikasi, seperti XGBoost, LightGBM, Logistic Regression, dan Random Forest, untuk menentukan model terbaik berdasarkan berbagai metrik evaluasi seperti Accuracy, Precision, Recall, F1-Score, dan ROC AUC.

- Menjadikan F1-Score dan ROC AUC sebagai metrik evaluasi utama dalam proses pemilihan model, mengingat keduanya mampu memberikan gambaran menyeluruh tentang performa model, terutama dalam konteks data tidak seimbang dan kebutuhan untuk menyeimbangkan antara false positive dan false negative.

- Model dengan performa terbaik akan diuji dan divalidasi secara menyeluruh sebelum diintegrasikan ke dalam sistem pendukung keputusan, guna membantu proses deteksi dini dan diagnosis risiko penyakit jantung.

## Data Understanding
dataset ini berisi pasien yang diduga mengidap penyakit jantung. struktur dataset ini memiliki lebih dari 70000 baris dan 13 kolom. kondisi awal dataset ini tidak ada missing value dan data duplicated. dataset berasal dari [Kaggle Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada cardiovascular dataset adalah sebagai berikut:
- Id (no unik pasien) : setiap pasien memiliki no id yang berbeda (tipe data: integer)
- Age (umur): Usia pasien cardiovascular (tipe data: integer).
- Height (tinggi badan): Tinggi badan pasien dalam cm (integer).
- Weight (berat badan): Berat badan pasien dalam kg (float).
- Gender (jenis kelamin): jenis kelamin pasien, dengan nilai 0 = perempuan, 1 = laki-laki.
- Systolic blood pressure (tekanan darah sistolik): Tekanan darah atas dalam satuan mmHg (integer).
- Diastolic blood pressure (tekanan darah diastolik): Tekanan darah bawah dalam satuan mmHg (integer).
- Cholesterol (kolesterol): Kadar kolesterol, dengan nilai 1 = normal, 2 = di atas normal, 3 = jauh di atas normal.
- Glucose (glukosa): Kadar gula darah, dengan nilai 1 = normal, 2 = di atas normal, 3 = jauh di atas normal.
- Smoking (merokok): Status merokok, 0 = tidak, 1 = ya.
- Alcohol intake (konsumsi alkohol): Status konsumsi alkohol, 0 = tidak, 1 = ya.
- Physical activity (aktivitas fisik): Status aktivitas fisik, 0 = tidak aktif, 1 = aktif.
- Cardiovascular disease (penyakit jantung): Target variabel, 0 = tidak memiliki penyakit kardiovaskular, 1 = memiliki penyakit kardiovaskular.

### Target Cardiovascular
berdasarkan dari grafik distribusi target terdapat lebih dari 20000 pasian tidak mengidap penyakit cardiovascular dan terindikasi sebanyak 17500 pasien mengidap penyakit cardiovascular. Kondisi ini mendukung validitas data untuk training model klasifikasi dan menegaskan pentingnya prediksi yang akurat terhadap kasus penyakit cardiovascular yang jumlahnya cukup signifikan dalam dataset.

### Usia Cardiovascular
Terlihat bahwa jumlah kasus cardiovascular cenderung meningkat seiring bertambahnya usia, khususnya pada rentang usia 50–60 tahun, di mana jumlah penderita (biru) cukup tinggi dibandingkan kelompok usia yang lebih muda. Hal ini menunjukkan adanya korelasi antara usia yang lebih tua dan peningkatan risiko penyakit kardiovaskular.

### Corellation Matrix
Terlihat bahwa tekanan darah sistolik (ap_hi) dan diastolik (ap_lo) memiliki korelasi tinggi (0.71), menandakan keterkaitan erat. BMI juga sangat berkorelasi dengan weight (0.84), karena BMI dihitung dari berat dan tinggi badan. Fitur cardio (penyakit jantung) menunjukkan korelasi sedang dengan ap_hi (0.45), ap_lo (0.35), dan cholesterol (0.22), menunjukkan bahwa tekanan darah dan kadar kolesterol berpengaruh terhadap risiko penyakit kardiovaskular

## Data Preparation
Proses data preparation yang dilakukan dalam membangun model klasifikasi ini antara lain:

### Cek Missing Value
cardio_disease.isnull().sum() kode ini digunakan untuk mengecek apakah ada nilai yang kosong pada data. proses ini wajib dilakukan sebelum melakukan pemodelan karena algoritma machine learning tidak dapat menangani data kosong.

### Cek Dupllikasi Data
cardio_disease.duplicated().sum()) kode ini digunakan untuk mengecek berapa banyak baris yang mengalami duplikasi. proses ini dilakukan untuk menghindari model berlebihan yang sering muncul berulang ulang sehingga performa model menurun.

### Menghapus Kolom
cardio_disease.drop(['id', 'active', 'smoke', 'alco'], axis=1, inplace=True) mengahpus kolom yang tidak digunakan pada saat modeling prosses
cardio_disease['age'] = (cardio_disease['age'] / 365).astype(int) mengubah age dari hari ke tahun
cardio_disease['BMI'] = cardio_disease['weight'] / ((cardio_disease['height'] / 100) ** 2) membuat kolom baru bernama BMI

### Feature Scaling
normalisasi yang mengubah nilai fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1. Tujuan dilakukan feature scaling adalah untuk membuat semua fitur numerik berada dalam skala yang sebanding sehingga dapat meningkatkan performa dan kecepatan konvergensi algoritma selama training model dan untuk menghindari dominasi fitur dengan nilai besar terhadap fitur dengan nilai kecil dalam perhitungan jarak atau bobot.

### Data Split
Proses ini membagi dataset menjadi 80% data latih yang digunakan untuk melatih model dan 20% data uji yang digunakan untuk mengevaluasi performa model. Kode random_state=42 digunakan untuk memastikan pembagian data selalu konsisten jika kode dijalankan ulang. Tujuan dilakukan data spliting adalah untuk melatih dan menguji model secara adil dimana data latih digunakan untuk membangun model dan data uji digunakan untuk mengevaluasi kinerja model terhadap data yang belum pernah dilihat sbeelumnya. Proses ini dapat mencegah terjadinya overfitting dan menentukkan akurasi performa awal.

## Modeling

### XGBoost
Parameter yang digunakan dalam proses pemodelan menggunakan XGBoost (Extreme Gradient Boosting) adalah use_label_encoder=False, eval_metric='logloss', dan random_state=42. Model ini merupakan algoritma boosting yang bekerja dengan cara membangun beberapa pohon keputusan secara bertahap, di mana setiap pohon baru mencoba memperbaiki kesalahan dari pohon sebelumnya. Parameter use_label_encoder=False digunakan untuk menonaktifkan encoder label bawaan yang sudah deprecated, sedangkan eval_metric='logloss' digunakan untuk mengukur kesalahan prediksi dalam bentuk logaritmik loss.

Model ini dilatih menggunakan data pelatihan (X_train, y_train) dan diuji pada data pengujian X_test. Proses evaluasi dilakukan menggunakan beberapa metrik seperti akurasi, presisi, recall, F1-score, dan ROC AUC.
mampu menangani data tidak seimbang, mendukung regularisasi (sehingga mengurangi overfitting), dan bekerja sangat baik pada data tabular. Namun, kekurangannya adalah waktu pelatihan yang relatif lamaa dibanding model lain, serta kompleksitas parameter yang tinggi yang membutuhkan tuning lebih cermat.

### LightGBM 
Parameter yang digunakan dalam proses pemodelan menggunakan LightGBM (Light Gradient Boosting Machine) adalah random_state=42 yang digunakan untuk memastikan hasil yang konsisten saat model dilatih ulang. LightGBM merupakan algoritma boosting berbasis pohon keputusan yang dirancang untuk efisiensi dan kecepatan, terutama pada dataset berukuran besar. LightGBM menggunakan teknik histogram-based learning dan leaf-wise tree growth, yang membuatnya lebih cepat dan efisien dibandingkan metode boosting tradisional.

Keunggulan LightGBM kecepatan training yang sangat tinggi, penggunaan memori yang lebih hemat, serta performa prediksi yang sangat kompetitif. Kekurangannya, LightGBM cenderung lebih sensitif terhadap data yang tidak terurut dan dapat menghasilkan hasil yang kurang stabil pada dataset kecil atau fitur kategorikal yang tidak ditangani dengan benar.

### Logistic Regression
Model yang digunakan adalah Logistic Regression, dengan parameter max_iter=1000 untuk memastikan konvergensi model selama proses pelatihan dan random_state=42 untuk menjaga konsistensi hasil. Logistic Regression adalah algoritma klasifikasi linier yang paling sederhana namun sangat efektif, terutama ketika hubungan antara fitur dan target bersifat linier. Model ini menghitung probabilitas suatu kelas menggunakan fungsi sigmoid, yang membuatnya cocok untuk klasifikasi biner seperti prediksi penyakit jantung.
Kelebihan dari model logistic regression adalah model cepat dilarih dan cocok untuk baseline model. Memberikan probabilitas, cocok untuk ROC AUC. Sedangkan kekurangan dari model ini adalah model ini tidak mampu menangkap hbungan non-linear yang kompleks sehingga performa model bisa buruk jika terdapat outlier ekstrem atau korelasi multikolinearitas tinggi.

### Random Forest
Tahapan untuk membuat model RandomForestClassifier dengan parameter random_state=42 untuk memastikan hasil yang konsisten setiap kali model dijalankan. Random Forest adalah algoritma ensemble learning berbasis pohon keputusan (Decision Tree), yang membangun banyak pohon keputusan saat pelatihan dan menggabungkan prediksi dari semua pohon untuk menentukan hasil akhir. Pendekatan ini meningkatkan akurasi dan mengurangi risiko overfitting dibandingkan model pohon tunggal.
Kelebihan random forest mampu menangani data yang kompleks, dapat menangai fitur numerikal dan kategorikal, serta relatif tahan terhadap overfitting. Akan tetapi, kekurangannya terletak pada kebutuhan memori yang tinggi, waktu prediksi yang bisa lebih lambat untuk model besar, dan interpretasi model yang tidak sesederhana logistic regression.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Dalam proyek klasifikasi ini, digunakan lima metrik utama untuk mengevaluasi performa model: accuracy, precision, recall, F1-score, dan ROC AUC. Accuracy mengukur persentase prediksi yang benar, namun bisa menyesatkan jika data tidak seimbang. Precision penting untuk meminimalkan kesalahan prediksi positif (false positive), sedangkan recall fokus pada seberapa banyak data positif yang berhasil terdeteksi (false negative). F1-score menggabungkan precision dan recall secara seimbang, cocok untuk data tidak seimbang. Terakhir, ROC AUC digunakan untuk menilai kemampuan model membedakan kelas positif dan negatif berdasarkan probabilitas, dengan nilai antara 0.5 hingga 1.0 semakin tinggi nilainya, semakin baik performa model.

​Empat algoritma yang digunakan dalam proyek ini adalah XGBoost, LightGBM ,Logistic Regression dan Random Forest. Berikut adalah hasil metrik evaluasi dari empat algoritma klasifikasi yang telah dilatih:
| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| XGBoost             | 0.7318   | 0.7338    | 0.7318 | 0.7293   | 0.7872  |
| LightGBM            | 0.7370   | 0.7393    | 0.7370 | 0.7345   | 0.7940  |
| Logistic Regression | 0.7322   | 0.7360    | 0.7322 | 0.7288   | 0.7911  |
| Random Forest       | 0.7007   | 0.7002    | 0.7007 | 0.7002   | 0.7577  |

Berdasarkan hasil evaluasi performa model, LightGBM menunjukkan kinerja terbaik secara keseluruhan dengan nilai tertinggi pada hampir semua metrik, terutama pada ROC AUC sebesar 0.7940 yang mencerminkan kemampuan model membedakan kelas dengan baik. XGBoost dan Logistic Regression juga memberikan performa yang kompetitif dengan nilai akurasi dan F1-Score di atas 0.72, menunjukkan keseimbangan antara precision dan recall. Sementara itu, Random Forest memiliki performa paling rendah di antara keempat model, terutama pada ROC AUC yang hanya mencapai 0.7577, menunjukkan kemampuan klasifikasi yang kurang optimal pada data ini. Secara umum, LightGBM merupakan pilihan terbaik untuk model klasifikasi pada dataset ini.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

