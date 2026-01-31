# Laporan Proyek Machine Learning - Supriyono

## Project Overview
Dalam ekosistem digital saat ini, pertumbuhan jumlah konten informasi dan hiburan meningkat secara eksponensial. Platform penyedia layanan film seperti Netflix, Disney+, atau IMDb memiliki ribuan hingga jutaan katalog film. Hal ini memicu fenomena yang dikenal sebagai Choice Overload atau Information Overload, di mana pengguna justru merasa kewalahan dan sulit menentukan pilihan karena terlalu banyaknya opsi yang tersedia. Implementasi sistem ini diharapkan dapat meningkatkan retensi pengguna pada platform dan memberikan nilai tambah berupa penemuan konten baru yang sesuai dengan selera unik setiap individu


**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
  Masalah ini harus diselesaikan karena berdampak langsung pada dua sisi:
  Sisi Pengguna: Mengurangi kelelahan kognitif (decision fatigue) dan meningkatkan kepuasan dalam menggunakan platform.
  Sisi Bisnis: Sistem rekomendasi yang efektif merupakan mesin pertumbuhan utama. Sebagai contoh, sekitar 75-80% aktivitas menonton di Netflix berasal dari rekomendasi otomatis, yang     secara signifikan menurunkan tingkat pembatalan langganan (churn rate).
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  Pengembangan sistem ini didasarkan pada temuan riset berikut:
  Pentingnya Personalisasi: Riset oleh Gomez-Uribe dan Hunt menunjukkan bahwa algoritma rekomendasi mampu menghemat biaya operasional hingga $1 miliar per tahun bagi penyedia layanan streaming dengan meningkatkan retensi pelanggan [1].
Efektivitas Deep Learning: Penggunaan Neural Networks dalam Collaborative Filtering (seperti yang digunakan dalam proyek ini melalui RecommenderNet) terbukti mampu menangkap hubungan non-linear antara pengguna dan item yang tidak bisa ditangkap oleh metode matriks konvensional [2].
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.
Sistem rekomendasi menjadi tulang punggung bagi platform hiburan modern untuk mempertahankan pengguna. Tanpa filter yang tepat, pengguna cenderung meninggalkan platform karena kesulitan menemukan konten yang relevan. Menurut riset oleh McKinsey & Company, sekitar 75% dari konten yang ditonton pengguna di platform streaming didorong oleh mesin rekomendasi [4]. Oleh karena itu, efisiensi algoritma dalam menyaring informasi menjadi kunci keberlanjutan bisnis.
Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
-Berdasarkan latar belakang di atas, proyek ini akan menjawab permasalahan berikut:
 Relevansi Konten: Bagaimana memberikan rekomendasi film yang memiliki karakteristik serupa dengan film yang telah ditonton pengguna sebelumnya agar pengalaman menonton tetap konsisten?
 Personalisasi Skala Besar: Bagaimana memprediksi preferensi film bagi pengguna secara personal dengan memanfaatkan pola rating dari ribuan pengguna lain tanpa harus bergantung hanya pada kemiripan genre?

### Goals
Tujuan utama dari proyek ini adalah:

-Menghasilkan daftar Top-N Recommendation film yang memiliki kemiripan genre menggunakan teknik Content-Based Filtering.
-Membangun model Collaborative Filtering berbasis Deep Learning yang mampu memprediksi rating film dengan nilai Root Mean Squared Error (RMSE) di bawah 0.25 (pada skala rating yang telah dinormalisasi).

- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
  Untuk mencapai tujuan tersebut, diajukan dua solusi algoritma:
- Content-Based Filtering:
Algoritma ini akan merepresentasikan setiap film ke dalam vektor fitur menggunakan TF-IDF Vectorizer.
- Menggunakan Cosine Similarity untuk menghitung kemiripan antara satu film dengan film lainnya. Solusi ini sangat baik untuk memberikan rekomendasi yang sangat spesifik pada preferensi genre pengguna.
Collaborative Filtering (Neural Collaborative Filtering):
- Membangun model jaringan saraf (RecommenderNet) untuk mempelajari Latent Features dari pengguna dan film.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah MovieLens Small Latest Dataset yang dikelola oleh GroupLens Research. Dataset ini merupakan versi ringkas yang sering digunakan untuk tolok ukur sistem rekomendasi karena ukurannya yang efisien namun tetap representatif untuk menggambarkan interaksi pengguna dan film.
Sumber Data: Kaggle - Movie Lens Small Latest Dataset
Informasi Dataset
Dataset ini terdiri dari beberapa file CSV, namun dalam proyek ini kita fokus pada dua berkas utama:
Jumlah Data Film (movies.csv): Memiliki 9.742 baris data film yang berbeda.
Jumlah Data Rating (ratings.csv): Memiliki 100.836 baris data rating yang diberikan oleh 610 pengguna kepada 9.724 film unik.
Rentang Waktu: Data dikumpulkan antara 29 Maret 1996 hingga 24 September 2018.

**Rubrik/Kriteria Tambahan (Opsional)**:


## Data Preparation
1. Pembersihan Data (Data Cleaning)
Kita harus memastikan tidak ada data duplikat atau nilai kosong yang dapat mengganggu performa model.

Langkah: Menghapus baris yang memiliki nilai null (jika ada) dan memastikan movieId yang ada di file ratings juga tersedia di file movies.

2. Persiapan untuk Content-Based Filtering
Untuk merekomendasikan film berdasarkan genre, kita perlu mengubah teks genre menjadi representasi angka menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).

Proses: 1. Ekstraksi fitur genre menggunakan TfidfVectorizer. 2. Melakukan fit dan transformasi data genre menjadi matriks TF-IDF. 3. Matriks ini akan merepresentasikan seberapa penting sebuah genre bagi suatu film relatif terhadap seluruh katalog film yang ada.

3. Persiapan untuk Collaborative Filtering
Model Deep Learning (Neural Network) memerlukan input berupa angka (integer) yang berurutan untuk diproses oleh Embedding Layer.

Label Encoding: Kita mengonversi userId dan movieId menjadi index yang dimulai dari 0 hingga jumlah total data.

Normalisasi Rating: Rating asli (0.5 - 5.0) diubah skalanya menjadi 0 hingga 1 menggunakan teknik Min-Max Scaling.

Alasan: Skala yang seragam membantu model Neural Network untuk konvergen (mencapai hasil optimal) lebih cepat dan stabil selama proses pelatihan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- 1. Feature Engineering: TF-IDF VectorizationPada tahap ini, data teks pada kolom genres dikonversi menjadi representasi numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).Proses: Menghitung frekuensi kemunculan sebuah genre dalam satu film dan membandingkannya dengan distribusi genre di seluruh dataset.Alasan: Komputer tidak dapat memproses data string secara langsung. TF-IDF digunakan karena mampu memberikan bobot lebih rendah pada genre yang terlalu umum dan bobot lebih tinggi pada genre yang lebih spesifik, sehingga pencarian kemiripan film menjadi lebih akurat dibandingkan hanya menghitung frekuensi kata biasa.2. Encoding UserID dan MovieIDProses ini memetakan (mapping) userId dan movieId yang asli ke dalam indeks integer yang berurutan (misalnya 0 hingga jumlah total user).Proses: Membuat dictionary untuk memetakan ID asli ke ID baru yang unik dan berurutan.Alasan: Layer Embedding pada model Deep Learning memerlukan input berupa indeks integer yang berurutan untuk membuat matriks representasi (vektor laten). Jika kita menggunakan ID asli yang seringkali memiliki lompatan angka (tidak berurutan), hal ini akan menyebabkan pemborosan memori dan error pada struktur matriks model.3. Min-Max Scaling (Normalisasi Rating)Mengubah skala rating asli (0.5 – 5.0) ke dalam rentang 0 hingga 1.Proses: Mengurangi setiap nilai rating dengan nilai minimum, lalu membaginya dengan selisih nilai maksimum dan minimum.$$x_{norm} = \frac{x - min(x)}{max(x) - min(x)}$$Alasan: Algoritma Deep Learning (jaringan saraf) menggunakan fungsi aktivasi seperti Sigmoid di lapisan terakhirnya. Normalisasi ke rentang 0-1 membantu mempercepat proses konvergensi (pembelajaran) model dan mencegah masalah gradient explosion selama proses training.4. Data Splitting (Pembagian Data)Membagi data interaksi menjadi dua set: Data Training (80%) dan Data Validasi (20%).Proses: Mengacak data secara random (menggunakan random_state untuk konsistensi) agar distribusi rating tetap seimbang di kedua set.Alasan: Data training digunakan untuk melatih model agar mengenali pola preferensi, sedangkan data validasi digunakan sebagai "ujian" untuk mengukur sejauh mana model dapat memprediksi data yang belum pernah dilihat sebelumnya. Hal ini sangat penting untuk mendeteksi apakah model mengalami Overfitting (hanya menghafal data latihan).Tips untuk Reviewer:
## Modeling
Pada tahap ini, dikembangkan dua model sistem rekomendasi dengan pendekatan yang berbeda untuk memberikan hasil yang komprehensif.1. Content-Based Filtering (Cosine Similarity)Model ini bekerja dengan cara merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu.Algoritma: Menggunakan Cosine Similarity. Algoritma ini menghitung sudut antara dua vektor (film) dalam ruang multidimensi TF-IDF. Semakin kecil sudutnya, semakin besar nilai kesamaannya.Proses:Membangun matriks TF-IDF dari fitur genres.Menghitung skor kesamaan antar film dengan rumus:$$Similarity(A, B) = \frac{A \cdot B}{||A|| ||B||}$$Mengambil $N$ film dengan skor kesamaan tertinggi terhadap film yang diinputkan.

Kelebihan: Tidak membutuhkan data pengguna lain dan mampu merekomendasikan film yang sangat baru (selama genre tersedia).
Kekurangan: Terbatas pada fitur yang tersedia; tidak bisa memberikan rekomendasi yang benar-benar "mengejutkan" di luar genre yang biasa ditonton.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan Hasil Top-N Recommendation
Berikut adalah contoh output dari kedua model tersebut:
Hasil Content-Based Filtering: Jika pengguna menyukai film "Toy Story (1995)" (Genre: Adventure, Animation, Children, Comedy, Fantasy), maka sistem memberikan 5 rekomendasi teratas:
Antz (1998)
Toy Story 2 (1999)
The Emperor's New Groove (2000)
Monsters, Inc. (2001)
Shrek (2001)
Hasil Collaborative Filtering: Sistem memberikan prediksi untuk User ID 123. Berdasarkan pola rating user lain yang mirip, sistem menyarankan film yang mungkin belum ditonton namun akan mendapatkan rating tinggi dari user tersebut:
The Godfather (1972): Prediksi Rating 4.8
The Shawshank Redemption (1994): Prediksi Rating 4.7
... (dan seterusnya)

## Evaluation
Pada tahap evaluasi, digunakan metrik yang berbeda untuk menilai performa model Content-Based Filtering dan Collaborative Filtering.

1. Content-Based Filtering (Metrik: Precision)
Karena Content-Based Filtering memberikan rekomendasi dalam bentuk daftar (bukan prediksi skor), metrik yang paling relevan adalah Precision.
Definisi: Perbandingan antara jumlah item rekomendasi yang relevan (memiliki kemiripan genre) dengan total item yang direkomendasikan.
$$Precision = \frac{\text{Jumlah Rekomendasi yang Relevan}}{\text{Total Item yang Direkomendasikan}}$$****
**Rubrik/Kriteria Tambahan (Opsional)**: 
- . Precision (Content-Based Filtering)
Metrik ini mengukur seberapa banyak film yang direkomendasikan oleh sistem benar-benar relevan terhadap film yang disukai pengguna sebelumnya.
<img width="541" height="89" alt="image" src="https://github.com/user-attachments/assets/ec75c166-3e7e-4aca-812a-b7a5922d34c9" />


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
