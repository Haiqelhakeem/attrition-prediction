# Attrition Prediction at Edutech Company
## Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan yang tersebar di seluruh Indonesia. Meskipun telah mengalami pertumbuhan pesat, perusahaan ini masih menghadapi tantangan dalam pengelolaan sumber daya manusia, terutama terkait tingkat retensi karyawan.

Tingkat attrition (keluarnya karyawan) yang melebihi 10% menjadi sorotan utama, karena berpotensi menurunkan efisiensi operasional dan meningkatkan biaya rekrutmen. Oleh karena itu, divisi HR ingin menggali lebih dalam mengenai faktor-faktor yang memengaruhi keputusan karyawan untuk mengundurkan diri.

### Permasalahan Bisnis

Permasalahan utama yang menjadi fokus dalam proyek ini meliputi:

- Mengidentifikasi variabel-variabel kunci yang berkontribusi terhadap tingginya tingkat attrition.
- Menyediakan visualisasi informatif untuk membantu tim HR dalam memantau kondisi tenaga kerja secara menyeluruh.
- Memberikan rekomendasi berbasis data untuk mengurangi angka attrition secara berkelanjutan.

### Cakupan Proyek

Lingkup proyek ini mencakup:

- Analisis eksploratif terhadap data karyawan untuk menemukan pola dan wawasan penting.
- Pembuatan dashboard interaktif sebagai alat bantu dalam pengambilan keputusan strategis oleh tim HR.
- Pengembangan model prediktif sederhana untuk mengestimasi kemungkinan attrition.
- Dokumentasi hasil analisis dalam format markdown (.md).

### Persiapan

**Sumber Data**:  
Dataset diambil dari repository GitHub [dicoding](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee), yang mencakup informasi demografis, jabatan, kepuasan kerja, dan status attrition karyawan. Dataset ini umum digunakan dalam analisis HR.

**Setup Environment (Shell/Terminal - venv)**:
```bash
python -m venv env
# Aktivasi environment:
# Windows: .\env\Scripts\activate
# MacOS/Linux: source env/bin/activate
pip install -r requirements.txt
```

## ## Business Dashboard

Dashboard **Employee Attrition Prediction** ini dikembangkan untuk membantu tim Human Resource (HR) dalam:
- Mengidentifikasi karyawan yang berisiko tinggi untuk keluar (attrition),
- Menyediakan prediksi tingkat risiko berdasarkan data aktual,
- Memberikan rekomendasi tindakan strategis secara otomatis berdasarkan hasil prediksi,
- Menganalisis fitur-fitur terpenting yang memengaruhi kemungkinan attrition,
- Mengeksplorasi data berdasarkan filter interaktif.

Dengan menggabungkan model machine learning dan visualisasi data, dashboard ini menjadi alat pendukung keputusan yang berbasis data untuk strategi retensi karyawan.


🧩 Fitur Utama Dashboard

1. **📁 Data Upload & Model Integration**
   - Menggunakan model prediktif (`attrition_best_model.pkl`) untuk melakukan klasifikasi terhadap data karyawan yang dimuat dari `employee_data.csv`.
   - Model dilengkapi dengan pipeline preprocessing dan klasifikasi.

2. **🔍 Sidebar Filter Interaktif**
   - Pengguna dapat menyaring data berdasarkan:
     - **Department** (Departemen),
     - **Job Role** (Peran Jabatan),
     - **OverTime** (Lembur).
   - Penyaringan ini memudahkan fokus analisis pada kelompok karyawan tertentu.

3. **📊 Prediksi Attrition dan Risiko**
   - Sistem memprediksi apakah seorang karyawan berpotensi keluar.
   - Menghitung dan menampilkan **persentase risiko attrition**.
   - Memberikan **rekomendasi tindakan** otomatis:
     - 🚨 *Reduce workload* – jika lembur tinggi,
     - 💬 *Conduct 1-on-1* – jika kepuasan kerja rendah,
     - 🕒 *Improve balance* – jika work-life balance buruk,
     - 🎯 *Offer career path* – untuk kasus lainnya,
     - ✅ *No immediate action* – jika risiko rendah.

4. **📥 Ekspor Hasil**
   - Tabel hasil prediksi dapat diunduh sebagai file `.csv` untuk dokumentasi atau pelaporan lebih lanjut.

5. **📈 Analisis Feature Importance**
   - Visualisasi fitur-fitur paling berpengaruh dalam prediksi, berdasarkan bobot feature importance dari model Random Forest.
   - Slider disediakan untuk menyesuaikan jumlah fitur yang ingin ditampilkan.
   - Menyediakan tabel dan diagram batang horizontal untuk interpretasi mudah.

6. **📉 Visualisasi Tambahan**
   - Diagram *countplot* menunjukkan distribusi attrition berdasarkan departemen, memperkuat insight dari visualisasi eksploratif.


Dashboard ini mempermudah tim HR dalam mengidentifikasi dan menangani risiko turnover secara proaktif. Dengan integrasi model prediktif dan sistem rekomendasi otomatis, perusahaan dapat:
- Menyusun strategi retensi berbasis data,
- Meningkatkan kepuasan kerja,
- Mengurangi biaya yang ditimbulkan oleh pergantian karyawan.

## Conclusion

Berdasarkan analisis terhadap data karyawan Jaya Jaya Maju, proyek ini bertujuan untuk mengidentifikasi faktor-faktor utama yang memengaruhi tingkat attrition serta memberikan rekomendasi yang bersifat strategis dan berbasis data. Berikut adalah poin-poin utama yang berhasil diperoleh:

1. **Faktor-Faktor yang Mempengaruhi Attrition**  
   Hasil analisis menunjukkan bahwa beberapa variabel memiliki korelasi yang signifikan terhadap kemungkinan karyawan mengundurkan diri, antara lain:
   - Karyawan dengan usia yang lebih matang, pengalaman kerja yang lebih lama, dan jabatan yang lebih tinggi cenderung memiliki tingkat attrition yang lebih rendah.
   - Sebaliknya, karyawan yang relatif muda dan memiliki masa kerja yang pendek menunjukkan kecenderungan lebih tinggi untuk keluar dari perusahaan.
   - Jarak tempat tinggal dari kantor merupakan faktor penting; semakin jauh jaraknya, semakin besar kecenderungan untuk mengalami attrition.

2. **Insight Berdasarkan Visualisasi Data**  
   Visualisasi interaktif mengungkapkan bahwa variabel **DistanceFromHome** merupakan salah satu indikator paling signifikan terhadap risiko attrition. Selain itu, ditemukan bahwa:
   - **OverTime** memiliki hubungan kuat dengan peningkatan tingkat attrition, khususnya pada karyawan dengan Work-Life Balance yang rendah.
   - Variabel kategorikal seperti **Department**, **BusinessTravel**, dan **Gender** menunjukkan variasi dalam distribusi attrition, namun tidak terdapat satu pun kategori yang secara konsisten mendominasi tingkat keluarnya karyawan.
   - Kombinasi beberapa faktor — bukan satu atribut tunggal — menjadi penyebab utama attrition, sehingga pendekatan komprehensif diperlukan dalam menyusun strategi retensi.

Kesimpulan ini menjadi landasan penting bagi tim HR untuk merancang kebijakan yang lebih tepat sasaran, termasuk penyesuaian beban kerja, program pengembangan karier, serta kebijakan fleksibilitas kerja guna meningkatkan retensi dan kepuasan kerja karyawan.


3. **Kesimpulan Umum**  
   Proyek ini memberikan wawasan kritis bagi tim HR untuk memahami kompleksitas di balik attrition karyawan. Diperlukan pendekatan yang berbasis data dan komprehensif untuk merancang strategi retensi yang lebih efektif.

Dengan adanya dashboard yang telah dikembangkan, perusahaan kini memiliki alat pemantauan real-time terhadap faktor-faktor kritis yang mempengaruhi kepuasan dan loyalitas karyawan.

## Rekomendasi Tindak Lanjut

Berdasarkan hasil analisis, berikut adalah beberapa tindakan strategis yang direkomendasikan:

- **🚨 Reduce workload**: Jika karyawan berisiko tinggi mengalami attrition dan bekerja lembur secara konsisten, maka disarankan untuk mengurangi beban kerja atau memberikan fleksibilitas waktu.
- **💬 Conduct 1-on-1**: Jika kepuasan kerja karyawan rendah (skor 1 atau 2), HR disarankan untuk mengadakan sesi percakapan pribadi guna memahami masalah secara langsung.
- **🕒 Improve balance**: Untuk karyawan dengan WorkLifeBalance rendah, dapat dilakukan perbaikan kebijakan kerja, misalnya cuti tambahan atau fleksibilitas jam kerja.
- **🎯 Offer career path**: Jika tidak ada faktor risiko spesifik namun model memprediksi kemungkinan attrition, perusahaan dapat menawarkan jalur karier yang jelas atau program pengembangan diri.
- **✅ No immediate action**: Untuk karyawan yang diprediksi tetap bertahan, tidak diperlukan tindakan khusus saat ini.

Dengan langkah-langkah tersebut, diharapkan perusahaan dapat menekan angka attrition dan menciptakan lingkungan kerja yang lebih produktif dan berkelanjutan.
