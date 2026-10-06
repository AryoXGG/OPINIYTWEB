# Web Visualisasi — Analisis Opini Influencer Keuangan

Aplikasi Streamlit untuk memvisualkan hasil skripsi:
halaman **Prediksi Opini** (klasifikasi live memakai model SVM + pembanding 5 algoritma),
**Dashboard Hasil** (metrik, grafik, sebaran per influencer), dan **Metodologi**.

## 1. Train ulang + ekspor artefak (Google Colab)

1. Buka `DATA/Machine_Learning.ipynb` hasil patch (sudah bersih dari tugas kedua).
2. Run All. Sel terakhir menyimpan 7 file `.pkl`:
   `tfidf_vectorizer.pkl`, `model_nb_opini.pkl`, `model_lr_opini.pkl`,
   `model_svm_opini.pkl`, `model_rf_opini.pkl`, `model_xgb_opini.pkl`, `label_classes.pkl`.
3. Catat angka metrik baru. Jika bergeser dari laporan, perbarui Tabel 4.4 skripsi
   (tabel `METRICS` di `app.py` dan grafik Dashboard mengikuti angka final).
4. Salin ketujuh `.pkl` ke folder `Web/artifacts/`.

> Catatan: `Labeling.ipynb` dan `dataset_labeled.csv` TIDAK perlu di-run ulang
> (pelabelan RoBERTa berat; label yang ada sudah final).

## 2. Jalankan lokal

```bash
cd Web
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`. Saat pertama dipakai, NLTK mengunduh paket
`stopwords` otomatis (butuh internet sekali saja).

## 3. Deploy gratis (Streamlit Community Cloud)

1. Push folder `Web/` ke GitHub (beserta isi `artifacts/`).
2. Buka share.streamlit.io → New app → pilih repo, file `app.py`.
3. Tuliskan `requirements.txt` sebagai dependensi. Selesai, dapat URL publik
   untuk demo sidang.

## Struktur

```
Web/
├── app.py            # aplikasi (3 halaman)
├── requirements.txt
├── README.md         # file ini
└── artifacts/        # 7 file .pkl + METRICS (diisi setelah train ulang)
```
