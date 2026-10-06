# -*- coding: utf-8 -*-
"""Aplikasi web visualisasi hasil: Analisis Opini Audiens Influencer Keuangan.
Jalankan: streamlit run app.py
Artefak model (.pkl) ditaruh di folder artifacts/ (dihasilkan sel ekspor Machine_Learning.ipynb).
"""
import pathlib
import re
import html

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = pathlib.Path(__file__).parent
ART = BASE / 'artifacts'

st.set_page_config(page_title='Analisis Opini Influencer Keuangan', page_icon='📊', layout='wide')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(120deg, #0f2027, #203a43, #2c5364); border-radius: 16px;
        padding: 28px 32px; color: white; margin-bottom: 18px; }
.hero h1 { font-size: 1.7rem; font-weight: 800; margin: 0 0 6px 0; }
.hero p { opacity: .85; margin: 0; font-size: .95rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2027, #2c5364); }
section[data-testid="stSidebar"] * { color: #eef4f7 !important; }
div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #1f3864, #2c5364);
    border: none; border-radius: 10px; padding: .55rem 1.6rem; font-weight: 600; }
div.stButton > button { border-radius: 10px; }
.result-card { background: #f0f7f2; border-left: 6px solid #2ca02c; border-radius: 12px;
               padding: 16px 20px; margin: 12px 0; }
.result-card.neg { background: #fdf0f0; border-left-color: #d62728; }
.result-card.neu { background: #eef4fb; border-left-color: #1f77b4; }
.footer { text-align: center; opacity: .6; font-size: .8rem; margin-top: 30px; }
</style>
''', unsafe_allow_html=True)

# ---------------- util: artefak ----------------
@st.cache_resource
def load_artifacts():
    out = {}
    try:
        cands = [ART, BASE]
        base = next(d for d in cands if (d / 'tfidf_vectorizer.pkl').exists())
        out['tfidf'] = joblib.load(base / 'tfidf_vectorizer.pkl')
        for m in ['nb', 'lr', 'svm', 'rf', 'xgb']:
            out[m] = joblib.load(base / f'model_{m}_opini.pkl')
        out['classes'] = list(joblib.load(base / 'label_classes.pkl'))
        ok = True
    except Exception as e:  # noqa: BLE001
        ok, out['error'] = False, str(e)
    out['ok'] = ok
    return out

# ---------------- util: prapemrosesan (sama dgn notebook) ----------------
NORMALISASI = {
    'yg': 'yang', 'dr': 'dari', 'dri': 'dari', 'dg': 'dengan', 'dgn': 'dengan',
    'utk': 'untuk', 'jd': 'jadi', 'org': 'orang', 'krn': 'karena', 'tp': 'tapi',
    'tdk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'aja': 'saja', 'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'bgt': 'banget', 'trs': 'terus', 'sm': 'sama', 'amp': 'sampai',
    'bagis': 'bagus', 'nyas': 'nya', 'kokoh': 'koko', 'yaaa': 'ya', 'yaa': 'ya',
    'yaaah': 'ya', 'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'gmn': 'bagaimana',
    'gmna': 'bagaimana', 'gimana': 'bagaimana', 'gimna': 'bagaimana', 'bnyk': 'banyak',
    'banyakk': 'banyak', 'thn': 'tahun', 'jt': 'juta', 'jtan': 'jutaan', 'rb': 'ribu',
    'makasi': 'terima kasih', 'makasih': 'terima kasih', 'dratis': 'drastis',
    'vedio': 'video', 'inves': 'investasi', 'dlm': 'dalam', 'bljr': 'belajar',
    'trsnya': 'terusnya',
}
CUSTOM_STOP = {'nih', 'sih', 'deh', 'dong', 'kok', 'lah', 'yah', 'ya', 'nya', 'pun',
               'kan', 'mah', 'weh', 'woi'}

@st.cache_resource
def get_nlp():
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as _sw
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return set(_sw.words('indonesian')) | CUSTOM_STOP, StemmerFactory().create_stemmer()

def preprocess(text, stop_words, stemmer):
    t = text.lower()
    t = html.unescape(t)
    t = re.sub(r'http\S+|www\S+', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = re.sub(r'\b\d{1,2}:\d{2}\b', ' ', t)
    t = re.sub(r'(.)\1{2,}', r'\1', t)
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    toks = [NORMALISASI.get(w, w) for w in t.split()]
    toks = [w for w in toks if w not in stop_words]
    return ' '.join(stemmer.stem(w) for w in toks)

# ---------------- data ringkas terverifikasi ----------------
METRICS = pd.DataFrame([
    ['Naive Bayes', 0.6931, 0.6841, 0.6931, 0.6413],
    ['Logistic Regression', 0.7417, 0.7398, 0.7417, 0.7178],
    ['Random Forest', 0.7570, 0.7533, 0.7570, 0.7447],
    ['XGBoost', 0.7468, 0.7415, 0.7468, 0.7273],
    ['SVM', 0.7596, 0.7524, 0.7596, 0.7521],
], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
INFLUENCER = pd.DataFrame([
    ['Leon Hartono', 669, 1039, 317, 2025],
    ['Timothy Ronald', 581, 645, 406, 1632],
    ['Doddy Bicara Investasi', 59, 176, 25, 260],
], columns=['Influencer', 'Positif', 'Netral', 'Negatif', 'Total'])
CONTOH = {
    'Positif': 'Terima kasih penjelasannya sangat bermanfaat, akhirnya paham bedanya emas fisik dan digital',
    'Netral': 'Min, untuk pemula mulai dari emas atau reksadana dulu ya?',
    'Negatif': 'Penjelasannya menyesatkan, datanya tidak lengkap dan bikin bingung pemula',
}
WARNA = {'Positif': '#2ca02c', 'Netral': '#1f77b4', 'Negatif': '#d62728'}

# ---------------- UI ----------------
st.markdown('''<div class="hero"><h1>📊 Analisis Opini Audiens terhadap Influencer Keuangan</h1>
<p>Klasifikasi opini komentar YouTube (Positif · Negatif · Netral) — Naive Bayes · Logistic Regression · SVM · Random Forest · XGBoost</p></div>''',
    unsafe_allow_html=True)
menu = st.sidebar.radio('Navigasi', ['🔮 Prediksi Opini', '📈 Dashboard Hasil', 'ℹ️ Metodologi'])
st.sidebar.markdown('---')
st.sidebar.caption('Model terbaik: **SVM** (Accuracy 0,7596 · F1 0,7522) · Data uji: 782 komentar')

if menu == '🔮 Prediksi Opini':
    st.header('Prediksi opini komentar')
    art = load_artifacts()
    if not art['ok']:
        st.warning('Artefak model belum ditemukan (7 file .pkl di folder aplikasi). Jalankan sel ekspor di `Machine_Learning.ipynb`, '
                   'lalu upload ketujuh `.pkl` ke repo. Detail: ' + art.get('error', ''))
    col1, col2 = st.columns([3, 1])
    with col2:
        st.write('Contoh cepat:')
        for k in ['Positif', 'Netral', 'Negatif']:
            if st.button(k, key='c_' + k):
                st.session_state['teks'] = CONTOH[k]
    with col1:
        teks = st.text_area('Tulis komentar YouTube berbahasa Indonesia:', key='teks', height=120,
                            placeholder='misal: terima kasih ilmunya sangat bermanfaat bang...')
    if st.button('Klasifikasikan', type='primary'):
        if not teks.strip():
            st.error('Isi dulu komentarnya.')
        elif not art['ok']:
            st.error('Model belum tersedia.')
        else:
            stop_words, stemmer = get_nlp()
            bersih = preprocess(teks, stop_words, stemmer)
            if not bersih:
                st.error('Teks habis setelah prapemrosesan (mis. hanya emoji/URL).')
            else:
                st.write('Teks bersih:', f'`{bersih}`')
                X = art['tfidf'].transform([bersih])
                pred_svm = art['svm'].predict(X)[0]
                cls = 'neg' if pred_svm == 'Negatif' else ('neu' if pred_svm == 'Netral' else '')
                st.markdown(f'<div class="result-card {cls}"><b>Hasil model terbaik (SVM): {pred_svm}</b></div>',
                            unsafe_allow_html=True)
                st.write('Perbandingan kelima model:')
                hasil = []
                svm_dec, svm_cls = None, None
                for key, nama in [('nb', 'Naive Bayes'), ('lr', 'Logistic Regression'), ('svm', 'SVM'),
                                  ('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
                    m = art[key]
                    pred = m.predict(X)[0]
                    if hasattr(m, 'predict_proba'):
                        conf = float(np.max(m.predict_proba(X)))
                    else:  # LinearSVC: softmax atas decision_function sbg pendekatan keyakinan
                        dec = np.asarray(m.decision_function(X)[0], dtype=float)
                        e = np.exp(dec - dec.max())
                        proba = e / e.sum()
                        conf = float(proba[list(m.classes_).index(pred)])
                        if key == 'svm':
                            svm_dec, svm_cls = dec, list(m.classes_)
                    hasil.append((nama, pred, round(conf, 4)))
                df_h = pd.DataFrame(hasil, columns=['Model', 'Prediksi', 'Keyakinan'])
                st.dataframe(df_h, use_container_width=True)
                if svm_dec is not None:
                    with st.expander('Lihat skor keputusan SVM per kelas'):
                        fig, ax = plt.subplots()
                        ax.bar(svm_cls, svm_dec, color=['#d62728' if c == 'Negatif' else '#1f77b4' if c == 'Netral' else '#2ca02c' for c in svm_cls])
                        ax.set_ylabel('Skor decision_function')
                        ax.set_title('Jarak margin SVM (makin besar = makin yakin)')
                        st.pyplot(fig)
                st.caption('*Keyakinan SVM dihitung via softmax atas decision_function (pendekatan, bukan probabilitas terkalibrasi). '
                           'Sistem hanya mengklasifikasikan teks, tanpa menyimpulkan hal di luar teks.')

elif menu == '📈 Dashboard Hasil':
    st.header('Hasil evaluasi (data uji, 782 komentar)')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Model Terbaik', 'SVM')
    m2.metric('Accuracy', '0,7596')
    m3.metric('F1-Score', '0,7521')
    m4.metric('Data Uji', '782')
    st.dataframe(METRICS, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.bar(METRICS['Model'], METRICS['Accuracy'], color=['#A6A6A6'] * 4 + ['#1F3864'])
        ax.set_ylim(0.6, 0.82)
        ax.set_title('Accuracy')
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.bar(METRICS['Model'], METRICS['F1-Score'], color=['#A6A6A6'] * 4 + ['#1F3864'])
        ax.set_ylim(0.55, 0.82)
        ax.set_title('F1-Score (weighted)')
        plt.xticks(rotation=20)
        st.pyplot(fig)
    st.subheader('Sebaran opini per influencer (n = 3.917)')
    st.dataframe(INFLUENCER, use_container_width=True)
    fig, ax = plt.subplots()
    x = np.arange(len(INFLUENCER))
    ax.bar(x, INFLUENCER['Positif'], label='Positif', color='#2ca02c')
    ax.bar(x, INFLUENCER['Netral'], bottom=INFLUENCER['Positif'], label='Netral', color='#1f77b4')
    ax.bar(x, INFLUENCER['Negatif'], bottom=INFLUENCER['Positif'] + INFLUENCER['Netral'], label='Negatif', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(['Leon\nHartono', 'Timothy\nRonald', 'Doddy Bicara\nInvestasi'])
    ax.legend()
    st.pyplot(fig)

else:
    st.header('Metodologi singkat')
    st.markdown('''
1. **Pengumpulan**: 4.186 komentar via YouTube Data API v3 (21 video, 3 influencer).
2. **Pelabelan**: model RoBERTa Bahasa Indonesia + penyempurnaan kaidah → 3.917 berlabel (Netral 47,5%, Positif 33,4%, Negatif 19,1%).
3. **Prapemrosesan**: case folding → cleansing → normalisasi slang → tokenisasi → stopword removal → stemming Sastrawi.
4. **Fitur**: TF-IDF unigram+bigram (`min_df=2`), fit hanya pada data latih.
5. **Split**: stratified 80:20 (3.128 latih / 782 uji, `random_state=42`).
6. **Model**: Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost — terbaik **SVM** (Accuracy 0,7596; F1 0,7521).
''')
st.markdown('<div class="footer">Skripsi Informatika UISI · Dinunaya Syuja Aryoko (3012210012) · 2026</div>',
            unsafe_allow_html=True)
