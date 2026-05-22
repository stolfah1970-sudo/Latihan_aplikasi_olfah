import streamlit as st
import pandas as pd
import pickle
import altair as alt
import subprocess

st.title("Dashboard Analitik Belanja")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

# Load Model & Data
# model = pickle.load(open("models/model.pkl", "rb"))
data = pd.read_csv("data/02_realisasi_anggaran_klasifikasi.csv")
st.dataframe(data.head())

# Visualisasi dan Prediksi dalam Tabs
tab1, tab2 = st.tabs(["Visualisasi", "Prediksi"])

with tab1:
    st.header("Scatter Plot: Skor IKPA vs Deviasi RPD")
    # Pastikan domain sumbu x minimal 70
    xmax = data['skor_ikpa'].max()
    if xmax < 70:
        xmax = 70
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X('skor_ikpa', scale=alt.Scale(domain=[70, float(xmax)]), axis=alt.Axis(title='skor_ikpa')),
        y=alt.Y('deviasi_rpd_persen', axis=alt.Axis(title='deviasi_rpd_persen')),
        tooltip=list(data.columns)
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.header("Menu Prediksi")
    st.write("Isi data entri di bawah untuk melakukan prediksi menggunakan model di folder model.")

    # Ambil opsi kategori dari data jika tersedia
    if 'tipe_satker' in data.columns:
        tipe_options = list(data['tipe_satker'].dropna().unique())
    else:
        tipe_options = ["Kantor Pusat", "Kantor Daerah", "Dekonsentrasi", "Tugas Pembantuan"]

    with st.form(key='predict_form'):
        jumlah_spm = st.number_input('Jumlah SPM', min_value=0, value=int(data['jumlah_spm'].median() if 'jumlah_spm' in data.columns else 50))
        revisi_dipa = st.selectbox('Revisi DIPA', options=sorted(list(set(data['revisi_dipa'].dropna().astype(int).tolist())) ) if 'revisi_dipa' in data.columns else [0,1,2,3,4,5])
        deviasi_rpd = st.number_input('Deviasi RPD (%)', format="%.2f", value=float(data['deviasi_rpd_persen'].median() if 'deviasi_rpd_persen' in data.columns else 0.0))
        skor_ikpa = st.number_input('Skor IKPA', min_value=0.0, max_value=100.0, format="%.2f", value=float(data['skor_ikpa'].median() if 'skor_ikpa' in data.columns else 80.0))
        tipe_satker = st.selectbox('Tipe Satker', options=tipe_options)

        submit = st.form_submit_button('Prediksi')

    if submit:
        # Coba load model
        try:
            model = pickle.load(open('model/Best_model.pkcls', 'rb'))
        except Exception as e:
            err_msg = str(e)
            st.error(f"Gagal memuat model: {err_msg}")
        else:
            # Buat DataFrame input sesuai kolom yang mungkin diperlukan (untuk ditampilkan)
            input_df = pd.DataFrame([{
                'jumlah_spm': int(jumlah_spm),
                'revisi_dipa': int(revisi_dipa),
                'deviasi_rpd_persen': float(deviasi_rpd),
                'skor_ikpa': float(skor_ikpa),
                'tipe_satker': tipe_satker
            }])

            try:
                # Jika model adalah model Orange yang menyimpan domain, bangun array numerik
                # sesuai urutan atribut di domain (mis. indikator tipe_satker=...)
                if hasattr(model, 'domain') and getattr(model, 'domain') is not None:
                    attrs = model.domain.attributes
                    row = []
                    for a in attrs:
                        aname = a.name
                        if '=' in aname:
                            # indikator dari variabel kategorikal, format: "var=Value"
                            base, val = aname.split('=', 1)
                            if base in input_df.columns:
                                # cocokkan nilai kategorikal
                                row.append(1.0 if str(input_df.iloc[0][base]) == val else 0.0)
                            else:
                                row.append(0.0)
                        else:
                            # variabel kontinu, ambil dari input_df jika ada
                            row.append(float(input_df.iloc[0].get(aname, 0.0)))

                    import numpy as np
                    X = np.array([row], dtype=float)
                    preds = model.predict(X)
                else:
                    # fallback: lewati DataFrame langsung ke model
                    preds = model.predict(input_df)
            except Exception as e:
                st.error(f"Prediksi gagal: {e}")
                st.write("Coba periksa format fitur yang diharapkan model. Menampilkan input yang dikirim:")
                st.dataframe(input_df)
            else:
                # Format hasil prediksi agar lebih ramah: label prediksi dan probabilitas per kelas
                try:
                    import numpy as _np
                    # dapatkan nama kelas jika tersedia
                    try:
                        class_names = list(model.domain.class_var.values)
                    except Exception:
                        class_names = None

                    # model Orange bisa mengembalikan (labels, probs)
                    if isinstance(preds, tuple) and len(preds) >= 2:
                        labels, probs = preds[0], _np.array(preds[1])
                    else:
                        labels = _np.array(preds)
                        probs = None

                    # ambil prediksi pertama
                    pred_idx = int(_np.asarray(labels).ravel()[0])
                    pred_label = class_names[pred_idx] if class_names else str(pred_idx)

                    st.success(f"Prediksi: {pred_label}")

                    if probs is not None:
                        probs = _np.asarray(probs)
                        probs_row = probs.reshape(probs.shape[0], -1)[0]
                        # buat DataFrame probabilitas
                        try:
                            import pandas as _pd
                            prob_df = _pd.DataFrame({
                                'kelas': class_names if class_names else [str(i) for i in range(len(probs_row))],
                                'probabilitas (%)': (probs_row * 100).round(4)
                            })
                            st.write('Probabilitas per kelas:')
                            st.dataframe(prob_df)
                            # tampilkan sebagai bar chart
                            st.bar_chart(prob_df.set_index('kelas'))
                        except Exception:
                            st.write('Probabilitas:', probs_row)
                except Exception as e:
                    st.write('Hasil prediksi:', preds)
                    st.error(f'Gagal memformat hasil prediksi: {e}')