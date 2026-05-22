import streamlit as st
import pandas as pd
import pickle

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

# Load Model & Data
model = pickle.load(open("models/model.pkl", "rb"))
data = pd.read_csv("data/dataset.csv")
st.dataframe(data.head())

# Prediksi
if st.button("Jalankan Prediksi"):
 st.success("Prediksi berhasil!")
 st.write(model.predict(data))