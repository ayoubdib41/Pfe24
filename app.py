
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prédiction Superstore", layout="centered")
st.title("📈 Prédiction des Quantités et Profits Superstore")

# Charger les modèles
quantity_model = joblib.load("quantity_pipeline.pkl")
profit_model = joblib.load("profit_pipeline.pkl")

# Interface utilisateur
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Catégorie", ["Furniture", "Office Supplies", "Technology"])
    region = st.selectbox("Région", ["East", "West", "Central", "South"])
with col2:
    sub_category = st.selectbox("Sous-catégorie", ["Chairs", "Phones", "Binders", "Paper", "Storage"])
    year = st.selectbox("Année", [2014, 2015, 2016, 2017])
    month = st.selectbox("Mois", list(range(1, 13)))

input_df = pd.DataFrame({
    'Category': [category],
    'Sub-Category': [sub_category],
    'Region': [region],
    'Order_Year': [year],
    'Order_Month': [month]
})

if st.button("📊 Prédire"):
    quantity = quantity_model.predict(input_df)[0]
    profit = profit_model.predict(input_df)[0]

    st.success(f"📦 Quantité prédite : {quantity:.0f}")
    st.success(f"💰 Profit prédit : {profit:.2f} $")
