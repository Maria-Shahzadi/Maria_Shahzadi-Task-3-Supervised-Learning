import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🏠 Advanced House Price App", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
model = pickle.load(open("house_price_model.pkl", "rb"))

# ==============================
# UI STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right,#1e3a8a, #3b82f6);
    color: white;
}

h1 {
    text-align:center;
    color:#38bdf8;
}

.block {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
}

/* FIX INPUT TEXT VISIBILITY */
input, textarea {
    color: black !important;
    background-color: white !important;
}

label {
    color: #e2e8f0 !important;
    font-weight: 600;
}

.stNumberInput input {
    color: black !important;
}

.stSelectbox div {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# NAVIGATION
# ==============================
menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "📊 Prediction", "📈 Analytics"])

# ==============================
# HOME SCREEN
# ==============================
if menu == "🏠 Home":
    st.title("🏠 House Price Prediction App")
    st.markdown("<div class='block'>Welcome to your smart ML dashboard 💙</div>", unsafe_allow_html=True)

# ==============================
# PREDICTION SCREEN
# ==============================
elif menu == "📊 Prediction":
    st.title("🔮 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-122.0)
        latitude = st.number_input("Latitude", value=37.0)
        housing_median_age = st.number_input("Housing Age", step=1)
        total_rooms = st.number_input("Total Rooms", step=1)

    with col2:
        total_bedrooms = st.number_input("Bedrooms", step=1)
        population = st.number_input("Population", step=1)
        households = st.number_input("Households", step=1)
        median_income = st.number_input("Income")

    ocean = st.selectbox("Ocean Proximity", ["INLAND", "NEAR BAY"])
    ocean_encoded = 1 if ocean == "INLAND" else 0

    user_input = np.array([[longitude, latitude, housing_median_age, total_rooms,
                            total_bedrooms, population, households, median_income, ocean_encoded]])

    if st.button("Predict"):
        try:
            prediction = model.predict(user_input)
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")
        except:
            st.error("Prediction failed")

# ==============================
# ANALYTICS SCREEN
# ==============================
elif menu == "📈 Analytics":
    st.title("📈 Data Insights")

    # Dummy data for visualization
    data = pd.DataFrame({
        "Rooms": [100,200,300,400,500],
        "Prices": [100000,200000,300000,400000,500000]
    })

    st.subheader("📊 Price vs Rooms")

    fig, ax = plt.subplots()
    ax.plot(data["Rooms"], data["Prices"])
    ax.set_xlabel("Rooms")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    st.subheader("📉 Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(data["Prices"])
    st.pyplot(fig2)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("Made with 💙 by Maria")
