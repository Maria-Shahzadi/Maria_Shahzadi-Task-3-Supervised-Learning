import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🏠 House Price Prediction</h1>
    <h4 style='text-align: center; color: #2e7d32;'>Made by Maria ✨</h4>
    <p style='text-align: center; color: gray;'>Enter property details to predict house price</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Property Details")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0, format="%.2f")
    HouseAge = st.number_input("House Age", min_value=0, format="%.2f")
    AveRooms = st.number_input("Average Rooms", min_value=0, format="%.2f")
    AveBedrms = st.number_input("Average Bedrooms", min_value=0, format="%.2f")

with col2:
    Population = st.number_input("Population", min_value=0, format="%.2f")
    AveOccup = st.number_input("Average Occupancy", min_value=0, format="%.2f")
    Latitude = st.number_input("Latitude", format="%.4f")
    Longitude = st.number_input("Longitude", format="%.4f")

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price", use_container_width=True):

    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(input_data)

    st.success("🎯 Prediction Completed!")

    st.markdown(f"""
        <div style='text-align:center; padding:20px; background-color:#e8f5e9; border-radius:10px;'>
            <h2 style='color:#2e7d32;'>Predicted Price</h2>
            <h1 style='color:#1b5e20;'>${prediction[0]:.2f}</h1>
        </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by Maria | Streamlit ML Project</p>", unsafe_allow_html=True)