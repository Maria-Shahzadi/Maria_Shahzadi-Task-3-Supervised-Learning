import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Maria ML Studio",
    layout="wide",
    page_icon="⚡"
)

# ==============================
# ULTRA MODERN CSS (GLASSMORPHISM)
# ==============================
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

/* Title */
h1 {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
}

/* Glass Cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    font-size: 16px;
    padding: 10px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0b1220;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("⚡ Maria ML Studio")
st.write("Ultra Modern Regularisation Dashboard (Ridge vs Lasso)")

# ==============================
# LOAD MODEL
# ==============================
model = pickle.load(open("lasso_model.pkl", "rb"))

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("🎛 Feature Controls")

features = []
for i in range(20):
    val = st.sidebar.slider(f"Feature {i+1}", -10.0, 10.0, 0.0)
    features.append(val)

features = np.array(features).reshape(1, -1)

# ==============================
# PREDICTION
# ==============================
if st.button("🚀 Run Prediction"):
    pred = model.predict(features)

    st.markdown(f"""
    <div class="card">
        <h2 style='text-align:center;'>Prediction Result</h2>
        <h1 style='color:#22c55e;text-align:center;'>{pred[0]:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# FEATURE IMPORTANCE (MODERN CHART)
# ==============================
st.subheader("📊 Feature Importance (Lasso)")

coefs = model.coef_

fig = go.Figure()

fig.add_trace(go.Bar(
    x=[f"F{i+1}" for i in range(len(coefs))],
    y=coefs,
    marker_color=["#38bdf8" if c != 0 else "#ef4444" for c in coefs]
))

fig.update_layout(
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font_color="white",
    title="Lasso Feature Selection",
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# STATS CARDS
# ==============================
zero_features = np.sum(coefs == 0)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <h3>Features</h3>
        <h2>{len(coefs)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Removed Features</h3>
        <h2>{zero_features}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>Model Type</h3>
        <h2>LassoCV</h2>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# INFO SECTION
# ==============================
with st.expander("📘 How it works"):
    st.write("""
    ✔ Lasso removes useless features (sets them to 0)  
    ✔ Blue bars = important features  
    ✔ Red bars = eliminated features  
    ✔ This is feature selection automatically  
    """)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("<h5 style='text-align:center;color:gray;'>Made with 💙 by Maria</h5>", unsafe_allow_html=True)