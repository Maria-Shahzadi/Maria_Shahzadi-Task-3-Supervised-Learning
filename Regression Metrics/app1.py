import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="ML Dashboard by Maria",
    page_icon="✨",
    layout="wide"
)

# ==============================
# CUSTOM CSS (BEAUTIFUL UI)
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
}
footer {
    visibility: hidden;
}
.footer-text {
    text-align: center;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Controls")

n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
noise = st.sidebar.slider("Noise Level", 0.1, 5.0, 1.0)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1)
iterations = st.sidebar.slider("Iterations", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by Maria")

# ==============================
# TITLE
# ==============================
st.title("✨ Machine Learning Dashboard")
st.write("Linear Regression + Gradient Descent Visualization")

# ==============================
# DATA GENERATION
# ==============================
np.random.seed(42)

X = 2 * np.random.rand(n_samples, 1)
y = 4 + 3 * X + noise * np.random.randn(n_samples, 1)

# ==============================
# SKLEARN MODEL
# ==============================
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ==============================
# METRICS
# ==============================
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R²", f"{r2:.2f}")

# ==============================
# GRADIENT DESCENT
# ==============================
m = 0
b = 0
n = len(X)

cost_history = []

for i in range(iterations):
    y_pred_gd = m * X + b

    dm = (-2/n) * np.sum(X * (y - y_pred_gd))
    db = (-2/n) * np.sum(y - y_pred_gd)

    m = m - learning_rate * dm
    b = b - learning_rate * db

    cost = (1/n) * np.sum((y - y_pred_gd)**2)
    cost_history.append(cost)

# ==============================
# PLOTS
# ==============================
st.subheader("📊 Regression Line")

fig1 = plt.figure()
plt.scatter(X, y)
plt.plot(X, m * X + b)
plt.title("Gradient Descent Fit")
st.pyplot(fig1)

st.subheader("📉 Cost Reduction")

fig2 = plt.figure()
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Curve")
st.pyplot(fig2)

# ==============================
# DETAILS SECTION
# ==============================
with st.expander("📘 Show Details"):
    st.write(f"Final Slope (m): {m}")
    st.write(f"Final Intercept (b): {b}")

# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="footer-text">
    🚀 Developed by <b>Maria</b> | Machine Learning Dashboard 💙
</div>
""", unsafe_allow_html=True)