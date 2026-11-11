# foundations_app.py
# Streamlit app: Foundations (Perceptron, MLP, Activations) + decision boundary visualizer

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="NN Foundations: Perceptron & MLP", layout="wide")

st.title("Foundations: Perceptrons, MLPs, and Activation Functions")
st.write("Explore linear vs. non-linear decision boundaries and activation effects.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Dataset")
    ds = st.selectbox("Select toy dataset",
                      ["Linear (2 classes)", "Moons (nonlinear)", "Circles (nonlinear)"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)

    st.header("Model")
    model_type = st.selectbox("Model", ["Perceptron (linear)", "MLP (nonlinear)"])
    activation = st.selectbox("MLP activation", ["relu", "tanh", "logistic", "identity"])
    hidden = st.text_input("Hidden layer sizes (e.g., 16,16)", value="16,16")
    alpha = st.slider("L2 regularization (alpha)", 0.0001, 0.01, 0.0005, 0.0001)
    max_iter = st.slider("Max Iterations/Epochs", 100, 3000, 800, 100)
    random_state = st.number_input("Random seed", 0, 9999, 42)

def make_linear(n, noise, rs):
    # truly linear-separable dataset
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=2.0, random_state=rs)
    if noise > 0:
        rng = np.random.RandomState(rs)
        X += noise * rng.randn(*X.shape)
    return X, y

# --- Data ---
if ds == "Linear (2 classes)":
    X, y = make_linear(n_samples, noise, random_state)
elif ds == "Moons (nonlinear)":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)

# --- Model ---
if model_type.startswith("Perceptron"):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", Perceptron(max_iter=max_iter, random_state=random_state))
    ])
else:
    hidden_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip().isdigit())
    if not hidden_sizes:
        hidden_sizes = (16, 16)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                              activation=activation,
                              alpha=alpha,
                              learning_rate_init=0.01,
                              max_iter=max_iter,
                              random_state=random_state))
    ])

# --- Fit ---
clf.fit(X, y)
score = clf.score(X, y)

# --- Plot decision boundary ---
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_light = ListedColormap(["#FFEEEE", "#EEFFEE"])
cmap_bold = ["#CC0000", "#007700"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
ax.scatter(X[:,0], X[:,1], c=y, s=25, edgecolor="k", cmap=ListedColormap(cmap_bold))
ax.set_title(f"{model_type} (acc={score:.3f})")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
st.pyplot(fig)

# --- Activation function viewer ---
st.subheader("ReLU Activation Function")
xgrid = np.linspace(-6, 6, 401)
activation = np.maximum(0, xgrid)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(xgrid, activation, label="ReLU")
ax2.set_xlabel("Input z")
ax2.set_ylabel("Output f(z)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
plt.close(fig2)
