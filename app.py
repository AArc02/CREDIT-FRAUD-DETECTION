import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
DATA_PATH = "data/creditcard.csv"
SEED = 123


# ──────────────────────────────────────────────────────────────
# 1. Chargement des données
# ──────────────────────────────────────────────────────────────
@st.cache_data(persist=True)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge le CSV de transactions.
    La première ligne est malformée (8 champs) → skiprows=1.
    """
    df = pd.read_csv(path, header=None, skiprows=1, names=COLUMNS)
    df.dropna(subset=["Class"], inplace=True)
    df["Class"] = df["Class"].astype(int)
    return df


# ──────────────────────────────────────────────────────────────
# 2. Split train / test
# ──────────────────────────────────────────────────────────────
@st.cache_data
def split(df: pd.DataFrame):
    y = df["Class"]
    X = df.drop(columns=["Class"])
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


# ──────────────────────────────────────────────────────────────
# 3. Entraînement
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(classifier_name: str, params: dict, _X_train, _y_train):
    if classifier_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=SEED,
        )
    elif classifier_name == "SVM":
        model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
    else:
        model = LogisticRegression(C=params["C_lr"], max_iter=1000)

    model.fit(_X_train, _y_train)
    return model


# ──────────────────────────────────────────────────────────────
# 4. Interface principale
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Credit Fraud Detection",
        page_icon="💳",
        layout="wide",
    )

    # ── Header ─────────────────────────────────────────────────
    st.title("💳 Détection de défaut de paiement")
    st.markdown(
        "Analyse et classification de transactions frauduleuses "
        "via **Machine Learning**."
    )
    st.divider()

    # ── Chargement ─────────────────────────────────────────────
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            f"❌ Fichier `{DATA_PATH}` introuvable. "
            "Placez `creditcard.csv` dans le dossier `data/`."
        )
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    st.sidebar.title("⚙️ Paramètres")

    # Données brutes
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("📋 Échantillon — 100 observations")
        st.dataframe(df.sample(100, random_state=SEED), use_container_width=True)

    # ── Statistiques générales ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    fraudes = df["Class"].sum()
    col1.metric("Total transactions", f"{total:,}")
    col2.metric("Transactions normales", f"{total - fraudes:,}")
    col3.metric("Fraudes détectées", f"{fraudes}")
    col4.metric("Taux de fraude", f"{fraudes/total*100:.2f}%")

    st.divider()

    # ── Choix du classificateur ────────────────────────────────
    st.sidebar.subheader("Classificateur")
    classifier_name = st.sidebar.selectbox(
        "Modèle",
        ("Random Forest", "SVM", "Logistic Regression"),
    )

    params = {}
    if classifier_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Nombre d'arbres", 10, 300, 100, 10)
        params["max_depth"] = st.sidebar.slider("Profondeur max", 1, 30, 5)
    elif classifier_name == "SVM":
        params["C"] = st.sidebar.slider("C (régularisation)", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))
    else:
        params["C_lr"] = st.sidebar.slider("C (régularisation)", 0.01, 10.0, 1.0, 0.01)

    # ── Entraînement ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = split(df)

    with st.spinner(f"⏳ Entraînement du modèle **{classifier_name}**…"):
        model = train_model(classifier_name, params, X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Métriques ──────────────────────────────────────────────
    st.subheader(f"📊 Résultats — {classifier_name}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
    m3.metric("Recall",    f"{recall_score(y_test, y_pred, zero_division=0):.2%}")
    m4.metric("F1-Score",  f"{f1_score(y_test, y_pred, zero_division=0):.2%}")
    m5.metric("ROC-AUC",   f"{roc_auc_score(y_test, y_proba):.4f}")

    st.divider()

    # ── Visualisations ─────────────────────────────────────────
    st.subheader("📈 Visualisations")
    metrics_options = st.multiselect(
        "Graphiques à afficher",
        ["Matrice de confusion", "Courbe ROC", "Courbe Précision-Rappel"],
        default=["Matrice de confusion", "Courbe ROC"],
    )

    if metrics_options:
        fig, axes = plt.subplots(1, len(metrics_options),
                                 figsize=(6 * len(metrics_options), 5))
        if len(metrics_options) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_options):
            if metric == "Matrice de confusion":
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
                ax.set_title("Matrice de confusion")
            elif metric == "Courbe ROC":
                RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
                ax.set_title("Courbe ROC")
            else:
                PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
                ax.set_title("Courbe Précision-Rappel")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.caption("Projet Data Science — Détection de fraude par carte de crédit")


if __name__ == "__main__":
    main()
