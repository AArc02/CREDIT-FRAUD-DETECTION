"""
Tests unitaires — Détection de fraude
Lancer avec : pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
SEED = 123


@pytest.fixture
def sample_df():
    """Génère un DataFrame de test synthétique."""
    np.random.seed(SEED)
    n = 200
    df = pd.DataFrame(np.random.randn(n, 30), columns=COLUMNS[:-1])
    df["Class"] = 0
    df.loc[:4, "Class"] = 1  # 5 fraudes sur 200
    return df


def test_dataframe_shape(sample_df):
    """Le DataFrame doit avoir 31 colonnes."""
    assert sample_df.shape[1] == 31


def test_class_column_exists(sample_df):
    """La colonne Class doit exister."""
    assert "Class" in sample_df.columns


def test_class_binary(sample_df):
    """Class ne doit contenir que 0 et 1."""
    assert set(sample_df["Class"].unique()).issubset({0, 1})


def test_no_nulls(sample_df):
    """Pas de valeurs manquantes."""
    assert sample_df.isnull().sum().sum() == 0


def test_split_proportions(sample_df):
    """Le split 80/20 doit être respecté."""
    X = sample_df.drop(columns=["Class"])
    y = sample_df["Class"]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=SEED)
    assert abs(len(X_test) / len(sample_df) - 0.2) < 0.05


def test_model_trains(sample_df):
    """Le modèle doit s'entraîner sans erreur."""
    X = sample_df.drop(columns=["Class"])
    y = sample_df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_amount_positive(sample_df):
    """Les montants négatifs ne devraient pas exister."""
    assert (sample_df["Amount"] >= 0).all()
