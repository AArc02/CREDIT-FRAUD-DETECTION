# 💳 Détection de Défaut de Paiement

> Application interactive de détection de fraudes par carte de crédit basée sur le Machine Learning.

[![CI](https://github.com/VOTRE_USERNAME/credit-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/VOTRE_USERNAME/credit-fraud-detection/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Présentation

Cette application Streamlit permet d'explorer, entraîner et évaluer des modèles de Machine Learning pour la **détection de transactions frauduleuses** sur un dataset de transactions bancaires.

### Fonctionnalités
- 📊 Visualisation interactive des données (distribution, statistiques)
- 🤖 3 classificateurs : Random Forest, SVM, Régression Logistique
- ⚙️ Hyperparamètres configurables via la sidebar
- 📈 Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🔲 Matrice de confusion, Courbe ROC, Courbe Précision-Rappel

---

## 📁 Structure du projet

```
credit-fraud-detection/
│
├── app.py                    # Application Streamlit principale
├── requirements.txt          # Dépendances Python
├── .gitignore
├── README.md
│
├── data/
│   └── README.md             # Instructions pour obtenir le dataset
│
├── notebooks/
│   └── 01_exploration.ipynb  # Analyse exploratoire (EDA)
│
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py      # Tests unitaires (pytest)
│
└── .github/
    └── workflows/
        └── ci.yml            # Pipeline CI/CD (GitHub Actions)
```

---

## ⚡ Installation & Lancement rapide

### Prérequis
- Python 3.11+
- Git

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/credit-fraud-detection.git
cd credit-fraud-detection
```

### 2. Créer et activer l'environnement virtuel

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer le dataset

Téléchargez `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le dans le dossier `data/` :

```
data/
└── creditcard.csv   ← ici
```

> ⚠️ Le fichier CSV n'est **pas versionné** sur GitHub (taille + confidentialité).

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application est accessible sur : **http://localhost:8501**

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📊 Dataset

| Propriété         | Valeur                        |
|-------------------|-------------------------------|
| Source            | Kaggle — ULB Machine Learning |
| Transactions      | ~285 000 (extrait : 1 981)    |
| Features          | V1–V28 (PCA), Amount, Time    |
| Variable cible    | Class (0 = normal, 1 = fraude)|
| Taux de fraude    | ~0.40%                        |
| Valeurs nulles    | Aucune                        |

---

## 🤖 Modèles disponibles

| Modèle               | Hyperparamètres configurables        |
|----------------------|--------------------------------------|
| Random Forest        | n_estimators, max_depth              |
| SVM                  | C (régularisation), kernel           |
| Logistic Regression  | C (régularisation)                   |

---

## 📈 Métriques d'évaluation

| Métrique   | Description                                          |
|------------|------------------------------------------------------|
| Accuracy   | % de prédictions correctes (trompeuse si déséquilibre)|
| Precision  | Parmi les alertes, combien sont de vraies fraudes ?  |
| Recall     | Parmi les fraudes, combien sont détectées ?          |
| F1-Score   | Moyenne harmonique Precision / Recall                |
| ROC-AUC    | Capacité discriminante à tous les seuils             |

> 💡 Sur ce dataset fortement déséquilibré, **Recall** et **ROC-AUC** sont les métriques les plus pertinentes.

---

## 🚀 Déploiement sur Streamlit Cloud

1. Poussez votre code sur GitHub
2. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Cliquez **"New app"** → sélectionnez votre dépôt
4. Définissez `app.py` comme fichier principal
5. Uploadez votre CSV via **"Advanced settings → Secrets"** ou utilisez un dataset public

---

## 🛠️ Technologies

- [Streamlit](https://streamlit.io/) — Interface web
- [scikit-learn](https://scikit-learn.org/) — Modèles ML
- [pandas](https://pandas.pydata.org/) — Manipulation de données
- [matplotlib](https://matplotlib.org/) — Visualisations
- [pytest](https://pytest.org/) — Tests unitaires
- [GitHub Actions](https://github.com/features/actions) — CI/CD

---

## 📄 Licence

Distribué sous licence **MIT**. Voir [LICENSE](LICENSE) pour plus d'informations.

---

## 👤 Auteur : Arsene ALLAHNDIGUIM

**postgres** — Projet Data Science  
* Machine Learning — 2026*
