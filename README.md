# 💳 Deployment-Oriented Evaluation of Fraud Detection Systems

[![Journal](https://img.shields.io/badge/Intelligent%20Systems%20with%20Applications-Under%20Review-blue?style=flat-square)](https://www.sciencedirect.com/journal/intelligent-systems-with-applications)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Amit2004k/fraud-detection-cost-sensitive-xai?style=flat-square)](https://github.com/Amit2004k/fraud-detection-cost-sensitive-xai/stargazers)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME%20%7C%20Counterfactual-purple?style=flat-square)]()

> **Under Review — Intelligent Systems with Applications (Elsevier)**
> *Deployment-Oriented Evaluation of Fraud Detection Systems: Cost-Sensitive Modeling, Calibration, and Robustness Under Distribution Shift*

---

## 🧠 Problem Statement

Most fraud detection research optimizes for accuracy on clean, balanced datasets. **Real-world deployment is a different beast:**

- 🏦 Class imbalance is extreme (fraud is <0.2% of transactions)
- 💸 False negatives (missed fraud) cost orders of magnitude more than false positives
- 📉 Distribution shift — fraud patterns evolve constantly
- ⚖️ Regulators require explainable, auditable decisions

This paper tackles all four simultaneously with a **deployment-oriented evaluation framework**.

---

## 🔥 Key Contributions

- ✅ **Cost-sensitive learning** — asymmetric loss functions calibrated to real financial costs
- ✅ **Probability calibration** — ensures predicted fraud scores are reliable for risk thresholds
- ✅ **Distribution shift robustness** — evaluated under temporal drift and synthetic covariate shift
- ✅ **Counterfactual explanations** — DICE-generated "what-if" explanations for fraud officers
- ✅ **Deployment metrics** — Average Precision, PR-AUC, cost curves (not just AUC-ROC)
- ✅ **Ablation study** — validates each component's individual contribution

---

## 📊 Results at a Glance

### IEEE-CIS Fraud Detection Dataset (Kaggle)

| Model | PR-AUC | F1 (Fraud) | Cost Savings | Calibration ECE |
|-------|--------|------------|-------------|-----------------|
| Logistic Regression (baseline) | 0.612 | 0.58 | — | 0.087 |
| XGBoost (standard) | 0.891 | 0.79 | — | 0.043 |
| **Ours (Cost-Sensitive + Calibrated)** | **0.921** | **0.83** | **+31% vs baseline** | **0.018** |

> Under simulated distribution shift (6-month temporal drift), our model retains **94.3%** of its clean-data PR-AUC vs **76.1%** for the uncalibrated baseline.

---

## 🏗️ Framework Architecture

```
Raw Transaction Data
        │
        ▼
┌──────────────────────┐
│  Feature Engineering │  ← Time-based, velocity, aggregation features
└──────────┬───────────┘
           │
        ▼
┌──────────────────────┐
│  Imbalance Handling  │  ← SMOTE, cost-sensitive weighting, threshold tuning
└──────────┬───────────┘
           │
        ▼
┌──────────────────────┐
│  Model Training      │  ← LR, RF, XGBoost, LightGBM (cost-sensitive)
│                      │  ← Custom loss: FN cost = 100× FP cost
└──────────┬───────────┘
           │
        ▼
┌──────────────────────┐
│  Calibration         │  ← Isotonic Regression, Platt Scaling
│                      │  ← Reliability diagrams, ECE/Brier per decile
└──────────┬───────────┘
           │
        ▼
┌──────────────────────┐
│  Robustness Testing  │  ← Temporal drift, covariate shift, noise injection
└──────────┬───────────┘
           │
        ▼
┌──────────────────────┐
│  XAI & Explainability│  ← SHAP, LIME, DICE counterfactuals
└──────────────────────┘
```

---

## 📁 Repository Structure

```
📦 fraud-detection-cost-sensitive-xai
├── 📂 src/
│   ├── features.py               # Feature engineering pipeline
│   ├── models.py                 # Cost-sensitive model wrappers
│   ├── calibration.py            # Calibration utilities
│   ├── robustness.py             # Distribution shift simulation
│   └── explainability.py        # SHAP, LIME, DICE wrappers
├── 📂 notebooks/
│   ├── 01_eda_and_features.ipynb
│   ├── 02_cost_sensitive_training.ipynb
│   ├── 03_calibration_analysis.ipynb
│   ├── 04_robustness_evaluation.ipynb
│   ├── 05_xai_explanations.ipynb
│   └── 06_ablation_study.ipynb
├── 📂 data/
│   └── README.md                 # Dataset download instructions
├── 📂 results/
│   ├── 📂 figures/               # PR curves, cost curves, SHAP plots
│   └── 📂 tables/                # Performance tables
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amit2004k/fraud-detection-cost-sensitive-xai.git
cd fraud-detection-cost-sensitive-xai
pip install -r requirements.txt
```

### Download Dataset

```bash
# Option 1: Kaggle CLI (IEEE-CIS Fraud Detection)
kaggle competitions download -c ieee-fraud-detection -p data/

# Option 2: Use the smaller creditcard.csv benchmark
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### Run cost-sensitive training:

```python
from src.models import CostSensitiveFraudDetector

detector = CostSensitiveFraudDetector(
    model_type="xgboost",
    fn_cost=100,     # missing fraud costs 100× more than a false alarm
    fp_cost=1,
)
detector.fit(X_train, y_train)
y_pred = detector.predict(X_test, threshold=0.3)  # lower threshold = catch more fraud
```

---

## 🔍 Explainability

### SHAP — Global Feature Importance
```python
import shap
explainer = shap.TreeExplainer(detector.model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### DICE — Counterfactual Explanations
```python
# "What would need to change for this transaction to be classified as legitimate?"
import dice_ml
# See notebooks/05_xai_explanations.ipynb for full walkthrough
```

---

## 📐 Evaluation Metrics (Why Not Just AUC-ROC?)

| Metric | Why it matters for fraud |
|--------|--------------------------|
| **PR-AUC** | Better than ROC-AUC under extreme imbalance |
| **Average Precision** | Summarizes precision-recall trade-off |
| **Cost Curve** | Directly models financial impact |
| **ECE / Brier Score** | Calibration quality — essential for risk thresholds |
| **Robustness Score** | Performance under distribution shift |

---

## 📖 Citation

```bibtex
@article{kalita2026fraud,
  title   = {Deployment-Oriented Evaluation of Fraud Detection Systems: Cost-Sensitive Modeling, Calibration, and Robustness Under Distribution Shift},
  author  = {Kalita, Amit and others},
  journal = {Intelligent Systems with Applications},
  year    = {2026},
  note    = {Under Review},
  publisher = {Elsevier}
}
```

---

## 🙋 Author

**Amit Kalita**
B.Tech CSE (8th Semester), Dibrugarh University
[GitHub](https://github.com/Amit2004k)

> 📌 *Part of a series of published ML research repos. See also:
> [Fairness Optimization](https://github.com/Amit2004k/fairness-threshold-optimization) |
> [Breast Cancer Classification](https://github.com/Amit2004k/decision-aware-breast-cancer-classification) |
> DDI Prediction | Alzheimer's Detection | and more.*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

⭐ **If this helped your fraud detection research, please star the repo!**
