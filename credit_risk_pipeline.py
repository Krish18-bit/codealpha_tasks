"""
Creditworthiness Prediction Pipeline

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn --break-system-packages
    python credit_risk_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. DATA LOADING  (swap this out for your real source)
# ---------------------------------------------------------------------------
def load_data(n_samples: int = 5000) -> pd.DataFrame:
    """Synthetic stand-in dataset. Replace with pd.read_csv('your_file.csv')."""
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(21, 70, n_samples)
    annual_income = rng.normal(55000, 20000, n_samples).clip(12000, None)
    employment_years = rng.integers(0, 35, n_samples)
    existing_loans = rng.integers(0, 6, n_samples)
    total_debt = rng.normal(15000, 12000, n_samples).clip(0, None)
    credit_limit = rng.normal(20000, 9000, n_samples).clip(1000, None)
    monthly_expenses = rng.normal(2000, 800, n_samples).clip(300, None)
    num_late_payments_12m = rng.poisson(0.6, n_samples)
    num_missed_payments_12m = rng.poisson(0.2, n_samples)
    credit_history_years = rng.integers(0, 30, n_samples)
    num_credit_inquiries_6m = rng.poisson(1.2, n_samples)

    # Latent "risk score" driving the true label, so the model has real signal to learn
    risk = (
        -0.00002 * annual_income
        + 0.00004 * total_debt
        + 0.4 * num_late_payments_12m
        + 0.9 * num_missed_payments_12m
        + 0.15 * num_credit_inquiries_6m
        - 0.05 * credit_history_years
        - 0.03 * employment_years
        + rng.normal(0, 1, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-risk))
    target = (rng.random(n_samples) < prob_default).astype(int)
    target = 1 - target  # 1 = creditworthy (majority class), 0 = default risk

    return pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "total_debt": total_debt.round(2),
        "credit_limit": credit_limit.round(2),
        "monthly_expenses": monthly_expenses.round(2),
        "num_late_payments_12m": num_late_payments_12m,
        "num_missed_payments_12m": num_missed_payments_12m,
        "credit_history_years": credit_history_years,
        "num_credit_inquiries_6m": num_credit_inquiries_6m,
        "target": target,
    })


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Debt-to-income ratio — core underwriting signal
    df["debt_to_income"] = df["total_debt"] / df["annual_income"].replace(0, np.nan)

    # Credit utilization — how much of available credit is being used
    df["credit_utilization"] = df["total_debt"] / df["credit_limit"].replace(0, np.nan)

    # Disposable income after debt & expenses
    df["disposable_income"] = (
        df["annual_income"] / 12 - df["monthly_expenses"] - df["total_debt"] / 60
    )

    # Payment reliability score (fewer late/missed = higher)
    df["payment_reliability"] = 1 / (
        1 + df["num_late_payments_12m"] + 2 * df["num_missed_payments_12m"]
    )

    # Credit history depth relative to age (maturity of credit behavior)
    df["credit_history_ratio"] = df["credit_history_years"] / df["age"].replace(0, np.nan)

    # Loan burden per year employed (stability proxy)
    df["loans_per_employment_year"] = df["existing_loans"] / (df["employment_years"] + 1)

    # Recent credit-seeking intensity (many inquiries can signal distress)
    df["inquiry_intensity"] = df["num_credit_inquiries_6m"] / 6

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


# ---------------------------------------------------------------------------
# 3. TRAIN / EVALUATE MULTIPLE MODELS
# ---------------------------------------------------------------------------
def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                 random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                                 class_weight="balanced",
                                                 random_state=RANDOM_STATE),
    }


def evaluate_model(name, model, X_test, y_test, X_test_scaled=None, use_scaled=False):
    X_eval = X_test_scaled if use_scaled else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def main():
    # --- Load & engineer ---
    df = load_data()
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale (Logistic Regression benefits from this; trees don't need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    results = []
    roc_curves = {}

    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        if use_scaled:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(
            name, model, X_test, y_test, X_test_scaled, use_scaled
        )
        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, metrics["ROC-AUC"])

        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(classification_report(y_test, y_pred, target_names=["High Risk", "Creditworthy"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        if name == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            print("\nTop 8 Feature Importances (Random Forest):")
            print(importances.sort_values(ascending=False).head(8))

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print(f"\n{'=' * 60}\nModel Comparison\n{'=' * 60}")
    print(results_df)

    # --- Plot ROC curves ---
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Creditworthiness Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/roc_curves.png", dpi=150)
    print("\nSaved ROC curve plot to roc_curves.png")

    results_df.to_csv("/mnt/user-data/outputs/model_comparison.csv")
    print("Saved metrics table to model_comparison.csv")


if __name__ == "__main__":
    main()
