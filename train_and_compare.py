"""
Customer Churn Prediction - Ensemble Learning ile Model Karsilastirma

Logistic Regression, Decision Tree, Random Forest ve Gradient Boosting
modellerini egitir, karsilastirir ve metrikleri/grafikleri outputs/ klasorune kaydeder.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

from preprocessing import prepare_pipeline

OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def get_models():
    """Karsilastirilacak modelleri dondurur."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_model(model, X_test, y_test):
    """Bir model icin performans metriklerini hesaplar."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    return metrics, y_pred, y_proba


def plot_confusion_matrices(models_preds, y_test, output_dir):
    """Tum modeller icin confusion matrix grafiklerini tek figurde cizer."""
    n_models = len(models_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()


def plot_metrics_comparison(results_df, output_dir):
    """Modellerin metriklerini bar chart olarak karsilastirir."""
    metrics_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in results_df.columns]
    plot_df = results_df[metrics_cols]

    ax = plot_df.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set_title("Model Performans Karsilastirmasi")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close()


def plot_roc_curves(models_proba, y_test, output_dir):
    """Tum modeller icin ROC egrilerini tek grafikte cizer."""
    plt.figure(figsize=(7, 6))

    for name, y_proba in models_proba.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Egrileri - Model Karsilastirmasi")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Agac tabanli modeller icin feature importance grafigi cizer."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], color="steelblue")
    plt.title(f"{model_name} - Ozellik Onem Siralamasi (Top 15)")
    plt.xlabel("Importance")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"feature_importance_{safe_name}.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Veri yukleniyor ve on isleniyor...")
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names = prepare_pipeline()

    models = get_models()
    results = {}
    models_preds = {}
    models_proba = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"Egitiliyor: {name}")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        results[name] = metrics
        models_preds[name] = y_pred
        models_proba[name] = y_proba
        fitted_models[name] = model

        print(f"  {name} sonuclari:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))
    print("\nTum modellerin metrik ozeti:")
    print(results_df.round(4))

    print("\nGrafikler olusturuluyor...")
    plot_confusion_matrices(models_preds, y_test, OUTPUT_DIR)
    plot_metrics_comparison(results_df, OUTPUT_DIR)
    plot_roc_curves(models_proba, y_test, OUTPUT_DIR)

    for name, model in fitted_models.items():
        plot_feature_importance(model, feature_names, name, OUTPUT_DIR)

    best_model_name = results_df["F1-Score"].idxmax()
    print(f"\nEn iyi model (F1-Score'a gore): {best_model_name}")
    print(f"Sonuclar '{OUTPUT_DIR}/' klasorune kaydedildi.")


if __name__ == "__main__":
    main()
