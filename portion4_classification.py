"""
PORTION 4: CLASSIFICATION MODELING (PREDICT ORDER PROFITABILITY)
Retail Sales & Profitability Intelligence

Tasks:
- Formulate binary classification problem (1 = Profitable, 0 = Loss)
- Stratified train/test split (80% / 20%)
- Train and evaluate:
    1. Logistic Regression (Baseline)
    2. Decision Tree Classifier
    3. Random Forest Classifier
- Calculate Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Plot confusion matrix heatmaps
- Extract and plot feature importance drivers
- Serialize classification models to outputs/models/
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    make_preprocessor,
    save_plot,
)


def compute_metrics(y_true, y_pred):
    """Calculate Accuracy, Precision, Recall, F1, and Confusion Matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
    }


def run_classification(df=None, reg_models=None, reg_feat_names=None):
    """
    Train and evaluate classification pipelines predicting 'is_profitable'.
    
    Args:
        df (pd.DataFrame, optional): Preprocessed DataFrame.
        reg_models (dict, optional): Fitted regression models for feature importance comparison.
        reg_feat_names (tuple, optional): (numeric_features, categorical_features) from regression.
        
    Returns:
        tuple: (results_dict, best_model_name, fitted_models, feature_names)
    """
    print("\n" + "=" * 65)
    print("PORTION 4: CLASSIFICATION MODELING (PREDICT PROFITABILITY)")
    print("=" * 65)

    if df is None:
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}. Run Portion 1 first.")
        df = pd.read_csv(CLEANED_DATA_PATH)

    numeric_features = ["Sales", "Order Quantity", "Unit Price", "Shipping Cost", "Order Month", "Order Year"]
    categorical_features = ["Order Priority", "Ship Mode", "Customer Segment", "Product Category"]

    X = df[numeric_features + categorical_features]
    y = df["is_profitable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ]),
        "Decision Tree Classifier": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
        "Random Forest Classifier": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("classifier", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=RANDOM_STATE)),
        ]),
    }

    results = {}
    fitted_models = {}
    predictions = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_m = compute_metrics(y_train, y_train_pred)
        test_m = compute_metrics(y_test, y_test_pred)

        report = classification_report(
            y_test, y_test_pred, target_names=["Unprofitable (0)", "Profitable (1)"], output_dict=True
        )

        results[name] = {"train": train_m, "test": test_m, "classification_report": report}
        fitted_models[name] = pipeline
        predictions[name] = (y_test, y_test_pred)

        print(f"\n--- {name} ---")
        print(f"  Train: Accuracy = {train_m['accuracy']:.4f} | F1 = {train_m['f1']:.4f}")
        print(f"  Test:  Accuracy = {test_m['accuracy']:.4f} | F1 = {test_m['f1']:.4f} | Precision = {test_m['precision']:.4f} | Recall = {test_m['recall']:.4f}")
        print(f"  Confusion Matrix: {test_m['confusion_matrix']}")

    # Determine best classifier based on Test F1 Score
    best_model_name = max(results.keys(), key=lambda k: results[k]["test"]["f1"])
    best_pipeline = fitted_models[best_model_name]
    print(f"\n>>> Best Classification Model: {best_model_name} (Test F1: {results[best_model_name]['test']['f1']:.4f})")

    # Serialize models to outputs/models/
    joblib.dump(best_pipeline, MODELS_DIR / "best_profitability_classifier.joblib")
    joblib.dump(fitted_models["Logistic Regression"], MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(fitted_models["Decision Tree Classifier"], MODELS_DIR / "decision_tree_classifier.joblib")
    joblib.dump(fitted_models["Random Forest Classifier"], MODELS_DIR / "random_forest_classifier.joblib")
    print(f"[SAVED] Classification models serialized to {MODELS_DIR}")

    # Plot Confusion Matrices for all 3 models side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = ["Loss (0)", "Profitable (1)"]
    palettes = ["Blues", "Greens", "Purples"]

    for idx, (m_name, (act, pred)) in enumerate(predictions.items()):
        cm = confusion_matrix(act, pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=palettes[idx],
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{m_name}\nAcc: {results[m_name]['test']['accuracy']:.1%} | F1: {results[m_name]['test']['f1']:.4f}",
            fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("Actual Label")

    save_plot("09_classification_confusion_matrices.png")

    # Plot Feature Importance
    _plot_feature_importance(fitted_models, (numeric_features, categorical_features), reg_models, reg_feat_names)

    return results, best_model_name, fitted_models, (numeric_features, categorical_features)


def _plot_feature_importance(clf_models, clf_feat_names, reg_models=None, reg_feat_names=None):
    """Internal helper to plot feature importances from ensemble or tree models."""
    try:
        model_key = "Random Forest Classifier" if "Random Forest Classifier" in clf_models else "Decision Tree Classifier"
        estimator = clf_models[model_key].named_steps["classifier"]
        preprocessor = clf_models[model_key].named_steps["preprocessor"]
        ohe_cat_clf = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(clf_feat_names[1])
        all_clf_features = clf_feat_names[0] + list(ohe_cat_clf)
        importances_clf = pd.Series(estimator.feature_importances_, index=all_clf_features).sort_values(ascending=False).head(10)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Check if regression model has feature importances
        if reg_models:
            reg_key = "Random Forest Regressor" if "Random Forest Regressor" in reg_models else "Decision Tree Regressor"
            if reg_key in reg_models and reg_feat_names:
                reg_est = reg_models[reg_key].named_steps["regressor"]
                prep_reg = reg_models[reg_key].named_steps["preprocessor"]
                ohe_cat_reg = prep_reg.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(reg_feat_names[1])
                all_reg_features = reg_feat_names[0] + list(ohe_cat_reg)
                importances_reg = pd.Series(reg_est.feature_importances_, index=all_reg_features).sort_values(ascending=False).head(10)

                sns.barplot(x=importances_reg.values, y=importances_reg.index, ax=axes[0], palette="Blues_r")
                axes[0].set_title(f"Top 10 Drivers of Sales ({reg_key})", fontweight="bold")
                axes[0].set_xlabel("Relative Importance Score")
        else:
            axes[0].text(0.5, 0.5, "Regression feature importances available during pipeline run", ha="center", va="center")
            axes[0].set_axis_off()

        sns.barplot(x=importances_clf.values, y=importances_clf.index, ax=axes[1], palette="Greens_r")
        axes[1].set_title(f"Top 10 Drivers of Profitability ({model_key})", fontweight="bold")
        axes[1].set_xlabel("Relative Importance Score")

        save_plot("10_feature_importance.png")
    except Exception as e:
        print(f"  [NOTE] Feature importance visualization skipped: {e}")


if __name__ == "__main__":
    run_classification()
