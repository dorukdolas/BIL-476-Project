from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

import config as cfg

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

rng = cfg.RANDOM_STATE


def load_xy():
    df = pd.read_csv(cfg.DATA_PATH)
    y = (df[cfg.TARGET] == cfg.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[cfg.TARGET])
    return X, y


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def model_grid():
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rng),
            {"clf__max_depth": [5, 10, 20, None],
             "clf__min_samples_leaf": [1, 20]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "k-NN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [15, 25, 51],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=rng, n_jobs=-1),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [10, 20, None]},
        ),
    }


def run_scenario(name, numeric_features, categorical_features):
    X, y = load_xy()
    X = X[numeric_features + categorical_features]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=rng, stratify=y
    )

    pre = build_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=rng)

    rows = []
    cv_rows = []
    roc_data = {}
    fitted = {}

    for model_name, (estimator, grid) in model_grid().items():
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc",
                              cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        fitted[model_name] = best
        
        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]

        rows.append({
            "Scenario": name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred),
            "Recall": recall_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred),
            "ROC_AUC": roc_auc_score(y_te, y_proba),
        })
        cv_rows.append({
            "Scenario": name,
            "Model": model_name,
            "CV_ROC_AUC_mean": search.best_score_,
            "Best_Params": str(search.best_params_),
        })
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        roc_data[model_name] = (fpr, tpr, roc_auc_score(y_te, y_proba))

    results = pd.DataFrame(rows)
    cv_results = pd.DataFrame(cv_rows)
    return results, cv_results, roc_data, fitted, (X_te, y_te)


def plot_roc(roc_data, scenario, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 4))
    for model_name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=1.6, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {scenario}")
    ax.legend(loc="lower right", fontsize=8)
    if own:
        fig.savefig(cfg.FIGURES_DIR / f"roc_{scenario}.png")
        plt.close(fig)


def plot_confusion_matrices(fitted, test_data, scenario):
    X_te, y_te = test_data
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (model_name, model) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_te, model.predict(X_te))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=ax, xticklabels=["no", "yes"],
                    yticklabels=["no", "yes"])
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices - {scenario}", y=1.05)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / f"confusion_{scenario}.png")
    plt.close(fig)


def plot_feature_importance(fitted, scenario, numeric_features,
                            categorical_features, top_n=15):
    rf = fitted["Random Forest"]
    pre = rf.named_steps["pre"]
    ohe = pre.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    feat_names = np.concatenate([numeric_features, cat_names])
    importances = rf.named_steps["clf"].feature_importances_

    imp = (pd.Series(importances, index=feat_names)
           .sort_values(ascending=False).head(top_n))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax,
                hue=imp.index, palette="rocket", legend=False)
    ax.set_title(f"Random Forest Feature Importance - {scenario}")
    ax.set_xlabel("Importance")
    fig.savefig(cfg.FIGURES_DIR / f"feature_importance_{scenario}.png")
    plt.close(fig)


def plot_scenario_comparison(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["F1", "ROC_AUC"]):
        sns.barplot(data=all_results, x="Model", y=metric,
                    hue="Scenario", ax=ax, palette="Set1")
        ax.set_title(f"{metric} by Model and Scenario")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "scenario_comparison.png")
    plt.close(fig)


def to_latex_table(df, scenario, path):
    sub = df[df["Scenario"] == scenario].drop(columns=["Scenario"]).copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        sub[c] = sub[c].map(lambda v: f"{v:.3f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(sub.to_latex(index=False, escape=True))


def main():
    all_cat = cfg.CATEGORICAL_FEATURES
    full_num = cfg.NUMERIC_FEATURES
    no_dur_num = [c for c in cfg.NUMERIC_FEATURES if c != cfg.LEAKY_FEATURE]

    scenarios = {
        "full": (full_num, all_cat),
        "no_duration": (no_dur_num, all_cat),
    }

    all_results = []
    all_cv = []

    for scen, (num_f, cat_f) in scenarios.items():
        print(f"\n===== Senaryo: {scen} =====")
        results, cv_results, roc_data, fitted, test_data = run_scenario(
            scen, num_f, cat_f
        )
        print(results.to_string(index=False))

        plot_roc(roc_data, scen)
        plot_confusion_matrices(fitted, test_data, scen)
        plot_feature_importance(fitted, scen, num_f, cat_f)
        to_latex_table(results, scen,
                       cfg.RESULTS_DIR / f"results_{scen}.tex")

        all_results.append(results)
        all_cv.append(cv_results)

    combined = pd.concat(all_results, ignore_index=True)
    combined_cv = pd.concat(all_cv, ignore_index=True)
    combined.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)
    combined_cv.to_csv(cfg.RESULTS_DIR / "cv_results.csv", index=False)

    plot_scenario_comparison(combined)

    print("\n===== Tum sonuclar =====")
    print(combined.to_string(index=False))
    print(f"\nSonuclar: {cfg.RESULTS_DIR}\nFigurler: {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
