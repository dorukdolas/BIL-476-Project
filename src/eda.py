import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(cfg.DATA_PATH)
    return df


def summarize(df: pd.DataFrame) -> None:
    n_records, n_cols = df.shape
    n_features = n_cols - 1
    pos_rate = (df[cfg.TARGET] == cfg.POSITIVE_LABEL).mean()

    unknown_counts = {
        c: int((df[c] == "unknown").sum())
        for c in cfg.CATEGORICAL_FEATURES
        if (df[c] == "unknown").any()
    }

    summary = pd.DataFrame(
        {
            "Property": [
                "Number of Records",
                "Number of Attributes",
                "Numeric Attributes",
                "Categorical Attributes",
                "Target Variable",
                "Positive Class Rate (deposit=yes)",
                "Missing Values (NaN)",
                "'unknown' encoded values",
            ],
            "Value": [
                n_records,
                n_features,
                len(cfg.NUMERIC_FEATURES),
                len(cfg.CATEGORICAL_FEATURES),
                f"{cfg.TARGET} (yes/no)",
                f"{pos_rate:.1%}",
                int(df.isnull().sum().sum()),
                "; ".join(f"{k}={v}" for k, v in unknown_counts.items()) or "none",
            ],
        }
    )
    summary.to_csv(cfg.RESULTS_DIR / "dataset_summary.csv", index=False)

    desc = df[cfg.NUMERIC_FEATURES].describe().T
    desc = desc.round(2)
    desc.to_csv(cfg.RESULTS_DIR / "numeric_descriptive_stats.csv")

    print("=== Veri Seti Ozeti ===")
    print(summary.to_string(index=False))
    print("\n=== Sayisal Betimsel Istatistikler ===")
    print(desc.to_string())


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.2))
    order = df[cfg.TARGET].value_counts().index
    sns.countplot(data=df, x=cfg.TARGET, order=order, ax=ax,
                  hue=cfg.TARGET, palette="Set2", legend=False)
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("deposit")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.savefig(cfg.FIGURES_DIR / "class_distribution.png")
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    n = len(cfg.NUMERIC_FEATURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cfg.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=cfg.TARGET, kde=False,
                     bins=30, ax=axes[i], palette="Set2",
                     element="step", stat="density", common_norm=False)
        axes[i].set_title(col)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions by Target", y=1.01)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "numeric_distributions.png")
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[cfg.NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (Numeric Features)")
    fig.savefig(cfg.FIGURES_DIR / "correlation_matrix.png")
    plt.close(fig)


def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    cols = ["job", "education", "housing", "loan", "contact", "poutcome"]
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        rate = (df.assign(pos=(df[cfg.TARGET] == cfg.POSITIVE_LABEL))
                  .groupby(col)["pos"].mean().sort_values())
        sns.barplot(x=rate.values, y=rate.index, ax=axes[i],
                    hue=rate.index, palette="viridis", legend=False)
        axes[i].axvline(0.474, color="red", ls="--", lw=1,
                        label="overall rate")
        axes[i].set_title(f"deposit=yes rate by {col}")
        axes[i].set_xlabel("P(deposit=yes)")
        axes[i].set_ylabel("")
    fig.suptitle("Subscription Rate by Categorical Feature", y=1.01)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "categorical_vs_target.png")
    plt.close(fig)


def plot_duration_analysis(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=df, x=cfg.TARGET, y="duration", ax=ax,
                hue=cfg.TARGET, palette="Set2", legend=False)
    ax.set_title("Call Duration vs Subscription (potential leakage)")
    ax.set_xlabel("deposit")
    ax.set_ylabel("duration (seconds)")
    fig.savefig(cfg.FIGURES_DIR / "duration_leakage.png")
    plt.close(fig)


def main() -> None:
    df = load_data()
    summarize(df)
    plot_class_distribution(df)
    plot_numeric_distributions(df)
    plot_correlation(df)
    plot_categorical_vs_target(df)
    plot_duration_analysis(df)
    print(f"\nEDA tamamlandi. Figurler: {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
