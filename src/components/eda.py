import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.logger import logger
from src.exception import CustomException

EDA_OUTPUT_DIR = os.path.join("artifacts", "eda_plots")

PALETTE = {"<=50K": "#4C9BE8", ">50K": "#E8724C"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def run_full_eda(df: pd.DataFrame, output_dir: str = EDA_OUTPUT_DIR) -> dict:
    """Run complete EDA and save all plots. Returns dict of plot paths."""
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    try:
        # Clean for display — handle both str and already-encoded int
        df = df.copy()
        if "income" in df.columns:
            if df["income"].dtype == object:
                df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
            else:
                df["income"] = df["income"].map({0: "<=50K", 1: ">50K"}).fillna(df["income"].astype(str))

        target = "income" if "income" in df.columns else None
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)

        # ── 1. Target Distribution ─────────────────────────────────────────────
        if target:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            counts = df[target].value_counts()
            axes[0].pie(
                counts, labels=counts.index,
                colors=list(PALETTE.values()), autopct="%1.1f%%",
                startangle=90, explode=[0.05, 0.05]
            )
            axes[0].set_title("Income Class Distribution", fontweight="bold")
            sns.countplot(x=target, data=df, palette=PALETTE, ax=axes[1])
            axes[1].set_title("Count by Income Class", fontweight="bold")
            axes[1].bar_label(axes[1].containers[0])
            plt.tight_layout()
            path = os.path.join(output_dir, "01_target_distribution.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["target_distribution"] = path

        # ── 2. Numerical Distributions ────────────────────────────────────────
        if num_cols:
            n = len(num_cols)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                if target:
                    for label, grp in df.groupby(target):
                        axes[i].hist(grp[col].dropna(), bins=30, alpha=0.6,
                                     label=label, color=PALETTE.get(label, "steelblue"))
                    axes[i].legend(fontsize=8)
                else:
                    axes[i].hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
                axes[i].set_title(col, fontweight="bold")
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.suptitle("Numerical Feature Distributions by Income", fontsize=14, fontweight="bold", y=1.01)
            plt.tight_layout()
            path = os.path.join(output_dir, "02_numerical_distributions.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["numerical_distributions"] = path

        # ── 3. Correlation Heatmap ────────────────────────────────────────────
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 9))
            corr = df[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, ax=ax,
                        linewidths=0.5, square=True)
            ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(output_dir, "03_correlation_heatmap.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["correlation_heatmap"] = path

        # ── 4. Categorical Features vs Target ─────────────────────────────────
        important_cats = ["workclass", "education", "marital_status",
                          "occupation", "relationship", "race", "sex"]
        available = [c for c in important_cats if c in df.columns and target]
        if available:
            fig, axes = plt.subplots(len(available), 1, figsize=(14, len(available) * 4))
            if len(available) == 1:
                axes = [axes]
            for ax, col in zip(axes, available):
                order = (df.groupby(col)[target]
                         .apply(lambda x: (x == ">50K").mean())
                         .sort_values(ascending=False).index)
                ct = pd.crosstab(df[col], df[target], normalize="index") * 100
                ct.loc[order].plot(kind="bar", ax=ax, color=list(PALETTE.values()),
                                   edgecolor="white", width=0.7)
                ax.set_title(f"{col.replace('_', ' ').title()} vs Income (%)", fontweight="bold")
                ax.set_ylabel("Percentage (%)")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)
                ax.legend(title="Income", loc="upper right")
            plt.suptitle("Categorical Features vs Income", fontsize=14, fontweight="bold", y=1.01)
            plt.tight_layout()
            path = os.path.join(output_dir, "04_categorical_vs_target.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["categorical_vs_target"] = path

        # ── 5. Age Distribution by Income ─────────────────────────────────────
        if "age" in df.columns and target:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for label, grp in df.groupby(target):
                axes[0].hist(grp["age"].dropna(), bins=30, alpha=0.7,
                             label=label, color=PALETTE.get(label, "steelblue"), edgecolor="white")
            axes[0].set_title("Age Distribution by Income", fontweight="bold")
            axes[0].legend(); axes[0].set_xlabel("Age")

            sns.boxplot(x=target, y="age", data=df, palette=PALETTE, ax=axes[1])
            axes[1].set_title("Age Boxplot by Income", fontweight="bold")
            axes[1].set_xlabel("Income Class")
            plt.tight_layout()
            path = os.path.join(output_dir, "05_age_analysis.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["age_analysis"] = path

        # ── 6. Hours per Week Analysis ─────────────────────────────────────────
        if "hours_per_week" in df.columns and target:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for label, grp in df.groupby(target):
                axes[0].hist(grp["hours_per_week"].dropna(), bins=30, alpha=0.7,
                             label=label, color=PALETTE.get(label, "steelblue"), edgecolor="white")
            axes[0].axvline(40, color="red", linestyle="--", label="40 hrs/wk")
            axes[0].set_title("Hours/Week Distribution", fontweight="bold")
            axes[0].legend(); axes[0].set_xlabel("Hours per Week")

            sns.boxplot(x=target, y="hours_per_week", data=df, palette=PALETTE, ax=axes[1])
            axes[1].axhline(40, color="red", linestyle="--")
            axes[1].set_title("Hours/Week Boxplot", fontweight="bold")
            plt.tight_layout()
            path = os.path.join(output_dir, "06_hours_analysis.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["hours_analysis"] = path

        # ── 7. Capital Gain/Loss Analysis ──────────────────────────────────────
        if "capital_gain" in df.columns and target:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for col, ax in zip(["capital_gain", "capital_loss"], axes):
                nonzero = df[df[col] > 0]
                if len(nonzero):
                    sns.boxplot(x=target, y=col, data=nonzero, palette=PALETTE, ax=ax)
                    ax.set_title(f"{col.replace('_', ' ').title()} (>0 only)", fontweight="bold")
                else:
                    ax.set_title(f"{col} (all zero)")
            plt.suptitle("Capital Gain & Loss by Income", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(output_dir, "07_capital_analysis.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["capital_analysis"] = path

        # ── 8. Education Level Analysis ────────────────────────────────────────
        if "education_num" in df.columns and target:
            fig, ax = plt.subplots(figsize=(14, 6))
            edu_income = (df.groupby("education_num")[target]
                          .apply(lambda x: (x == ">50K").mean() * 100)
                          .reset_index())
            edu_income.columns = ["education_num", "pct_over_50k"]
            bars = ax.bar(edu_income["education_num"], edu_income["pct_over_50k"],
                          color=plt.cm.RdYlGn(edu_income["pct_over_50k"] / 100),
                          edgecolor="white")
            ax.set_xlabel("Education Level (num)"); ax.set_ylabel("% Earning >50K")
            ax.set_title("Percentage Earning >50K by Education Level", fontweight="bold")
            ax.axhline(df[target].eq(">50K").mean() * 100, color="red",
                       linestyle="--", label="Overall avg")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, "08_education_analysis.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["education_analysis"] = path

        # ── 9. Missing Values Heatmap ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct):
            bars = ax.barh(missing_pct.index, missing_pct.values, color="tomato", edgecolor="white")
            ax.bar_label(bars, fmt="%.1f%%")
            ax.set_title("Missing Values (%)", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "✓ No missing values found!", ha="center", va="center",
                    fontsize=18, color="green", transform=ax.transAxes)
            ax.set_title("Missing Values Check", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "09_missing_values.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        plots["missing_values"] = path

        # ── 10. Occupation Income Rate ─────────────────────────────────────────
        if "occupation" in df.columns and target:
            fig, ax = plt.subplots(figsize=(14, 6))
            occ_income = (df.groupby("occupation")[target]
                          .apply(lambda x: (x == ">50K").mean() * 100)
                          .sort_values(ascending=True))
            colors = plt.cm.RdYlGn(occ_income.values / 100)
            bars = ax.barh(occ_income.index, occ_income.values, color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%.1f%%", padding=3)
            ax.axvline(df[target].eq(">50K").mean() * 100, color="red",
                       linestyle="--", label="Overall avg")
            ax.set_title("% Earning >50K by Occupation", fontweight="bold")
            ax.legend(); ax.set_xlabel("% Earning >50K")
            plt.tight_layout()
            path = os.path.join(output_dir, "10_occupation_income.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            plots["occupation_income"] = path

        logger.info(f"EDA complete. {len(plots)} plots saved to {output_dir}")
        return plots

    except Exception as e:
        raise CustomException(e, sys)