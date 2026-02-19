import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.q1.utils import CAT_COLS, NUM_COLS, TARGET_COL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.train)

    # text summary (easy to paste into report)
    with open(os.path.join(args.out, "partA_dataset_summary.txt"), "w") as f:
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n\n")
        f.write("Dtypes:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Missing values:\n")
        f.write(str(df.isna().sum()) + "\n\n")
        f.write("Class distribution:\n")
        f.write(str(df[TARGET_COL].value_counts().sort_index()) + "\n")

    # class distribution
    counts = df[TARGET_COL].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("price_class (0=Budget,1=Moderate,2=Premium,3=Luxury)")
    plt.ylabel("Count")
    plt.title("Class distribution (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "A1_class_distribution.png"), dpi=200)
    plt.close()

    # missing values
    miss = df.isna().sum()
    plt.figure(figsize=(7,4))
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Missing count")
    plt.title("Missing values per feature")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "A2_missing_values.png"), dpi=200)
    plt.close()

    # numeric vs target (boxplots)
    for col in NUM_COLS:
        plt.figure(figsize=(6,4))
        data = [df.loc[df[TARGET_COL]==k, col].dropna().values for k in sorted(df[TARGET_COL].unique())]
        plt.boxplot(data, tick_labels=[str(k) for k in sorted(df[TARGET_COL].unique())], showfliers=False)
        plt.xlabel("price_class")
        plt.ylabel(col)
        plt.title(f"{col} vs price_class")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"A3_box_{col}.png"), dpi=200)
        plt.close()

    # categorical vs target (stacked bars, top 10)
    for col in CAT_COLS:
        tmp = df[[col, TARGET_COL]].copy()
        tmp[col] = tmp[col].fillna("Unknown").astype(str)
        ctab = pd.crosstab(tmp[col], tmp[TARGET_COL])
        top = ctab.sum(axis=1).sort_values(ascending=False).head(10).index
        ctab = ctab.loc[top]

        plt.figure(figsize=(7,4))
        bottom = np.zeros(len(ctab))
        for k in ctab.columns:
            plt.bar(ctab.index.astype(str), ctab[k].values, bottom=bottom, label=str(k))
            bottom += ctab[k].values
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count")
        plt.title(f"Top {col} categories by price_class")
        plt.legend(title="price_class", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"A4_stacked_{col}.png"), dpi=200)
        plt.close()

    # correlation matrix (numerical)
    corr = df[NUM_COLS].corr()
    plt.figure(figsize=(5,4))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(NUM_COLS)), NUM_COLS, rotation=30, ha="right")
    plt.yticks(range(len(NUM_COLS)), NUM_COLS)
    plt.colorbar(label="Pearson r")
    for i in range(len(NUM_COLS)):
        for j in range(len(NUM_COLS)):
            plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation matrix (numerical features)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "A5_corr_matrix.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
