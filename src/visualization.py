import matplotlib.pyplot as plt
import numpy as np

def plot_hist(col, df, bins=20):
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=bins)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

def plot_box(col, df):
    plt.figure(figsize=(6,2))
    plt.boxplot(df[col], vert=False)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.grid(alpha=0.3)
    plt.show()

def plot_density(col, df):
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30, density=True, alpha=0.6)
    plt.title(f"Density (Histogram Approx) of {col}")
    plt.xlabel(col)
    plt.grid(alpha=0.3)
    plt.show()

def plot_all(col, df, bins=20):
    plt.figure(figsize=(16,4))

    plt.subplot(1,3,1)
    plt.hist(df[col], bins=bins)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.subplot(1,3,2)
    plt.boxplot(df[col], vert=False)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.subplot(1,3,3)
    plt.hist(df[col], bins=30, density=True, alpha=0.6)
    plt.title(f"Density Approx of {col}")
    plt.xlabel(col)


    plt.suptitle(f"Distribution Analysis: {col}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_all_numeric(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != "ID"]  # loại ID nếu chưa drop

    for col in numeric_cols:
        plot_all(col, df)

def plot_categorical(df, col):
    counts = df[col].value_counts()

    plt.figure(figsize=(7,4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Count Plot of {col}")
    plt.ylabel("Frequency")
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_heatmap(corr):
    plt.figure(figsize=(14,10))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.columns)

    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()
