import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

project_root = os.path.abspath("..")
sys.path.append(project_root)

from src.visualization import *

# 1.Hàm phát hiện outliers bằng IQR Method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    return outliers, lower, upper

# 2. Hàm phát hiện outliers bằng Z-Score
def detect_outliers_zscore(series, threshold=3):
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    outliers = series[abs(z_scores) > threshold]
    return outliers
    
# 3. Hàm tổng hợp kiểm tra
def check_outliers_all(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        print(f"\n===== OUTLIER ANALYSIS: {col} =====")

        # IQR method
        outliers_iqr, lower, upper = detect_outliers_iqr(df[col])
        print(f"IQR Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
        print(f"IQR Outliers count: {len(outliers_iqr)}")

        # Z-score method
        outliers_z = detect_outliers_zscore(df[col])
        print(f"Z-score Outliers count: {len(outliers_z)}")

        # Boxplot (visual)
        plt.figure(figsize=(6,2))
        plt.boxplot(df[col], vert=False)
        plt.title(f"Boxplot: {col}")
        plt.grid(alpha=0.3)
        plt.show()