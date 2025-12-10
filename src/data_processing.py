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

# 4. Hàm in thống kê mô tả (mean, median) cho các biến câu 4.
def describe_purposes(df, purposes):
    print("===== DESCRIPTIVE STATISTICS FOR USAGE PURPOSE =====\n")

    for col in purposes:
        mean_val = df[col].mean()
        median_val = df[col].median()

        print(f"{col}:")
        print(f"  Mean:   {mean_val:.2f} hours")
        print(f"  Median: {median_val:.2f} hours\n")

# 5. Hàm in thống kê mô tả cho các biến câu 5.        
def describe_screen_time_groups(df):
    print("\n===== DESCRIPTIVE STATISTICS BY SCREEN-TIME GROUPS =====\n")

    # Tính quantile
    q33 = df['Screen_Time_Before_Bed'].quantile(0.33)
    q67 = df['Screen_Time_Before_Bed'].quantile(0.67)

    print(f"Quantile boundaries: q33={q33:.2f}, q67={q67:.2f}\n")

    # Chia nhóm 
    def group_screen(x):
        if x <= q33:
            return 'Low'
        elif x <= q67:
            return 'Medium'
        return 'High'

    df_temp = df.copy()
    df_temp["Screen_Group"] = df_temp["Screen_Time_Before_Bed"].apply(group_screen)

    # Các biến quan tâm
    target_vars = [
        "Sleep_Hours",
        "Academic_Performance",
        "Anxiety_Level",
        "Depression_Level"
    ]

    # Lặp qua từng nhóm
    for group in ["Low", "Medium", "High"]:
        g = df_temp[df_temp["Screen_Group"] == group]
        print(f"\n--- {group} Screen-Time Group ---")
        print(f"Sample size: {len(g)}")

        for var in target_vars:
            print(f"{var}: mean={g[var].mean():.2f}, median={g[var].median():.2f}")

# 6. Chia mức độ nghiện thành 3 nhóm dựa trên phân vị          
def split_addiction_by_quantiles(df, low_q=0.33, high_q=0.67):
    print("===== SPLIT GROUPS =====\n")
    # Tính giá trị phân vị thấp và cao
    q_low = df["Addiction_Level"].quantile(low_q)
    q_high = df["Addiction_Level"].quantile(high_q)

     # Chia nhóm theo ngưỡng
    low_group = df[df["Addiction_Level"] <= q_low]
    medium_group = df[(df["Addiction_Level"] > q_low) & (df["Addiction_Level"] <= q_high)]
    high_group = df[df["Addiction_Level"] > q_high]

    groups = {
        "Low": low_group,
        "Medium": medium_group,
        "High": high_group
    }

    return groups, (q_low, q_high)

# 7. Tính thống kê mô tả cho từng biến trong mỗi nhóm nghiện
def summarize_addiction_groups(groups, variables):
    print("\n\n===== DESCRIPTIVE STATISTICS =====\n")
    rows = []

    for var in variables:
        row = {"Variable": var}
        for name, g in groups.items():
            series = g[var].dropna()
            row[f"{name}_n"] = len(series)
            row[f"{name}_mean"] = series.mean()
            row[f"{name}_median"] = series.median()
            row[f"{name}_std"] = series.std()
        rows.append(row)

    return pd.DataFrame(rows).set_index("Variable")
