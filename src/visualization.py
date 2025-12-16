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

def plot_comm_vs_addiction_scatter(df):
    plt.figure(figsize=(8,5))
    
    # Vẽ scatter plot
    plt.scatter(df["Family_Communication"], df["Addiction_Level"], alpha=0.5, s=20)
    
    # Thêm đường hồi quy tuyến tính
    z = np.polyfit(df["Family_Communication"], df["Addiction_Level"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["Family_Communication"].min(), df["Family_Communication"].max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.2f}")
    
    plt.xlabel("Family Communication")
    plt.ylabel("Addiction Level")
    plt.title("Family Communication vs Addiction Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_comm_vs_addiction_boxplot(df):
    # Chia thành 3 nhóm theo phân vị
    q33 = df["Family_Communication"].quantile(0.33)
    q67 = df["Family_Communication"].quantile(0.67)
    
    # Tạo label cho từng nhóm
    labels = []
    for val in df["Family_Communication"]:
        if val <= q33:
            labels.append("Low")
        elif val <= q67:
            labels.append("Medium")
        else:
            labels.append("High")
    
    # Tạo dataframe tạm để vẽ
    temp_df = df[["Addiction_Level"]].copy()
    temp_df["Communication_Group"] = labels
    
    # Chuẩn bị dữ liệu cho boxplot
    low_data = temp_df[temp_df["Communication_Group"] == "Low"]["Addiction_Level"]
    med_data = temp_df[temp_df["Communication_Group"] == "Medium"]["Addiction_Level"]
    high_data = temp_df[temp_df["Communication_Group"] == "High"]["Addiction_Level"]
    
    plt.figure(figsize=(8,5))
    plt.boxplot([low_data, med_data, high_data], labels=["Low", "Medium", "High"])
    plt.xlabel("Family Communication Level")
    plt.ylabel("Addiction Level")
    plt.title("Addiction Level by Family Communication Group")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mental_health_scatter(df):
    """
    Vẽ scatter plots kiểm tra mối quan hệ giữa sức khỏe tâm thần
    (Anxiety, Depression) và hành vi sử dụng điện thoại
    (Daily Usage, Addiction Level).
    """
    pairs = [
        ("Anxiety_Level", "Daily_Usage_Hours"),
        ("Anxiety_Level", "Addiction_Level"),
        ("Depression_Level", "Daily_Usage_Hours"),
        ("Depression_Level", "Addiction_Level"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (x_var, y_var) in zip(axes, pairs):
        x = df[x_var]
        y = df[y_var]

        ax.scatter(x, y, alpha=0.4, s=15)

        # Đường hồi quy tuyến tính
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2)

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"{x_var} vs {y_var}")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Mental Health vs Phone Usage",
        fontsize=16,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

def plot_mental_health_by_anxiety_group(df):
    # Chia thành 3 nhóm theo phân vị
    q33 = df["Anxiety_Level"].quantile(0.33)
    q67 = df["Anxiety_Level"].quantile(0.67)
    
    # Tạo label cho từng nhóm
    labels = []
    for val in df["Anxiety_Level"]:
        if val <= q33:
            labels.append("Low")
        elif val <= q67:
            labels.append("Medium")
        else:
            labels.append("High")
    
    # Tạo dataframe tạm
    temp_df = df[["Daily_Usage_Hours", "Addiction_Level"]].copy()
    temp_df["Anxiety_Group"] = labels
    
    # Chuẩn bị dữ liệu cho boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot cho Daily_Usage_Hours
    low_usage = temp_df[temp_df["Anxiety_Group"] == "Low"]["Daily_Usage_Hours"]
    med_usage = temp_df[temp_df["Anxiety_Group"] == "Medium"]["Daily_Usage_Hours"]
    high_usage = temp_df[temp_df["Anxiety_Group"] == "High"]["Daily_Usage_Hours"]
    
    axes[0].boxplot([low_usage, med_usage, high_usage], labels=["Low", "Medium", "High"])
    axes[0].set_xlabel("Anxiety Level Group")
    axes[0].set_ylabel("Daily Usage Hours")
    axes[0].set_title("Daily Usage Hours by Anxiety Level Group")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Boxplot cho Addiction_Level
    low_addiction = temp_df[temp_df["Anxiety_Group"] == "Low"]["Addiction_Level"]
    med_addiction = temp_df[temp_df["Anxiety_Group"] == "Medium"]["Addiction_Level"]
    high_addiction = temp_df[temp_df["Anxiety_Group"] == "High"]["Addiction_Level"]
    
    axes[1].boxplot([low_addiction, med_addiction, high_addiction], labels=["Low", "Medium", "High"])
    axes[1].set_xlabel("Anxiety Level Group")
    axes[1].set_ylabel("Addiction Level")
    axes[1].set_title("Addiction Level by Anxiety Level Group")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle("Phone Usage and Addiction by Anxiety Level", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_mental_health_by_depression_group(df):
    # Chia thành 3 nhóm theo phân vị
    q33 = df["Depression_Level"].quantile(0.33)
    q67 = df["Depression_Level"].quantile(0.67)
    
    # Tạo label cho từng nhóm
    labels = []
    for val in df["Depression_Level"]:
        if val <= q33:
            labels.append("Low")
        elif val <= q67:
            labels.append("Medium")
        else:
            labels.append("High")
    
    # Tạo dataframe tạm
    temp_df = df[["Daily_Usage_Hours", "Addiction_Level"]].copy()
    temp_df["Depression_Group"] = labels
    
    # Chuẩn bị dữ liệu cho boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot cho Daily_Usage_Hours
    low_usage = temp_df[temp_df["Depression_Group"] == "Low"]["Daily_Usage_Hours"]
    med_usage = temp_df[temp_df["Depression_Group"] == "Medium"]["Daily_Usage_Hours"]
    high_usage = temp_df[temp_df["Depression_Group"] == "High"]["Daily_Usage_Hours"]
    
    axes[0].boxplot([low_usage, med_usage, high_usage], labels=["Low", "Medium", "High"])
    axes[0].set_xlabel("Depression Level Group")
    axes[0].set_ylabel("Daily Usage Hours")
    axes[0].set_title("Daily Usage Hours by Depression Level Group")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Boxplot cho Addiction_Level
    low_addiction = temp_df[temp_df["Depression_Group"] == "Low"]["Addiction_Level"]
    med_addiction = temp_df[temp_df["Depression_Group"] == "Medium"]["Addiction_Level"]
    high_addiction = temp_df[temp_df["Depression_Group"] == "High"]["Addiction_Level"]
    
    axes[1].boxplot([low_addiction, med_addiction, high_addiction], labels=["Low", "Medium", "High"])
    axes[1].set_xlabel("Depression Level Group")
    axes[1].set_ylabel("Addiction Level")
    axes[1].set_title("Addiction Level by Depression Level Group")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle("Phone Usage and Addiction by Depression Level", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_purpose_scatter_comparison(df):
    """Vẽ 3 scatter plots so sánh mối quan hệ giữa từng mục đích sử dụng và Addiction Level"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Time_on_Social_Media vs Addiction_Level
    axes[0].scatter(df["Time_on_Social_Media"], df["Addiction_Level"], alpha=0.5, s=20)
    # Đường hồi quy
    z = np.polyfit(df["Time_on_Social_Media"], df["Addiction_Level"], 1)
    p = np.poly1d(z)
    axes[0].plot(df["Time_on_Social_Media"], p(df["Time_on_Social_Media"]), "r--", linewidth=2)
    axes[0].set_xlabel("Time on Social Media (hours)")
    axes[0].set_ylabel("Addiction Level")
    axes[0].set_title("Social Media vs Addiction Level")
    axes[0].grid(alpha=0.3)
    
    # Time_on_Gaming vs Addiction_Level
    axes[1].scatter(df["Time_on_Gaming"], df["Addiction_Level"], alpha=0.5, s=20, color='orange')
    z = np.polyfit(df["Time_on_Gaming"], df["Addiction_Level"], 1)
    p = np.poly1d(z)
    axes[1].plot(df["Time_on_Gaming"], p(df["Time_on_Gaming"]), "r--", linewidth=2)
    axes[1].set_xlabel("Time on Gaming (hours)")
    axes[1].set_ylabel("Addiction Level")
    axes[1].set_title("Gaming vs Addiction Level")
    axes[1].grid(alpha=0.3)
    
    # Time_on_Education vs Addiction_Level
    axes[2].scatter(df["Time_on_Education"], df["Addiction_Level"], alpha=0.5, s=20, color='green')
    z = np.polyfit(df["Time_on_Education"], df["Addiction_Level"], 1)
    p = np.poly1d(z)
    axes[2].plot(df["Time_on_Education"], p(df["Time_on_Education"]), "r--", linewidth=2)
    axes[2].set_xlabel("Time on Education (hours)")
    axes[2].set_ylabel("Addiction Level")
    axes[2].set_title("Education vs Addiction Level")
    axes[2].grid(alpha=0.3)
    
    plt.suptitle("Phone Usage Purpose vs Addiction Level", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_purpose_by_usage_group(df):
    """Vẽ boxplot so sánh Addiction Level theo các nhóm thời gian sử dụng cho từng mục đích"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Chia nhóm cho Social Media
    q33_sm = df["Time_on_Social_Media"].quantile(0.33)
    q67_sm = df["Time_on_Social_Media"].quantile(0.67)
    
    sm_labels = []
    for val in df["Time_on_Social_Media"]:
        if val <= q33_sm:
            sm_labels.append("Low")
        elif val <= q67_sm:
            sm_labels.append("Medium")
        else:
            sm_labels.append("High")
    
    temp_df_sm = df[["Addiction_Level"]].copy()
    temp_df_sm["SM_Group"] = sm_labels
    
    low_sm = temp_df_sm[temp_df_sm["SM_Group"] == "Low"]["Addiction_Level"]
    med_sm = temp_df_sm[temp_df_sm["SM_Group"] == "Medium"]["Addiction_Level"]
    high_sm = temp_df_sm[temp_df_sm["SM_Group"] == "High"]["Addiction_Level"]
    
    axes[0].boxplot([low_sm, med_sm, high_sm], labels=["Low", "Medium", "High"])
    axes[0].set_xlabel("Social Media Time Group")
    axes[0].set_ylabel("Addiction Level")
    axes[0].set_title("Addiction by Social Media Usage")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Chia nhóm cho Gaming
    q33_gm = df["Time_on_Gaming"].quantile(0.33)
    q67_gm = df["Time_on_Gaming"].quantile(0.67)
    
    gm_labels = []
    for val in df["Time_on_Gaming"]:
        if val <= q33_gm:
            gm_labels.append("Low")
        elif val <= q67_gm:
            gm_labels.append("Medium")
        else:
            gm_labels.append("High")
    
    temp_df_gm = df[["Addiction_Level"]].copy()
    temp_df_gm["GM_Group"] = gm_labels
    
    low_gm = temp_df_gm[temp_df_gm["GM_Group"] == "Low"]["Addiction_Level"]
    med_gm = temp_df_gm[temp_df_gm["GM_Group"] == "Medium"]["Addiction_Level"]
    high_gm = temp_df_gm[temp_df_gm["GM_Group"] == "High"]["Addiction_Level"]
    
    axes[1].boxplot([low_gm, med_gm, high_gm], labels=["Low", "Medium", "High"])
    axes[1].set_xlabel("Gaming Time Group")
    axes[1].set_ylabel("Addiction Level")
    axes[1].set_title("Addiction by Gaming Usage")
    axes[1].grid(axis='y', alpha=0.3)
    
    # Chia nhóm cho Education
    q33_ed = df["Time_on_Education"].quantile(0.33)
    q67_ed = df["Time_on_Education"].quantile(0.67)
    
    ed_labels = []
    for val in df["Time_on_Education"]:
        if val <= q33_ed:
            ed_labels.append("Low")
        elif val <= q67_ed:
            ed_labels.append("Medium")
        else:
            ed_labels.append("High")
    
    temp_df_ed = df[["Addiction_Level"]].copy()
    temp_df_ed["ED_Group"] = ed_labels
    
    low_ed = temp_df_ed[temp_df_ed["ED_Group"] == "Low"]["Addiction_Level"]
    med_ed = temp_df_ed[temp_df_ed["ED_Group"] == "Medium"]["Addiction_Level"]
    high_ed = temp_df_ed[temp_df_ed["ED_Group"] == "High"]["Addiction_Level"]
    
    axes[2].boxplot([low_ed, med_ed, high_ed], labels=["Low", "Medium", "High"])
    axes[2].set_xlabel("Education Time Group")
    axes[2].set_ylabel("Addiction Level")
    axes[2].set_title("Addiction by Education Usage")
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle("Addiction Level by Phone Usage Purpose Groups", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_screen_before_bed_scatter(df):
    """Vẽ scatter plot + regression line cho Screen_Time_Before_Bed vs 4 outcomes"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Screen Time Before Bed: Scatter Plots with Regression Lines', fontsize=16, fontweight='bold')
    
    # Các cặp biến cần vẽ
    pairs = [
        ('Screen_Time_Before_Bed', 'Sleep_Hours', 'Sleep Hours'),
        ('Screen_Time_Before_Bed', 'Academic_Performance', 'Academic Performance'),
        ('Screen_Time_Before_Bed', 'Anxiety_Level', 'Anxiety Level'),
        ('Screen_Time_Before_Bed', 'Depression_Level', 'Depression Level')
    ]
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (x_var, y_var, y_label) in enumerate(pairs):
        row, col = positions[idx]
        ax = axes[row, col]
        
        # Scatter plot
        ax.scatter(df[x_var], df[y_var], alpha=0.5, s=20)
        
        # Regression line
        z = np.polyfit(df[x_var], df[y_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
        
        # Tính correlation
        corr = df[x_var].corr(df[y_var])
        
        ax.set_xlabel('Screen Time Before Bed (hours)', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f'{y_label} (r = {corr:.4f})', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_screen_before_bed_analysis(df):
    """Vẽ 4 boxplot so sánh ảnh hưởng của Screen Time Before Bed đến Sleep, Academic, Anxiety, Depression"""
    # Chia thành 3 nhóm theo phân vị
    q33 = df["Screen_Time_Before_Bed"].quantile(0.33)
    q67 = df["Screen_Time_Before_Bed"].quantile(0.67)
    
    # Tạo label cho từng nhóm
    labels = []
    for val in df["Screen_Time_Before_Bed"]:
        if val <= q33:
            labels.append("Low")
        elif val <= q67:
            labels.append("Medium")
        else:
            labels.append("High")
    
    # Tạo dataframe tạm
    temp_df = df[["Sleep_Hours", "Academic_Performance", "Anxiety_Level", "Depression_Level"]].copy()
    temp_df["Screen_Before_Bed_Group"] = labels
    
    # Vẽ 4 boxplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Screen Time Before Bed vs Health & Academic Outcomes', fontsize=16, fontweight='bold')
    
    # Sleep Hours
    low_sleep = temp_df[temp_df["Screen_Before_Bed_Group"] == "Low"]["Sleep_Hours"]
    med_sleep = temp_df[temp_df["Screen_Before_Bed_Group"] == "Medium"]["Sleep_Hours"]
    high_sleep = temp_df[temp_df["Screen_Before_Bed_Group"] == "High"]["Sleep_Hours"]
    
    axes[0, 0].boxplot([low_sleep, med_sleep, high_sleep], labels=["Low", "Medium", "High"])
    axes[0, 0].set_xlabel("Screen Time Before Bed Group")
    axes[0, 0].set_ylabel("Sleep Hours")
    axes[0, 0].set_title("Sleep Hours by Screen Time Before Bed")
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Academic Performance
    low_academic = temp_df[temp_df["Screen_Before_Bed_Group"] == "Low"]["Academic_Performance"]
    med_academic = temp_df[temp_df["Screen_Before_Bed_Group"] == "Medium"]["Academic_Performance"]
    high_academic = temp_df[temp_df["Screen_Before_Bed_Group"] == "High"]["Academic_Performance"]
    
    axes[0, 1].boxplot([low_academic, med_academic, high_academic], labels=["Low", "Medium", "High"])
    axes[0, 1].set_xlabel("Screen Time Before Bed Group")
    axes[0, 1].set_ylabel("Academic Performance")
    axes[0, 1].set_title("Academic Performance by Screen Time Before Bed")
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Anxiety Level
    low_anxiety = temp_df[temp_df["Screen_Before_Bed_Group"] == "Low"]["Anxiety_Level"]
    med_anxiety = temp_df[temp_df["Screen_Before_Bed_Group"] == "Medium"]["Anxiety_Level"]
    high_anxiety = temp_df[temp_df["Screen_Before_Bed_Group"] == "High"]["Anxiety_Level"]
    
    axes[1, 0].boxplot([low_anxiety, med_anxiety, high_anxiety], labels=["Low", "Medium", "High"])
    axes[1, 0].set_xlabel("Screen Time Before Bed Group")
    axes[1, 0].set_ylabel("Anxiety Level")
    axes[1, 0].set_title("Anxiety Level by Screen Time Before Bed")
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Depression Level
    low_depression = temp_df[temp_df["Screen_Before_Bed_Group"] == "Low"]["Depression_Level"]
    med_depression = temp_df[temp_df["Screen_Before_Bed_Group"] == "Medium"]["Depression_Level"]
    high_depression = temp_df[temp_df["Screen_Before_Bed_Group"] == "High"]["Depression_Level"]
    
    axes[1, 1].boxplot([low_depression, med_depression, high_depression], labels=["Low", "Medium", "High"])
    axes[1, 1].set_xlabel("Screen Time Before Bed Group")
    axes[1, 1].set_ylabel("Depression Level")
    axes[1, 1].set_title("Depression Level by Screen Time Before Bed")
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_addiction_level_boxplots(groups, comparison_vars):
    group_names = list(groups.keys())
    group_data = list(groups.values())

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    colors = ["#2ecc71", "#e74c3c"]  

    for idx, var in enumerate(comparison_vars):
        if idx >= len(axes):
            break
        ax = axes[idx]
        data_to_plot = [
            group_data[0][var],
            group_data[1][var]
        ]
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(group_names)

        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)

        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel(var, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Tắt các subplot dư
        for idx in range(len(comparison_vars), len(axes)):
            axes[idx].axis("off")
    
    plt.suptitle(
        "Behavioral / Psychological / Social Factors by Addiction Level",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


def plot_addiction_level_radar(groups, variables):
    """ Vẽ radar chart thể hiện profile đa chiều của 3 nhóm Addiction (Low, Medium, High)."""
    group_names = list(groups.keys())
    group_data = list(groups.values())
    colors = ["#2ecc71", "#e74c3c"]

    # Tính mean cho từng nhóm
    mean_matrix = []
    for g in group_data:
        mean_matrix.append([g[var].mean() for var in variables])

    # Normalize theo scale thực (absolute normalization)
    scaled = [[], []]

    for j, var in enumerate(variables):
        if var == "Academic_Performance":
            max_scale = 100
        else:
            max_scale = 10  # các biến còn lại đều scale 0–10

        for i in range(2):
            val = mean_matrix[i][j]
            scaled[i].append(0.0 if np.isnan(val) else val / max_scale)

    # Chuẩn bị góc radar
    num_vars = len(variables)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # đóng vòng

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    # Vẽ radar cho từng nhóm
    for i in range(2):
        values = scaled[i] + [scaled[i][0]]
        ax.plot(angles, values, linewidth=2, label=group_names[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(variables, fontsize=9)
    ax.set_yticklabels([])

    ax.set_title(
        "Multi-dimensional Profile by Addiction Level (Normalized)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ Coefficients cho Linear Regression 
def plot_lr_coefficients(coef_df, title="Linear Regression - Feature Coefficients"):

    plt.figure(figsize=(10, 5))
    colors = ['skyblue' if c > 0 else 'red' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)  # đường mốc 0
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Vẽ biểu đồ Feature Importance cho Random Forest 
def plot_rf_feature_importance(fi_df, title="Random Forest - Feature Importance"):
    
    plt.figure(figsize=(10, 5))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue', edgecolor='black')
    for i, val in enumerate(fi_df['Importance']):
        plt.text(val + 0.005, i, f"{val:.3f}", va='center')
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ so sánh 2 models
def compare_model_metrics(mae_list, rmse_list, r2_list, model_names=['Linear Regression', 'Random Forest']):
    
    metrics = ['MAE', 'RMSE', 'R²']
    values = [mae_list, rmse_list, r2_list]
    colors = ['#3498db', '#2ecc71']  # màu cho các model

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 cột

    for i, metric in enumerate(metrics):
        axes[i].bar(model_names, values[i], color=colors, edgecolor='black', linewidth=1.5)

        # In giá trị lên trên bar
        for j, val in enumerate(values[i]):
            axes[i].text(j, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        # Highlight model tốt nhất
        if metric == 'R²':
            best_idx = np.argmax(values[i])  # R² càng cao càng tốt
        else:
            best_idx = np.argmin(values[i])  # MAE/RMSE càng thấp càng tốt
        axes[i].patches[best_idx].set_edgecolor('gold')
        axes[i].patches[best_idx].set_linewidth(3)

        axes[i].set_ylabel(metric, fontweight='bold')
        axes[i].set_title(f'{metric} Comparison', fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
