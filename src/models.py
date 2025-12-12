import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Huấn luyện mô hình Linear Regression trên các feature đã chọn 
def train_linear_regression(df, features, target='Addiction_Level', test_size=0.2, random_state=42):
    
    X = df[features]
    y = df[target]

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Chuẩn hóa dữ liệu để Linear Regression ổn định hơn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit trên train để tránh data leakage
    X_test_scaled  = scaler.transform(X_test)       # transform test theo scaler đã học

    # Tạo và huấn luyện Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Dự đoán trên tập test
    y_pred = lr.predict(X_test_scaled)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Phân tích hệ số (Coefficient)
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': lr.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False).reset_index(drop=True)

    return lr, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, mae, rmse, r2, coef_df


# Huấn luyện mô hình Random Forest Regression trên các feature đã chọn
def train_random_forest(df, features, target='Addiction_Level', test_size=0.2, random_state=42,
                        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, n_jobs=-1):
    
    X = df[features]
    y = df[target]

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Tạo và huấn luyện Random Forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)

    # Dự đoán
    y_pred = rf.predict(X_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Feature importance
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return rf, X_train, X_test, y_train, y_test, y_pred, mae, rmse, r2, fi_df
