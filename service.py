import yfinance as yf
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')

def get_stock_data(ticker, period='1mo'):
    try:
       
        stock_data = yf.download(ticker, period=period)
        print(stock_data)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None
def plot_predictions(y_train, y_pred, model_name):
    img_buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.title(f'{model_name}')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    img_buffer.close()
    return img_base64


def prepare_data(stock_data):
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    target = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def predict_next_day_price(model, X_last_day):
    return model.predict(X_last_day)

def linear_regression(X_train, X_test, y_train, y_test, X_last_day):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    next_day_prediction = predict_next_day_price(model, X_last_day)
    img_base64 = plot_predictions(y_test, y_pred, 'Linear Regression')
    return mse, r2, next_day_prediction, img_base64, rmse, mae

def ridge_regression(X_train, X_test, y_train, y_test, X_last_day):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    next_day_prediction = predict_next_day_price(model, X_last_day)
    img_base64 = plot_predictions(y_test, y_pred, 'Ridge Regression')
    return mse, r2, next_day_prediction, img_base64, rmse, mae

def neural_network(X_train, X_test, y_train, y_test, X_last_day):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_last_day_scaled = scaler.transform(X_last_day)
    
    model = MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    next_day_prediction = model.predict(X_last_day_scaled.reshape(1, -1))
    
    img_base64 = plot_predictions(y_test, y_pred, 'Neural Network')
    return mse, r2, next_day_prediction, img_base64, rmse, mae

def ensemble_model(X_train, X_test, y_train, y_test, X_last_day):
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    predictions_test = []
    individual_predictions = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        predictions_test.append(pred_test)
        
        individual_predictions[model_name] = pred_test
    
    # Tính trung bình của dự đoán từ các mô hình
    ensemble_predictions_test = np.mean(predictions_test, axis=0)
    
    # Tính các chỉ số đánh giá
    mse = mean_squared_error(y_test, ensemble_predictions_test)
    r2 = r2_score(y_test, ensemble_predictions_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, ensemble_predictions_test)
    
    # Vẽ biểu đồ kết quả, chỉ tập trung vào mô hình tuyến tính
    img_buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    
    # Vẽ giá trị thực tế
    plt.plot(np.arange(len(y_test)), y_test.values, label='Actual Prices', color='blue', linewidth=2)
    
    # Vẽ dự đoán của Linear Regression (mô hình tuyến tính)
    plt.plot(np.arange(len(y_test)), individual_predictions['linear'], label='Linear Model', color='green', linestyle='--')
    
    # Vẽ dự đoán của mô hình Ensemble
    plt.plot(np.arange(len(y_test)), ensemble_predictions_test, label='Ensemble Model', color='black', linewidth=2)
    
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    
    # Lưu biểu đồ vào buffer
    plt.savefig(img_buffer, format='png')
    plt.close()  # Đóng hình ảnh để giải phóng bộ nhớ
    
    img_buffer.seek(0)
    
    # Mã hóa hình ảnh thành chuỗi base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Đóng buffer
    img_buffer.close()
    
    return mse, r2, img_base64, rmse, mae
