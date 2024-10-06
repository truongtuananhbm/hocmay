import yfinance as yf
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import pandas as pd
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingRegressor
import getdata as gd


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

def stacking(X_train, X_test, y_train, y_test, X_last_day):
    stock_data = yf.download(tickers="AAPL", period="6mo")
    stock_data_cleaned = gd.get_cleaned_stock_data()

    features = stock_data_cleaned[['Open', 'High', 'Low', 'Volume']]
    target = stock_data_cleaned['Close']

    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=43)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43)
    scaler = MinMaxScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_val = scaler.transform(X_val)
    scaled_X_test = scaler.transform(X_test)
    X_last_day_scaled = scaler.transform(X_last_day)

    mlp = MLPRegressor(random_state=1, max_iter=3000)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], store_cv_values=True)
    linreg = LinearRegression()

    stacking_model = StackingRegressor(
        estimators=[('mlp', mlp), ('ridge', ridge), ('linreg', linreg)],
        final_estimator=RidgeCV()
    )

    stacking_model.fit(scaled_X_train, y_train)

    y_train_pred = stacking_model.predict(scaled_X_train)
    y_val_pred = stacking_model.predict(scaled_X_val)
    y_test_pred = stacking_model.predict(scaled_X_test)

    y_pred = stacking_model.predict(scaled_X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    next_day_prediction = stacking_model.predict(X_last_day_scaled)

    img_base64 = plot_predictions(y_test, y_pred, 'Stacking Model')

    return mse, r2, next_day_prediction, img_base64, rmse, mae
