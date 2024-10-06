from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from service import get_stock_data, prepare_data, linear_regression, ridge_regression, neural_network, stacking


templates = Jinja2Templates(directory="./")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

class StockRequest(BaseModel):
    stock_name: str
    model_type: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("/index.html", {"request": request})

@app.post("/predict")
def predict_stock(data: StockRequest):
    stock_name = data.stock_name
    model_type = data.model_type
    days = '6mo'
    
    stock_data = get_stock_data(stock_name, days)
    if stock_data is None:
        return {"error": "Failed to retrieve stock data"}
    

    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    

    X_last_day = np.array([stock_data[['Open', 'High', 'Low', 'Volume']].iloc[-1]]).reshape(1, -1)
    

    if model_type == "linear":
        mse, r2, next_day_prediction, img_base64, rmse, mae = linear_regression(X_train, X_test, y_train, y_test, X_last_day)
    elif model_type == "ridge":
        mse, r2, next_day_prediction, img_base64, rmse, mae = ridge_regression(X_train, X_test, y_train, y_test, X_last_day)
    elif model_type == "neural":
        mse, r2, next_day_prediction, img_base64, rmse, mae = neural_network(X_train, X_test, y_train, y_test, X_last_day)
    elif model_type == "stacking":
        mse, r2, next_day_prediction, img_base64, rmse, mae = stacking(X_train, X_test, y_train, y_test, X_last_day)
        
    else:
        return {"error": "Model not supported"}

    return {
        "stock_name": stock_name,
        "mse": mse.tolist() if isinstance(mse, np.ndarray) else mse,
        "rmse": rmse.tolist() if isinstance(rmse, np.ndarray) else rmse,
        "r2": r2.tolist() if isinstance(r2, np.ndarray) else r2,
        "mae": mae.tolist() if isinstance(mae, np.ndarray) else mae,
        "next_day_prediction": next_day_prediction.tolist() if isinstance(next_day_prediction, np.ndarray) else next_day_prediction,
        "chart": img_base64
    }
