import yfinance as yf
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from sklearn.preprocessing import MinMaxScaler

def get_cleaned_stock_data():
    # Tải dữ liệu cổ phiếu
    stock_data = yf.download(tickers="AAPL", period="6mo")

    # Chọn các cột liên quan
    print(stock_data)

    # Kiểm tra giá trị NaN trong toàn bộ dữ liệu trước khi xử lý
    print("Số lượng NaN trong dữ liệu trước khi xử lý:")
    print(stock_data.isnull().sum())

    # Xóa các hàng có NaN trong toàn bộ dữ liệu
    stock_data_cleaned = stock_data.dropna()  # Không dùng inplace để tránh mất dữ liệu gốc

    # Kiểm tra lại sau khi xóa NaN
    print("\nSố lượng NaN trong dữ liệu sau khi xử lý:")
    print(stock_data_cleaned.isnull().sum())

    # In dữ liệu sau khi xử lý NaN
    print("\nDữ liệu sau khi xử lý NaN:")
    print(stock_data_cleaned)


    # X_train_scaled, X_val_scaled, và X_test_scaled là dữ liệu đã được tiền xử lý, sẵn sàng để đưa vào mô hình.


    # Phát hiện và xử lý outliers bằng IQR
    Q1 = stock_data_cleaned.quantile(0.25)
    Q3 = stock_data_cleaned.quantile(0.75)
    IQR = Q3 - Q1

    # Xác định outliers (bất kỳ giá trị nào nằm ngoài 1.5 lần khoảng IQR)
    outliers = (stock_data_cleaned < (Q1 - 1.5 * IQR)) | (stock_data_cleaned > (Q3 + 1.5 * IQR))

    # Loại bỏ các hàng chứa outliers
    stock_data_cleaned1 = stock_data_cleaned[~outliers.any(axis=1)]

    print(f"Số giá trị ngoại lai đã loại bỏ: {len(stock_data_cleaned) - len(stock_data_cleaned1)}")
    print("\nDữ liệu sau khi loại bỏ giá trị ngoại lai:")
    print(stock_data_cleaned1)

    return stock_data_cleaned1












