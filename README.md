# Supply Chain Demand Forecasting with Prophet

This repository contains a Kaggle notebook for demand forecasting in a supply chain context using the Prophet library. The goal of this project is to predict future demand for products based on historical order data. The model uses Facebook's Prophet, a robust forecasting tool designed for time series data with strong seasonal effects.

## Project Overview

The project includes the following key steps:
1. **Data Preparation & Cleaning**: Standardizing column names, handling missing values, and converting date columns to datetime format.
2. **Exploratory Data Analysis (EDA)**: Analyzing the historical data to understand trends and seasonality.
3. **Forecasting**: Applying Prophet to forecast future demand.
4. **Evaluation**: Measuring the performance of the model using metrics such as MAE, MSE, and RMSE.
5. **Visualization**: Plotting forecast results and components.

## Data

The dataset used in this project is a CSV file containing historical order data with columns such as:
- `ORDER_DATE_(DATEORDERS)`
- `CATEGORY_NAME`
- `CATEGORY_ID`
- `ORDER_ITEM_QUANTITY`
- `ORDER_REGION`
- `ORDER_STATUS`
- `PRODUCT_NAME`
- `PRODUCT_CARD_ID`
- `DAYS_FOR_SHIPPING_(REAL)`
- `DAYS_FOR_SHIPMENT_(SCHEDULED)`

## Requirements

To run this project, you need to have the following Python packages installed:

- `pandas`
- `numpy`
- `plotly`
- `prophet`
- `sklearn`
- `streamlit`