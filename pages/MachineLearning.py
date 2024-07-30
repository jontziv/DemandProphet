import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title('Demand Forecasting')
st.subheader("Prophet", divider=True)

multi = '''Forecasting future demand is a fundamental business problem and any solution that is successful in tackling this will find valuable commercial \n    
applications in diverse business segments. In the retail context, Demand Forecasting methods are implemented to make decisions regarding buying, provisioning,\n  
replenishment, and financial planning. Some of the common time-series methods applied for Demand Forecasting and provisioning include Moving Average, Exponential Smoothing, \n  
and ARIMA. The most popular models in Kaggle competitions for time-series forecasting have been Gradient Boosting models that convert time-series data into tabular data, with \n  
lag terms in the time-series as ‘features’ or columns in the table.
\n  
\nThe Facebook Prophet model is a type of GAM (Generalized Additive Model) that specializes in solving business/econometric — time-series problems.
'''
st.markdown(multi)

df1= pd.read_csv('data/DataCoSupplyChainDatasetFPt1.csv', encoding='ISO-8859-1')
df2= pd.read_csv('data/DataCoSupplyChainDatasetFPt2.csv', encoding='ISO-8859-1')
df3= pd.read_csv('data/DataCoSupplyChainDatasetFPt3.csv', encoding='ISO-8859-1')
df4= pd.read_csv('data/DataCoSupplyChainDatasetFPt4.csv', encoding='ISO-8859-1')
dataco_supply_chain=pd.concat([df1, df2,df3,df4], 
                  axis = 1)


#dataco_supply_chain = pd.read_csv('data/DataCoSupplyChainDatasetFcopy.csv', encoding='ISO-8859-1')
dataco_supply_chain.columns = dataco_supply_chain.columns.str.upper().str.replace(' ', '_')
# select columns to use
dataco_supply_chain = dataco_supply_chain[
    ['ORDER_DATE_(DATEORDERS)'
     , 'CATEGORY_NAME','CATEGORY_ID'
     ,'ORDER_ITEM_QUANTITY'
     ,'ORDER_REGION'
     ,'ORDER_STATUS'
     ,'PRODUCT_NAME','PRODUCT_CARD_ID'
     ,'DAYS_FOR_SHIPPING_(REAL)','DAYS_FOR_SHIPMENT_(SCHEDULED)'
    ]
]


dataco_supply_chain['ORDER_DATE_(DATEORDERS)'] = pd.to_datetime(dataco_supply_chain['ORDER_DATE_(DATEORDERS)'])
# Extracting year, month, day, and weekday from the order date
dataco_supply_chain['ORDER_YEAR'] = dataco_supply_chain['ORDER_DATE_(DATEORDERS)'].dt.year
dataco_supply_chain['ORDER_MONTH'] = dataco_supply_chain['ORDER_DATE_(DATEORDERS)'].dt.month
dataco_supply_chain['ORDER_DAY'] = dataco_supply_chain['ORDER_DATE_(DATEORDERS)'].dt.day
dataco_supply_chain['ORDER_WEEKDAY'] = dataco_supply_chain['ORDER_DATE_(DATEORDERS)'].dt.weekday
dataco_supply_chain['ORDER_DATE'] = dataco_supply_chain['ORDER_DATE_(DATEORDERS)'].dt.date
dataco_supply_chain.drop(columns='ORDER_DATE_(DATEORDERS)', inplace=True)
orders_over_time = dataco_supply_chain.groupby('ORDER_DATE')['ORDER_ITEM_QUANTITY'].sum().reset_index()

# remove outliers
# Aggregate order quantities by date
daily_orders = dataco_supply_chain.groupby(['ORDER_DATE'])['ORDER_ITEM_QUANTITY'].sum().reset_index()
filtered_daily_orders = daily_orders.copy()
# Calculate Q1, Q3, and IQR
Q1 = filtered_daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.25)
Q3 = filtered_daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
cleaned_daily_orders = filtered_daily_orders[(filtered_daily_orders['ORDER_ITEM_QUANTITY'] >= lower_bound) & 
                               (filtered_daily_orders['ORDER_ITEM_QUANTITY'] <= upper_bound)]


# remove outliers
# Aggregate order quantities by date
daily_orders = dataco_supply_chain.groupby(['ORDER_DATE'])['ORDER_ITEM_QUANTITY'].sum().reset_index()
filtered_daily_orders = daily_orders.copy()
# Calculate Q1, Q3, and IQR
Q1 = filtered_daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.25)
Q3 = filtered_daily_orders['ORDER_ITEM_QUANTITY'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
cleaned_daily_orders = filtered_daily_orders[(filtered_daily_orders['ORDER_ITEM_QUANTITY'] >= lower_bound) & 
                               (filtered_daily_orders['ORDER_ITEM_QUANTITY'] <= upper_bound)]
# create monthly, weekly data
cleaned_daily_orders = cleaned_daily_orders.copy()
cleaned_daily_orders['ORDER_DATE'] = pd.to_datetime(cleaned_daily_orders['ORDER_DATE'])
cleaned_daily_orders['YEAR_MONTH']=cleaned_daily_orders['ORDER_DATE'].dt.to_period('M')
cleaned_daily_orders['YEAR_WEEK']=cleaned_daily_orders['ORDER_DATE'].dt.to_period('W')
# Group by 'year-month' and sum the ORDER_ITEM_QUANTITY
monthly_orders = cleaned_daily_orders.groupby('YEAR_MONTH')['ORDER_ITEM_QUANTITY'].sum()
weekly_orders = cleaned_daily_orders.groupby('YEAR_WEEK')['ORDER_ITEM_QUANTITY'].sum()

# Convert the series to a DataFrame
weekly_orders_df = weekly_orders.reset_index()
weekly_orders_df.columns = ['ds', 'y']

# Convert the 'ds' column to datetime format
weekly_orders_df['ds'] = weekly_orders_df['ds'].dt.to_timestamp()

# Split the data (holding out the last 20% for testing)
split_point = int(len(weekly_orders_df) * 0.80)
train = weekly_orders_df.iloc[:split_point]
test = weekly_orders_df.iloc[split_point:]

# Initialize and fit the Prophet model
weekly_model = Prophet()
weekly_model.fit(train)

# Create future dates for prediction (entire duration: train + test)
weekly_future = weekly_model.make_future_dataframe(periods=len(test), freq='W-SUN')

# Predict
weekly_forecast = weekly_model.predict(weekly_future)

# Evaluate on Training data
y_pred_train = weekly_forecast['yhat'][:split_point]
mae_train = mean_absolute_error(train['y'], y_pred_train)
mse_train = mean_squared_error(train['y'], y_pred_train)
rmse_train = np.sqrt(mse_train)


# Create a DataFrame with the metrics
metrics_tedf = pd.DataFrame({
    'Metric': ['MAE_Train', 'MSE_Train', 'RMSE_Train'],
    'Training': [mae_train, mse_train, rmse_train]
})
# Display the metrics as a table in Streamlit
st.write("Training Metrics")
st.table(metrics_tedf)

#print(f"Training MAE: {mae_train}")
#print(f"Training MSE: {mse_train}")
#print(f"Training RMSE: {rmse_train}")

# Evaluate on Testing data
y_pred_test = weekly_forecast['yhat'][split_point:]
mae_test = mean_absolute_error(test['y'], y_pred_test)
mse_test = mean_squared_error(test['y'], y_pred_test)
rmse_test = np.sqrt(mse_test)

#print(f"\nTesting MAE: {mae_test}")
#print(f"Testing MSE: {mse_test}")
#print(f"Testing RMSE: {rmse_test}")


# Create a DataFrame with the metrics
metrics_trdf = pd.DataFrame({
    'Metric': ['MAE_Testing', 'MSE_Testing', 'RMSE_Testing'],
    'Training': [mae_test, mse_test, rmse_test]
})
# Display the metrics as a table in Streamlit
st.write("Training Metrics")
st.table(metrics_trdf)



fig_forecast = go.Figure()

# Add the forecasted values
fig_forecast.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))

# Add the confidence intervals
fig_forecast.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='lightgray')))
fig_forecast.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='lightgray')))

# Add the historical data
fig_forecast.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='markers', name='Training Data', marker=dict(color='blue')))
fig_forecast.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='markers', name='Actual Test Data', marker=dict(color='orange')))

fig_forecast.update_layout(
    title='Weekly Orders Forecast',
    xaxis_title='Date',
    yaxis_title='Order Quantity'
)

# Show the forecast plot
st.plotly_chart(fig_forecast, use_container_width=True)

#fig_forecast.show()

# 2. Plot Components of the Forecast using Plotly
# Extract trend and seasonal components from the forecast
fig_components = go.Figure()

# Trend component
if 'trend' in weekly_forecast.columns:
    fig_components.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')))

# Yearly seasonality (if applicable)
if 'yearly' in weekly_forecast.columns:
    fig_components.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['yearly'], mode='lines', name='Yearly Seasonality', line=dict(color='orange')))

# Weekly seasonality (if applicable)
if 'weekly' in weekly_forecast.columns:
    fig_components.add_trace(go.Scatter(x=weekly_forecast['ds'], y=weekly_forecast['weekly'], mode='lines', name='Weekly Seasonality', line=dict(color='green')))

fig_components.update_layout(
    title='Forecast Components',
    xaxis_title='Date',
    yaxis_title='Value'
)

# Show the components plot
#fig_components.show()
st.plotly_chart(fig_components, use_container_width=True)

# 3. Plotting Actual vs Predicted Values using Plotly
fig_actual_vs_predicted = go.Figure()

# Actual training data
fig_actual_vs_predicted.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Training Data', line=dict(color='blue')))

# Actual test data
fig_actual_vs_predicted.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual Test Data', line=dict(color='orange')))

# Predicted training data
fig_actual_vs_predicted.add_trace(go.Scatter(x=train['ds'], y=y_pred_train, mode='lines', name='Predicted Training Data', line=dict(color='red', dash='dash')))

# Predicted test data
fig_actual_vs_predicted.add_trace(go.Scatter(x=test['ds'], y=y_pred_test, mode='lines', name='Predicted Test Data', line=dict(color='green', dash='dash')))

fig_actual_vs_predicted.update_layout(
    title='Actual vs Predicted Weekly Orders',
    xaxis_title='Date',
    yaxis_title='Order Quantity'
)

# Show the actual vs predicted plot
#fig_actual_vs_predicted.show()
st.plotly_chart(fig_actual_vs_predicted, use_container_width=True)

st.table(weekly_orders_df.describe())

multi1 = '''Mean Absolute Error (MAE): The model's testing MAE deviates by an average of 173.65 units from the actual weekly orders. Given the average weekly orders of 2,601, this represents an error of about 6.7%.\n

Root Mean Squared Error (RMSE): The model's testing RMSE of 204.48 indicates to instances where the model's predictions have larger deviations from the actual values.\n  
In other words our RMSE is higher than the standard deviation of the data, indicating the model might be missing some variability, fact that can be attributed to the initial dataset.\n

To enhance the model's accuracy, the dataset can undergo a stationarity review ( A stationary process has the property that the mean, variance and autocorrelation structure do not change over time), incorporating additional dimensions, tuning hyperparameters, or assess other prediction algorithms that maybe better suited.\n
\n  
'''
st.markdown(multi1)