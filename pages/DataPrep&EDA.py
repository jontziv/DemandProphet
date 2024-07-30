import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('Data Preperation & Exploratory Data Analysis')
st.subheader("Data Preperation", divider=True)

multi = '''Once the raw csv is loaded the preperation face involves, bringing the data in a format suitable for EDA.\n  
    Steps followed:\n  
        1) # Standardize column names for removing spaces\n 
        2) Creating sub-set of our dataset using only the columns: \n  
                'ORDER_DATE_(DATEORDERS)'\n  
            , 'CATEGORY_NAME','CATEGORY_ID'\n  
            ,'ORDER_ITEM_QUANTITY'\n  
            ,'ORDER_REGION'\n  
            ,'ORDER_STATUS'\n  
            ,'PRODUCT_NAME','PRODUCT_CARD_ID'\n
            ,'DAYS_FOR_SHIPPING_(REAL)','DAYS_FOR_SHIPMENT_(SCHEDULED)'\n  
        3)Remova Not a Number/Null values  \n  
        4) and finally making timestamp fields from string to datetime type. \n   
\n  
Two (or more) newline characters in a row will result in a hard return.
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
     ,
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


st.subheader("Exploratory Data Analysis", divider=True)
orders_over_time = dataco_supply_chain.groupby('ORDER_DATE')['ORDER_ITEM_QUANTITY'].sum().reset_index()

# Create a line plot using Plotly
figa = go.Figure()

figa.add_trace(go.Scatter(
    x=orders_over_time['ORDER_DATE'],
    y=orders_over_time['ORDER_ITEM_QUANTITY'],
    mode='lines',
    name='Total Order Quantity'
))

# Update layout for better appearance
figa.update_layout(
    title='Total Orders Over Time',
    xaxis_title='Order Date',
    yaxis_title='Total Order Quantity',
    plot_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
    xaxis=dict(
        showgrid=False  # Hide x-axis grid
    ),
    yaxis=dict(
        showgrid=True  # Show y-axis grid
    ),
    margin=dict(l=0, r=0, t=30, b=0)  # Adjust margins
)
st.plotly_chart(figa, use_container_width=True)
# Show the plot

# Assuming 'dataco_supply_chain' is your DataFrame
figb = px.histogram(dataco_supply_chain, y='ORDER_REGION', title='Distribution of Orders by Region')

# Update layout for better appearance
figb.update_layout(
    xaxis_title='Count',
    yaxis_title='Region',
    bargap=0.2,  # gap between bars
    plot_bgcolor='rgba(0,0,0,0)'  # set background color to transparent
)

# Show the plot
st.plotly_chart(figb, use_container_width=True)

# Assuming 'dataco_supply_chain' is your DataFrame
figc = px.histogram(
    dataco_supply_chain, 
    x='ORDER_STATUS', 
    title='Distribution of Order Status',
    labels={'ORDER_STATUS': 'Order Status'},
    category_orders={"ORDER_STATUS": dataco_supply_chain['ORDER_STATUS'].value_counts().index}  # To preserve the order of the categories as in the countplot
)

# Update layout for better appearance
figc.update_layout(
    yaxis_title='Count',
    xaxis_title='Order Status',
    bargap=0.2,  # Gap between bars
    plot_bgcolor='rgba(0,0,0,0)'  # Set background color to transparent
)

# Show the plot
st.plotly_chart(figc, use_container_width=True)
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
figd = px.line(
    cleaned_daily_orders, 
    x='ORDER_DATE', 
    y='ORDER_ITEM_QUANTITY', 
    title='Total Orders Over Time (Without Outliers)',
    labels={'ORDER_DATE': 'Order Date', 'ORDER_ITEM_QUANTITY': 'Total Order Quantity'}
)

# Update layout for better appearance
figd.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
    xaxis=dict(
        showgrid=False  # Hide x-axis grid
    ),
    yaxis=dict(
        showgrid=True  # Show y-axis grid
    ),
    margin=dict(l=0, r=0, t=30, b=0)  # Adjust margins
)
st.plotly_chart(figd, use_container_width=True)
