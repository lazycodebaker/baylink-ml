# %%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from prophet import Prophet
import streamlit as st
import os 

def install_requirements():
    os.system('pip3 install -r requirements.txt')


def load_data(path):
    orders = pd.read_excel(path)   
    orders['Invoice Date'] = pd.to_datetime(orders['Invoice Date'])
    orders = preprocess_data(orders) 
    orders = orders[['Product Name','Qty','Invoice Date']]

    return orders

def abs(x):
    return np.abs(x)

def preprocess_data(orders): 
    orders['Qty'] = orders['Qty'].fillna(orders['Qty'].mean(), inplace=False)
    
    q_low = orders['Qty'].quantile(0.01)
    q_high = orders['Qty'].quantile(0.99)

    orders = orders[(orders['Qty'] >= q_low) & (orders['Qty'] <= q_high)]
    
    return orders

def create_models(orders):
    product_models = {}
    products = orders['Product Name'].unique()
    products = [product for product in products if len(orders[orders['Product Name'] == product]) > 2]
    
    for product in products:
        product_data = orders[orders['Product Name'] == product]
        if len(product_data) > 2:
            aggregated_data = product_data.groupby('Invoice Date')['Qty'].sum().reset_index()
            aggregated_data_prophet = aggregated_data.rename(columns={'Invoice Date': 'ds', 'Qty': 'y'})
            aggregated_data_prophet['cap'] = aggregated_data_prophet['y'].max()

            changepoints = aggregated_data_prophet['ds'].quantile(np.linspace(0.1, 0.9, 10)).tolist()

            model = Prophet(
                weekly_seasonality=True,
                daily_seasonality=True,  
                growth='linear',
            )
            
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(aggregated_data_prophet) 

            product_models[product] = model
    
    return product_models, products

def predict_next_30_days(product_name, product_models,orders):
    if product_name in product_models:
        model = product_models[product_name]

        future = model.make_future_dataframe(periods=30)
        future['cap'] = orders[orders['Product Name'] == product_name].groupby('Invoice Date')['Qty'].sum().max()
       
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].astype(int)

        accuracy = model.plot_components(forecast)
        st.pyplot(accuracy)
        
        fig1 = model.plot(forecast)
        plt.title(f'{product_name} Orders Forecast')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        st.pyplot(fig1)
        
        next_30_days = forecast[['ds', 'yhat']].tail(30)
 
        orders_sum = abs(next_30_days['yhat'].sum())

        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_30_days['ds'], next_30_days['yhat'], marker='o', linestyle='-', color='b')
        ax.set_title(f'Predicted Order Quantities for the Next 30 Days for {product_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Quantity')
        ax.grid(True)
        plt.xticks(rotation=45)
        
        return fig1, fig2, next_30_days , orders_sum
    else:
        st.write(f"Product '{product_name}' not found in the data.")
        return None, None, None

        
def main():
    path = 'orders.xlsx'

    st.title("Product Quantity Forecast")
    st.write("Select a product to view the quantity forecast for the next 30 days.")
    
    orders = load_data(path=path)
    product_models, products = create_models(orders)
    
    product_name = st.selectbox("Select a Product", products)
    
    if product_name:
        fig1, fig2, predictions , orders_sum  = predict_next_30_days(product_name, product_models,orders)
        if fig1 and fig2:
            st.write("Forecast for the next 30 days:")
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.write(predictions)

        st.write(f'Predicted next 30 days: {orders_sum}')

if __name__ == "__main__": 
    install_requirements()
    main()
