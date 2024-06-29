import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import streamlit as st
import os

def install_requirements():
    os.system('pip3 install -r requirements.txt')

def load_data():
    orders_file_path = 'orders.xlsx'
    orders = pd.read_excel(orders_file_path)
    orders['Invoice Date'] = pd.to_datetime(orders['Invoice Date'])
    
    return orders

def preprocess_data(orders):
    # Handle missing values
    orders['Qty'].fillna(orders['Qty'].mean(), inplace=True)
    
    # Remove outliers
    q_low = orders['Qty'].quantile(0.01)
    q_high = orders['Qty'].quantile(0.99)
    orders = orders[(orders['Qty'] >= q_low) & (orders['Qty'] <= q_high)]
    
    return orders

def create_models(orders):
    product_models = {}
    products = orders['Product Name'].unique()
    
    for product in products:
        product_data = orders[orders['Product Name'] == product]
        if len(product_data) > 2:
            aggregated_data = product_data.groupby('Invoice Date')['Qty'].sum().reset_index()
            aggregated_data_prophet = aggregated_data.rename(columns={'Invoice Date': 'ds', 'Qty': 'y'})
            aggregated_data_prophet = aggregated_data_prophet[['ds', 'y']]

            # Prophet Model
            prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            prophet_model.add_country_holidays(country_name='US')
            prophet_model.fit(aggregated_data_prophet)
            
            # ARIMA Model
            arima_model = ARIMA(aggregated_data_prophet['y'])
            arima_model = arima_model.fit()
            
            product_models[product] = {
                'prophet': prophet_model,
                'arima': arima_model
            }
    
    return product_models, products

def predict_next_30_days(product_name, product_models, orders):
    if product_name in product_models:
        models = product_models[product_name]
        
        # Prophet Prediction
        prophet_model = models['prophet']
        future = prophet_model.make_future_dataframe(periods=30)
        prophet_forecast = prophet_model.predict(future)
        prophet_forecast['yhat'] = prophet_forecast['yhat'].astype(int)
        
        # ARIMA Prediction
        arima_model = models['arima']
        arima_forecast = arima_model.forecast(steps=30)
        arima_forecast = pd.DataFrame({
            'ds': future['ds'].tail(30).values,
            'yhat': arima_forecast.astype(int)
        })
        
        # Ensemble Prediction
        ensemble_forecast = (prophet_forecast[['ds', 'yhat']].tail(30)['yhat'] + arima_forecast['yhat']) / 2
        
        # Evaluation metrics
        '''
        historical = prophet_forecast[prophet_forecast['ds'] <= orders['Invoice Date'].max()]
        y_true = orders[orders['Product Name'] == product_name]['Qty']
        y_pred = historical['yhat'].values[:len(y_true)]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2%}")
        '''
        
        accuracy = prophet_model.plot_components(prophet_forecast)
        st.pyplot(accuracy)
        
        fig1 = prophet_model.plot(prophet_forecast)
        plt.title(f'Quantity Forecast for the Next 30 Days for {product_name}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        
        next_30_days = pd.DataFrame({
            'ds': future['ds'].tail(30).values,
            'yhat': ensemble_forecast.astype(int)
        })
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_30_days['ds'], next_30_days['yhat'], marker='o', linestyle='-', color='b')
        ax.set_title(f'Predicted Order Quantities for the Next 30 Days for {product_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Quantity')
        ax.grid(True)
        
        plt.xticks(rotation=45)
        
        total_orders = next_30_days['yhat'].sum()
        
        return fig1, fig2, next_30_days, total_orders
    else:
        st.write(f"Product '{product_name}' not found in the data.")
        return None, None, None

def main():
    st.title("Product Quantity Forecast")
    st.write("Select a product to view the quantity forecast for the next 30 days.")
    
    orders = load_data()
    orders = preprocess_data(orders)
    product_models, products = create_models(orders)
    
    product_name = st.selectbox("Select a Product", products)
    
    if product_name:
        fig1, fig2, predictions, total_orders = predict_next_30_days(product_name, product_models, orders)
        if fig1 and fig2:
            st.write("Forecast for the next 30 days:")
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.write(predictions)
            st.write(f"Total Orders: {total_orders}")

if __name__ == "__main__":
    #install_requirements()
    main()
