import streamlit as st
import pandas as pd
import numpy as np 
import joblib
from sklearn.ensemble import GradientBoostingClassifier 
import sklearn.ensemble._gb_losses 
import random

# Load the trained model
model = joblib.load('brand_prediction_model.pkl')

# Function to preprocess user input and make predictions
def predict_store_mapping(brand_details):
    model = {}
    # Process brand details and convert to DataFrame format expected by the model
    df_input = pd.DataFrame(brand_details, index=[0])
    
    # Preprocess input (similar to training data preprocessing steps)
    categorical_cols = [ 'Headquarters_Location', 'Product_Category', 'Product_Sub_Category',
                        'Ideal_Store_Types', 'Sampling_Interest', 'Purchase_Patterns', 'Social_Media_Presence',
                        'Advertising_Channels', 'Certifications_Awards', 'Special_Requirements', 'Geographical_Coverage',
                         'Brand_Collaborations']

    # One-hot encode categorical columns
    df_input_encoded = pd.get_dummies(df_input, columns=categorical_cols)

    # Ensure all expected features are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0  # Set default value or handle missing columns appropriately

    # Perform prediction
    prediction = model.predict(df_input_encoded)

    return int(prediction[0])

# Streamlit app
def main():
    st.title('Brand Store Mapping Prediction')
    
    # Example input fields for brand details
    funding_amount = st.number_input('Funding Amount', min_value=0, step=1)
    brand_age = st.number_input('Brand Age', min_value=0, step=1)
    revenue_last_month = st.number_input('Revenue Last Month', min_value=0, step=1)
    arr = st.number_input('ARR (Predicted)', min_value=0, step=1)
    number_of_employees = st.number_input('Number of Employees', min_value=0, step=1)
    headquarters_location = st.selectbox('Headquarters Location',['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata','USA'])
    product_category = st.selectbox('Product Category',['Food', 'Beverages', 'Others'])
    product_sub_category = st.selectbox('Product Sub Category',['RTE & Instant Food mixes', 'Pickles, Jams and Ketchups', 'Spices and masala', 'Snacks and namkeen', 'Chocolate, Buiscuits and Cookies', 'Bakery and confectionary products', 'Juices, Soups','Milk based beverages', 'Carbonated and Energy Drinks', 'Tea & Coffee', 'Ayurvedic & Herbal, Health supplements', 'Fresh, Dried & Preserved Fruits', 'Pet Food','Nutraceuticals'  ])
    average_price_point = st.number_input('Average Price Point', min_value=0.0, step=0.01)
    shelf_life = st.number_input('Shelf Life', min_value=6, step=1)
    product_dimensions = st.text_input('Product Dimensions')
    special_requirements = st.selectbox('Special Requirements',['Yes', 'No'])
    ideal_store_types = st.selectbox('Ideal Store Types',['Medical', 'Grocery', 'Daily Needs', 'Paan Store','Confectionary', 'Bakery'])
    minimum_order_quantity = st.number_input('Minimum Order Quantity', min_value=10000, step=5000)
    retailer_profit_margin = st.number_input('Retailer Profit Margin', min_value=10.0, step=1.00)
    online_sales_availability = st.checkbox('Online Sales Availability')
    geographical_coverage = st.selectbox('Geographical Coverage',['Regional', 'National', 'International'])
    delivery_time = st.number_input('Delivery Time', min_value=1, step=1)
    sampling_interest = st.checkbox('Sampling Interest')
    purchase_patterns = st.selectbox('Purchase Patterns',['Regular', 'Seasonal', 'One-time','Impulse'])
    promotional_budget = st.number_input('Promotional Budget', min_value=0, step=1)
    marketing_materials_availability = st.checkbox('Marketing Materials Availability')
    social_media_presence = st.checkbox('Social Media Presence')
    consumer_engagement_programs = st.checkbox('Consumer Engagement Programs')
    brand_collaborations = st.checkbox('Brand Collaborations')
    advertising_channels = st.selectbox('Advertising Channels',['Social Media', 'Print', 'Online Ads'])
    certifications_awards = st.selectbox('Certifications & Awards', ['FSSAI', 'ISO', 'AGMARK', 'USFDA','None'])
    customer_reviews_ratings = st.number_input('Customer Reviews & Ratings', min_value=0.0, max_value=5.0, step=0.01)
    compliance_with_regulations = st.checkbox('Compliance with Regulations')
    
    # Example button to trigger prediction
    if st.button('Predict'):
        # Prepare input dictionary for prediction
        brand_details = {
            'Funding_Amount': funding_amount,
            'Brand_Age': brand_age,
            'Revenue_Last_Month': revenue_last_month,
            'ARR': arr,
            'Number_of_Employees': number_of_employees,
            'Headquarters_Location': headquarters_location,
            'Product_Category': product_category,
            'Product_Sub_Category': product_sub_category,
            'Average_Price_Point': average_price_point,
            'Shelf_Life in months': shelf_life, 
            'Special_Requirements': special_requirements,
            'Ideal_Store_Types': ideal_store_types,
            'Minimum_Order_Quantity': minimum_order_quantity,
            'Retailer_Profit_Margin': retailer_profit_margin,
            'Online_Sales_Availability': online_sales_availability,
            'Geographical_Coverage': geographical_coverage,
            'Delivery_Time': delivery_time,
            'Sampling_Interest': sampling_interest,
            'Purchase_Patterns': purchase_patterns,
            'Promotional_Budget': promotional_budget,
            'Marketing_Materials_Availability': marketing_materials_availability,
            'Social_Media_Presence': social_media_presence,
            'Consumer_Engagement_Programs': consumer_engagement_programs,
            'Brand_Collaborations': brand_collaborations,
            'Advertising_Channels': advertising_channels,
            'Certifications_Awards': certifications_awards,
            'Customer_Reviews_Ratings': customer_reviews_ratings,
            'Compliance_with_Regulations': compliance_with_regulations,
        }

         #  'Product_Dimensions': product_dimensions,# '23x45x65'
        dimensions = product_dimensions.split('x')
        brand_details['length'] = dimensions[0]
        brand_details['width'] = dimensions[1]
        brand_details['height'] = dimensions[2]

        print(brand_details) 
        
        # Call prediction function
        prediction = predict_store_mapping(brand_details)
        
        # Display prediction result
        st.success(f'The brand can be mapped to approximately {brand_details} stores.')

if __name__ == '__main__':
    main()


brand_details = {
            'Funding_Amount': 12000000,
            'Brand_Age': 24,
            'Revenue_Last_Month': 40000,
            'ARR': 10000,
            'Number_of_Employees': 1000,
            'Headquarters_Location': 23,
            'Product_Category': 3,
            'Product_Sub_Category': 10,
            'Average_Price_Point': 243,
            'Shelf_Life': 3, 
            'Special_Requirements': 0,
            'Ideal_Store_Types': 0,
            'Minimum_Order_Quantity': 12,
            'Retailer_Profit_Margin': 33.3,
            'Online_Sales_Availability': 1,
            'Geographical_Coverage': 2,
            'Delivery_Time': 17,
            'Purchase_Patterns_x': 0,
            'Sampling_Interest': 0,
            'Purchase_Patterns_y': 0,
            'Promotional_Budget': 1245,
            'Marketing_Materials_Availability': 1,
            'Social_Media_Presence': 1,
            'Consumer_Engagement_Programs': 1,
            'Brand_Collaborations': 1,
            'Advertising_Channels': 1,
            'Certifications_Awards': 1,
            'Customer_Reviews_Ratings': 1.7,
            'Compliance_with_Regulations': 1,
            'length' : 32,
            'width' : 14,
            'height' : 46
        }