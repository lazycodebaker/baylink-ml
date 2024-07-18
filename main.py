# %%
import pandas as pd 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os 

data_path = os.path.abspath('brand_prediction_model_latest.pkl')

model = joblib.load(data_path)
data = pd.read_excel("latest.xlsx")
mapping_score = data['mapping_score']
data.drop('Unnamed: 0',axis=1,inplace=True)
data.drop('mapping_score',axis=1,inplace=True)
data.dropna(inplace=True)


def install_requirements():
    os.system('pip3 install -r requirements.txt')


Headquarters_Location = {
 'Delhi'  : 5, 
 'Mumbai' : 0,
 'Bangalore': 4,
 'Chennai' : 3,
 'Kolkata' : 1,
 'USA' : 2
}

Product_Sub_Category = {
 'RTE & Instant Food mixes' : 1 , 
 'Pickles, Jams and Ketchups' : 3 , 
 'Spices and masala': 0,
 'Snacks and namkeen' : 2,
 'Chocolate, Buiscuits and Cookie' : 4,
 'Bakery and confectionary products': 5,
 'Juices, Soups' : 6,
 'Milk based beverages' : 7,
 'Carbonated and Energy Drinks' : 8,
 'Tea & Coffee' : 9,
 'Ayurvedic & Herbal' : 11,
 'Health supplements' : 10,
 'Fresh, Dried & Preserved Fruits' : 13,
 'Pet Food' : 14 ,
 'Nutraceuticals' : 12
}

Product_Category = {
    'Food' : 1 ,
    'Beverages' : 0,
    'Others' : 2
}

Special_Requirements = {
    'Yes' : 0,
    'No' : 2,
    'None' : 1
}

Ideal_Store_Types = {
    'Medical' : 0 , 
    'Grocery' : 1,
    'Daily Needs' : 2,
    'Paan Store' : 3,
    'Confectionary' : 4,
    'Bakery' : 5
}

Online_Sales_Availability = {
    'No' : 0,
    'Yes' : 1
}

Geographical_Coverage = {
    'Regional' : 0,
    'National' :  1, 
    'International' :  2
}

Purchase_Patterns_x = {
    'Regular' : 1 , 
    'Seasonal' : 2 ,
    'One-time' : 0,
    'Impulse' : 3
}

Sampling_Interest = {
    'No' : 0,
    'Yes' : 1
}

Purchase_Patterns_y = {
    'Impulse'  : 0,
    'One-time'  : 2 ,
    'Regular' : 1 ,
    'Seasonal' : 3
}

Marketing_Materials_Availability = {
    'No' : 0,
    'Yes' : 1
}

Brand_Collaborations = {
    'No' : 0,
    'Yes' : 1

}
 
Advertising_Channels = { # social_media_presence
    'No' : 0,
    'Yes' : 1,
    'Social Media' : 2,
    'Print' : 3,
    'Online Ads' : 4
}
 
 #  ['FSSAI', 'ISO', 'AGMARK', 'USFDA','None'])
Certifications_Awards = { # consumer_engagement_programs
    'No' : 0,
    'Yes' : 1,
    'FSSAI' : 2,
    'ISO' : 3,
    'AGMARK' : 4,
    'USFDA' : 5,
    'None' : 6
}

Compliance_with_Regulations = {
    'No' : 0,
    'Yes' : 1
}

# %%
import streamlit as st

def main():
    st.title('Brand Store Mapping Prediction')
    
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
    target_customer_age  = st.selectbox('Target Customer Age',['18-25', '26-35', '36-45', '46-55', '56+', '65+','all'])
    product_dimensions = st.text_input('Product Dimensions -- L x W x H')
    special_requirements = st.selectbox('Special Requirements',['Yes', 'No'])
    ideal_store_types = st.selectbox('Ideal Store Types',['Medical', 'Grocery', 'Daily Needs', 'Paan Store','Confectionary', 'Bakery'])
    minimum_order_quantity = st.number_input('Minimum Order Quantity', min_value=10000, step=5000)
    retailer_profit_margin = st.number_input('Retailer Profit Margin', min_value=10.0, step=1.00)
    online_sales_availability = st.selectbox('Online Sales Availability',['Yes', 'No'])
    geographical_coverage = st.selectbox('Geographical Coverage',['Regional', 'National', 'International'])
    delivery_time = st.number_input('Delivery Time', min_value=1, step=1)
    sampling_interest = st.selectbox('Sampling Interest',['Yes', 'No'])
    purchase_patterns = st.selectbox('Purchase Patterns',['Regular', 'Seasonal', 'One-time','Impulse'])
    promotional_budget = st.number_input('Promotional Budget', min_value=0, step=1)
    marketing_materials_availability = st.selectbox('Marketing Materials Availability',['Yes', 'No'])
    social_media_presence = st.selectbox('Social Media Presence',['Yes', 'No'])
    consumer_engagement_programs = st.selectbox('Consumer Engagement Programs',['Yes', 'No'])
    brand_collaborations = st.selectbox('Brand Collaborations',['Yes', 'No'])
    advertising_channels = st.selectbox('Advertising Channels',['Social Media', 'Print', 'Online Ads'])
    certifications_awards = st.selectbox('Certifications & Awards', ['FSSAI', 'ISO', 'AGMARK', 'USFDA','None'])
    customer_reviews_ratings = st.number_input('Customer Reviews & Ratings', min_value=0.0, max_value=5.0, step=0.01)
    compliance_with_regulations = st.selectbox('Compliance with Regulations',['Yes', 'No'])

    btn = st.button('Predict')

    if btn:
        brand_details = {
            'Funding_Amount': funding_amount,
            'Brand_Age': brand_age,
            'Revenue_Last_Month': revenue_last_month,
            'ARR': arr,
            'Number_of_Employees': number_of_employees,
            'Headquarters_Location': Headquarters_Location[headquarters_location],
            'Product_Category': Product_Category[product_category],
            'Product_Sub_Category': Product_Sub_Category[product_sub_category],
            'Average_Price_Point': average_price_point,
            'Shelf_Life': shelf_life,
            'Special_Requirements': Special_Requirements[special_requirements],
            'Ideal_Store_Types': Ideal_Store_Types[ideal_store_types],
            'Minimum_Order_Quantity': minimum_order_quantity,
            'Retailer_Profit_Margin': retailer_profit_margin,
            'Online_Sales_Availability': Online_Sales_Availability[online_sales_availability],
            'Geographical_Coverage': Geographical_Coverage[geographical_coverage],
            'Delivery_Time': delivery_time,
            'Sampling_Interest': Sampling_Interest[sampling_interest],
            'Purchase_Patterns_x': Purchase_Patterns_x[purchase_patterns],
            'Promotional_Budget': promotional_budget,
            'Marketing_Materials_Availability': Marketing_Materials_Availability[marketing_materials_availability],
            'Social_Media_Presence': Advertising_Channels[social_media_presence],
            'Consumer_Engagement_Programs': Certifications_Awards[consumer_engagement_programs],
            'Brand_Collaborations': Brand_Collaborations[brand_collaborations],
            'Advertising_Channels': Advertising_Channels[advertising_channels],
            'Certifications_Awards': Certifications_Awards[certifications_awards],
            'Customer_Reviews_Ratings': customer_reviews_ratings,
            'Compliance_with_Regulations': Compliance_with_Regulations[compliance_with_regulations]
        }

        length , width , height = product_dimensions.split('x')
        brand_details['length'] = float(length)
        brand_details['width'] = float(width)
        brand_details['height'] = float(height)
        
        brand_cols = brand_details.keys()

        retailer_col = []

        for col in data.columns:
            if col not in brand_cols:
                retailer_col.append(col)

        retailer_threshold_data = data[:200][retailer_col]

        brand_details_df = pd.DataFrame(brand_details, index=[0])

        brand_details_df_replicated = pd.concat([brand_details_df]*len(retailer_threshold_data), ignore_index=True)

        input_brand = pd.concat([brand_details_df_replicated, retailer_threshold_data], axis=1)

        input_brand = input_brand[data.columns]
        
        y_input_data = model.predict(input_brand)

        retailer_threshold_data['mapping_score'] = mapping_score

        s = 0
        for i in y_input_data:
            if i > 60:
                s +=1

        # give me all unique ids of stores only whose mapping_score is greater than 60
        store_ids = retailer_threshold_data[retailer_threshold_data['mapping_score'] > 60]['Store_ID'].unique()
        
        st.write(f"Your brand can be placed in stores")
        st.write(store_ids)

        st.success(f"Number of stores your brand can be placed are : {s}")

if __name__ == '__main__':
    # install_requirements()
    main()
