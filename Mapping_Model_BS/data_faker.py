import pandas as pd
import numpy as np
from faker import Faker
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

fake = Faker()

# Number of records to generate
num_records = 200

# Generate dummy data for Retailer Table
retailer_data = {
    'Store_ID': [fake.uuid4() for _ in range(num_records)],
    'Location_Score': [random.uniform(1, 5) for _ in range(num_records)],
    'Store_Area': [random.randint(200, 5000) for _ in range(num_records)],
    'Store_Age': [random.randint(1, 20) for _ in range(num_records)],
    'Store_Review_Among_Sales_People': [random.uniform(0, 5) for _ in range(num_records)],
    'Store_Review': [random.uniform(0, 5) for _ in range(num_records)],
    'Overall_Sales_Data GST score': [random.randint(0, 5) for _ in range(num_records)],
    'Store_Format': [random.choice(['Walk-in', 'Counter Top']) for _ in range(num_records)],
    'Operating_Hours': [random.randint(8, 24) for _ in range(num_records)],
    'Sales_Data': [random.randint(30000, 500000) for _ in range(num_records)],
    'Daily_Foot_Traffic': [random.randint(100, 5000) for _ in range(num_records)],
    'Conversion_Rates': [random.uniform(0, 1) for _ in range(num_records)], # calculate with sales_data and daily_foot_traffic
    'Average_Bill_Value': [random.uniform(100, 5000) for _ in range(num_records)],
    'Population_Density': [random.randint(1, 5) for _ in range(num_records)],
    'Area_Population': [random.randint(1000, 2000000) for _ in range(num_records)],
    'Age_Distribution': [random.choice(['<18','18-25', '25-40', '40-60', '>60']) for _ in range(num_records)],
    'Average_Household_Income': [random.randint(200000, 20000000) for _ in range(num_records)],
    'Education_Levels': [random.choice(['High School', 'Bachelor', 'Master', 'PhD']) for _ in range(num_records)],
    'Occupation_Types': [random.choice(['Farmer','Self-Employeed','Salaried']) for _ in range(num_records)],
    'Purchase_Patterns': [random.choice(['Regular', 'Seasonal', 'One-time']) for _ in range(num_records)],
    'Customer_Feedback_and_Reviews': [random.uniform(0, 5) for _ in range(num_records)],
    'Loyalty_Program_Data': [random.choice([True, False]) for _ in range(num_records)],
    'Property_Prices_Data': [random.randint(50000, 1000000) for _ in range(num_records)],
    'RTO_Sales_Data': [random.randint(100, 10000) for _ in range(num_records)],
    'Presence_of_Businesses': [random.randint(10, 1000) for _ in range(num_records)],
    'Dominating business type': [random.choice(['Clothing', 'Restraunts', 'Retail', 'F&B', 'Healthcare']) for _ in range(num_records)],
    'Presence_of_Schools': [random.randint(1, 100) for _ in range(num_records)],
    'Local_Employment_Rates': [random.uniform(0.5, 10) for _ in range(num_records)],
    'Real_Estate_Score': [random.uniform(0.5, 10) for _ in range(num_records)], # LIG Flats, HIG Flats etc 
    'UPI_Growth': [random.uniform(0.5, 10) for _ in range(num_records)],
    'Marketing_Efforts': [random.uniform(0.5, 5) for _ in range(num_records)],
    'Promotional_Activity_Score': [random.uniform(0.5, 10) for _ in range(num_records)],
    'Store_Category': [random.choice(['Medical', 'Grocery', 'Daily Needs', 'Paan Store','Confectionary', 'Bakery']) for _ in range(num_records)],
    'Is_Modern_Trade': [random.choice([True, False]) for _ in range(num_records)],
}

retailer_df = pd.DataFrame(retailer_data)

num_records_brand = 6

# Generate dummy data for Brand Table
brand_data = {
    'Brand_ID': [fake.uuid4() for _ in range(num_records_brand)],
    'Funding_Amount': [random.randint(100000, 10000000) for _ in range(num_records_brand)],
    'Brand_Age': [random.randint(1, 10) for _ in range(num_records_brand)],
    'Revenue_Last_Month': [random.randint(100000, 10000000) for _ in range(num_records_brand)],
    'ARR': [random.randint(100000, 10000000) for _ in range(num_records_brand)],
    'Target_Audience': [random.choice(['Teens', 'Adults', 'Seniors', 'All']) for _ in range(num_records_brand)],
    'Headquarters_Location': [fake.city() for _ in range(num_records_brand)],
    'Number_of_Employees': [random.randint(10, 10000) for _ in range(num_records_brand)],
    'Product_Category': [random.choice(['Food', 'Beverages', 'Others']) for _ in range(num_records_brand)],
    'Product_Sub_Category': [random.choice(['RTE & Instant Food mixes', 'Pickles, Jams and Ketchups', 'Spices and masala', 'Snacks and namkeen', 'Chocolate, Buiscuits and Cookies', 'Bakery and confectionary products', 'Juices, Soups','Milk based beverages', 'Carbonated and Energy Drinks', 'Tea & Coffee', 'Ayurvedic & Herbal, Health supplements', 'Fresh, Dried & Preserved Fruits', 'Pet Food','Nutraceuticals'  ]) for _ in range(num_records_brand)],
    'Product_ID': [fake.uuid4() for _ in range(num_records_brand)],
    'Number_of_SKUs': [random.randint(1, 1000) for _ in range(num_records_brand)],
    'Bestselling_Products': [fake.word() for _ in range(num_records_brand)],
    'Average_Price_Point': [random.uniform(10, 1000) for _ in range(num_records_brand)],
    'Shelf_Life': [random.randint(1, 365) for _ in range(num_records_brand)],
    'Seasonality': [random.choice(['Summer', 'Winter', 'All Year']) for _ in range(num_records_brand)],
    'Product_Dimensions': [f"{random.randint(1, 100)}x{random.randint(1, 100)}x{random.randint(1, 100)}" for _ in range(num_records_brand)],
    'Special_Requirements': [random.choice(['None', 'Refrigeration', 'Fragile']) for _ in range(num_records_brand)],
    'Ideal_Store_Types': [random.choice(['Supermarkets', 'Convenience Stores', 'Specialty Stores']) for _ in range(num_records_brand)],
    'Minimum_Order_Quantity': [random.randint(1, 100) for _ in range(num_records_brand)],
    'Retailer_Profit_Margin': [random.uniform(5, 50) for _ in range(num_records_brand)],
    'Online_Sales_Availability': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Geographical_Coverage': [random.choice(['Local', 'Regional', 'National']) for _ in range(num_records_brand)],
    'Delivery_Time': [random.randint(1, 30) for _ in range(num_records_brand)],
    'Store_Category': [random.choice(['Dairy', 'Chemist', 'Grocery']) for _ in range(num_records_brand)],
    'Sampling_Interest': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Purchase_Patterns': [random.choice(['Regular', 'Seasonal', 'One-time','Impulse']) for _ in range(num_records_brand)],
    'Promotional_Budget': [random.randint(1000, 1000000) for _ in range(num_records_brand)],
    'Marketing_Materials_Availability': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Social_Media_Presence': [random.choice(['Low', 'Medium', 'High']) for _ in range(num_records_brand)],
    'Consumer_Engagement_Programs': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Brand_Collaborations': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Advertising_Channels': [random.choice(['Social Media', 'Print', 'Online Ads']) for _ in range(num_records_brand)],
    'Certifications_Awards': [random.choice(['None', 'Organic', 'Fair Trade']) for _ in range(num_records_brand)],
    'Company_Website': [fake.url() for _ in range(num_records_brand)],
    'Customer_Reviews_Ratings': [random.uniform(0, 5) for _ in range(num_records_brand)],
    'Compliance_with_Regulations': [random.choice([True, False]) for _ in range(num_records_brand)],
    'Product_Pipeline': [random.choice(['Yes', 'No']) for _ in range(num_records_brand)],
    'Customer_Support_Services': [random.choice(['Email', 'Phone', 'Online Chat']) for _ in range(num_records_brand)],
}

brand_df = pd.DataFrame(brand_data)
retailer_df.to_json('retailer_data.json', orient='records', lines=False, indent=4)
brand_df.to_json('brand_data.json', orient='records', lines=False, indent=4)

print("Dummy data saved to JSON files successfully.")
# Save the data to CSV

brand_to_store_mapping = {
    'Store_ID': [],
    'Brand_IDs': [],
}

for store_id in retailer_df['Store_ID']:
    '''
    num_brands = random.randint(1, 5)  # Random number of brands per store
    brand_ids = random.sample(list(brand_df['Brand_ID']), num_brands)
    brand_to_store_mapping['Brand_IDs'].append(brand_ids)
    brand_to_store_mapping['Store_ID'].append(store_id)
    '''

    for brand_id in brand_df['Brand_ID']:
        brand_to_store_mapping['Store_ID'].append(store_id)
        brand_to_store_mapping['Brand_IDs'].append(brand_id)
 

brand_to_store_df = pd.DataFrame(brand_to_store_mapping)

# Explode the Brand_IDs to create a one-to-many relationship
exploded_df = brand_to_store_df.explode('Brand_IDs') 

# Merge retailer and brand datasets based on mapping
combined_df = pd.merge(exploded_df, retailer_df, on='Store_ID')
combined_df = pd.merge(combined_df, brand_df, left_on='Brand_IDs', right_on='Brand_ID')

def calculate_mapping_score(row):
    return random.randint(0,100)

combined_df['mapping_score'] = combined_df.apply(lambda row: calculate_mapping_score(row), axis=1)

# Save the combined data to a JSON file with pretty formatting
combined_df.to_json('combined_data_new.json', orient='records', lines=False, indent=4)


'''

df = combined_df.copy()
df['Loyalty_Program_Data'] = df['Loyalty_Program_Data'].astype(int)
df['Is_Modern_Trade'] = df['Is_Modern_Trade'].astype(int)
df['Online_Sales_Availability'] = df['Online_Sales_Availability'].astype(int)
df['Sampling_Interest'] = df['Sampling_Interest'].astype(int)
df['Marketing_Materials_Availability'] = df['Marketing_Materials_Availability'].astype(int)
df['Consumer_Engagement_Programs'] = df['Consumer_Engagement_Programs'].astype(int)
df['Brand_Collaborations'] = df['Brand_Collaborations'].astype(int)
df['Compliance_with_Regulations'] = df['Compliance_with_Regulations'].astype(int)

# Drop identifier columns
df = df.drop(columns=['Store_ID', 'Brand_ID', 'Product_ID', 'Company_Website','Product_Dimensions'])

# Convert categorical columns using one-hot encoding
categorical_cols = ['Store_Format', 'Age_Distribution', 'Education_Levels', 'Occupation_Types', 'Purchase_Patterns_x',
                    'Dominating business type', 'Store_Category_x', 'Target_Audience', 'Headquarters_Location',
                    'Product_Category', 'Product_Sub_Category', 'Seasonality', 'Ideal_Store_Types',
                    'Store_Category_y', 'Purchase_Patterns_y', 'Social_Media_Presence', 'Advertising_Channels',
                    'Product_Pipeline', 'Customer_Support_Services', 'Bestselling_Products', 'Sampling_Interest', 'Certifications_Awards', 'Promotional_Budget','Special_Requirements', 'Geographical_Coverage']

df = pd.get_dummies(df, columns=categorical_cols)

# Separate features and target variable
X = df.drop(columns=['Brand_IDs'])
y = df['Brand_IDs']

# Encode target variable
y_encoded = y.astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(clf, 'brand_prediction_model.pkl')

'''