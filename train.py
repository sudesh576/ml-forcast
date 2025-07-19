import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from prophet import Prophet

# # Load the dataset
# file_path = "your_file.csv"  # Replace with actual file path
df = pd.read_csv('ecom_sales_dnd.csv')

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Aggregate sales by month and product
df['Month'] = df['Timestamp'].dt.to_period('M').dt.to_timestamp()
monthly_sales = df.groupby(['Month', 'Product Name'])['Total Sales Value'].sum().reset_index()

# Rename columns for Prophet
monthly_sales.rename(columns={'Month': 'ds', 'Total Sales Value': 'y', 'Product Name': 'product'}, inplace=True)

# Train the one-hot encoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_products = encoder.fit_transform(monthly_sales[['product']])
encoded_df = pd.DataFrame(encoded_products, columns=encoder.get_feature_names_out(['product']))

# Merge encoded features
monthly_sales = pd.concat([monthly_sales, encoded_df], axis=1)

# Save the encoder for future use
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Train the Prophet model
model = Prophet()

# Add categorical regressors
for col in encoded_df.columns:
    model.add_regressor(col)

# Fit the model
model.fit(monthly_sales)

# Save the trained Prophet model
with open("monthly_sales_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and Encoder saved successfully!")