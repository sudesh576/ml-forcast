import pandas as pd
import pickle

# Load the trained Prophet model
with open("monthly_sales_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Load the one-hot encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Function to make predictions
def predict_sales(product_name, months_ahead):
    # Get the last date from the trained dataset
    last_date = loaded_model.history['ds'].max()

    # Generate future dates (Monthly)
    future_dates = pd.date_range(start=last_date, periods=months_ahead, freq='M')
    future_df = pd.DataFrame({'ds': future_dates})

    # One-hot encode product for prediction
    product_columns = encoder.get_feature_names_out(['product'])
#     print(f"product_columns:{product_columns}")
    product_encoded = pd.DataFrame(0, index=[0], columns=product_columns)
#     print(f"product_encoded:{product_encoded}")

    # Set the specific product column to 1
    product_col = f'product_{product_name}'
    if product_col in product_encoded.columns:
        product_encoded.loc[0, product_col] = 1
    else:
        raise ValueError(f"Product '{product_name}' not found in trained model!")

    #  FIX: Properly repeat the row for all future months
    product_encoded = pd.concat([product_encoded.iloc[0:1]] * len(future_df), ignore_index=True)

    # Merge future dates with encoded product data
    future_encoded = pd.concat([future_df, product_encoded], axis=1)

    # Make predictions
    forecast = loaded_model.predict(future_encoded)
    
    return forecast[['ds', 'yhat']]

# Example usage
user_input = ["Monitor", 3]  # Forecast for "Tablet" for 3 months
predicted_sales = predict_sales(user_input[0], user_input[1])
print(predicted_sales)
