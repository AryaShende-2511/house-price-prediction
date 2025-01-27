# import numpy as np
# import joblib
# import pandas as pd
# import streamlit as st
# from sklearn.preprocessing import LabelEncoder

# # Add custom CSS for background image
# def add_bg_image(image_file):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url({image_file});
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Call the function with your image
# add_bg_image("https://www.shutterstock.com/image-illustration/white-empty-room-scandinavian-interior-260nw-1908939706.jpg")  # Replace with the path to your image or a URL

# # Load necessary components
# df = pd.read_csv("data.csv")  # Load your dataset
# model = joblib.load("house_price_model.pkl")  # Load your trained model

# # Ensure correct data types
# df['price'] = pd.to_numeric(df['price'], errors='coerce')
# df['bedrooms'] = df['bedrooms'].astype(int)
# df['bathrooms'] = df['bathrooms'].astype(int)
# df['sqft_living'] = pd.to_numeric(df['sqft_living'], errors='coerce')
# df['sqft_lot'] = pd.to_numeric(df['sqft_lot'], errors='coerce')
# df['condition'] = df['condition'].astype(int)
# df['yr_built'] = df['yr_built'].astype(int)
# df['city'] = df['city'].astype(str)

# # Initialize LabelEncoder for 'city'
# label_encoders = {'city': LabelEncoder()}
# label_encoders['city'].fit(df['city'])  # Fit LabelEncoder for 'city'

# # Function to encode input values
# def encode_input(value, column):
#     try:
#         return label_encoders[column].transform([value])[0]
#     except ValueError:
#         st.warning(f"Warning: The value '{value}' for {column} is unseen. Using default encoding (0).")
#         return 0  # Default encoding for unseen values

# # Streamlit UI
# st.title("🏠 House Price Prediction App")
# st.markdown("### Enter the house details below to predict its price:")

# # Collect user inputs
# bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=10, value=3, step=1)
# bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=10, value=2, step=1)
# city = st.text_input("City:", placeholder="Enter city name here...")

# # Default values for other features
# st.markdown("### Additional Details (Default values assumed):")
# sqft_living = st.number_input("Square feet of living area:", min_value=500.0, value=2000.0, step=100.0)
# sqft_lot = st.number_input("Square feet of lot area:", min_value=500.0, value=5000.0, step=100.0)
# floors = st.number_input("Number of floors:", min_value=1, max_value=4, value=2, step=1)
# age_of_property = st.number_input("Age of the property (in years):", min_value=0, value=10, step=1)
# condition = st.number_input("Condition of the house (1-5):", min_value=1, max_value=5, value=3, step=1)
# year_built = st.number_input("Year the house was built:", min_value=1800, max_value=2024, value=2000, step=1)
# sqft_above = st.number_input("Square feet above ground:", min_value=500.0, value=1500.0, step=100.0)
# yr_renovated = st.number_input("Year the house was renovated (0 if not renovated):", min_value=0, value=0, step=1)

# # Price range inputs
# st.markdown("### Filter Houses by Price Range:")
# min_price_value = st.number_input("Minimum price:", min_value=0.0, value=100000.0, step=10000.0)
# max_price_value = st.number_input("Maximum price:", min_value=0.0, value=1000000.0, step=10000.0)

# # Debugging: Print the selected filters
# st.write(f"Filtering houses with the following parameters:")
# st.write(f"Price Range: ${min_price_value:,.2f} - ${max_price_value:,.2f}")
# st.write(f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}")
# st.write(f"City: {city if city else 'All cities'}")

# # Filter the dataframe based on user inputs
# available_houses = df[
#     (df['price'] >= min_price_value) & 
#     (df['price'] <= max_price_value) & 
#     (df['bedrooms'] == bedrooms) & 
#     (df['bathrooms'] == bathrooms)
# ]

# # Optionally, filter by city if provided
# if city.strip():
#     available_houses = available_houses[available_houses['city'].str.lower() == city.strip().lower()]

# # Debugging: Check how many houses remain after applying filters
# st.write(f"Remaining houses after filtering: {len(available_houses)}")

# # Add other conditions like sqft_living, condition, etc., if needed
# if sqft_living:
#     available_houses = available_houses[available_houses['sqft_living'].between(sqft_living - 500, sqft_living + 500)]
# if sqft_lot:
#     available_houses = available_houses[available_houses['sqft_lot'].between(sqft_lot - 1000, sqft_lot + 1000)]
# if condition:
#     available_houses = available_houses[available_houses['condition'] == condition]
# if year_built:
#     available_houses = available_houses[available_houses['yr_built'] == year_built]

# # Final check: Display a warning if no houses match
# if available_houses.empty:
#     st.warning("No houses found matching the criteria.")

# # Display results
# st.write(f"🏘 Number of houses available: *{len(available_houses)}*")

# # Optional: Display filtered house details
# if st.checkbox("Show filtered house details"):
#     st.dataframe(available_houses)

import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Add custom CSS for background image
def add_bg_image(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_file});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image
add_bg_image("https://www.shutterstock.com/image-illustration/white-empty-room-scandinavian-interior-260nw-1908939706.jpg")  # Replace with the path to your image or a URL

# Load necessary components
df = pd.read_csv("data.csv")  # Load your dataset
model = joblib.load("house_price_model.pkl")  # Load your trained model

# Ensure correct data types
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['bedrooms'] = df['bedrooms'].astype(int)
df['bathrooms'] = df['bathrooms'].astype(int)
df['sqft_living'] = pd.to_numeric(df['sqft_living'], errors='coerce')
df['sqft_lot'] = pd.to_numeric(df['sqft_lot'], errors='coerce')
df['condition'] = df['condition'].astype(int)
df['yr_built'] = df['yr_built'].astype(int)
df['city'] = df['city'].astype(str)

# Initialize LabelEncoder for 'city'
label_encoders = {'city': LabelEncoder()}
label_encoders['city'].fit(df['city'])  # Fit LabelEncoder for 'city'

# Function to encode input values
def encode_input(value, column):
    try:
        return label_encoders[column].transform([value])[0]
    except ValueError:
        st.warning(f"Warning: The value '{value}' for {column} is unseen. Using default encoding (0).")
        return 0  # Default encoding for unseen values

# Streamlit UI
st.title("🏠 House Price Prediction App")
st.markdown("### Enter the house details below to predict its price:")

# Collect user inputs
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=10, value=2, step=1)

# Replace city text input with a selectbox
city = st.selectbox("City:", options=["All"] + sorted(df['city'].unique()), index=0)

# Default values for other features
st.markdown("### Additional Details (Default values assumed):")
sqft_living = st.number_input("Square feet of living area:", min_value=500.0, value=2000.0, step=100.0)
sqft_lot = st.number_input("Square feet of lot area:", min_value=500.0, value=5000.0, step=100.0)
floors = st.number_input("Number of floors:", min_value=1, max_value=4, value=2, step=1)
age_of_property = st.number_input("Age of the property (in years):", min_value=0, value=10, step=1)
condition = st.number_input("Condition of the house (1-5):", min_value=1, max_value=5, value=3, step=1)
year_built = st.number_input("Year the house was built:", min_value=1800, max_value=2024, value=2000, step=1)
sqft_above = st.number_input("Square feet above ground:", min_value=500.0, value=1500.0, step=100.0)
yr_renovated = st.number_input("Year the house was renovated (0 if not renovated):", min_value=0, value=0, step=1)

# Price range inputs
st.markdown("### Filter Houses by Price Range:")
min_price_value = st.number_input("Minimum price:", min_value=0.0, value=100000.0, step=10000.0)
max_price_value = st.number_input("Maximum price:", min_value=0.0, value=1000000.0, step=10000.0)

# Debugging: Print the selected filters
st.write(f"Filtering houses with the following parameters:")
st.write(f"Price Range: ${min_price_value:,.2f} - ${max_price_value:,.2f}")
st.write(f"Bedrooms: {bedrooms}, Bathrooms: {bathrooms}")
st.write(f"City: {city if city != 'All' else 'All cities'}")

# Filter the dataframe based on user inputs
available_houses = df[
    (df['price'] >= min_price_value) & 
    (df['price'] <= max_price_value) & 
    (df['bedrooms'] == bedrooms) & 
    (df['bathrooms'] == bathrooms)
]

# Optionally, filter by city if not "All"
if city != "All":
    available_houses = available_houses[available_houses['city'].str.lower() == city.lower()]

# Debugging: Check how many houses remain after applying filters
st.write(f"Remaining houses after filtering: {len(available_houses)}")

# Add other conditions like sqft_living, condition, etc., if needed
if sqft_living:
    available_houses = available_houses[available_houses['sqft_living'].between(sqft_living - 500, sqft_living + 500)]
if sqft_lot:
    available_houses = available_houses[available_houses['sqft_lot'].between(sqft_lot - 1000, sqft_lot + 1000)]
if condition:
    available_houses = available_houses[available_houses['condition'] == condition]
if year_built:
    available_houses = available_houses[available_houses['yr_built'] == year_built]

# Final check: Display a warning if no houses match
if available_houses.empty:
    st.warning("No houses found matching the criteria.")

# Display results
st.write(f"🏧 Number of houses available: *{len(available_houses)}*")

# Optional: Display filtered house details
if st.checkbox("Show filtered house details"):
    st.dataframe(available_houses)