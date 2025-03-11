import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
data = pd.read_csv("Makaan_Properties_No_Duplicates.csv")

# Data Cleaning
data["Size"] = pd.to_numeric(data["Size"].str.replace(",", "").str.extract(r"(\d+)")[0], errors="coerce")
data["No_of_BHK"] = pd.to_numeric(data["No_of_BHK"].str.extract(r"(\d+)")[0], errors="coerce")
data["Price"] = pd.to_numeric(data["Price"].str.replace(",", "", regex=True), errors="coerce")

# Drop NaN values
data.dropna(subset=["Size", "No_of_BHK", "Price", "Latitude", "Longitude"], inplace=True)

# Feature Selection
features = ["Size", "No_of_BHK", "City_name", "Property_type"]
target = "Price"

numerical_cols = ["Size", "No_of_BHK"]
categorical_cols = ["City_name", "Property_type"]

# Preprocessor
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
encoder = OneHotEncoder(handle_unknown="ignore")
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), numerical_cols),
    ("cat", Pipeline([("imputer", cat_imputer), ("encoder", encoder)]), categorical_cols)
])

# Load or Train Model
try:
    knn_model = joblib.load("knn_model.pkl")
except:
    knn_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor(n_neighbors=5, metric="euclidean"))
    ])
    X = data[features]
    y = data[target]
    knn_model.fit(X, y)
    joblib.dump(knn_model, "knn_model.pkl")

# Function to format price
def format_price(price):
    if price >= 1_00_00_000:
        return f"₹{price / 1_00_00_000:.2f} Cr"
    elif price >= 1_00_000:
        return f"₹{price / 1_00_000:.2f} L"
    else:
        return f"₹{price:.2f}"

# Streamlit UI
st.title("🏠 Property Recommendation")

# Sidebar Filters
st.sidebar.header("🔍 Search Filters")
city = st.sidebar.selectbox("Select City", sorted(data["City_name"].dropna().unique()))
property_type = st.sidebar.selectbox("Property Type", sorted(data["Property_type"].dropna().unique()))
bhk = st.sidebar.selectbox("BHK", sorted(data["No_of_BHK"].dropna().unique()))
size = st.sidebar.number_input("Enter Size (sqft)", min_value=300, max_value=10000, step=100)
search_btn = st.sidebar.button("Search")

if search_btn:
    # AI-Based Price Prediction
    input_data = pd.DataFrame([[size, bhk, city, property_type]], columns=features)
    predicted_price = knn_model.predict(input_data)[0]

    # Filter recommended properties
    recommended_properties = data[
        (data["City_name"] == city) &
        (data["Property_type"] == property_type) &
        (data["No_of_BHK"] == bhk) &
        (data["Price"].between(predicted_price * 0.8, predicted_price * 1.2))
    ].copy()

    # Calculate future price after 5 years
    appreciation_rate = 0.05  # 5% annual increase
    years = 5
    recommended_properties["Future_Price"] = recommended_properties["Price"] * ((1 + appreciation_rate) ** years)

    # Format prices
    recommended_properties["Price"] = recommended_properties["Price"].apply(format_price)
    recommended_properties["Future_Price"] = recommended_properties["Future_Price"].apply(format_price)

    # Display Results
    if recommended_properties.empty:
        st.warning("No matching properties found.")
    else:
        st.markdown("### 🏡 Recommended Properties")
        for _, row in recommended_properties.iterrows():
            price_per_sqft = float(row["Price"].replace("₹", "").replace(" Cr", "e7").replace(" L", "e5")) / row["Size"] if row["Size"] > 0 else 0
            st.markdown(
                f"""
                <div style="background-color: #fff; border: 1px solid #ddd; border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); padding: 15px; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px; color: #333; font-size: 22px;">🏠 {row['Property_Name']}</h4>
                    <p style="margin: 5px 0; color: #555; font-size: 18px;"><b>📍 City:</b> {row['City_name']} | <b>🏡 Type:</b> {row['Property_type']} | <b>🛏 BHK:</b> {row['No_of_BHK']}</p>
                    <p style="margin: 5px 0; color: #555; font-size: 18px;"><b>📈 Price per sqft:</b> ₹{price_per_sqft:,.2f}</p>
                    <p style="margin: 5px 0; color: #555; font-size: 18px;"><b>📏 Size:</b> {row['Size']} sqft | <b>💰 Price:</b> {row['Price']}</p>
                    <p style="margin: 5px 0; color: #555; font-size: 18px;"><b>📌 Locality:</b> {row['Locality_Name']} | <b>📢 Status:</b> {row['Property_status']}</p>
                    <a href="https://www.google.com/maps?q={row['Latitude']},{row['Longitude']}" target="_blank">
                        <button style="background-color: #4285F4; color: white; border: none; padding: 8px 12px; font-size: 16px; border-radius: 5px; cursor: pointer;">
                            📍 View on Map
                        </button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
