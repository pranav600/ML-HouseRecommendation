import streamlit as st
import pandas as pd

# Load the dataset
data = pd.read_csv("Makaan_Properties_No_Duplicates.csv")

# Function to clean and preprocess the dataset
def clean_size(size_str):
    return int(str(size_str).replace(",", "").replace(" sq ft", "").strip())

def clean_bhk(bhk_str):
    return int(str(bhk_str).split()[0])

def clean_price(price_str):
    price_str = str(price_str).strip()
    if "Cr" in price_str:
        return float(price_str.replace(" Cr", "")) * 10**7
    elif "Lakh" in price_str:
        return float(price_str.replace(" Lakh", "")) * 10**5
    return float(price_str.replace(",", ""))

def format_price(price):
    if price >= 10**7:
        return f"\u20B9{price / 10**7:.2f} Cr"
    elif price >= 10**5:
        return f"\u20B9{price / 10**5:.2f} Lakh"
    return f"\u20B9{price:,.0f}"

# Apply cleaning functions
data["Size"] = data["Size"].apply(clean_size)
data["No_of_BHK"] = data["No_of_BHK"].apply(clean_bhk)
data["Price"] = data["Price"].apply(clean_price)

# Calculate price per square foot
data["Price_per_sqft"] = data["Price"] / data["Size"]

st.set_page_config(page_title="Property Recommendation", layout="wide")

# Header
st.markdown("### Property Recommendation")

# UI Layout - Centered Input Bar
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    city = st.selectbox("City", sorted(data["City_name"].unique()))
    property_type = st.selectbox("Property Type", sorted(data["Property_type"].unique()))
    bhk = st.selectbox("BHK", sorted(data["No_of_BHK"].unique()))
    size_input = st.text_input("Enter Size (e.g., '2000' or '2000-3000' sqft):")
    property_name = st.text_input("Search by Property Name", placeholder="Enter property name...")
    search_btn = st.button("Search")

if search_btn:
    # Determine size range
    if "-" in size_input:
        try:
            size_min, size_max = map(int, size_input.split("-"))
        except ValueError:
            st.error("Invalid size range format. Please enter like '2000-3000'.")
            st.stop()
    else:
        try:
            exact_size = int(size_input)
            size_min = int(exact_size * 0.9)  # 10% lower
            size_max = int(exact_size * 1.1)  # 10% higher
        except ValueError:
            st.error("Invalid size format. Please enter a valid number.")
            st.stop()

    # Filter properties
    filtered_data = data[
        (data["City_name"] == city) &
        (data["Property_type"] == property_type) &
        (data["No_of_BHK"] == bhk) &
        (data["Size"].between(size_min, size_max))
    ]
    
    # Apply property name search if input is provided
    if property_name:
        filtered_data = filtered_data[filtered_data["Property_Name"].str.contains(property_name, case=False, na=False)]

    if filtered_data.empty:
        st.warning("No matching properties found.")
    else:
        st.markdown("### 🔍 Matching Properties")
        cols = st.columns(2)  # Two cards per row
        for index, (_, row) in enumerate(filtered_data.iterrows()):
            price_formatted = format_price(row['Price'])
            price_per_sqft_formatted = f"\u20B9{row['Price_per_sqft']:,.2f} per sqft"
            
            with cols[index % 2]:
                st.markdown(
                    f"""
                    <div style="background-color: #fff; border: 1px solid #ddd; border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); padding: 15px; margin-bottom: 15px;">
                        <h4 style="margin: 0 0 10px; color: #333;">🏠 {row['Property_Name']}</h4>
                        <p style="margin: 5px 0; color: #555;"><b>📍 City:</b> {row['City_name']} | <b>🏡 Type:</b> {row['Property_type']} | <b>🛏 BHK:</b> {row['No_of_BHK']}</p>
                        <p style="margin: 5px 0; color: #555"><b>📈 Price per sqft:</b> {price_per_sqft_formatted}</p>
                        <p style="margin: 5px 0; color: #555"><b>📏 Size:</b> {row['Size']} sqft | <b>💰 Price:</b> {price_formatted}</p>
                        <p style="margin: 5px 0; color: #555"><b>📌 Locality:</b> {row['Locality_Name']} | <b>📢 Status:</b> {row['Property_status']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
