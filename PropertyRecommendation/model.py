from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load dataset
file_path = "Makaan_Properties_No_Duplicates.csv"
data = pd.read_csv(file_path)

# Data Cleaning
data["Size"] = data["Size"].str.replace(",", "").str.extract(r"(\d+)").astype(float)
data["No_of_BHK"] = data["No_of_BHK"].str.extract(r"(\d+)").astype(int)
data["Price"] = data["Price"].str.replace(",", "", regex=True).astype(float)

# Define columns
numerical_cols = ["Size", "No_of_BHK"]
categorical_cols = ["City_name", "Property_type"]

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

joblib.dump(encoders, "label_encoders.pkl")  # Save encoders

# Define features and target
features = ["Size", "No_of_BHK", "City_name", "Property_type"]
target = "Price"

# Scale numerical features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Train models
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])

        # Load encoders and scaler
        encoders = joblib.load("label_encoders.pkl")
        scaler = joblib.load("scaler.pkl")

        # Encode categorical features (handle unseen values)
        for col in categorical_cols:
            if col in input_df:
                if input_df[col][0] in encoders[col].classes_:
                    input_df[col] = encoders[col].transform(input_df[col])
                else:
                    input_df[col] = -1  # Assign -1 for unseen values

        # Ensure correct feature order
        input_df = input_df[features]

        # Scale numerical features
        input_df[features] = scaler.transform(input_df)

        # Load models
        rf_model = joblib.load("rf_model.pkl")
        gb_model = joblib.load("gb_model.pkl")

        # Predict and take an average
        rf_pred = rf_model.predict(input_df)[0]
        gb_pred = gb_model.predict(input_df)[0]
        final_pred = (rf_pred + gb_pred) / 2

        return jsonify({'predicted_price': final_pred})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)