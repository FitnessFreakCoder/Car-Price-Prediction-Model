from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import os

app = Flask(__name__)

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.num_attribs = None
        self.cat_attribs = None
        
    def extract_number(self, x):
        """Extract numeric value from strings like '3990 cc', '250 km/h', etc."""
        if pd.isnull(x):
            return None
        x = str(x)
        nums = re.findall(r"[\d\.]+", x.replace(",", ""))
        if len(nums) == 0:
            return None
        if "–" in x or "-" in x:
            nums = [float(n) for n in nums]
            return sum(nums) / len(nums)
        return float(nums[0])
    
    def load_and_prepare_data(self, csv_path):
        """Load and clean the dataset"""
        try:
            data = pd.read_csv(csv_path, encoding="ISO-8859-1")
        except:
            data = pd.read_csv(csv_path, encoding="utf-8")
        
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Extract numeric values from string columns
        if "CC/Battery Capacity" in data.columns:
            data["CC"] = data["CC/Battery Capacity"].apply(self.extract_number)
        
        if "HorsePower" in data.columns:
            data["HorsePower"] = data["HorsePower"].apply(self.extract_number)
        
        if "Total Speed" in data.columns:
            data["TopSpeed"] = data["Total Speed"].apply(self.extract_number)
        
        if "Performance(0 - 100 )KM/H" in data.columns:
            data["Acceleration"] = data["Performance(0 - 100 )KM/H"].apply(self.extract_number)
        
        if "Torque" in data.columns:
            data["Torque"] = data["Torque"].apply(self.extract_number)
        
        if "Cars Prices" in data.columns:
            data["Price"] = data["Cars Prices"].apply(self.extract_number)
        
        # Drop original string columns
        cols_to_drop = ["CC/Battery Capacity", "Cars Prices", "Total Speed", "Performance(0 - 100 )KM/H"]
        data_clean = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)
        
        # Convert Seats to numeric
        data_clean["Seats"] = pd.to_numeric(data_clean["Seats"], errors="coerce")
        
        return data_clean
    
    def train_model(self, csv_path):
        """Train the car price prediction model"""
        # Load and prepare data
        data_clean = self.load_and_prepare_data(csv_path)
        
        # Create fuel type bins for stratification
        data_clean["Fuel_Code"] = data_clean["Fuel Types"].astype("category").cat.codes
        data_clean["Fuel_Binned"] = pd.cut(
            data_clean["Fuel_Code"],
            bins=[-1, 5, 10, 15, np.inf],
            labels=["Fuel_A", "Fuel_B", "Fuel_C", "Fuel_Other"]
        )
        
        # Stratified split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data_clean, data_clean["Fuel_Binned"]):
            strat_train_set = data_clean.loc[train_index].drop(["Fuel_Binned", "Fuel_Code"], axis=1)
            strat_test_set = data_clean.loc[test_index].drop(["Fuel_Binned", "Fuel_Code"], axis=1)

        strat_test_set.to_csv("test_data.csv", index=False)    
        
        # Prepare training data
        df = strat_train_set.copy()
        df = df.dropna(subset=["Price"])
        df_labels = df["Price"]
        df = df.drop('Price', axis=1)
        
        # Identify numeric and categorical columns
        self.num_attribs = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.cat_attribs = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Create preprocessing pipelines
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        
        cat_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_attribs),
            ("cat", cat_pipeline, self.cat_attribs),
        ])
        
        # Create full pipeline
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", LinearRegression())
        ])
        
        # Train the model
        self.model.fit(df, df_labels)
        
        # Calculate RMSE
        predictions = self.model.predict(df)
        rmse = np.sqrt(mean_squared_error(df_labels, predictions))
        print(f"Training RMSE: {rmse}")
        
        return self.model
    
    def predict_price(self, car_data):
        """Predict car price from input data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create DataFrame from input
        df_input = pd.DataFrame([car_data])
        
        # Make prediction
        prediction = self.model.predict(df_input)
        return prediction[0]
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'num_attribs': self.num_attribs,
            'cat_attribs': self.cat_attribs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.num_attribs = model_data['num_attribs']
        self.cat_attribs = model_data['cat_attribs']

# Initialize the predictor
predictor = CarPricePredictor()

# Train model on startup (you'll need to have the CSV file)
try:
    if os.path.exists("model.pkl"):
        predictor.load_model("model.pkl")
        print("Model loaded from file")
    elif os.path.exists("Cardataset.csv"):
        print("Training new model...")
        predictor.train_model("Cardataset.csv")
        predictor.save_model("model.pkl")
        print("Model trained and saved")
    else:
        print("Warning: No model or dataset found. Please upload Cardataset.csv")
except Exception as e:
    print(f"Error loading/training model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        car_data = {
            'Company Names': request.form['company'],
            'Cars Names': request.form['car_name'],
            'Engines': request.form['engine'],
            'HorsePower': float(request.form['horsepower']),
            'Fuel Types': request.form['fuel_type'],
            'Seats': int(request.form['seats']),
            'Torque': float(request.form['torque']),
            'CC': float(request.form['cc']),
            'TopSpeed': float(request.form['top_speed']),
            'Acceleration': float(request.form['acceleration'])
        }
        
        # Make prediction
        if predictor.model is None:
            return jsonify({'error': 'Model not available. Please ensure Cardataset.csv is present and restart the app.'})
        
        predicted_price = predictor.predict_price(car_data)
        
        return jsonify({
            'predicted_price': f"${predicted_price:,.2f}",
            'car_details': car_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/companies')
def get_companies():
    """API endpoint to get list of car companies"""
    companies = [
        'FERRARI', 'ROLLS ROYCE', 'Ford', 'MERCEDES', 'AUDI', 'LAMBORGHINI', 
        'Porsche', 'Chevrolet', 'Jaguar Land Rover', 'Cadillac', 'TOYOTA',
        'Kia', 'Nissan', 'BMW', 'Honda', 'Hyundai', 'Volkswagen', 'Tesla'
    ]
    return jsonify(companies)

@app.route('/api/fuel-types')
def get_fuel_types():
    """API endpoint to get list of fuel types"""
    fuel_types = [
        'Petrol', 'Electric', 'Hybrid', 'Diesel', 'Plug-in Hybrid',
        'Gas / Hybrid', 'Petrol/Diesel'
    ]
    return jsonify(fuel_types)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)