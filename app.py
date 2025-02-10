from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load and prepare SEPHER data
def prepare_model():
    try:
        df = pd.read_csv('sepher_data.csv')
        
        # Select most recent year's socioeconomic features (2018)
        features = [
            'EP_POV_2018',      # Poverty rate
            'EP_UNEMP_2018',    # Unemployment rate
            'E_PCI_2018',       # Per capita income
            'EP_NOHSDP_2018',   # No high school diploma
            'EP_AGE65_2018',    # Population over 65
            'EP_MINRTY_2018',   # Minority population
            'EP_NOVEH_2018',    # No vehicle access
            'EP_MUNIT_2018',    # Multi-unit structures
            'BUILDVALUE',       # Building value
            'AREA'             # Area
        ]
        
        # Remove rows with missing values
        df_clean = df[features + ['RISK_SCORE']].dropna()
        
        # Store risk score statistics for later use
        risk_stats = {
            'min': df_clean['RISK_SCORE'].min(),
            'max': df_clean['RISK_SCORE'].max(),
            '25th': df_clean['RISK_SCORE'].quantile(0.25),
            '75th': df_clean['RISK_SCORE'].quantile(0.75)
        }
        
        # Print risk score distribution for debugging
        print("Risk Score Statistics:")
        print(f"Min: {risk_stats['min']}")
        print(f"25th percentile: {risk_stats['25th']}")
        print(f"75th percentile: {risk_stats['75th']}")
        print(f"Max: {risk_stats['max']}")
        
        # Split features and target
        X = df_clean[features]
        y = df_clean['RISK_SCORE']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create polynomial features (degree=2 for simplicity)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_scaled)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        return poly, model, scaler, features, risk_stats
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Add route for home page
@app.route('/')
def home():
    return render_template('index.html')

poly, model, scaler, features, risk_stats = prepare_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input array from request data
        input_data = np.array([[
            float(data['poverty']),
            float(data['unemployment']),
            float(data['income']),
            float(data['education']),
            float(data['elderly']),
            float(data['minority']),
            float(data['transport']),
            float(data['housing']),
            float(data['buildvalue']),
            float(data['area'])
        ]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Transform input data
        input_poly = poly.transform(input_scaled)
        
        # Make prediction
        prediction = model.predict(input_poly)[0]
        
        # Get risk rating based on quartiles from the data
        if prediction >= risk_stats['75th']:
            risk_rating = "High"
        elif prediction <= risk_stats['25th']:
            risk_rating = "Low"
        else:
            risk_rating = "Medium"
        
        return jsonify({
            'risk_score': round(prediction, 2),
            'risk_rating': risk_rating,
            'score_range': {
                'min': round(risk_stats['min'], 2),
                'max': round(risk_stats['max'], 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 