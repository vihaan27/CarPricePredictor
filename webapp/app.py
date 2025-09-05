from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

model = joblib.load("model.joblib")
brand_freq_map = joblib.load("brand_freq_map.joblib")

## Standardise brand names to uppercase
brand_freq_map = {k.upper(): v for k, v in brand_freq_map.items()}
brand_freq_map["OTHER"] = 0

# My App
app = Flask(__name__)

# One hot encoding maps
fuel_map = {"Diesel":  [1, 0, 0, 0],
            "Electric":[0, 1, 0, 0],
            "LPG":     [0, 0, 1, 0],
            "Petrol":  [0, 0, 0, 1],
            "CNG":     [0, 0, 0, 0]}

seller_map = {"Dealer": [0, 0],
              "Individual": [1, 0],
              "Trustmark Dealer": [0, 1]}

transmission_map = {"Manual": [1],
                    "Automatic": [0]}

@app.route("/")
def index():
    brands = sorted(brand_freq_map.keys())
    return render_template("index.html", brands = brands)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        ## Testing model
        # sample = np.array([[12,73000,20.36,1197,78.9,5,2982,0,0,0,0,0,1,1]])

        # in order
        vehicle_age = int(request.form["vehicle_age"])
        km_driven = int(request.form["km_driven"])
        mileage = float(request.form["mileage"])
        engine = int(request.form["engine"])
        max_power = float(request.form["max_power"])
        seats = int(request.form["seats"])
        brand = request.form["brand"] 
        seller = request.form["seller"]
        fuel = request.form["fuel"]
        transmission = request.form["transmission"]

        seller_encoded = seller_map[seller]
        fuel_encoded = fuel_map[fuel]
        transmission_encoded = transmission_map[transmission]
        brand_freq = brand_freq_map.get(brand, 0)

        sample = [vehicle_age, km_driven, mileage, engine, max_power,
                  seats, brand_freq] + seller_encoded + fuel_encoded + transmission_encoded
        
        columns = ["vehicle_age","km_driven","mileage","engine","max_power",
                   "seats","brand_freq","Individual","Trustmark Dealer",
                   "Diesel","Electric","LPG","Petrol","Manual"]


        input_df = pd.DataFrame([sample], columns=columns)
        prediction = model.predict(input_df)
        prediction_value = prediction.item()
        return render_template("predict.html", prediction_text=f"Predicted Price: ₹{int(prediction_value):,}")

if __name__ == "__main__":
    app.run(debug=True)