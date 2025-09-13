import os
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model#
loaded_model = pickle.load(open('Coffe_sales.sav', 'rb'))

# --- Step 1: Ensure scaler.sav exists ---
if not os.path.exists("scaler.sav") or os.path.getsize("scaler.sav") == 0:
    print("Scaler.sav not found or empty. Creating a new scaler...")

    #  training dataset 
    train_data = pd.read_csv("Coffe_sales.csv")

    # Extract 
    X_train = train_data[['hour_of_day', 'money', 'Weekdaysort', 'Monthsort']]

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save scaler
    with open("scaler.sav", "wb") as f:
        pickle.dump(scaler, f)

    print("Scaler fitted and saved to scaler.sav")
else:
    # Load existing scaler
    scaler = pickle.load(open("scaler.sav", "rb"))

# --- Step 2: Take user input ---
hour_of_day = int(input("Enter hour of day (6-22) = "))
money = float(input("Enter money spent = "))
Weekdaysort = int(input("Enter Weekday sort (1-7) = "))
Monthsort = int(input("Enter Month sort (1-12) = "))

# Create DataFrame for new data
new_data = pd.DataFrame([[hour_of_day, money, Weekdaysort, Monthsort]],
                        columns=['hour_of_day', 'money', 'Weekdaysort', 'Monthsort'])

# --- Step 3: Scale and predict ---
new_data_scaled = scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)

print("Prediction:", prediction)

