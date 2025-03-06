import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv(r"C:\Users\USER\Desktop\_\ML\.venv\ATMProject\FUTMINNA_ATM_Withdrawals_2020_2024.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Fill missing values
df["Total Amount Withdrawn (NGN)"] = df["Total Amount Withdrawn (NGN)"].fillna(df["Total Amount Withdrawn (NGN)"].median())

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Total Amount Withdrawn (NGN)']])

# Function to create yearly sequences for LSTM
def create_yearly_sequences(data, years_input, years_output):
    X, y = [], []
    num_days_per_year = 365  # Adjust for leap years if needed
    
    for i in range(len(data) - years_input * num_days_per_year - years_output * num_days_per_year):
        X.append(data[i:i + years_input * num_days_per_year, 0])  
        y.append(data[i + years_input * num_days_per_year : i + (years_input + years_output) * num_days_per_year, 0])  
    
    return np.array(X), np.array(y)

# Set input-output years
years_input = 3  # Use past 3 years
years_output = 1  # Predict 1 full year

# Prepare sequences
X, y = create_yearly_sequences(df_scaled, years_input, years_output)

# Ensure we have enough data
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Not enough data to create sequences. Adjust input-output years.")

# Split into train and test sets
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(100, return_sequences=False),
    Dense(50, activation="relu"),
    Dense(years_output * 365)  # Predicts full year (365 days)
])

# Compile Model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
if X_train.shape[0] > 0:
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict on Test Data
if X_test.shape[0] > 0:
    predictions = model.predict(X_test)

    # Reshape predictions and y_test
    predictions = predictions.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Convert back to original scale
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    # Reshape for plotting
    predictions = predictions.reshape(-1, years_output * 365)
    y_test_actual = y_test_actual.reshape(-1, years_output * 365)

    # Save predictions for 2025
    prediction_dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
    predictions_df = pd.DataFrame({
        "Date": prediction_dates,
        "Predicted Withdrawals": predictions[0]
    })
    predictions_df.to_csv("predictions.csv", index=False)

    print("Predictions saved to predictions.csv")

    # Plot Predictions vs Actual Data
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[0], label="Actual Withdrawals")
    plt.plot(predictions[0], label="Predicted Withdrawals", linestyle="dashed")
    plt.title("ATM Withdrawal Forecasting (Full Year)")
    plt.xlabel("Days")
    plt.ylabel("Withdrawals")
    plt.legend()
    plt.show()
else:
    print("Not enough data to generate predictions.")

# Save the trained model
model.save('atm_model.keras')
print("Model saved as atm_model.keras")
