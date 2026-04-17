import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

def generate_mock_data():
    """Generates 2 years of daily blood demand data for AI training"""
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq='D')
    
    # Base demand + seasonal fluctuation (e.g. malaria/dengue seasons) + random noise
    demand = np.random.poisson(lam=50, size=len(dates)) 
    demand = demand + np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) * 20
    demand = np.maximum(demand, 15) # Ensure no unrealistic low demand
    
    df = pd.DataFrame({'Date': dates, 'Demand': demand})
    df.to_csv('historical_blood_demand.csv', index=False)
    print("✅ Mock historical data generated: historical_blood_demand.csv")
    return df

def build_and_train_lstm():
    df = generate_mock_data()
    data = df['Demand'].values.reshape(-1, 1)
    
    # Scale Data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create Sequences (Look back 30 days to predict next day)
    seq_length = 30
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
        
    X = np.array(X)
    y = np.array(y)
    
    # Train-Test Split (80% Train)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 🧠 Initialize the LSTM Deep Learning Architecture
    model = Sequential()
    # Layer 1: LSTM Sequence
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2)) # Prevent overfitting
    # Layer 2: LSTM Core
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    # Dense Processing layers
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1)) # Output Layer (Predicts next unit demand)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("🧠 Starting Training for Deep Learning LSTM Model...")
    # Training the Model...
    model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the architecture and weights
    model.save('blood_demand_lstm.h5')
    print("✅ Model successfully trained and saved to blood_demand_lstm.h5")
    
    # Save the scaler for inference in the app
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    build_and_train_lstm()
