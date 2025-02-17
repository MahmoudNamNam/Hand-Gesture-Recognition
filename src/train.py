import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("../Data/gesture_data.csv")


# Ensure correct shape (21 landmarks * 3 features each = 63)
X = df.iloc[:, 1:].values  # Extract features (landmarks)
y = df.iloc[:, 0].astype(int).values  # Labels

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))  # Ensure it matches the dataset
y = keras.utils.to_categorical(y, num_classes=num_classes)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),  # Input shape must be 63
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')  # Adjust dynamically
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save the model and scaler
model.save("gesture_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model trained and saved successfully!")
