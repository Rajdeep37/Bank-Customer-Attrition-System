import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('Churn_Modelling.csv', delimiter=',')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Geography"] = le.fit_transform(data["Geography"])

X = data.drop('Exited', axis=1)
y = data['Exited']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Custom callback to print predictions
class PrintPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
    

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=1000, batch_size=32, verbose=1,
    callbacks=[PrintPredictionsCallback((X_val_scaled, y_val))]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy}")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the R² Score (for regression tasks)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Save the trained model
model.export('own_model/')

# Save the scaler object for future use
joblib.dump(scaler, "scaler.pkl")
