import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Example data
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95, 98],
    "score": [50, 55, 58, 65, 68, 75, 78, 85, 88, 95],
}

df = pd.DataFrame(data)

# Features and target
X = df[["hours_studied", "attendance"]]
y = df["score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "app/model.joblib")
print("Model trained and saved to app/model.joblib")