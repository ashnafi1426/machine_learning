import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. YouTube Dataset
data = {
    'Video_Duration_Min': [3, 5, 8, 10, 12, 15, 20, 25, 30, 45],
    'Likes_Count':        [50, 120, 300, 450, 600, 850, 1200, 1600, 2100, 3500],
    'Comments_Count':     [5, 12, 25, 40, 55, 75, 110, 150, 200, 320],
    'Views':  [1000, 2500, 6000, 9500, 13000, 18000, 26000, 34000, 45000, 72000]
}
df = pd.DataFrame(data)
print(" Data Tabel")
print(df.head(), "\n")


X = df[['Video_Duration_Min', 'Likes_Count', 'Comments_Count']]
y = df['Views']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Predicted Views:", y_pred)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Evaluation Metrics ")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")



new_video = pd.DataFrame({
    'Video_Duration_Min': [18],
    'Likes_Count': [1000],
    'Comments_Count': [90]
})

predicted_views = model.predict(new_video)
print(f"\n Predicted Views: {int(predicted_views[0]):,} Views")

plt.scatter(y_test, y_pred, color='purple', edgecolor='black', s=80)
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Actual vs Predicted YouTube Views")
plt.grid(True)
plt.show()