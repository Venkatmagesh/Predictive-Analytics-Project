import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../dataset/train.csv")

# -----------------------------
# Handle Missing Values
# -----------------------------
df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())

# -----------------------------
# Convert Order Date
# -----------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month
df["Order Day"] = df["Order Date"].dt.day

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

categorical_columns = [
    "Segment",
    "Region",
    "Category",
    "Ship Mode"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Features
# -----------------------------
X = df[
    [
        "Postal Code",
        "Segment",
        "Region",
        "Category",
        "Ship Mode",
        "Order Year",
        "Order Month",
        "Order Day"
    ]
]

# Target
y = df["Sales"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# Save Predictions
# -----------------------------
results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

results.to_csv("../output/predictions.csv", index=False)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label="Actual Sales")
plt.plot(y_pred[:100], label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.show()