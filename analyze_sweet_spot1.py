import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("ml_training_data.csv")

# Features (adjusted for your dataset)
X = df[[
    "failure_rate",
    "checkpoint_cost",
    "checkpoint_count",
    "loop_count",
    "cyclomatic_complexity"
]]

# Target
y = df["efficiency"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Extract parameters
intercept = model.intercept_
coeffs = dict(zip(X.columns, model.coef_))

print("\n=== REGRESSION MODEL PARAMETERS ===")
print(f"Intercept: {intercept}")
for k, v in coeffs.items():
    print(f"{k}: {v}")

# Save
with open("regression_params.txt", "w") as f:
    f.write(f"intercept={intercept}\n")
    for k, v in coeffs.items():
        f.write(f"{k}={v}\n")

print("\nSaved to regression_params.txt")