import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load dataset
df = pd.read_csv("ml_training_data.csv")

# Create labels
threshold = df["efficiency"].mean()
df["label"] = (df["efficiency"] >= threshold).astype(int)

# Features
X = df[[
    "failure_rate",
    "checkpoint_cost",
    "checkpoint_count"
]]

y = df["label"]

# Train classifier
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

print("\n=== CLASSIFICATION RULES ===")
rules = tree.export_text(clf, feature_names=list(X.columns))
print(rules)

# Save rules
with open("classification_rules.txt", "w") as f:
    f.write(rules)

print("\nSaved to classification_rules.txt")