import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = {
    "ambient_c": [22.0, 23.1, 25.0, 21.4, 24.2],
    "object_c":  [32.2, 27.8, 25.1, 30.2, 26.0],
}
df = pd.DataFrame(data)
df["diff_c"] = df["object_c"] - df["ambient_c"]
df["fault"] = [1, 0, 0, 1, 0]

X = df[["ambient_c", "object_c", "diff_c"]]
y = df["fault"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, "fault_model.pkl")
print("✅ Model saved as fault_model.pkl")
