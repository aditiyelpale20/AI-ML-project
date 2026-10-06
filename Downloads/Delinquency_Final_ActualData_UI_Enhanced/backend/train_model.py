from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent
data = pd.read_excel(BASE / "data" / "Delinquency_prediction_dataset.xlsx")

for col in ["Income", "Credit_Score", "Loan_Balance"]:
    data[col] = data[col].fillna(data[col].mean())

means = {c: float(data[c].mean()) for c in ["Income","Credit_Score","Loan_Balance"]}

data = data.drop("Customer_ID", axis=1)
data = pd.get_dummies(data, columns=["Employment_Status","Credit_Card_Type","Location"])

status_mapping = {"On-time":0,"Late":1,"Missed":2}
for col in ["Month_1","Month_2","Month_3","Month_4","Month_5","Month_6"]:
    data[col] = data[col].map(status_mapping)

X = data.drop("Month_6", axis=1)
y = data["Month_6"]

x_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

model=RandomForestClassifier(
    n_estimators=300,max_depth=10,min_samples_split=5,
    min_samples_leaf=2,random_state=42
)
model.fit(x_train,y_train)
accuracy=accuracy_score(y_test,model.predict(x_test))

artifact={
    "model":model,
    "feature_columns":list(X.columns),
    "numeric_means":means,
    "status_mapping":status_mapping,
    "status_labels":{0:"On-time",1:"Late",2:"Missed"}
}
joblib.dump(artifact,BASE/"model.joblib")

metadata={
    "rows":len(data),
    "accuracy":float(accuracy),
    "employment_options":sorted(pd.read_excel(BASE/"data"/"Delinquency_prediction_dataset.xlsx")["Employment_Status"].unique().tolist()),
    "card_options":sorted(pd.read_excel(BASE/"data"/"Delinquency_prediction_dataset.xlsx")["Credit_Card_Type"].unique().tolist()),
    "location_options":sorted(pd.read_excel(BASE/"data"/"Delinquency_prediction_dataset.xlsx")["Location"].unique().tolist()),
    "payment_options":["On-time","Late","Missed"],
    "target":"Month_6"
}
(BASE/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print(f"Model retrained. Test accuracy: {accuracy:.2%}")
