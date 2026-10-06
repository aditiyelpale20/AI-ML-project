# Delinquency Prediction — FINAL Actual Dataset Version

This version uses the **actual uploaded dataset**:

`backend/data/Delinquency_prediction_dataset.xlsx`

It contains **500 rows and 19 columns**.

The model preprocessing and Random Forest configuration follow the supplied notebook:
- Fill missing Income, Credit Score and Loan Balance with their column means
- Drop Customer_ID
- One-hot encode Employment_Status, Credit_Card_Type and Location
- Map On-time=0, Late=1, Missed=2
- Predict Month_6 from customer information + Month_1 to Month_5
- RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
- 80/20 stratified train/test split, random_state=42

## Architecture

User
→ Streamlit UI
→ HTTP POST `/predict`
→ FastAPI
→ Saved `model.joblib`
→ JSON response
→ Streamlit UI

The dataset is used **during training**. The running API loads the already-trained `model.joblib`, so it does not retrain for every user prediction.

## Run

### Terminal 1 — API

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### Terminal 2 — Streamlit

```powershell
cd frontend
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

API docs:
`http://127.0.0.1:8000/docs`

Streamlit:
`http://localhost:8501`

## Important

The saved model in this project was trained from your uploaded original dataset. If you replace the dataset later, run the included training workflow again before using the new data.
