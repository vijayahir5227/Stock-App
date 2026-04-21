from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import streamlit as st

def train_model(X, y):
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    st.sidebar.metric("Model MAE (test)", f"₹{mae:.2f}")

    model._scaler = scaler
    return model

def predict(model, X):
    X_scaled = model._scaler.transform(X)
    return model.predict(X_scaled)