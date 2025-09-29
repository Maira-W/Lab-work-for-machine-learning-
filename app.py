import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"C:\Users\Pc Planet\Downloads\Uni\sem 5\ML\50_Startups (1).csv")

# Train model
X = df[["R&D Spend"]]
y = df["Profit"]
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Startup Profit Predictor")
rnd_spend = st.number_input("Enter R&D Spend:", min_value=0.0, step=1000.0)

if st.button("Predict Profit"):
    profit = model.predict([[rnd_spend]])[0]
    st.success(f"Predicted Profit: {profit:.2f}")
    
    
