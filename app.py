import streamlit as st
import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Classification App")

st.write("Nhập thông số hoa để dự đoán loại:")

# Input từ user
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    classes = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Dự đoán: {classes[prediction]}")
