import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF

# Load Models
model = joblib.load("../Backend_Model/career_prediction_model.pkl")
gender_encoder = joblib.load("../Backend_Model/gender_encoder.pkl")
major_encoder = joblib.load("../Backend_Model/major_encoder.pkl")
domain_encoder = joblib.load("../Backend_Model/domain_encoder.pkl")
python_encoder = joblib.load("../Backend_Model/python_encoder.pkl")
sql_encoder = joblib.load("../Backend_Model/sql_encoder.pkl")
java_encoder = joblib.load("../Backend_Model/java_encoder.pkl")
career_encoder = joblib.load("../Backend_Model/label_encoder.pkl")

st.title("🎯 AI Career Prediction Chatbot 🤖")
st.markdown("## Hey Future Billionare 🔥 Tera Bhai Future Batayega!")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=30)
gpa = st.slider("GPA (0-4 scale)", 0.0, 4.0, 2.0)
major = st.selectbox("Major", ["Computer Science"])
domain = st.selectbox("Interested Domain", ["Data Science", "Web Development", "Mobile App Development", "Artificial Intellegence", "Quantum Computing", "Machine Learning", "IoT (Internet of Things)", "Software Engineering", "Natural Language Processing", "Database Management", "Cybersecurity", "Computer Graphics", "Cloud Computing", "Biomedical Computing", "Blockchain Technology"])
python = st.selectbox("Python Skill", ["Weak", "Average", "Strong"])
sql = st.selectbox("SQL Skill", ["Weak", "Average", "Strong"])
java = st.selectbox("Java Skill", ["Weak", "Average", "Strong"])

if st.button("Predict Career 🚀"):
    user_data = [[
        gender_encoder.transform([gender])[0],
        age,
        gpa,
        major_encoder.transform([major])[0],
        domain_encoder.transform([domain])[0],
        python_encoder.transform([python])[0],
        sql_encoder.transform([sql])[0],
        java_encoder.transform([java])[0]
    ]]

    prediction = model.predict(user_data)[0]
    career = career_encoder.inverse_transform([prediction])[0]
    st.success(f"🎯 Predicted Career: **{career}**")

    # PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Career Prediction Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Predicted Career: {career}", ln=True)
    pdf.output("Career_Report.pdf")
    st.download_button("Download Report PDF 📄", data=open("Career_Report.pdf", "rb"), file_name="Career_Report.pdf")

st.info("✅ This Model is developed by **Akshat Soni** 🔥")

