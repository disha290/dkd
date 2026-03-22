import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered",
    page_icon="🏦"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}

.main-card {
    background-color: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #0d47a1;
}

.stButton>button {
    background: linear-gradient(90deg, #1565c0, #0d47a1);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 25px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0d47a1, #002171);
    transform: scale(1.03);
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}

.footer-text {
    text-align: center;
    font-size: 13px;
    color: gray;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD FILES --------------------
model = joblib.load("loan_approval_model.pkl")
model_columns = joblib.load("model_columns.pkl")
label_encoders = joblib.load("D:/final dkd/label_encoders.pkl")

# -------------------- MAIN CARD START --------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.title("🏦 Loan Approval Prediction")
st.caption("Enter applicant details to predict loan approval status")

# -------------------- INPUT FORM --------------------
with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ['','Male', 'Female'])
        Married = st.selectbox("Married", ['','Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['','0', '1', '2', '3+'])
        Education = st.selectbox("Education", ['','Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox("Self Employed", ['','Yes', 'No'])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.number_input("Loan Amount (in Rupees)", min_value=1000, step=1000)
        Loan_Amount_Term = st.number_input("Loan Term (in Months)", min_value=12)
        Credit_History = st.selectbox("Credit History", [1, 0])
        Property_Area = st.selectbox("Property Area", ['','Urban', 'Semiurban', 'Rural'])

    submit = st.form_submit_button("Predict Loan Status")

# -------------------- PREDICTION --------------------
if submit:

    LoanAmount = LoanAmount / 1000  # match training scale

    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })

    # Label Encoding
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # Align columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    confidence = prediction_proba * 100

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class='result-box' style='background-color:#e8f5e9; color:#1b5e20;'>
        ✅ Loan Approved<br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-box' style='background-color:#ffebee; color:#b71c1c;'>
        ❌ Loan Not Approved<br>
        Confidence: {100-confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='footer-text'>⚠ Prediction is based on historical data and is not final approval.</div>", unsafe_allow_html=True)

# -------------------- MAIN CARD END --------------------
st.markdown("</div>", unsafe_allow_html=True)