import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="wide")

# ------------------ Custom CSS Styling ------------------
st.markdown("""
<style>
.stApp {
    background-color: #F8F9FA;
}
.main .block-container {
    background-color: #FFFFFF;
    padding: 2.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}
h1, h2, h3, h4, h5, h6 {
    color: #212529;
}
p, label {
    color: #343A40;
}
.stButton > button {
    background-color: #007BFF;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    font-size: 1.1em;
    width: 100%;
}
.stButton > button:hover {
    background-color: #0056b3;
}
.st-emotion-cache-eczf16 {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 20px;
    box-shadow: 0 1px 5px rgba(0, 0, 0, 0.03);
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💼 Employee Salary Prediction System")
st.markdown("A simple tool to estimate employee salaries based on key attributes.")

# ------------------ Generate Sample Data ------------------
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 300
    data = {
        "Experience": np.random.randint(0, 31, n),
        "Education": np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], n),
        "Role": np.random.choice(["Software Engineer", "Data Analyst", "Manager", "HR", "Product Owner", "Designer"], n),
        "Location": np.random.choice(["Bangalore", "Hyderabad", "Chennai", "Delhi", "Pune", "Mumbai"], n),
        "Certifications": np.random.choice(["None", "AWS", "Azure", "PMP", "Scrum Master", "Google Cloud"], n),
        "Skills": np.random.choice(["Python", "Java", "SQL", "Excel", "Tableau", "None"], n),
    }
    df = pd.DataFrame(data)

    base_salary = 25000 + df["Experience"] * 2200
    edu_factor = df["Education"].map({
        "High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3
    })
    role_factor = df["Role"].map({
        "Software Engineer": 1.3, "Data Analyst": 1.1, "Manager": 1.6,
        "HR": 1.0, "Product Owner": 1.5, "Designer": 1.2
    })
    loc_factor = df["Location"].map({
        "Bangalore": 1.3, "Hyderabad": 1.2, "Chennai": 1.1,
        "Delhi": 1.4, "Pune": 1.1, "Mumbai": 1.35
    })
    cert_factor = df["Certifications"].map({
        "None": 0, "AWS": 5000, "Azure": 4500,
        "PMP": 7000, "Scrum Master": 4000, "Google Cloud": 4800
    })
    skill_factor = df["Skills"].map({
        "None": 0, "Python": 4000, "Java": 3800,
        "SQL": 3500, "Excel": 2000, "Tableau": 3000
    })

    df["Salary"] = ((base_salary + edu_factor * 6000 + cert_factor + skill_factor) * role_factor * loc_factor).astype(int)
    return df

df = generate_sample_data()

# Label Encoding
label_encoders = {}
df_encoded = df.copy()
for col in ["Education", "Role", "Location", "Certifications", "Skills"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Model Training
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ------------------ Prediction UI ------------------
with st.container():
    st.header("⚙️ Enter Employee Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        exp = st.slider("Years of Experience", 0, 40, 5)
        education = st.selectbox("Education Level", label_encoders["Education"].classes_)
    with col2:
        role = st.selectbox("Job Role", label_encoders["Role"].classes_)
        location = st.selectbox("Location", label_encoders["Location"].classes_)
    with col3:
        certification = st.selectbox("Certification", label_encoders["Certifications"].classes_)
        skill = st.selectbox("Primary Skill", label_encoders["Skills"].classes_)

    input_df = pd.DataFrame({
        "Experience": [exp],
        "Education": [label_encoders["Education"].transform([education])[0]],
        "Role": [label_encoders["Role"].transform([role])[0]],
        "Location": [label_encoders["Location"].transform([location])[0]],
        "Certifications": [label_encoders["Certifications"].transform([certification])[0]],
        "Skills": [label_encoders["Skills"].transform([skill])[0]],
    })

    if st.button("💰 Predict Salary"):
        pred_salary = model.predict(input_df)[0]
        st.success(f"**Predicted Salary: ₹ {int(pred_salary):,}**")

# ------------------ Show Data ------------------
with st.expander("📄 Show Sample Dataset"):
    st.write(df.head(20))

# ------------------ Visualization Tabs ------------------
with st.container():
    st.header("📊 Salary Insights")
    tab1, tab2, tab3 = st.tabs(["Distribution", "By Role", "Feature Importance"])

    with tab1:
        fig, ax = plt.subplots()
        sns.histplot(df["Salary"], kde=True, ax=ax, color="skyblue")
        ax.set_title("Salary Distribution")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots()
        role_avg = df.groupby("Role")["Salary"].mean().sort_values()
        sns.barplot(x=role_avg.values, y=role_avg.index, ax=ax, palette="viridis")
        ax.set_title("Average Salary by Role")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots()
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        sns.barplot(x=imp.values, y=imp.index, ax=ax, palette="magma")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
