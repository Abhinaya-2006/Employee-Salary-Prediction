import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="💼 Employee Salary Predictor",
    layout="wide",
    initial_sidebar_state="collapsed" # Ensure sidebar is collapsed or not shown
)

# --- Custom CSS for Styling (Minimal for clean modules & READABILITY) ---
st.markdown(
    """
    <style>
    /* General App Background */
    .stApp {
        background-image: none;
        background-color: #F8F9FA; /* A very light, almost white background for the entire app */
    }

    /* Main content area (container for all modules) */
    .main .block-container {
        background-color: #FFFFFF; /* Pure white background for main content, essential for readability */
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04); /* Even lighter shadow */
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Hide the sidebar completely */
    .st-emotion-cache-1ldv1c7, /* Target sidebar background */
    .st-emotion-cache-vk337f, /* Target sidebar text color */
    .st-emotion-cache-j7q0jx { /* Another common sidebar element class */
        display: none !important; /* Force hide sidebar elements */
    }

    /* Adjust main content padding when no sidebar to fill the width */
    .st-emotion-cache-18ni7ap { /* Main content padding div - common class for the overall app content area */
        padding-left: 5rem !important; /* Adjust as needed */
        padding-right: 5rem !important; /* Adjust as needed */
        padding-top: 2rem !important; /* Ensure top padding */
        padding-bottom: 2rem !important; /* Ensure bottom padding */
    }

    /* Headers (h1 to h6) */
    h1, h2, h3, h4, h5, h6 {
        color: #212529; /* Very dark gray almost black for strong contrast */
        text-shadow: none; /* No text shadow */
        margin-top: 1.5rem; /* Consistent spacing */
        margin-bottom: 0.8rem;
    }

    /* General text (paragraphs, labels, markdown, etc.) */
    p, label, .stMarkdown, .st-emotion-cache-nahz7x, .st-emotion-cache-1pxazr6, .st-emotion-cache-j7q0jx, .st-emotion-cache-gh2j6y, .st-emotion-cache-1u4u5r, .st-emotion-cache-1t38b1f {
        color: #343A40; /* Dark gray for readability */
    }

    /* Predicted Salary display */
    .st-emotion-cache-1c7y2gy p { /* This targets the text within st.success */
        font-size: 2.5em !important;
        font-weight: bold;
        color: #007BFF !important; /* A clear, vibrant blue for visibility */
        text-align: center;
        padding: 15px;
        border: 1px solid #007BFF; /* Matching border color */
        border-radius: 5px;
        background-color: #E7F3FF; /* Very light blue background */
        margin-top: 1.5em;
        margin-bottom: 1.5em;
    }

    /* Input widgets (sliders, selectboxes, text inputs) */
    .stSelectbox > div > div > div,
    .stSlider > div > div,
    .stTextInput > div > div > input, /* Target the input field directly */
    .stTextInput > div > div { /* Target the container of text input */
        background-color: #F8F9FA; /* Slightly off-white background for inputs */
        border-radius: 5px;
        border: 1px solid #CED4DA; /* Standard light grey border */
        color: #212529; /* Dark text within inputs */
    }
    /* Ensure selectbox text itself is dark */
    .st-emotion-cache-1t38b1f { /* Specific class for selectbox current value text */
        color: #212529 !important;
    }
    /* Slider numbers */
    .st-emotion-cache-1k46wkc, .st-emotion-cache-1fo1vxt { /* Slider value labels */
        color: #212529 !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #007BFF; /* Primary blue button */
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 1.1em;
        font-weight: normal;
        letter-spacing: normal;
        transition: background-color 0.2s ease;
        box-shadow: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }

    /* Card-like containers for sections (modules) */
    .st-emotion-cache-eczf16 { /* Common container class for st.container() */
        background-color: #FFFFFF; /* Pure white background for cards */
        border: 1px solid #E0E0E0; /* Light gray border for separation */
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 20px;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.03); /* Even lighter shadow */
    }

    /* Styling for Streamlit's expander (if used, though not in this version) */
    .streamlit-expanderHeader {
        background-color: #F8F9FA; /* Light background for expander header */
        color: #212525;
        border-radius: 5px;
    }

    /* Specific styling for tabs content area */
    .st-emotion-cache-1kyxmo3 { /* This targets the content area of tabs */
        background-color: #FDFDFD; /* Very light off-white for tab content */
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #ECECEC;
        box-shadow: none;
    }

    /* Make plot backgrounds white (explicitly, as matplotlib/seaborn can vary) */
    .st-emotion-cache-ocqkz7 { /* Targets plot containers created by Streamlit */
        background-color: white !important;
        border-radius: 8px;
        padding: 10px;
        box-shadow: none;
    }

    /* Footer text color */
    .st-emotion-cache-h5g5z2 p {
        color: #555555;
        font-size: 0.9em;
    }

    /* Remove extra padding from columns if it creates gaps between modules */
    .st-emotion-cache-16ids9j { /* This targets the outer div of columns */
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Employee Salary Prediction System")
st.markdown("A simple tool to estimate employee salaries based on key attributes.")

# --- Generate richer sample dataset ---
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

    # Base salary logic
    base_salary = 25000 + df["Experience"] * 2200
    edu_factor = df["Education"].map({
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    })
    role_factor = df["Role"].map({
        "Software Engineer": 1.3,
        "Data Analyst": 1.1,
        "Manager": 1.6,
        "HR": 1.0,
        "Product Owner": 1.5,
        "Designer": 1.2,
    })
    loc_factor = df["Location"].map({
        "Bangalore": 1.3,
        "Hyderabad": 1.2,
        "Chennai": 1.1,
        "Delhi": 1.4,
        "Pune": 1.1,
        "Mumbai": 1.35,
    })
    cert_factor = df["Certifications"].map({
        "None": 0,
        "AWS": 5000,
        "Azure": 4500,
        "PMP": 7000,
        "Scrum Master": 4000,
        "Google Cloud": 4800,
    })
    skill_factor = df["Skills"].map({
        "None": 0,
        "Python": 4000,
        "Java": 3800,
        "SQL": 3500,
        "Excel": 2000,
        "Tableau": 3000,
    })

    df["Salary"] = ((base_salary + edu_factor * 6000 + cert_factor + skill_factor) * role_factor * loc_factor).astype(int)
    return df

df = generate_sample_data()

# Encode categorical columns
label_encoders = {}
df_encoded = df.copy()
for col in ["Education", "Role", "Location", "Certifications", "Skills"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# --- Main Content Layout ---

# Input and Prediction Module/Card
with st.container():
    st.header("⚙️ Configure Employee Details & Get Prediction")
    st.write("Enter the employee's characteristics below to estimate their annual salary.")

    # Arrange inputs in columns for a compact layout
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

    st.markdown("---") # Visual separator within the card

    # Prepare input dataframe for prediction
    input_df = pd.DataFrame({
        "Experience": [exp],
        "Education": [label_encoders["Education"].transform([education])[0]],
        "Role": [label_encoders["Role"].transform([role])[0]],
        "Location": [label_encoders["Location"].transform([location])[0]],
        "Certifications": [label_encoders["Certifications"].transform([certification])[0]],
        "Skills": [label_encoders["Skills"].transform([skill])[0]],
    })

    # Prediction Button and Result Area
    col_pred_btn, col_pred_result = st.columns([1, 2])

    with col_pred_btn:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        if st.button("Calculate Salary"):
            with st.spinner("Calculating..."):
                predicted_salary = model.predict(input_df)[0]
                # No balloons for a super clean look, but you can add it back if desired

    with col_pred_result:
        if 'predicted_salary' in locals():
            st.success(f"**Predicted Annual Salary: ₹ {int(predicted_salary):,}**")
        else:
            st.info("Fill in the details and click 'Calculate Salary' to see the prediction.")


st.markdown("---") # Separator between modules

# Dataset Overview Module/Card
with st.container():
    st.header("📊 Dataset Overview")
    st.write("Get a quick glance at the underlying data and its statistical properties.")

    col_summary, col_raw_data = st.columns(2)

    with col_summary:
        if st.checkbox("Show Dataset Summary Statistics"):
            st.subheader("Descriptive Statistics")
            st.write(df.describe())
    with col_raw_data:
        if st.checkbox("Show Raw Dataset Table"):
            st.subheader("Raw Sample Data (First 20 Rows)")
            st.write(df.head(20))


st.markdown("---") # Separator between modules

# Visualizations Module/Card with Tabs
with st.container():
    st.header("📈 Key Salary Insights")
    st.write("Explore how different factors influence salary through these charts.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Salary Distribution",
        "Salary by Role",
        "Salary by Education",
        "Salary by Location",
        "Feature Importance"
    ])

    with tab1:
        st.subheader("Salary Distribution Across Employees")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Salary"], bins=30, kde=True, ax=ax1, color='#007BFF') # Consistent blue
        ax1.set_xlabel("Salary (INR)", color='#212529') # Explicitly set plot label color
        ax1.set_ylabel("Number of Employees", color='#212529')
        ax1.tick_params(axis='x', colors='#212529') # Set tick label colors
        ax1.tick_params(axis='y', colors='#212529')
        st.pyplot(fig1)

    with tab2:
        st.subheader("Average Salary by Job Role")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        role_salary = df.groupby("Role")["Salary"].mean().sort_values(ascending=False)
        sns.barplot(x=role_salary.values, y=role_salary.index, ax=ax2, palette="viridis")
        ax2.set_xlabel("Average Salary (INR)", color='#212529')
        ax2.set_ylabel("Job Role", color='#212529')
        ax2.tick_params(axis='x', colors='#212529')
        ax2.tick_params(axis='y', colors='#212529')
        st.pyplot(fig2)

    with tab3:
        st.subheader("Average Salary by Education Level")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        edu_salary = df.groupby("Education")["Salary"].mean().reindex(["High School", "Bachelor's", "Master's", "PhD"])
        sns.barplot(x=edu_salary.values, y=edu_salary.index, ax=ax3, palette="plasma")
        ax3.set_xlabel("Average Salary (INR)", color='#212529')
        ax3.set_ylabel("Education Level", color='#212529')
        ax3.tick_params(axis='x', colors='#212529')
        ax3.tick_params(axis='y', colors='#212529')
        st.pyplot(fig3)

    with tab4:
        st.subheader("Average Salary by Location")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        loc_salary = df.groupby("Location")["Salary"].mean().sort_values(ascending=False)
        sns.barplot(x=loc_salary.values, y=loc_salary.index, ax=ax4, palette="cividis")
        ax4.set_xlabel("Average Salary (INR)", color='#212529')
        ax4.set_ylabel("Location", color='#212529')
        ax4.tick_params(axis='x', colors='#212529')
        ax4.tick_params(axis='y', colors='#212529')
        st.pyplot(fig4)

    with tab5:
        st.subheader("Feature Importance (Model Insights)")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax5, palette="magma")
        ax5.set_xlabel("Importance Score", color='#212529')
        ax5.set_ylabel("Feature", color='#212529')
        ax5.tick_params(axis='x', colors='#212529')
        ax5.tick_params(axis='y', colors='#212529')
        st.pyplot(fig5)

st.markdown("---") # Final separator

