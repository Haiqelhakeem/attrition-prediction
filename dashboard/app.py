import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

DATA_PATH = os.path.join("..", "data", "employee_data.csv")
MODEL_PATH = "../data/attrition_best_model.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# Load data
df = pd.read_csv(DATA_PATH)

# Drop rows with missing target if present
df = df.dropna(subset=['Attrition'])

st.title("💼 Employee Attrition Prediction Dashboard")
st.markdown("Use the filters below to explore employee data and get attrition predictions with recommended actions.")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Employees")
    selected_department = st.selectbox("Department", options=["All"] + sorted(df['Department'].unique().tolist()))
    selected_job_role = st.selectbox("Job Role", options=["All"] + sorted(df['JobRole'].unique().tolist()))
    selected_overtime = st.selectbox("OverTime", options=["All"] + sorted(df['OverTime'].unique().tolist()))

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_department]
if selected_job_role != "All":
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_job_role]
if selected_overtime != "All":
    filtered_df = filtered_df[filtered_df['OverTime'] == selected_overtime]

# Predict
X_input = filtered_df.drop(columns=['Attrition', 'EmployeeId'])
X_input['Over18'] = X_input['Over18'].map({'Y': 1})

# Get all required columns from model pipeline
preprocessor = model.named_steps['preprocessor']
expected_cols = preprocessor.feature_names_in_

# Default values for missing columns
default_values = {
    'BusinessTravel': 'Travel_Rarely',
    'Department': 'Research & Development',
    'EducationField': 'Life Sciences',
    'Gender': 'Male',
    'JobRole': 'Sales Executive',
    'MaritalStatus': 'Single',
    'OverTime': 'No',
    'Over18': 'Y',
    'DailyRate': 800,
    'DistanceFromHome': 5,
    'Education': 3,
    'EmployeeCount': 1,
    'EnvironmentSatisfaction': 3,
    'HourlyRate': 60,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobSatisfaction': 3,
    'MonthlyIncome': 5000,
    'MonthlyRate': 15000,
    'NumCompaniesWorked': 2,
    'PercentSalaryHike': 11,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StandardHours': 40,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 2,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 3,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 3,
}

# Add missing columns with defaults
for col in expected_cols:
    if col not in X_input.columns:
        X_input[col] = default_values.get(col, 0)

# Ensure column order matches
X_input = X_input[expected_cols]

# Run predictions
preds = model.predict(X_input)
probas = model.predict_proba(X_input)[:, 1]

# Add predictions to dataframe
filtered_df = filtered_df.reset_index(drop=True)
filtered_df['Predicted Attrition'] = preds
filtered_df['Attrition Risk (%)'] = np.round(probas * 100, 2)

# Recommend actions
def recommend_action(row):
    if row['Predicted Attrition'] == 1:
        if row['OverTime'] == 'Yes':
            return "🚨 Reduce workload"
        elif row['JobSatisfaction'] <= 2:
            return "💬 Conduct 1-on-1"
        elif row['WorkLifeBalance'] <= 2:
            return "🕒 Improve balance"
        else:
            return "🎯 Offer career path"
    else:
        return "✅ No immediate action"

filtered_df['Recommended Action'] = filtered_df.apply(recommend_action, axis=1)

# Display predictions
st.subheader("📊 Prediction Results")

st.dataframe(filtered_df[['EmployeeId', 'JobRole', 'Department', 'OverTime', 'Attrition Risk (%)', 'Recommended Action']])

# Download CSV
st.download_button(
    label="📥 Download Results as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='attrition_predictions.csv',
    mime='text/csv'
)

# ======================
# Feature Importance Section
# ======================
st.subheader("📈 Feature Importance (Top Predictors)")

# Get feature names from ColumnTransformer
feature_names = []
for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    elif name == 'cat':
        ohe = transformer
        ohe_feature_names = ohe.get_feature_names_out(columns)
        feature_names.extend(ohe_feature_names)
    elif name == 'over18':
        feature_names.extend(columns)

# Get importances
classifier = model.named_steps['classifier']
importances = classifier.feature_importances_

# Build DataFrame
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Select top N features
top_n = st.slider("Top N features to display", min_value=5, max_value=30, value=15)

# Show table
st.dataframe(feat_imp_df.head(top_n), use_container_width=True)

# Plot bar chart
top_feats = feat_imp_df.head(top_n)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_feats['Feature'][::-1], top_feats['Importance'][::-1], color='skyblue')
ax.set_xlabel("Importance")
ax.set_title(f"Top {top_n} Important Features")
st.pyplot(fig)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.xticks(rotation=45)
plt.title("Attrition based on Departement")
st.pyplot(plt)