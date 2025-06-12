import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HR Attrition Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">👥 HR Attrition Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sample data generation function (replace with your actual data loading)
df = pd.read_csv('C:\Project\LaskarAI\DataScience_Submission1_HaiqelAH\employee_data.csv')

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Department filter
departments = st.sidebar.multiselect(
    "Select Departments",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

# Age range filter
age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

# Business travel filter
travel_options = st.sidebar.multiselect(
    "Business Travel",
    options=df['BusinessTravel'].unique(),
    default=df['BusinessTravel'].unique()
)

# Filter data
filtered_df = df[
    (df['Department'].isin(departments)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['BusinessTravel'].isin(travel_options))
]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_employees = len(filtered_df)
    st.metric("👥 Total Employees", total_employees)

with col2:
    attrition_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
    st.metric("📉 Attrition Count", attrition_count)

with col3:
    attrition_rate = (attrition_count / total_employees) * 100 if total_employees > 0 else 0
    st.metric("📊 Attrition Rate", f"{attrition_rate:.1f}%")

with col4:
    avg_tenure = filtered_df['YearsAtCompany'].mean()
    st.metric("⏱️ Avg Tenure", f"{avg_tenure:.1f} years")

# Visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Deep Dive", "🤖 ML Insights", "💡 Recommendations"])

with tab1:
    st.header("📈 Attrition Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by Department
        dept_attrition = filtered_df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        fig_dept = px.bar(
            dept_attrition.reset_index(), 
            x='Department', 
            y=['Yes', 'No'],
            title="Attrition by Department",
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
        )
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        # Attrition by Age Group
        filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
        age_attrition = filtered_df.groupby(['AgeGroup', 'Attrition']).size().unstack(fill_value=0)
        fig_age = px.bar(
            age_attrition.reset_index(),
            x='AgeGroup',
            y=['Yes', 'No'],
            title="Attrition by Age Group",
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Attrition by Job Satisfaction
        if 'JobSatisfaction' in filtered_df.columns:
            satisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
            filtered_df['JobSatisfactionLabel'] = filtered_df['JobSatisfaction'].map(satisfaction_labels)
            sat_attrition = filtered_df.groupby('JobSatisfactionLabel')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            fig_sat = px.bar(
                x=sat_attrition.index,
                y=sat_attrition.values,
                title="Attrition Rate by Job Satisfaction",
                color=sat_attrition.values,
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_sat, use_container_width=True)
    
    with col4:
        # Attrition by Overtime
        if 'OverTime' in filtered_df.columns:
            overtime_attrition = filtered_df.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            fig_overtime = px.pie(
                values=overtime_attrition.values,
                names=overtime_attrition.index,
                title="Attrition Rate: Overtime vs No Overtime"
            )
            st.plotly_chart(fig_overtime, use_container_width=True)

with tab2:
    st.header("🔍 Deep Dive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = filtered_df[numeric_cols].corr()
            fig_corr = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Income distribution by attrition
        if 'MonthlyIncome' in filtered_df.columns:
            fig_income = px.box(
                filtered_df,
                x='Attrition',
                y='MonthlyIncome',
                title="Monthly Income Distribution by Attrition",
                color='Attrition',
                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
            )
            st.plotly_chart(fig_income, use_container_width=True)
    
    # Distance vs Attrition
    if 'DistanceFromHome' in filtered_df.columns:
        fig_distance = px.histogram(
            filtered_df,
            x='DistanceFromHome',
            color='Attrition',
            title="Distance from Home vs Attrition",
            marginal='box',
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#4ecdc4'}
        )
        st.plotly_chart(fig_distance, use_container_width=True)

with tab3:
    st.header("🤖 Machine Learning Insights")
    
    # Prepare data for ML
    def prepare_ml_data(data):
        """Prepare data for machine learning"""
        ml_data = data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ml_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Attrition':
                ml_data[col] = le.fit_transform(ml_data[col].astype(str))
        
        # Encode target variable
        ml_data['Attrition'] = le.fit_transform(ml_data['Attrition'])
        
        return ml_data
    
    try:
        ml_data = prepare_ml_data(filtered_df)
        
        # Features and target
        X = ml_data.drop(['Attrition', 'EmployeeId'], axis=1, errors='ignore')
        y = ml_data['Attrition']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance for Attrition Prediction",
                color='importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Model performance metrics
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
            
            # Prediction probabilities
            y_prob = rf_model.predict_proba(X_test)[:, 1]
            
            # ROC Curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Risk segmentation
        st.subheader("📊 Employee Risk Segmentation")
        
        # Predict attrition probability for all employees
        all_probs = rf_model.predict_proba(X)[:, 1]
        risk_df = filtered_df.copy()
        risk_df['Attrition_Risk'] = all_probs
        risk_df['Risk_Category'] = pd.cut(
            risk_df['Attrition_Risk'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Risk distribution
        risk_counts = risk_df['Risk_Category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Employee Risk Distribution",
            color_discrete_map={'Low Risk': '#4ecdc4', 'Medium Risk': '#ffe66d', 'High Risk': '#ff6b6b'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in ML analysis: {str(e)}")
        st.info("Please ensure your dataset has the required columns for ML analysis.")

with tab4:
    st.header("💡 Actionable Recommendations")
    
    # Calculate key insights
    attrition_by_dept = filtered_df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    highest_attrition_dept = attrition_by_dept.idxmax()
    
    if 'OverTime' in filtered_df.columns:
        overtime_attrition = filtered_df.groupby('OverTime')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    
    # Recommendations based on analysis
    recommendations = [
        {
            "category": "🏢 Department-Specific Actions",
            "priority": "High",
            "recommendations": [
                f"Focus retention efforts on {highest_attrition_dept} department (highest attrition rate: {attrition_by_dept[highest_attrition_dept]:.1f}%)",
                "Implement department-specific career development programs",
                "Conduct exit interviews to understand department-specific issues"
            ]
        },
        {
            "category": "💼 Work-Life Balance",
            "priority": "High",
            "recommendations": [
                "Reduce overtime requirements where possible",
                "Implement flexible working arrangements",
                "Provide work-life balance training for managers",
                "Monitor workload distribution across teams"
            ]
        },
        {
            "category": "📈 Career Development",
            "priority": "Medium",
            "recommendations": [
                "Create clear career progression pathways",
                "Increase promotion frequency for high performers",
                "Provide mentorship programs",
                "Offer skill development opportunities"
            ]
        },
        {
            "category": "💰 Compensation & Benefits",
            "priority": "Medium",
            "recommendations": [
                "Review compensation packages for competitive market rates",
                "Implement performance-based bonuses",
                "Provide additional benefits for remote workers",
                "Consider stock options for long-term retention"
            ]
        },
        {
            "category": "🎯 Predictive Interventions",
            "priority": "High",
            "recommendations": [
                "Implement early warning system using ML model",
                "Proactively engage with high-risk employees",
                "Create personalized retention plans",
                "Regular pulse surveys for at-risk employees"
            ]
        }
    ]
    
    for rec in recommendations:
        priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>{rec['category']} {priority_color[rec['priority']]} {rec['priority']} Priority</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for item in rec['recommendations']:
            st.write(f"• {item}")
        
        st.write("")
    
    # ROI Calculator
    st.subheader("💵 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_salary = st.number_input("Average Employee Salary ($)", value=50000, step=1000)
        replacement_cost_multiplier = st.slider("Replacement Cost Multiplier", 0.5, 2.0, 1.5, 0.1)
        
    with col2:
        retention_improvement = st.slider("Expected Retention Improvement (%)", 1, 50, 10)
        program_cost = st.number_input("Retention Program Cost ($)", value=100000, step=10000)
    
    # Calculate ROI
    current_attrition = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
    replacement_cost = avg_salary * replacement_cost_multiplier
    current_cost = current_attrition * replacement_cost
    
    improved_attrition = current_attrition * (1 - retention_improvement/100)
    future_cost = improved_attrition * replacement_cost
    
    savings = current_cost - future_cost - program_cost
    roi = (savings / program_cost) * 100 if program_cost > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💸 Current Annual Cost", f"${current_cost:,.0f}")
    
    with col2:
        st.metric("💰 Annual Savings", f"${savings:,.0f}")
    
    with col3:
        st.metric("📊 ROI", f"{roi:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>HR Attrition Analytics Dashboard | Built with Streamlit & ML</p>
    <p>Upload your dataset to get personalized insights and recommendations</p>
</div>
""", unsafe_allow_html=True)