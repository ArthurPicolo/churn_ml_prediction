#%%
# Importing Libs
#===========================
import streamlit as st
import pandas as pd
import joblib
import mlflow
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

#%%
# Page settings
#===========================
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

# Mlflow connection
#===========================
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Loading Model
#===========================
@st.cache_resource
def load_model():
    """ML Flow latest model"""
    try:
        models = mlflow.search_registered_models(filter_string="name='churn'")
        
        if not models:
            st.error("No model 'churn' found in MLflow!")
            st.stop()
        
        latest_version = max([int(v.version) for v in models[0].latest_versions])
        
        model = mlflow.sklearn.load_model(f'models:/churn/{latest_version}')
        
        features = list(model.feature_names_in_)
        
        return model, features, latest_version
    
    except Exception as e:
        st.error(f"Error while loading model: {str(e)}")
        st.info(f"Details: {e}")
        st.stop()

# Load Model
model_pipeline, best_features, model_version = load_model()

# Header
st.title("Churn Prediction System")
st.caption(f"Model: churn v{model_version} | Features: {len(best_features)}")

# Sidebar 
with st.sidebar:
    st.header("Information")
    st.metric("Model version", model_version)
    st.metric("Total Features", len(best_features))
    
    with st.expander("Features"):
        st.write(best_features)
    
    st.markdown("---")
    st.markdown("### Links")
    st.markdown(f"[MLflow UI](http://127.0.0.1:5000)")
    st.markdown(f"[LinkedIn](https://www.linkedin.com/in/arthur-picolo-dos-reis-00b601210/)")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Upload & Prediction", "Exploratory Analysis", "Individual Prediction"])

# TAB 1: Upload and Batch Prediction
with tab1:
    st.markdown("""
    ### How to use:
    1. Download the CSV template with the necessary features
    2. Fill it with your client data
    3. Upload the completed file
    4. Receive and analyze the predictions
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate template
        template_df = pd.DataFrame(columns=best_features)
        template_df.loc[0] = [0.0] * len(best_features)
        
        buffer = BytesIO()
        template_df.to_csv(buffer, index=False)
        
        st.download_button(
            label="Download CSV Template",
            data=buffer.getvalue(),
            file_name="template_churn.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.info("The template contains all required columns. Fill only with numeric data.")
    
    st.markdown("---")
    
    # File upload
    uploaded = st.file_uploader(
        "Upload the completed CSV file",
        type=['csv'],
        help="CSV file with the same columns as the template"
    )
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            
            # Validate columns
            missing_cols = set(best_features) - set(df.columns)
            extra_cols = set(df.columns) - set(best_features)
            
            if missing_cols:
                st.error(f"Missing columns in file: {missing_cols}")
                st.stop()
            
            if extra_cols:
                st.warning(f"Extra columns will be ignored: {extra_cols}")
            
            st.success(f"{len(df)} records loaded successfully")
            
            # Data preview
            with st.expander("View loaded data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
            
            # Processing button
            if st.button("Process Predictions", type="primary", use_container_width=True):
                with st.spinner("Calculating churn probabilities..."):
                    
                    # Make predictions
                    df['churn_probability'] = model_pipeline.predict_proba(df[best_features])[:, 1]
                    df['churn_prediction'] = model_pipeline.predict(df[best_features])
                    df['risk'] = pd.cut(
                        df['churn_probability'],
                        bins=[0, 0.4, 0.7, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Main metrics
                    st.subheader("Overall Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(df)
                    churn_count = (df['churn_prediction'] == 1).sum()
                    churn_rate = (df['churn_prediction'] == 1).mean() * 100
                    high_risk = (df['risk'] == 'High').sum()
                    
                    col1.metric("Total Clients", f"{total:,}")
                    col2.metric("Predicted Churn", f"{churn_count:,}", 
                               delta=f"{churn_rate:.1f}%", delta_color="inverse")
                    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
                    col4.metric("High Risk", f"{high_risk:,}")
                    
                    # Charts
                    st.markdown("---")
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Risk distribution chart
                        risk_counts = df['risk'].value_counts()
                        fig1 = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            labels={'x': 'Risk Level', 'y': 'Count'},
                            title="Distribution by Risk Level",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Low': 'green',
                                'Medium': 'gold',
                                'High': 'red'
                            }
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_chart2:
                        # Probability histogram
                        fig2 = px.histogram(
                            df,
                            x='churn_probability',
                            nbins=30,
                            title="Probability Distribution",
                            labels={'churn_probability': 'Churn Probability'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Top risk clients
                    st.markdown("---")
                    st.subheader("Top 20 Clients with Highest Churn Risk")
                    
                    top_risk = df.nlargest(20, 'churn_probability').copy()
                    
                    # Create DataFrame for display
                    top_risk_display = top_risk[['churn_probability', 'churn_prediction', 'risk']].copy()
                    top_risk_display['churn_probability'] = top_risk_display['churn_probability'].apply(
                        lambda x: f"{x*100:.2f}%"
                    )
                    
                    st.dataframe(
                        top_risk_display,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Complete table
                    st.markdown("---")
                    st.subheader("All Results")
                    
                    # Prepare dataframe for display
                    df_display = df.copy()
                    df_display['prob_%'] = df_display['churn_probability'].apply(
                        lambda x: f"{x*100:.2f}%"
                    )
                    
                    # Reorder columns for display
                    display_cols = ['prob_%', 'churn_prediction', 'risk'] + [col for col in df_display.columns if col not in ['prob_%', 'churn_prediction', 'risk', 'churn_probability']]
                    
                    st.dataframe(
                        df_display[display_cols].sort_values('prob_%', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download results
                    st.markdown("---")
                    col_down1, col_down2 = st.columns(2)
                    
                    with col_down1:
                        csv_full = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Complete Results (CSV)",
                            data=csv_full,
                            file_name=f"churn_predictions_full.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_down2:
                        csv_risk = df[df['risk'] == 'High'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download High Risk Only (CSV)",
                            data=csv_risk,
                            file_name=f"clients_high_risk.csv",
                            mime="text/csv",
                            use_container_width=True,
                            disabled=(df['risk'] == 'High').sum() == 0
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

# TAB 2: Exploratory Analysis
with tab2:
    st.header("Exploratory Analysis")
    
    if uploaded and 'df' in locals():
        st.info("Analyze data patterns after processing predictions in the previous tab")
        
        if 'churn_probability' in df.columns:
            # Descriptive statistics
            st.subheader("Probability Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df['churn_probability'].mean()*100:.1f}%")
            col2.metric("Median", f"{df['churn_probability'].median()*100:.1f}%")
            col3.metric("Minimum", f"{df['churn_probability'].min()*100:.1f}%")
            col4.metric("Maximum", f"{df['churn_probability'].max()*100:.1f}%")
            
            # Feature correlation (top 10)
            st.subheader("Top 10 Features Correlated with Churn")
            
            correlations = df[best_features].corrwith(df['churn_probability']).abs().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation='h',
                labels={'x': 'Absolute Correlation', 'y': 'Feature'},
                title="Feature Correlation with Churn Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Process predictions first in the 'Upload & Prediction' tab")
    else:
        st.info("Upload a CSV file in the 'Upload & Prediction' tab to view analysis")

# TAB 3: Individual Prediction
with tab3:
    st.header("Individual Prediction")
    st.info("Fill in a single client's data to get churn prediction")
    
    with st.form("individual_prediction"):
        # Organize in columns
        num_cols = 3
        cols = st.columns(num_cols)
        
        input_data = {}
        
        for idx, feature in enumerate(best_features):
            col_idx = idx % num_cols
            with cols[col_idx]:
                input_data[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"input_{feature}"
                )
        
        submitted = st.form_submit_button("Calculate Probability", use_container_width=True)
        
        if submitted:
            df_input = pd.DataFrame([input_data])
            
            prob = model_pipeline.predict_proba(df_input[best_features])[0, 1]
            pred = model_pipeline.predict(df_input[best_features])[0]
            
            st.markdown("---")
            st.subheader("Result")
            
            col1, col2, col3 = st.columns(3)
            
            risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            
            col1.metric("Churn Probability", f"{prob*100:.2f}%")
            col2.metric("Prediction", "CHURN" if pred == 1 else "ACTIVE")
            col3.metric("Risk Level", risk)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if prob > 0.7:
                st.error("HIGH ALERT: Client with high churn probability")
                st.markdown("""
                **Recommended Actions:**
                - Immediate contact with client
                - Offer special benefits or discounts
                - Investigate dissatisfaction reasons
                - Escalate to account manager
                """)
            elif prob > 0.4:
                st.warning("ATTENTION: Client at moderate risk")
                st.markdown("""
                **Recommended Actions:**
                - Monitor behavior closely
                - Consider preventive retention actions
                - Send satisfaction survey
                """)
            else:
                st.success("LOW RISK: Client with low churn probability")
                st.markdown("""
                **Recommended Actions:**
                - Maintain service quality
                - Continue regular monitoring
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Churn Prediction System | Powered by MLflow & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
#%%