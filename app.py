import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="E-Commerce Revenue Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_models():
    try:
        model = joblib.load('revenue_prediction_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, label_encoders, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None, None

model, label_encoders, feature_names, metadata = load_models()

# Header
st.markdown('<div class="main-header">💰 E-Commerce Revenue Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict revenue using AI-powered machine learning</div>', unsafe_allow_html=True)

if model is not None:
    # Display model information in sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title("Model Information")
        st.markdown("---")
        st.metric("Model Type", metadata['model_name'])
        st.metric("R² Score", f"{metadata['test_r2']:.4f}")
        st.metric("MAE", f"₹{metadata['test_mae']:,.2f}")
        st.metric("RMSE", f"₹{metadata['test_rmse']:,.2f}")
        st.metric("MAPE", f"{metadata['test_mape']:.2f}%")
        
        st.markdown("---")
        st.markdown("### 📊 Features Used")
        st.info(f"Total Features: {len(feature_names)}")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "ℹ️ About"])

    # Tab 1: Single Prediction
    with tab1:
        st.header("Enter Product Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Location Info")
            state = st.selectbox(
                "State",
                options=sorted(label_encoders['state'].classes_),
                help="Select the state where the product is sold"
            )
            zone = st.selectbox(
                "Zone",
                options=sorted(label_encoders['zone'].classes_),
                help="Geographic zone"
            )
        
        with col2:
            st.subheader("🛍️ Product Info")
            category = st.selectbox(
                "Category",
                options=sorted(label_encoders['category'].classes_),
                help="Product category"
            )
            brand_type = st.selectbox(
                "Brand Type",
                options=sorted(label_encoders['brand_type'].classes_),
                help="Brand positioning"
            )
        
        with col3:
            st.subheader("👤 Customer Info")
            customer_gender = st.selectbox(
                "Customer Gender",
                options=sorted(label_encoders['customer_gender'].classes_)
            )
            customer_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=30,
                help="Age of the customer"
            )
        
        st.markdown("---")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.subheader("💵 Pricing Info")
            base_price = st.number_input(
                "Base Price (₹)",
                min_value=0.0,
                value=10000.0,
                step=100.0,
                help="Original product price"
            )
            discount_percent = st.slider(
                "Discount (%)",
                min_value=0,
                max_value=100,
                value=20,
                help="Discount percentage"
            )
            final_price = base_price * (1 - discount_percent / 100)
            st.metric("Final Price", f"₹{final_price:,.2f}")
        
        with col5:
            st.subheader("📦 Sales Info")
            units_sold = st.number_input(
                "Units Sold",
                min_value=1,
                value=10,
                help="Number of units sold"
            )
            sales_event = st.selectbox(
                "Sales Event",
                options=sorted(label_encoders['sales_event'].classes_),
                help="Type of sales event"
            )
        
        with col6:
            st.subheader("🎯 Market Info")
            competition_intensity = st.selectbox(
                "Competition Intensity",
                options=sorted(label_encoders['competition_intensity'].classes_),
                help="Level of market competition"
            )
            inventory_pressure = st.selectbox(
                "Inventory Pressure",
                options=sorted(label_encoders['inventory_pressure'].classes_),
                help="Inventory pressure level"
            )
        
        st.markdown("---")
        
        if st.button("🔮 Predict Revenue", type="primary", use_container_width=True):
            # Create input data
            current_date = datetime.now()
            
            input_data = {
                'state': state,
                'zone': zone,
                'category': category,
                'brand_type': brand_type,
                'customer_gender': customer_gender,
                'customer_age': customer_age,
                'base_price': base_price,
                'discount_percent': discount_percent,
                'final_price': final_price,
                'units_sold': units_sold,
                'sales_event': sales_event,
                'competition_intensity': competition_intensity,
                'inventory_pressure': inventory_pressure
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering
            input_df['month'] = current_date.month
            input_df['quarter'] = (current_date.month - 1) // 3 + 1
            input_df['year'] = current_date.year
            input_df['discount_amount'] = input_df['base_price'] - input_df['final_price']
            input_df['price_per_unit'] = input_df['final_price'] / input_df['units_sold']
            input_df['is_festival'] = (input_df['sales_event'] == 'Festival').astype(int)
            input_df['is_premium'] = (input_df['brand_type'] == 'Premium').astype(int)
            input_df['is_high_competition'] = (input_df['competition_intensity'] == 'High').astype(int)
            input_df['is_high_inventory'] = (input_df['inventory_pressure'] == 'High').astype(int)
            
            # Encode categorical features
            for col in metadata['categorical_features']:
                input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Select features in correct order
            input_features = input_df[feature_names]
            
            # Make prediction
            predicted_revenue = model.predict(input_features)[0]
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Predicted Revenue</h2>
                    <h1 style="font-size: 3.5rem; margin: 1rem 0;">₹{predicted_revenue:,.2f}</h1>
                    <p style="font-size: 1.2rem;">Estimated revenue for this transaction</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.subheader("📊 Transaction Insights")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                revenue_per_unit = predicted_revenue / units_sold
                st.metric("Revenue per Unit", f"₹{revenue_per_unit:,.2f}")
            
            with col_b:
                profit_margin = ((predicted_revenue - (final_price * units_sold)) / predicted_revenue * 100) if predicted_revenue > 0 else 0
                st.metric("Estimated Margin", f"{profit_margin:.2f}%")
            
            with col_c:
                discount_impact = base_price - final_price
                st.metric("Discount Impact", f"₹{discount_impact:,.2f}")
            
            with col_d:
                total_sale_value = final_price * units_sold
                st.metric("Total Sale Value", f"₹{total_sale_value:,.2f}")
            
            # Visualization
            st.subheader("📈 Revenue Breakdown")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Base Revenue', 'After Discount', 'Predicted Revenue'],
                    y=[base_price * units_sold, final_price * units_sold, predicted_revenue],
                    marker_color=['#ff7f0e', '#2ca02c', '#1f77b4'],
                    text=[f"₹{base_price * units_sold:,.0f}", 
                          f"₹{final_price * units_sold:,.0f}", 
                          f"₹{predicted_revenue:,.0f}"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Revenue Comparison",
                yaxis_title="Amount (₹)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Batch Prediction
    with tab2:
        st.header("📊 Batch Prediction")
        st.info("Upload a CSV file with multiple records to get bulk predictions")
        
        # Download sample template
        if st.button("📥 Download Sample Template"):
            sample_data = {
                'state': ['Maharashtra', 'Karnataka'],
                'zone': ['West', 'South'],
                'category': ['Electronics', 'Fashion'],
                'brand_type': ['Premium', 'Budget'],
                'customer_gender': ['Male', 'Female'],
                'customer_age': [30, 25],
                'base_price': [50000, 2000],
                'discount_percent': [20, 15],
                'final_price': [40000, 1700],
                'units_sold': [10, 5],
                'sales_event': ['Festival', 'Regular'],
                'competition_intensity': ['Medium', 'Low'],
                'inventory_pressure': ['Low', 'Medium']
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv,
                file_name="prediction_template.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} records")
                
                st.subheader("Preview Data")
                st.dataframe(df.head(10))
                
                if st.button("🔮 Generate Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Feature engineering
                        current_date = datetime.now()
                        df['month'] = current_date.month
                        df['quarter'] = (current_date.month - 1) // 3 + 1
                        df['year'] = current_date.year
                        df['discount_amount'] = df['base_price'] - df['final_price']
                        df['price_per_unit'] = df['final_price'] / df['units_sold']
                        df['is_festival'] = (df['sales_event'] == 'Festival').astype(int)
                        df['is_premium'] = (df['brand_type'] == 'Premium').astype(int)
                        df['is_high_competition'] = (df['competition_intensity'] == 'High').astype(int)
                        df['is_high_inventory'] = (df['inventory_pressure'] == 'High').astype(int)
                        
                        # Encode categorical features
                        df_encoded = df.copy()
                        for col in metadata['categorical_features']:
                            df_encoded[col] = label_encoders[col].transform(df[col])
                        
                        # Make predictions
                        predictions = model.predict(df_encoded[feature_names])
                        df['Predicted_Revenue'] = predictions
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display results
                        st.subheader("Results")
                        st.dataframe(df)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Avg Predicted Revenue", f"₹{df['Predicted_Revenue'].mean():,.2f}")
                        with col3:
                            st.metric("Total Predicted Revenue", f"₹{df['Predicted_Revenue'].sum():,.2f}")
                        with col4:
                            st.metric("Max Predicted Revenue", f"₹{df['Predicted_Revenue'].max():,.2f}")
                        
                        # Download results
                        csv_result = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_result,
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        fig = px.histogram(df, x='Predicted_Revenue', nbins=30, 
                                         title="Distribution of Predicted Revenue")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Tab 3: About
    with tab3:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application uses machine learning to predict e-commerce revenue based on various 
        product, customer, and market factors.
        
        ### 🧠 Model Information
        - **Algorithm**: {}
        - **Features**: {} input features
        - **Performance**: R² = {:.4f}, MAE = ₹{:,.2f}
        
        ### 📊 Key Features
        The model considers the following factors:
        
        **Location Factors**
        - State and Zone
        
        **Product Factors**
        - Category
        - Brand Type
        - Pricing (Base Price, Discount, Final Price)
        
        **Customer Factors**
        - Age
        - Gender
        
        **Market Factors**
        - Competition Intensity
        - Inventory Pressure
        - Sales Events
        
        ### 🚀 How to Use
        
        **Single Prediction**
        1. Enter product and customer details in the form
        2. Click "Predict Revenue"
        3. View the predicted revenue and insights
        
        **Batch Prediction**
        1. Download the CSV template
        2. Fill in your data
        3. Upload the file
        4. Get predictions for all records
        
        ### 📝 Notes
        - All monetary values are in Indian Rupees (₹)
        - The model is trained on historical e-commerce data
        - Predictions are estimates and should be used as guidance
        
        ### 👨‍💻 Developer
        Built with ❤️ using Streamlit and Scikit-learn
        """.format(
            metadata['model_name'],
            len(feature_names),
            metadata['test_r2'],
            metadata['test_mae']
        ))

else:
    st.error("⚠️ Model files not found. Please ensure all required files are in the application directory.")
    st.markdown("""
    ### Required Files:
    - revenue_prediction_model.pkl
    - label_encoders.pkl
    - feature_names.pkl
    - model_metadata.pkl
    """)