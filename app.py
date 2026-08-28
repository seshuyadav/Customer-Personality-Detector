import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.prediction import predict_customer_personality, load_artifacts, RECOMMENDATIONS
from src.data_preprocessing import fetch_or_generate_dataset, load_raw_data, clean_data
from src.feature_engineering import create_features

# Page Configuration
st.set_page_config(
    page_title="Customer Personality Detector & Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Aesthetic
st.markdown("""
    <style>
    /* Global Styles & Glassmorphism */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Header Container */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 24px;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Personality Result Badge */
    .result-badge {
        padding: 16px 24px;
        border-radius: 12px;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Action Points List */
    .action-item {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dashboard_data():
    processed_path = os.path.join("data", "processed_customers.csv")
    kmeans_path = os.path.join("models", "kmeans_model.pkl")

    # If processed data or models are missing (e.g. on Streamlit Cloud deploy), auto-train on startup
    if not (os.path.exists(processed_path) and os.path.exists(kmeans_path)):
        try:
            from train import main as run_training_pipeline
            run_training_pipeline()
        except Exception as e:
            st.warning(f"Auto-training on startup encountered an issue: {e}")

    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)

    # Fallback load raw & process on the fly
    raw_path = fetch_or_generate_dataset()
    raw_df = load_raw_data(raw_path)
    cleaned_df = clean_data(raw_df)
    df_feat = create_features(cleaned_df)
    return df_feat


def main():
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/user-group-man-man.png", width=70)
    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Machine Learning Customer Segmentation")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home / Overview",
            "📊 Analytics Dashboard",
            "🎯 Customer Personality Detector",
            "💡 Marketing Recommendations Matrix"
        ]
    )

    df_data = get_dashboard_data()

    if page == "🏠 Home / Overview":
        render_home_page()
    elif page == "📊 Analytics Dashboard":
        render_analytics_dashboard(df_data)
    elif page == "🎯 Customer Personality Detector":
        render_detector_page(df_data)
    elif page == "💡 Marketing Recommendations Matrix":
        render_strategy_matrix()


def render_home_page():
    st.markdown("""
        <div class="header-box">
            <h1 style="margin:0; font-size:2.5rem; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎯 Customer Personality Detector
            </h1>
            <p style="color:#94A3B8; font-size:1.1rem; margin-top:8px;">
                An end-to-end Data Science & Machine Learning platform for automated customer segmentation, behavioral analytics, and targeted marketing strategy generation.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📌 Project Objectives")
        st.markdown("""
        - **Data-Driven Customer Segmentation**: Group customers based on purchasing behavior, revenue contribution, demographics, and channel activity using **K-Means Clustering**.
        - **Dimensionality Reduction**: Apply **Principal Component Analysis (PCA)** to map high-dimensional customer features into intuitive 2D visual spaces.
        - **Personality Profiling**: Dynamically translate abstract cluster centroids into actionable business profiles (*Premium Loyal*, *High-Potential*, *Budget-Conscious*, *Occasional*, *At-Risk*).
        - **Real-Time Inference**: Enable marketers and sales teams to input single customer attributes and instantly receive cluster classifications and customized marketing recommendations.
        """)

        st.markdown("### ⚙️ Machine Learning Pipeline Architecture")
        st.markdown("""
        1. **Data Ingestion & Cleaning**: Automated handling of missing income data, outlier filtering, and data quality validation.
        2. **Feature Engineering**: Derivation of key metrics like `Total_Spending`, `Total_Purchases`, `Age`, `Children`, and `Total_Campaigns`.
        3. **Feature Normalization**: Standardizing numerical attributes using `StandardScaler`.
        4. **Model Optimization**: Finding optimal cluster counts `k` via **Elbow Method** (Inertia) and **Silhouette Score** analysis.
        5. **Model Persistence**: Serializing trained scikit-learn models and metadata with `joblib`.
        """)

    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.info("""
        - **Language**: Python 3.11
        - **Data Science**: Pandas, NumPy
        - **Machine Learning**: Scikit-Learn (K-Means, PCA, StandardScaler)
        - **Visualization**: Plotly Express, Seaborn, Matplotlib
        - **Web Application**: Streamlit Framework
        - **Model Persistence**: Joblib
        """)

        st.markdown("### 👥 Targeted Personality Profiles")
        for name, details in RECOMMENDATIONS.items():
            st.markdown(f"**<span style='color:{details['color']};'>■ {name}</span>**: *{details['tagline']}*", unsafe_allow_html=True)


def render_analytics_dashboard(df: pd.DataFrame):
    st.markdown("## 📊 Customer Analytics & Segmentation Dashboard")

    # KPI Top Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total_cust = len(df)
    avg_income = df['Income'].mean() if 'Income' in df else 0
    avg_spend = df['Total_Spending'].mean() if 'Total_Spending' in df else 0
    num_segments = df['Personality'].nunique() if 'Personality' in df else 5
    avg_recency = df['Recency'].mean() if 'Recency' in df else 0

    c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_income:,.0f}</div>
            <div class="metric-label">Avg Annual Income</div>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_spend:,.0f}</div>
            <div class="metric-label">Avg Total Spend</div>
        </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_segments}</div>
            <div class="metric-label">Customer Segments</div>
        </div>
    """, unsafe_allow_html=True)

    c5.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_recency:.1f} days</div>
            <div class="metric-label">Avg Recency</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 Charts: PCA Visualization & Segment Distribution
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📍 2D PCA Cluster Map")
        if 'PCA_1' in df.columns and 'PCA_2' in df.columns:
            fig_pca = px.scatter(
                df,
                x='PCA_1',
                y='PCA_2',
                color='Personality',
                hover_data=['Age', 'Income', 'Total_Spending', 'Total_Purchases'],
                title="Customer Clusters Projected onto First 2 Principal Components",
                template="plotly_dark",
                color_discrete_map={k: v['color'] for k, v in RECOMMENDATIONS.items()}
            )
            fig_pca.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA coordinates being computed dynamically...")

    with col2:
        st.subheader("🍩 Segment Share Distribution")
        if 'Personality' in df.columns:
            seg_counts = df['Personality'].value_counts().reset_index()
            seg_counts.columns = ['Personality', 'Count']
            fig_pie = px.pie(
                seg_counts,
                names='Personality',
                values='Count',
                hole=0.45,
                template="plotly_dark",
                color='Personality',
                color_discrete_map={k: v['color'] for k, v in RECOMMENDATIONS.items()}
            )
            fig_pie.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    # Row 2 Charts: Income vs Spending Scatter & Product Category Spending
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💰 Income vs. Total Spending Relationship")
        if 'Income' in df.columns and 'Total_Spending' in df.columns:
            fig_inc_spend = px.scatter(
                df,
                x='Income',
                y='Total_Spending',
                color='Personality' if 'Personality' in df else None,
                opacity=0.7,
                template="plotly_dark",
                title="Annual Income vs. Total Spending by Segment",
                color_discrete_map={k: v['color'] for k, v in RECOMMENDATIONS.items()}
            )
            fig_inc_spend.update_layout(height=400)
            st.plotly_chart(fig_inc_spend, use_container_width=True)

    with col4:
        st.subheader("🍷 Product Category Spending Breakdown")
        spend_cols = ['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        existing_spend = [c for c in spend_cols if c in df.columns]
        if existing_spend and 'Personality' in df.columns:
            spend_df = df.groupby('Personality')[existing_spend].mean().reset_index()
            spend_melted = spend_df.melt(id_vars='Personality', var_name='Category', value_name='Avg_Spend')
            spend_melted['Category'] = spend_melted['Category'].str.replace('Mnt', '')
            
            fig_bar = px.bar(
                spend_melted,
                x='Personality',
                y='Avg_Spend',
                color='Category',
                barmode='stack',
                template="plotly_dark",
                title="Average Spend per Product Category by Personality"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    # Row 3: Age & Purchase Behaviors
    c6, c7 = st.columns(2)
    with c6:
        st.subheader("🎂 Age Distribution per Personality")
        if 'Age' in df.columns and 'Personality' in df.columns:
            fig_box = px.box(
                df,
                x='Personality',
                y='Age',
                color='Personality',
                template="plotly_dark",
                title="Customer Age Box Plot across Segments",
                color_discrete_map={k: v['color'] for k, v in RECOMMENDATIONS.items()}
            )
            fig_box.update_layout(height=380)
            st.plotly_chart(fig_box, use_container_width=True)

    with c7:
        st.subheader("🌐 Web Visits vs. Store Purchases")
        if 'NumWebVisitsMonth' in df.columns and 'NumStorePurchases' in df.columns:
            has_personality = 'Personality' in df.columns
            fig_web_store = px.scatter(
                df,
                x='NumWebVisitsMonth',
                y='NumStorePurchases',
                color='Personality' if has_personality else None,
                size='Total_Spending' if 'Total_Spending' in df.columns else None,
                template="plotly_dark",
                title="Monthly Web Visits vs Store Purchases",
                color_discrete_map={k: v['color'] for k, v in RECOMMENDATIONS.items()} if has_personality else None
            )
            fig_web_store.update_layout(height=380)
            st.plotly_chart(fig_web_store, use_container_width=True)


def render_detector_page(df_data: pd.DataFrame):
    st.markdown("## 🎯 Customer Personality Detector")
    st.caption("Enter customer attributes to run real-time K-Means segmentation & strategy generation.")

    with st.form("customer_input_form"):
        st.markdown("### 👤 Demographic & Income Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            year_birth = st.number_input("Year of Birth", min_value=1940, max_value=2010, value=1980, step=1)
            education = st.selectbox("Education Level", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
        with col2:
            income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=55000, step=1000)
            marital_status = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow"])
        with col3:
            kidhome = st.number_input("Kids at Home", min_value=0, max_value=3, value=0, step=1)
            teenhome = st.number_input("Teens at Home", min_value=0, max_value=3, value=0, step=1)

        st.markdown("---")
        st.markdown("### 🛒 Product Category Spending ($)")
        c1, c2, c3 = st.columns(3)
        with c1:
            wines = st.number_input("Wine Spending ($)", min_value=0, max_value=1500, value=250, step=10)
            meat = st.number_input("Meat Products Spending ($)", min_value=0, max_value=1000, value=120, step=10)
        with c2:
            fruits = st.number_input("Fruits Spending ($)", min_value=0, max_value=300, value=25, step=5)
            fish = st.number_input("Fish Products Spending ($)", min_value=0, max_value=300, value=35, step=5)
        with c3:
            sweets = st.number_input("Sweets Spending ($)", min_value=0, max_value=300, value=20, step=5)
            gold = st.number_input("Gold Products Spending ($)", min_value=0, max_value=300, value=30, step=5)

        st.markdown("---")
        st.markdown("### 🌐 Purchasing Channels & Engagement Behavior")
        b1, b2, b3 = st.columns(3)
        with b1:
            web_purchases = st.number_input("Web Purchases", min_value=0, max_value=30, value=4, step=1)
            catalog_purchases = st.number_input("Catalog Purchases", min_value=0, max_value=30, value=2, step=1)
            store_purchases = st.number_input("Store Purchases", min_value=0, max_value=30, value=5, step=1)
        with b2:
            deals_purchases = st.number_input("Discount Deals Purchases", min_value=0, max_value=20, value=2, step=1)
            web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=30, value=5, step=1)
            recency = st.number_input("Recency (Days since last order)", min_value=0, max_value=100, value=30, step=1)
        with b3:
            st.markdown("**Campaign Acceptance History**")
            cmp1 = st.checkbox("Accepted Campaign 1")
            cmp2 = st.checkbox("Accepted Campaign 2")
            cmp3 = st.checkbox("Accepted Campaign 3")
            cmp4 = st.checkbox("Accepted Campaign 4")
            cmp5 = st.checkbox("Accepted Campaign 5")
            response = st.checkbox("Accepted Last Offer (Response)")

        submit_btn = st.form_submit_button("⚡ Detect Customer Personality", type="primary", use_container_width=True)

    if submit_btn:
        customer_payload = {
            "Year_Birth": year_birth,
            "Education": education,
            "Marital_Status": marital_status,
            "Income": income,
            "Kidhome": kidhome,
            "Teenhome": teenhome,
            "MntWines": wines,
            "MntMeatProducts": meat,
            "MntFruits": fruits,
            "MntFishProducts": fish,
            "MntSweetProducts": sweets,
            "MntGoldProds": gold,
            "NumWebPurchases": web_purchases,
            "NumCatalogPurchases": catalog_purchases,
            "NumStorePurchases": store_purchases,
            "NumDealsPurchases": deals_purchases,
            "NumWebVisitsMonth": web_visits,
            "Recency": recency,
            "AcceptedCmp1": int(cmp1),
            "AcceptedCmp2": int(cmp2),
            "AcceptedCmp3": int(cmp3),
            "AcceptedCmp4": int(cmp4),
            "AcceptedCmp5": int(cmp5),
            "Response": int(response)
        }

        try:
            result = predict_customer_personality(customer_payload, models_dir="models")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏆 Prediction Results & Customer Profile")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="result-badge" style="background-color: {result['color']};">
                        {result['personality_label']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.markdown(f"**Target Tagline**: *{result['tagline']}*")
                st.markdown(f"**Cluster ID**: `Cluster #{result['cluster_id']}`")
                st.markdown(f"**Centroid Fit Distance**: `{result['distance_to_centroid']} Euclidean Units`")
                st.markdown(f"**Core Strategy**: **{result['strategy']}**")

                st.markdown("#### 🎯 Strategic Action Points")
                for point in result['action_points']:
                    st.markdown(f"<div class='action-item'>✓ {point}</div>", unsafe_allow_html=True)

                st.markdown(f"**Primary Communication Channels**: {', '.join(result['primary_channels'])}")

            with res_col2:
                st.markdown("#### 📊 Customer vs. Segment Centroid Comparison")
                # Bar comparison chart
                comp_data = pd.DataFrame({
                    "Attribute": ["Income ($)", "Total Spend ($)", "Total Purchases", "Web Visits", "Recency (Days)"],
                    "Input Customer": [
                        result['input_features']['Income'],
                        result['input_features']['Total_Spending'],
                        result['input_features']['Total_Purchases'],
                        result['input_features']['NumWebVisitsMonth'],
                        result['input_features']['Recency']
                    ]
                })
                
                fig_comp = px.bar(
                    comp_data,
                    x="Attribute",
                    y="Input Customer",
                    text_auto=True,
                    template="plotly_dark",
                    title="Key Input Attribute Summary"
                )
                fig_comp.update_traces(marker_color=result['color'])
                fig_comp.update_layout(height=380)
                st.plotly_chart(fig_comp, use_container_width=True)

        except Exception as e:
            st.error(f"Error during personality prediction: {e}")


def render_strategy_matrix():
    st.markdown("## 💡 Marketing Strategy & Segment Playbook")
    st.caption("Customized engagement playbooks tailored for each customer personality category.")

    for name, details in RECOMMENDATIONS.items():
        with st.expander(f"🔹 {name} ({details['tagline']})", expanded=True):
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown(f"<h4 style='color:{details['color']};'>Strategy Objective</h4>", unsafe_allow_html=True)
                st.markdown(f"**{details['strategy']}**")
                st.markdown("**Preferred Channels:**")
                for ch in details['primary_channels']:
                    st.markdown(f"- {ch}")
            with cols[1]:
                st.markdown("#### Actionable Playbook Rules")
                for pt in details['action_points']:
                    st.markdown(f"• {pt}")


if __name__ == "__main__":
    main()
