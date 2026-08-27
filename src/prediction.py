import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.feature_engineering import create_features, CLUSTERING_FEATURES

RECOMMENDATIONS = {
    "Premium Loyal Customer": {
        "tagline": "High-Value VIP Advocate",
        "strategy": "Retention & Exclusive VIP Experience",
        "action_points": [
            "Enroll in Tier-1 VIP Loyalty Program with dedicated concierge support.",
            "Offer exclusive early access to luxury product launches and limited editions.",
            "Provide invitation-only VIP events and personalized thank-you gifts.",
            "Avoid low-value discount spam; emphasize quality, prestige, and premium service."
        ],
        "primary_channels": ["Direct Phone / VIP Concierge", "Personalized Email", "Exclusive Direct Mail"],
        "color": "#1E3A8A" # Navy blue
    },
    "High-Potential Customer": {
        "tagline": "Active Upsell Target",
        "strategy": "Cross-Selling & Upselling",
        "action_points": [
            "Deliver personalized product recommendations based on past category purchases.",
            "Offer spend-threshold incentives (e.g., 'Spend $150, get $30 bonus reward').",
            "Introduce subscription box or membership upgrade trial.",
            "Utilize retargeting ads highlighting premium complementary offerings."
        ],
        "primary_channels": ["Email Marketing", "In-App Popups", "Targeted Social Ads"],
        "color": "#0D9488" # Teal
    },
    "Budget-Conscious Customer": {
        "tagline": "Deal-Driven Saver",
        "strategy": "Value Bundles & Discount Promotions",
        "action_points": [
            "Send regular clearance, flash sale, and discount coupon notifications.",
            "Promote 'Buy One, Get One' (BOGO) deals and high-value product bundles.",
            "Highlight free shipping thresholds and cashback reward incentives.",
            "Optimize web landing pages with clear price savings and discount badges."
        ],
        "primary_channels": ["SMS Marketing", "Promo Email Newsletters", "Deal Push Notifications"],
        "color": "#D97706" # Amber
    },
    "Occasional Customer": {
        "tagline": "Low-Frequency Shopper",
        "strategy": "Re-Engagement & Frequency Nurturing",
        "action_points": [
            "Implement automated re-engagement drip email series after 30 days of inactivity.",
            "Share top-rated customer reviews, bestsellers, and social proof.",
            "Offer time-sensitive welcome-back discount vouchers.",
            "Gather feedback via quick interactive polls or single-click surveys."
        ],
        "primary_channels": ["Retargeting Ads", "Email Drip Campaigns", "Web Banners"],
        "color": "#6B7280" # Slate Gray
    },
    "At-Risk Customer": {
        "tagline": "Churn Risk Alert",
        "strategy": "Win-Back & Churn Prevention",
        "action_points": [
            "Deploy urgent win-back offers with steep discounts (e.g., 'We miss you! 25% off your next purchase').",
            "Send customer service check-in to identify recent satisfaction friction.",
            "Trigger automated exit-intent popups and personalized reactivation messages.",
            "Highlight new features or improved services since their last visit."
        ],
        "primary_channels": ["Reactivation Email", "Targeted SMS", "Social Media Retargeting"],
        "color": "#DC2626" # Crimson Red
    }
}


def load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    """
    Loads all trained model artifacts from the specified models directory.
    """
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    metadata_path = os.path.join(models_dir, "cluster_metadata.pkl")

    if not (os.path.exists(kmeans_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model artifacts not found. Please run model training script (train.py) first.")

    artifacts = {
        "kmeans": joblib.load(kmeans_path),
        "scaler": joblib.load(scaler_path),
        "pca": joblib.load(pca_path) if os.path.exists(pca_path) else None,
        "metadata": joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
    }
    return artifacts


def predict_customer_personality(input_dict: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """
    Takes raw customer input dictionary, applies feature engineering & scaling,
    and returns cluster prediction, personality label, confidence, and marketing recommendations.
    """
    artifacts = load_artifacts(models_dir)
    kmeans = artifacts["kmeans"]
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    metadata = artifacts["metadata"]

    df_input = pd.DataFrame([input_dict])
    
    # Feature engineering
    df_feat = create_features(df_input)

    # Ensure all required features are present
    for col in CLUSTERING_FEATURES:
        if col not in df_feat.columns:
            df_feat[col] = 0

    X_input = df_feat[CLUSTERING_FEATURES].values
    X_scaled = scaler.transform(X_input)

    # Predict cluster
    cluster_id = int(kmeans.predict(X_scaled)[0])

    # Calculate distance to centroid for interpretation/confidence
    centroid = kmeans.cluster_centers_[cluster_id]
    dist_to_centroid = float(np.linalg_norm(X_scaled[0] - centroid)) if hasattr(np, 'linalg_norm') else float(np.sqrt(np.sum((X_scaled[0] - centroid) ** 2)))

    # PCA 2D coordinates for interactive plot placement
    pca_coords = [0.0, 0.0]
    if pca is not None:
        coords = pca.transform(X_scaled)[0]
        pca_coords = [float(round(coords[0], 3)), float(round(coords[1], 3))]

    # Retrieve mapped personality label
    cluster_mapping = metadata.get("cluster_mapping", {})
    personality_label = cluster_mapping.get(cluster_id, f"Customer Segment {cluster_id + 1}")

    rec_info = RECOMMENDATIONS.get(personality_label, RECOMMENDATIONS["Occasional Customer"])

    # Prepare response summary
    return {
        "cluster_id": cluster_id,
        "personality_label": personality_label,
        "tagline": rec_info["tagline"],
        "strategy": rec_info["strategy"],
        "action_points": rec_info["action_points"],
        "primary_channels": rec_info["primary_channels"],
        "color": rec_info["color"],
        "distance_to_centroid": round(dist_to_centroid, 3),
        "pca_coords": pca_coords,
        "input_features": {
            "Age": int(df_feat['Age'].iloc[0]),
            "Income": float(df_feat['Income'].iloc[0]),
            "Total_Spending": float(df_feat['Total_Spending'].iloc[0]),
            "Total_Purchases": int(df_feat['Total_Purchases'].iloc[0]),
            "Children": int(df_feat['Children'].iloc[0]),
            "NumWebVisitsMonth": int(df_feat['NumWebVisitsMonth'].iloc[0]),
            "Recency": int(df_feat['Recency'].iloc[0]),
            "Total_Campaigns": int(df_feat['Total_Campaigns'].iloc[0]),
            "NumDealsPurchases": int(df_feat['NumDealsPurchases'].iloc[0])
        }
    }
