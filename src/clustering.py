import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, Any, List


def evaluate_elbow_and_silhouette(X_scaled: np.ndarray, max_k: int = 8) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Calculates Inertia (Elbow Method) and Silhouette Scores for k from 2 to max_k.
    """
    inertias = {}
    silhouette_scores = {}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias[k] = float(kmeans.inertia_)
        silhouette_scores[k] = float(silhouette_score(X_scaled, labels))

    return inertias, silhouette_scores


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    """
    Fits K-Means clustering model with fixed random state.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    kmeans.fit(X_scaled)
    return kmeans


def fit_pca_projection(X_scaled: np.ndarray, n_components: int = 2, random_state: int = 42) -> Tuple[PCA, np.ndarray]:
    """
    Fits PCA for dimensionality reduction and 2D visual projection.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def profile_and_label_clusters(df: pd.DataFrame, cluster_labels: np.ndarray) -> Tuple[Dict[int, str], pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes actual cluster characteristics and dynamically assigns meaningful
    customer personality labels based on income, spending, purchase, and web visit profiles.
    
    Personality Categories:
    - Premium Loyal Customer
    - High-Potential Customer
    - Budget-Conscious Customer
    - Occasional Customer
    - At-Risk Customer
    """
    df_analysis = df.copy()
    df_analysis['Cluster'] = cluster_labels

    # Compute mean stats per cluster
    metrics = ['Income', 'Total_Spending', 'Total_Purchases', 'NumWebVisitsMonth', 'Recency', 'Total_Campaigns', 'NumDealsPurchases']
    available_metrics = [m for m in metrics if m in df_analysis.columns]

    cluster_summary = df_analysis.groupby('Cluster')[available_metrics].mean().reset_index()

    n_clusters = len(cluster_summary)
    
    # Standard personality names pool
    personality_pool = [
        "Premium Loyal Customer",
        "High-Potential Customer",
        "Budget-Conscious Customer",
        "Occasional Customer",
        "At-Risk Customer"
    ]

    # Rank clusters based on composite score: Income & Total_Spending
    cluster_summary['Spend_Rank'] = cluster_summary['Total_Spending'].rank(ascending=False)
    cluster_summary['Income_Rank'] = cluster_summary['Income'].rank(ascending=False)
    cluster_summary['Composite_Score'] = (cluster_summary['Total_Spending'] * 0.6) + (cluster_summary['Income'] * 0.4)

    # Sort cluster IDs from highest value to lowest value
    sorted_clusters = cluster_summary.sort_values(by='Composite_Score', ascending=False)['Cluster'].tolist()

    cluster_mapping = {}
    
    if n_clusters == 5:
        # Match exactly 5 categories
        cluster_mapping[sorted_clusters[0]] = "Premium Loyal Customer"
        cluster_mapping[sorted_clusters[1]] = "High-Potential Customer"
        cluster_mapping[sorted_clusters[2]] = "Occasional Customer"
        cluster_mapping[sorted_clusters[3]] = "Budget-Conscious Customer"
        cluster_mapping[sorted_clusters[4]] = "At-Risk Customer"
    elif n_clusters == 4:
        cluster_mapping[sorted_clusters[0]] = "Premium Loyal Customer"
        cluster_mapping[sorted_clusters[1]] = "High-Potential Customer"
        cluster_mapping[sorted_clusters[2]] = "Budget-Conscious Customer"
        cluster_mapping[sorted_clusters[3]] = "At-Risk Customer"
    else:
        for idx, cid in enumerate(sorted_clusters):
            cluster_mapping[cid] = personality_pool[idx % len(personality_pool)]

    # Add Assigned Personality to summary table
    cluster_summary['Personality'] = cluster_summary['Cluster'].map(cluster_mapping)

    # Detailed characteristics dictionary
    cluster_details = {}
    for cid, label in cluster_mapping.items():
        c_data = df_analysis[df_analysis['Cluster'] == cid]
        cluster_details[label] = {
            'cluster_id': int(cid),
            'count': int(len(c_data)),
            'percentage': float(round(len(c_data) / len(df_analysis) * 100, 1)),
            'avg_income': float(round(c_data['Income'].mean(), 2)) if 'Income' in c_data else 0,
            'avg_spending': float(round(c_data['Total_Spending'].mean(), 2)) if 'Total_Spending' in c_data else 0,
            'avg_purchases': float(round(c_data['Total_Purchases'].mean(), 1)) if 'Total_Purchases' in c_data else 0,
            'avg_web_visits': float(round(c_data['NumWebVisitsMonth'].mean(), 1)) if 'NumWebVisitsMonth' in c_data else 0,
            'avg_recency': float(round(c_data['Recency'].mean(), 1)) if 'Recency' in c_data else 0,
            'avg_campaigns': float(round(c_data['Total_Campaigns'].mean(), 2)) if 'Total_Campaigns' in c_data else 0,
        }

    return cluster_mapping, cluster_summary, cluster_details
