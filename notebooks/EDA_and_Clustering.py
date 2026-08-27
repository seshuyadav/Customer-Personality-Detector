"""
Customer Personality Analysis - Exploratory Data Analysis & Clustering Notebook Script
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import fetch_or_generate_dataset, load_raw_data, clean_data
from src.feature_engineering import create_features, prepare_clustering_matrix
from src.clustering import evaluate_elbow_and_silhouette, train_kmeans, fit_pca_projection, profile_and_label_clusters


def run_eda_and_clustering():
    print("Fetching and loading data...")
    data_path = fetch_or_generate_dataset()
    df_raw = load_raw_data(data_path)
    print(f"Raw Data Shape: {df_raw.shape}")

    print("\nCleaning data...")
    df_clean = clean_data(df_raw)
    print(f"Clean Data Shape: {df_clean.shape}")

    print("\nEngineering features...")
    df_feat = create_features(df_clean)
    
    print("\nPreparing scaled matrix...")
    X_scaled, scaler, feature_names = prepare_clustering_matrix(df_clean)

    print("\nEvaluating Elbow Method & Silhouette Scores...")
    inertias, silhouette_scores = evaluate_elbow_and_silhouette(X_scaled, max_k=8)
    for k in range(2, 9):
        print(f"k={k}: Inertia={inertias[k]:,.2f}, Silhouette Score={silhouette_scores[k]:.4f}")

    print("\nTraining final K-Means model (k=5)...")
    kmeans = train_kmeans(X_scaled, n_clusters=5)
    
    print("\nProfiling clusters...")
    mapping, summary, details = profile_and_label_clusters(df_feat, kmeans.labels_)
    print("\nCluster Personalities:")
    for cid, label in mapping.items():
        print(f"  Cluster {cid}: {label}")

    print("\nEDA & Clustering script execution completed successfully.")


if __name__ == "__main__":
    run_eda_and_clustering()
