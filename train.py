import os
import joblib
import pandas as pd
import numpy as np
from src.data_preprocessing import fetch_or_generate_dataset, load_raw_data, clean_data
from src.feature_engineering import create_features, prepare_clustering_matrix
from src.clustering import evaluate_elbow_and_silhouette, train_kmeans, fit_pca_projection, profile_and_label_clusters

MODELS_DIR = "models"


def main():
    print("=" * 70)
    print("      CUSTOMER PERSONALITY DETECTOR - ML TRAINING PIPELINE      ")
    print("=" * 70)

    # 1. Fetch / Load Dataset
    data_path = fetch_or_generate_dataset()
    print(f"\n[1/6] Loading dataset from: {data_path}")
    raw_df = load_raw_data(data_path)
    print(f"      Raw dataset loaded successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")

    # 2. Clean Data
    print("\n[2/6] Performing Data Preprocessing & Cleaning...")
    cleaned_df = clean_data(raw_df)
    print(f"      Cleaned dataset shape: {cleaned_df.shape[0]} rows.")

    # 3. Feature Engineering & Scaling
    print("\n[3/6] Engineering Features & Scaling Numerical Attributes...")
    df_feat = create_features(cleaned_df)
    X_scaled, scaler, feature_names = prepare_clustering_matrix(cleaned_df, fit_scaler=True)
    print(f"      Clustering Matrix prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features.")
    print(f"      Features: {', '.join(feature_names)}")

    # 4. Clustering Evaluation (Elbow & Silhouette)
    print("\n[4/6] Evaluating K-Means Clustering Performance (k=2..8)...")
    inertias, silhouette_scores = evaluate_elbow_and_silhouette(X_scaled, max_k=8)
    for k in range(2, 9):
        print(f"      k={k} -> Inertia: {inertias[k]:,.2f} | Silhouette Score: {silhouette_scores[k]:.4f}")

    # Optimal k selection: 5 clusters for 5 distinct customer personalities
    optimal_k = 5
    print(f"\n[5/6] Training K-Means Model with optimal k={optimal_k}...")
    kmeans_model = train_kmeans(X_scaled, n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans_model.labels_

    # PCA 2D Projection
    pca_model, X_pca = fit_pca_projection(X_scaled, n_components=2)
    print(f"      PCA fitted: Explained Variance Ratio = {pca_model.explained_variance_ratio_.sum():.2%}")

    # Profile & Label Clusters
    cluster_mapping, cluster_summary, cluster_details = profile_and_label_clusters(df_feat, cluster_labels)
    
    print("\n" + "-" * 70)
    print("                      CLUSTER ANALYSIS & PERSONALITIES          ")
    print("-" * 70)
    print(cluster_summary.to_string(index=False))

    # 5. Save Artifacts
    print("\n[6/6] Saving ML Model Artifacts to 'models/' directory...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca_model, os.path.join(MODELS_DIR, "pca_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    
    metadata = {
        "optimal_k": optimal_k,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "cluster_mapping": cluster_mapping,
        "cluster_details": cluster_details,
        "feature_names": feature_names,
        "pca_explained_variance": list(pca_model.explained_variance_ratio_)
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "cluster_metadata.pkl"))

    # Also save cleaned dataset with cluster labels for dashboard visualizations
    df_feat['Cluster'] = cluster_labels
    df_feat['Personality'] = df_feat['Cluster'].map(cluster_mapping)
    df_feat['PCA_1'] = X_pca[:, 0]
    df_feat['PCA_2'] = X_pca[:, 1]
    
    processed_data_path = os.path.join("data", "processed_customers.csv")
    df_feat.to_csv(processed_data_path, index=False)

    print("\nSUCCESS: All model artifacts and processed dataset saved successfully!")
    print(f"Processed dataset saved to: {processed_data_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
