import os
import pandas as pd
from src.prediction import predict_customer_personality, load_artifacts


def test_end_to_end_pipeline():
    print("Running end-to-end ML pipeline verification test...")

    # 1. Load artifacts test
    artifacts = load_artifacts("models")
    assert "kmeans" in artifacts, "kmeans model missing"
    assert "scaler" in artifacts, "scaler missing"
    assert "pca" in artifacts, "pca model missing"
    print("[OK] Model artifacts loaded successfully.")

    # 2. Test input customer payloads
    test_customers = [
        {
            "name": "High Income VIP",
            "payload": {
                "Year_Birth": 1975,
                "Income": 95000,
                "Education": "Graduation",
                "Marital_Status": "Married",
                "Kidhome": 0,
                "Teenhome": 0,
                "MntWines": 700,
                "MntFruits": 80,
                "MntMeatProducts": 500,
                "MntFishProducts": 90,
                "MntSweetProducts": 70,
                "MntGoldProds": 80,
                "NumDealsPurchases": 1,
                "NumWebPurchases": 6,
                "NumCatalogPurchases": 8,
                "NumStorePurchases": 10,
                "NumWebVisitsMonth": 2,
                "Recency": 25,
                "AcceptedCmp1": 1,
                "AcceptedCmp2": 0,
                "AcceptedCmp3": 0,
                "AcceptedCmp4": 1,
                "AcceptedCmp5": 1,
                "Response": 1
            }
        },
        {
            "name": "Budget Saver",
            "payload": {
                "Year_Birth": 1988,
                "Income": 28000,
                "Education": "Basic",
                "Marital_Status": "Single",
                "Kidhome": 1,
                "Teenhome": 0,
                "MntWines": 30,
                "MntFruits": 10,
                "MntMeatProducts": 20,
                "MntFishProducts": 5,
                "MntSweetProducts": 5,
                "MntGoldProds": 10,
                "NumDealsPurchases": 6,
                "NumWebPurchases": 2,
                "NumCatalogPurchases": 0,
                "NumStorePurchases": 3,
                "NumWebVisitsMonth": 8,
                "Recency": 60,
                "AcceptedCmp1": 0,
                "AcceptedCmp2": 0,
                "AcceptedCmp3": 0,
                "AcceptedCmp4": 0,
                "AcceptedCmp5": 0,
                "Response": 0
            }
        }
    ]

    for tc in test_customers:
        res = predict_customer_personality(tc["payload"], models_dir="models")
        print(f"\n[Test Case] {tc['name']}:")
        print(f"  Predicted Cluster: {res['cluster_id']}")
        print(f"  Personality Label: {res['personality_label']}")
        print(f"  Tagline: {res['tagline']}")
        print(f"  Strategy: {res['strategy']}")
        print(f"  Distance to Centroid: {res['distance_to_centroid']}")
        print(f"  PCA Coordinates: {res['pca_coords']}")
        assert res["personality_label"] is not None
        assert len(res["action_points"]) > 0

    print("\n[OK] ALL END-TO-END PIPELINE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_end_to_end_pipeline()
