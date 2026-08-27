import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any, Optional

CLUSTERING_FEATURES = [
    'Age',
    'Income',
    'Total_Spending',
    'Total_Purchases',
    'Children',
    'NumWebVisitsMonth',
    'Recency',
    'Total_Campaigns',
    'NumDealsPurchases'
]


def create_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """
    Engineers meaningful features for customer personality analysis.
    """
    df = df.copy()

    # 1. Customer Age
    if 'Year_Birth' in df.columns:
        df['Age'] = reference_year - df['Year_Birth']
    elif 'Age' not in df.columns:
        df['Age'] = 45  # fallback default

    # 2. Total Spending
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    existing_spend_cols = [c for c in spending_cols if c in df.columns]
    if existing_spend_cols:
        df['Total_Spending'] = df[existing_spend_cols].sum(axis=1)
    elif 'Total_Spending' not in df.columns:
        df['Total_Spending'] = 0.0

    # 3. Total Purchases
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    existing_purch_cols = [c for c in purchase_cols if c in df.columns]
    if existing_purch_cols:
        df['Total_Purchases'] = df[existing_purch_cols].sum(axis=1)
    elif 'Total_Purchases' not in df.columns:
        df['Total_Purchases'] = 0

    # 4. Children & Parental Status
    kid_col = 'Kidhome' if 'Kidhome' in df.columns else None
    teen_col = 'Teenhome' if 'Teenhome' in df.columns else None
    
    if kid_col and teen_col:
        df['Children'] = df[kid_col] + df[teen_col]
    elif kid_col:
        df['Children'] = df[kid_col]
    elif teen_col:
        df['Children'] = df[teen_col]
    elif 'Children' not in df.columns:
        df['Children'] = 0
        
    df['Is_Parent'] = (df['Children'] > 0).astype(int)

    # 5. Total Campaign Acceptance
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
    existing_cmp_cols = [c for c in cmp_cols if c in df.columns]
    if existing_cmp_cols:
        df['Total_Campaigns'] = df[existing_cmp_cols].sum(axis=1)
    elif 'Total_Campaigns' not in df.columns:
        df['Total_Campaigns'] = 0

    # 6. Customer Tenure (Days)
    if 'Dt_Customer' in df.columns:
        dt = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        # Baseline ref date: max date + 1 day or 2015-01-01
        ref_date = dt.max() if pd.notnull(dt.max()) else pd.Timestamp('2015-01-01')
        df['Customer_Tenure_Days'] = (ref_date - dt).dt.days.fillna(365).astype(int)
    else:
        df['Customer_Tenure_Days'] = 365

    # 7. Education Encoding (Ordinal)
    edu_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
    if 'Education' in df.columns:
        df['Education_Level'] = df['Education'].map(edu_map).fillna(3)

    return df


def prepare_clustering_matrix(
    df: pd.DataFrame, 
    fit_scaler: bool = True, 
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Extracts numerical features and scales them using StandardScaler.
    """
    df_feat = create_features(df)
    
    # Ensure all required features exist
    for col in CLUSTERING_FEATURES:
        if col not in df_feat.columns:
            df_feat[col] = 0

    X = df_feat[CLUSTERING_FEATURES].values

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler, CLUSTERING_FEATURES
