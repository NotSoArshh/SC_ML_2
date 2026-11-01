"""
Customer Segmentation using K-Means Clustering
==============================================
This script implements K-Means Clustering to group retail store customers 
based on their annual income and spending score.

Dataset expected: Mall_Customers.csv

Usage:
    python customer_segmentation.py --data Mall_Customers.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_and_preprocess(filepath: str):
    """
    Load dataset and extract features for clustering.
    
    Parameters:
        filepath (str): Path to Mall_Customers.csv
        
    Returns:
        df (pd.DataFrame): Original DataFrame
        X (pd.DataFrame): Feature matrix (Annual Income and Spending Score)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
        
    df = pd.read_csv(filepath)
    
    # Standard columns in Mall_Customers.csv:
    # CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
    income_col = "Annual Income (k$)"
    spending_col = "Spending Score (1-100)"
    
    # Check for column existence (handles slight variations in naming)
    for col in df.columns:
        if "annual income" in col.lower():
            income_col = col
        elif "spending score" in col.lower():
            spending_col = col

    features = [income_col, spending_col]
    X = df[features].copy()
    
    return df, X, features


def find_optimal_clusters(X: pd.DataFrame, max_k: int = 10):
    """
    Determine the optimal number of clusters using the Elbow Method and Silhouette Score.
    """
    wcss = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    print("\n--- Silhouette Scores by Cluster Count ---")
    for k, score in zip(range(2, max_k + 1), silhouette_scores):
        print(f"Clusters: {k} | Silhouette Score: {score:.4f}")

    return wcss, silhouette_scores


def train_kmeans(X: pd.DataFrame, n_clusters: int = 5):
    """
    Fit K-Means clustering algorithm with the chosen number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    return kmeans, cluster_labels


def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation using K-Means Clustering")
    parser.add_argument("--data", type=str, default="Mall_Customers.csv", help="Path to Mall_Customers.csv")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for final model")
    args = parser.parse_args()

    try:
        df, X, features = load_and_preprocess(args.data)
        print(f"Dataset loaded successfully with {len(df)} records.")
        
        # Determine performance metrics across cluster counts
        find_optimal_clusters(X, max_k=10)
        
        # Train final model
        kmeans, labels = train_kmeans(X, n_clusters=args.clusters)
        df['Cluster'] = labels
        
        print("\n" + "="*45)
        print(f"       K-MEANS CLUSTERING COMPLETED (K={args.clusters})       ")
        print("="*45)
        
        # Cluster Analysis Summary
        summary = df.groupby('Cluster')[features].mean()
        summary['Customer Count'] = df.groupby('Cluster').size()
        print(summary)
        print("="*45 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()