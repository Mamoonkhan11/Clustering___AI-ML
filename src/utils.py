# Utility functions for clustering visualizations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def Plot_elbow(X_scaled, max_k=10, out_path="Outputs/elbow_method.png"):
    inertias = []
    K_values = range(1, max_k + 1)
    for k in K_values:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, inertias, 'bo-')
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def Plot_clusters(X, model, out_path="Outputs/clusters_visualization.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X["Annual Income (k$)"],
        y=X["Spending Score (1-100)"],
        hue=model.labels_,
        palette="Set1",
        s=80
    )
    plt.title("Customer Segments based on Annual Income and Spending Score")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Cluster")
    plt.savefig(out_path)
    plt.close()