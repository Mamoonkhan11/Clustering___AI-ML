# Main script to run K-Means clustering on Mall Customers dataset

from src.Load__data import Load_data
from src.models import Train_KMeans
from src.evaluate import Evaluate_KMeans
from src.utils import Plot_elbow, Plot_clusters

def main():
    print("Loading Mall Customers dataset...\n")
    X, X_scaled = Load_data("Data/Mall_Customers.csv")

    print("Finding optimal number of clusters using Elbow Method...\n")
    Plot_elbow(X_scaled, max_k=10, out_path="outputs/elbow_method.png")

    print("Training K-Means model with K=5...\n")
    model = Train_KMeans(X_scaled, n_clusters=5)

    print("Evaluating model...\n")
    Evaluate_KMeans(model, X_scaled)

    print("Visualizing clusters...\n")
    Plot_clusters(X, model, out_path="outputs/clusters_visualization.png")

    print("Clustering complete! Check 'outputs/' folder for results.\n")

if __name__ == "__main__":
    main()