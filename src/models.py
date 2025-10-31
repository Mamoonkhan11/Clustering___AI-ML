# Model training for K-Means clustering

from sklearn.cluster import KMeans

def Train_KMeans(X_scaled, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_scaled)
    print(f" Model trained successfully with {n_clusters} clusters.\n")
    return model