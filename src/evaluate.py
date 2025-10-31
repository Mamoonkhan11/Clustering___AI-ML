# Evaluate clustering model performance

from sklearn.metrics import silhouette_score

def Evaluate_KMeans(model, X_scaled):
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, model.labels_)
    print(f"Inertia: {inertia:.2f}\n")
    print(f"Silhouette Score: {silhouette:.3f}\n")
    return inertia, silhouette
