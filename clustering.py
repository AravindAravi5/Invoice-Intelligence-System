import logging
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, Any, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvoiceClusterer:
    """
    Clustering module for grouping similar invoices based on their dense vector embeddings.
    
    Why DBSCAN?
    Unlike K-Means, which forces every point into a cluster and requires you to know the 
    number of clusters in advance, DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    1. Does not require the number of clusters (k) to be specified.
    2. Can find clusters of arbitrary shape.
    3. Explicitly isolates outliers/noise (labeled as -1). This is incredibly useful 
       for financial documents, as noise points naturally highlight anomalies or rare vendors.
    """
    def __init__(self, eps: float = 0.3, min_samples: int = 3, metric: str = 'cosine'):
        """
        Initialize the clustering module.
        
        Parameter Explanations:
        - eps (epsilon): The maximum distance between two samples for one to be considered 
                         as in the neighborhood of the other. Since we use sentence embeddings,
                         we default to 'cosine' distance. A lower eps means points must be very 
                         similar to be clustered; a higher eps allows looser clusters.
        - min_samples: The number of samples (invoices) in a neighborhood for a point to be 
                       considered as a 'core point'. This includes the point itself. If a vendor
                       only has 2 invoices, and min_samples is 3, they will be marked as noise.
        - metric: The distance metric to use. 'cosine' is best for transformer vector embeddings.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        logging.info(f"Initialized DBSCAN (eps={self.eps}, min_samples={self.min_samples}, metric='{self.metric}')")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster the provided embeddings.
        
        Args:
            embeddings (np.ndarray): The 2D numpy array of embeddings (from embeddings.py).
            
        Returns:
            np.ndarray: An array of cluster labels. A label of -1 means the invoice is noise/anomaly.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Empty embeddings array provided.")
            
        num_samples = len(embeddings)
        logging.info(f"Clustering {num_samples} invoices...")

        # --- Edge Case Handling ---
        
        # Edge Case 1: Small Datasets
        # If we have fewer invoices than the required min_samples, DBSCAN will mark everything 
        # as noise. We gracefully handle this by temporarily reducing min_samples.
        current_min_samples = self.min_samples
        if num_samples < self.min_samples:
            logging.warning(f"Dataset too small ({num_samples} samples) for min_samples={self.min_samples}.")
            if num_samples >= 2:
                current_min_samples = num_samples
                logging.warning(f"Dynamically reducing min_samples to {current_min_samples}.")
            else:
                logging.warning("Only 1 sample provided. It will be labeled as noise (-1).")
                return np.array([-1])
                
        # Edge Case 2: Sparse clusters
        # If the data is extremely spread out, 'eps' might need adjusting. While we don't 
        # auto-tune eps here, using the 'cosine' metric inherently handles scaling issues better 
        # than 'euclidean' distance when dealing with high-dimensional transformer embeddings.

        # Temporarily update the model if we adjusted min_samples
        self.model.set_params(min_samples=current_min_samples)
        
        # Fit and predict
        labels = self.model.fit_predict(embeddings)
        return labels

    def get_cluster_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the resulting labels and return basic statistics.
        """
        if len(labels) == 0:
            return {}
            
        # DBSCAN labels noise as -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        stats = {
            "total_invoices": len(labels),
            "number_of_clusters": n_clusters,
            "noise_points": n_noise,
            "noise_percentage": round((n_noise / len(labels)) * 100, 2)
        }
        return stats


# Example Usage for testing the module
if __name__ == "__main__":
    # --- Simulating Embeddings ---
    # In a real scenario, this comes from embeddings.py. 
    # Here we simulate 10 invoices with 384 dimensions (MiniLM size).
    
    # We will create 2 distinct clusters and 2 random noise points
    np.random.seed(42)
    
    # Cluster 1: 4 very similar invoices (e.g., all from 'Acme Corp')
    cluster_1 = np.random.normal(loc=0.5, scale=0.05, size=(4, 384))
    
    # Cluster 2: 4 very similar invoices (e.g., all from 'Globex Inc')
    cluster_2 = np.random.normal(loc=-0.5, scale=0.05, size=(4, 384))
    
    # Noise: 2 random outliers (e.g., rare vendors or OCR errors)
    noise = np.random.uniform(low=-1.0, high=1.0, size=(2, 384))
    
    # Combine into a single batch
    sample_embeddings = np.vstack([cluster_1, cluster_2, noise])
    
    print("\n--- Running Clustering ---")
    # Initialize the clusterer. We use cosine distance.
    clusterer = InvoiceClusterer(eps=0.1, min_samples=3, metric='cosine')
    
    # Predict clusters
    labels = clusterer.fit_predict(sample_embeddings)
    
    # Print results
    print("\n--- Clustering Results ---")
    for i, label in enumerate(labels):
        if label == -1:
            status = "NOISE (Potential Anomaly)"
        else:
            status = f"Cluster {label}"
        print(f"Invoice {i+1:02d} -> {status}")
        
    print("\n--- Statistics ---")
    stats = clusterer.get_cluster_stats(labels)
    for k, v in stats.items():
        print(f"{k}: {v}")
