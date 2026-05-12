import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Union

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvoiceAnomalyDetector:
    """
    Anomaly Detection module for identifying unusual invoices.
    
    How Isolation Forest works:
    Instead of profiling 'normal' data and measuring distance from it, Isolation Forest 
    explicitly isolates anomalies. It does this by randomly selecting a feature and randomly 
    splitting the data. Because anomalies are "few and different", they get isolated much 
    faster (in fewer splits or shorter path lengths) than normal points.
    
    Why it is suitable for Invoice AI:
    1. It handles high-dimensional data (like our embeddings) very efficiently.
    2. It works exceptionally well even when there are very few anomalies in the dataset.
    3. It does not assume normal data follows a Gaussian distribution.
    """
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize the Anomaly Detector.
        
        Args:
            contamination (float): The proportion of outliers in the dataset. Used to define 
                                   the threshold on the scores of the samples. Default is 5%.
            random_state (int): Seed for reproducibility.
        """
        self.contamination = contamination
        # We set n_estimators=100 (100 trees) which is usually sufficient
        self.model = IsolationForest(
            n_estimators=100, 
            contamination=self.contamination, 
            random_state=random_state
        )
        
        # We use an imputer to handle missing values gracefully
        self.imputer = SimpleImputer(strategy='median')
        
        logging.info(f"Initialized Isolation Forest (contamination={self.contamination})")

    def fit_predict(self, data: Union[List[float], np.ndarray], is_1d_amounts: bool = False) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: A list of invoice amounts OR a 2D numpy array of embeddings.
            is_1d_amounts (bool): Set to True if providing a list of flat invoice amounts.
                                  Set to False if providing a matrix of embeddings.
            
        Returns:
            Dict containing:
                - 'labels': Array of 1 (normal) and -1 (anomaly).
                - 'scores': Array of anomaly scores. Lower scores mean more anomalous.
        """
        if data is None or len(data) == 0:
            raise ValueError("Empty data provided.")
            
        # Convert to numpy array
        X = np.array(data, dtype=float)
        
        # Reshape 1D arrays (like amounts) to 2D for scikit-learn
        if is_1d_amounts or len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_samples = len(X)
        logging.info(f"Analyzing {num_samples} records for anomalies...")

        # --- Edge Case 1: Small Datasets ---
        # Isolation forests require enough data to create meaningful splits.
        # If we have very few samples, we cannot confidently flag anomalies.
        if num_samples < 5:
            logging.warning(f"Dataset too small ({num_samples} samples). Unable to confidently detect anomalies.")
            return {
                "labels": np.ones(num_samples, dtype=int), # All normal (1)
                "scores": np.zeros(num_samples) # Neutral scores
            }

        # --- Edge Case 2: Missing Values ---
        # If OCR fails to extract an amount, it might come in as NaN. 
        # IsolationForest cannot handle NaNs natively. We impute them with the median.
        if np.isnan(X).any():
            logging.warning("Missing values (NaN) detected in data. Imputing with median values.")
            X = self.imputer.fit_transform(X)

        # Fit the model and predict labels (-1 for anomaly, 1 for normal)
        labels = self.model.fit_predict(X)
        
        # Get raw anomaly scores
        # Note: scikit-learn scores are negative. The lower (more negative) the score, 
        # the more abnormal the point is. Normal points have scores closer to 0.
        scores = self.model.score_samples(X)
        
        return {
            "labels": labels,
            "scores": scores
        }

    def summarize_results(self, result_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Summarize the anomaly detection results.
        """
        labels = result_dict.get("labels", [])
        if len(labels) == 0:
            return {}
            
        n_anomalies = list(labels).count(-1)
        total = len(labels)
        
        return {
            "total_analyzed": total,
            "anomalies_detected": n_anomalies,
            "anomaly_percentage": round((n_anomalies / total) * 100, 2)
        }


# Example Usage for testing the module
if __name__ == "__main__":
    detector = InvoiceAnomalyDetector(contamination=0.1) # Expecting ~10% anomalies for this test
    
    # --- Example 1: 1D Invoice Amounts ---
    print("\n--- Testing 1D Invoice Amounts ---")
    
    # Simulating 10 normal amounts (around $500) and 2 anomalous amounts (very high/low)
    # Also including a missing value (np.nan) to test edge case handling
    sample_amounts = [
        500.0, 510.5, 495.0, 505.0, 490.0, 
        np.nan, # Simulating OCR failure
        502.0, 498.0, 515.0, 500.0,
        50000.0, # Anomaly: Massively large amount
        5.0      # Anomaly: Suspiciously small amount
    ]
    
    amount_results = detector.fit_predict(sample_amounts, is_1d_amounts=True)
    labels = amount_results["labels"]
    scores = amount_results["scores"]
    
    for i, (amt, label, score) in enumerate(zip(sample_amounts, labels, scores)):
        status = "ANOMALY" if label == -1 else "Normal"
        print(f"Invoice {i+1:02d} | Amount: {str(amt):>8s} | Status: {status:>7s} | Score: {score:.3f}")
        
    print("\nAmount Stats:", detector.summarize_results(amount_results))


    # --- Example 2: 2D Embeddings ---
    print("\n--- Testing 2D Embeddings ---")
    
    np.random.seed(42)
    # 20 normal embeddings clustered together
    normal_embeddings = np.random.normal(loc=0.0, scale=0.1, size=(20, 10))
    # 2 anomalous embeddings far away
    anomaly_embeddings = np.random.uniform(low=2.0, high=3.0, size=(2, 10))
    
    sample_embeddings = np.vstack([normal_embeddings, anomaly_embeddings])
    
    embed_results = detector.fit_predict(sample_embeddings, is_1d_amounts=False)
    
    print("\nEmbedding Stats:", detector.summarize_results(embed_results))
    print(f"Total labeled as Anomaly (-1): {list(embed_results['labels']).count(-1)}")
