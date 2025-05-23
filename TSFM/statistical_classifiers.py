import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def fit_svm(
    features: npt.NDArray, 
    y: npt.NDArray, 
    max_samples: int = 10000,
    n_jobs: int = -1
) -> SVC:
    """
    Train SVM classifier with optional hyperparameter tuning.
    
    Args:
        features: Input features (n_samples, n_features)
        y: Target labels (n_samples,)
        max_samples: Maximum samples to use for training
        n_jobs: Number of parallel jobs for GridSearch
        
    Returns:
        Trained SVM classifier
    """
    n_classes = np.unique(y).shape[0]
    train_size = features.shape[0]
    
    # Simple SVM for small datasets
    if train_size // n_classes < 5 or train_size < 50:
        return SVC(C=100000, gamma="scale").fit(features, y)
    
    # Grid search for larger datasets
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'max_iter': [10000000]
    }
    
    if train_size > max_samples:
        features, _, y, _ = train_test_split(
            features, y, 
            train_size=max_samples, 
            stratify=y,
            random_state=42
        )
    
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        n_jobs=n_jobs,
        verbose=1
    )
    grid_search.fit(features, y)
    
    return grid_search.best_estimator_