"""
GLM Fitness Function for PonyGE2
Evaluates evolved expressions as features in a Generalized Linear Model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fitness.base_ff_classes.base_ff import base_ff
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')


class glm_fitness(base_ff):
    """
    Fitness function that parses evolved expressions and uses them
    as features in a GLM model.
    """
    
    def __init__(self):
        super().__init__()
        self.maximise = False  # We want to minimize error
        
        # Load training data
        self.train_data = pd.read_csv('./../datasets/eeg/train.csv')
        
        # Calculate actual number of features (exclude target column)
        num_features = len(self.train_data.columns) - 1
        
        self.X_train = self.train_data[[f'E{i}' for i in range(1, num_features + 1)]].values
        self.y_train = self.train_data['target'].values
        
        # Store variable names for evaluation - THIS WAS THE BUG
        self.var_names = [f'E{i}' for i in range(1, num_features + 1)]
        
        print(f"Loaded {num_features} features (E1 to E{num_features})")
        
    

    def evaluate(self, ind, metric='bic', **kwargs):
        phenotype = ind.phenotype
        
        try:
            train_feature = self._evaluate_expression(phenotype, self.X_train)

            if not self._is_valid_feature(train_feature):
                print(f"Invalid feature values encountered here: {phenotype}")
                return float('inf')

            train_feature = train_feature.reshape(-1, 1)
            
            # Add intercept
            train_feature_with_intercept = sm.add_constant(train_feature)

            # Gamma GLM with log link - IDEAL for response time data
            model = sm.GLM(
                self.y_train, 
                train_feature_with_intercept,
                family=sm.families.Gamma(link=sm.families.links.Log())
            )
            results = model.fit()

            # Predictions
            y_train_pred = results.predict(train_feature_with_intercept)

            # Metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            print(f"Train MSE: {train_mse:.4f} for phenotype: {phenotype}")
            from numpy import std
            from sklearn.metrics import root_mean_squared_error as rmse
            score = rmse(self.y_train, y_train_pred) / std(self.y_train)
            print(f"Train RMSE / STD: {score:.4f} for phenotype: {phenotype}")
            ind.training_mse = train_mse
            
            # Pseudo R-squared for GLM (more appropriate than standard R²)
            ind.r2_score = 1 - (results.deviance / results.null_deviance)
            
            # BIC and AIC from statsmodels
            ind.bic = results.bic
            ind.aic = results.aic

            # Coefficient
            coefficient = results.params[1]  # Interpretable as multiplicative effect
            print(f"Feature coefficient (log scale): {coefficient:.4f}")

            if metric == 'mse':
                return train_mse
            elif metric == 'bic':
                return results.bic
            elif metric == 'aic':
                return results.aic
            else:
                return train_mse

        except Exception as e:
            print(f"Error evaluating {phenotype}: {e}")
            return float('inf')
    
    def _evaluate_expression(self, expr, X):
        """
        Safely evaluate the evolved expression with data.
        
        Args:
            expr: String representation of the evolved expression
            X: Input data matrix (n_samples, n_features)
        
        Returns:
            Array of computed feature values
        """
        # Create local namespace with variables and safe functions
        namespace = {
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'exp': self._safe_exp,
            'pow': self._safe_pow,
            'sqrt': self._safe_sqrt,
        }
        
        # Add variables E1-E7 to namespace
        for i, var_name in enumerate(self.var_names):
            namespace[var_name] = X[:, i]
        
        # Evaluate the expression
        result = eval(expr, {"__builtins__": {}}, namespace)
        
        # Ensure result is a numpy array
        if not isinstance(result, np.ndarray):
            result = np.full(X.shape[0], result)
        
        return result
    
    def _safe_exp(self, x):
        """Exponential with overflow protection"""
        x = np.clip(x, -100, 100)
        return np.exp(x)
    
    def _safe_pow(self, x, power):
        """Power function with protection for negative bases"""
        return np.sign(x) * np.abs(x) ** power
    
    def _safe_sqrt(self, x):
        """Square root with protection for negative values"""
        return np.sqrt(np.abs(x))
    
    def _is_valid_feature(self, feature):
        """
        Check if feature contains valid numeric values.
        
        Args:
            feature: Numpy array of feature values
        
        Returns:
            Boolean indicating if feature is valid
        """
        if feature is None:
            return False
        # Check for NaN or Inf
        if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
            print(f"Invalid feature values encountered:")
            #replace nans with 0
            feature[np.isnan(feature)] = 0
            
           
        
        return True


# Example usage and dataset creation
if __name__ == "__main__":
    """
    Create sample training dataset for testing
    """
    np.random.seed(42)
    
    # Generate synthetic training data
    n_train = 1000
    
    train_data = pd.DataFrame({
        f'E{i}': np.random.randn(n_train) for i in range(1, 8)
    })
    # Example target: some nonlinear function
    train_data['target'] = (
        2 * train_data['E1'] + 
        np.sin(train_data['E2']) + 
        0.5 * train_data['E3'] ** 2 +
        np.random.randn(n_train) * 0.1
    )
    train_data.to_csv('datasets/train.csv', index=False)
    
    print("Sample training dataset created successfully!")
    print(f"Training samples: {n_train}")
    print(f"Features: E1-E7")
