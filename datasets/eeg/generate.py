"""
Generate Dummy Datasets for PonyGE2 GLM Fitness Function

This script creates synthetic training and test datasets with various
complexity levels and noise patterns.
"""

import numpy as np
import pandas as pd
import os
from typing import Callable, Tuple


class DataGenerator:
    """Generate synthetic datasets for testing GLM evolution"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_dataset(
        self, 
        n_samples: int,
        target_function: Callable,
        noise_level: float = 0.1,
        feature_range: Tuple[float, float] = (-5, 5)
    ) -> pd.DataFrame:
        """
        Generate a dataset with specified target function.
        
        Args:
            n_samples: Number of samples to generate
            target_function: Function that takes feature dict and returns target
            noise_level: Standard deviation of Gaussian noise
            feature_range: (min, max) range for feature values
        
        Returns:
            DataFrame with columns E1-E7 and target
        """
        # Generate features E1-E7
        data = {}
        for i in range(1, 8):
            data[f'E{i}'] = np.random.uniform(
                feature_range[0], 
                feature_range[1], 
                n_samples
            )
        
        df = pd.DataFrame(data)
        
        # Generate target using provided function
        df['target'] = target_function(df)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_samples)
            df['target'] += noise
        
        return df
    
    def save_datasets(
        self,
        train_size: int = 1000,
        test_size: int = 200,
        target_function: Callable = None,
        noise_level: float = 0.1,
        output_dir: str = 'datasets',
        dataset_name: str = ''
    ):
        """
        Generate and save training and test datasets.
        
        Args:
            train_size: Number of training samples
            test_size: Number of test samples
            target_function: Function to generate targets
            noise_level: Noise standard deviation
            output_dir: Directory to save datasets
            dataset_name: Optional prefix for filenames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use default function if none provided
        if target_function is None:
            target_function = self.simple_linear
        
        # Generate datasets
        train_df = self.generate_dataset(train_size, target_function, noise_level)
        test_df = self.generate_dataset(test_size, target_function, noise_level)
        
        # Construct filenames
        prefix = f"{dataset_name}_" if dataset_name else ""
        train_path = os.path.join(output_dir, f'{prefix}train.csv')
        test_path = os.path.join(output_dir, f'{prefix}test.csv')
        
        # Save to CSV
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✓ Generated {dataset_name if dataset_name else 'default'} dataset")
        print(f"  Training: {train_path} ({train_size} samples)")
        print(f"  Test: {test_path} ({test_size} samples)")
        print(f"  Noise level: {noise_level}")
        print()
        
        return train_df, test_df
    
    # ========== Target Functions ==========
    
    @staticmethod
    def simple_linear(df: pd.DataFrame) -> np.ndarray:
        """Simple linear combination"""
        return 2 * df['E1'] + 3 * df['E2'] - 1.5 * df['E3']
    
    @staticmethod
    def polynomial(df: pd.DataFrame) -> np.ndarray:
        """Polynomial relationship"""
        return (
            2 * df['E1'] + 
            0.5 * df['E2'] ** 2 - 
            0.3 * df['E3'] ** 2 +
            0.1 * df['E1'] * df['E2']
        )
    
    @staticmethod
    def trigonometric(df: pd.DataFrame) -> np.ndarray:
        """Trigonometric functions"""
        return (
            2 * np.sin(df['E1']) + 
            np.cos(2 * df['E2']) + 
            0.5 * df['E3']
        )
    
    @staticmethod
    def mixed_complexity(df: pd.DataFrame) -> np.ndarray:
        """Mixed complexity with multiple operations"""
        return (
            np.sin(df['E1'] + df['E2']) + 
            np.exp(0.1 * df['E3']) + 
            np.sqrt(np.abs(df['E4'])) +
            df['E5'] * df['E6']
        )
    
    @staticmethod
    def interaction_heavy(df: pd.DataFrame) -> np.ndarray:
        """Heavy feature interactions"""
        return (
            df['E1'] * df['E2'] + 
            df['E2'] * df['E3'] + 
            df['E3'] * df['E4'] +
            np.sin(df['E1']) * np.cos(df['E5'])
        )
    
    @staticmethod
    def sparse_features(df: pd.DataFrame) -> np.ndarray:
        """Only uses a subset of features"""
        return (
            3 * df['E1'] + 
            2 * np.sin(df['E3']) + 
            0.5 * df['E5'] ** 2
        )
    
    @staticmethod
    def complex_nonlinear(df: pd.DataFrame) -> np.ndarray:
        """Complex nonlinear relationships"""
        return (
            np.exp(0.2 * df['E1']) * np.sin(df['E2']) +
            np.sqrt(np.abs(df['E3'] * df['E4'])) +
            np.cos(df['E5'] + df['E6']) * df['E7']
        )
    
    @staticmethod
    def friedman_like(df: pd.DataFrame) -> np.ndarray:
        """Similar to Friedman #1 benchmark function"""
        return (
            10 * np.sin(np.pi * df['E1'] * df['E2']) +
            20 * (df['E3'] - 0.5) ** 2 +
            10 * df['E4'] +
            5 * df['E5']
        )


def main():
    """Generate multiple datasets with different characteristics"""
    
    generator = DataGenerator(seed=42)
    
    print("=" * 60)
    print("Generating Dummy Datasets for PonyGE2 GLM Evolution")
    print("=" * 60)
    print()
    
    # Configuration
    configs = [
        {
            'name': '',  # Default dataset (no prefix)
            'function': generator.simple_linear,
            'train_size': 1000,
            'test_size': 200,
            'noise': 0.1,
            'description': 'Simple linear combination (easy)'
        },
        {
            'name': 'polynomial',
            'function': generator.polynomial,
            'train_size': 1000,
            'test_size': 200,
            'noise': 0.2,
            'description': 'Polynomial with interactions (medium)'
        },
        {
            'name': 'trigonometric',
            'function': generator.trigonometric,
            'train_size': 1000,
            'test_size': 200,
            'noise': 0.15,
            'description': 'Trigonometric functions (medium)'
        },
        {
            'name': 'mixed',
            'function': generator.mixed_complexity,
            'train_size': 1500,
            'test_size': 300,
            'noise': 0.2,
            'description': 'Mixed complexity operations (hard)'
        },
        {
            'name': 'interactions',
            'function': generator.interaction_heavy,
            'train_size': 1200,
            'test_size': 250,
            'noise': 0.25,
            'description': 'Heavy feature interactions (medium-hard)'
        },
        {
            'name': 'sparse',
            'function': generator.sparse_features,
            'train_size': 800,
            'test_size': 200,
            'noise': 0.1,
            'description': 'Sparse features (only E1, E3, E5 relevant)'
        },
        {
            'name': 'complex',
            'function': generator.complex_nonlinear,
            'train_size': 2000,
            'test_size': 400,
            'noise': 0.3,
            'description': 'Complex nonlinear relationships (very hard)'
        },
        {
            'name': 'friedman',
            'function': generator.friedman_like,
            'train_size': 1500,
            'test_size': 300,
            'noise': 0.5,
            'description': 'Friedman-like benchmark (hard)'
        }
    ]
    
    # Generate all datasets
    for config in configs:
        print(f"Dataset: {config['description']}")
        generator.save_datasets(
            train_size=config['train_size'],
            test_size=config['test_size'],
            target_function=config['function'],
            noise_level=config['noise'],
            dataset_name=config['name']
        )
    
    print("=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)
    print()
    print("Usage:")
    print("  Default: python ponyge.py --parameters params/glm_params.txt")
    print("  Custom:  Modify fitness function to load different dataset")
    print()
    print("To use a specific dataset, modify glm_fitness.py:")
    print("  self.train_data = pd.read_csv('datasets/polynomial_train.csv')")
    print("  self.test_data = pd.read_csv('datasets/polynomial_test.csv')")


if __name__ == "__main__":
    main()