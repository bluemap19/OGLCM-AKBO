"""
Manual Feature Selection Data Loader and Preprocessing Module
Manual Feature Selection Data Loader and Preprocessing for OGLCM-AKBO

Features:
    1. Read CSV texture feature data
    2. Select data columns based on user-specified feature name list
    3. Data quality check (missing values, outliers)
    4. Feature standardization (Z-score)

Author: Modified from original version
Date: 2026-03-17
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class ManualFeaturePreprocessor:
    """
    Manual Feature Selection Preprocessor

    Attributes:
        scaler: StandardScaler object for feature standardization
        imputer: SimpleImputer object for missing value imputation
        selected_columns: user-specified feature column name list
    """

    def __init__(self, feature_columns):
        """
        Initialize the preprocessor

        Parameters:
            feature_columns: list, user-specified feature column name list
               e.g.: ['GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy']
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.selected_columns = feature_columns

    def load_and_select_data(self, file_path):
        """
        Load CSV data and select specified features

        Parameters:
            file_path: path to the CSV file

        Returns:
            features: DataFrame containing only the user-specified features

        Raises:
            ValueError: if any specified feature column does not exist in the data
        """
        print(f"[1/4] Loading data: {file_path}")
        df = pd.read_csv(file_path)
        print(f"      Raw data shape: {df.shape}")
        print(f"      Raw data columns: {df.columns.tolist()}")

        # Check that all specified feature columns exist in the data
        missing_columns = [col for col in self.selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following specified features do not exist in the data: {missing_columns}")

        # Select user-specified feature columns
        features = df[self.selected_columns].copy()

        print(f"      Number of selected features: {len(self.selected_columns)}")
        print(f"      Selected features: {self.selected_columns}")
        print(f"      Processed data shape: {features.shape}")

        return features

    def check_data_quality(self, features):
        """
        Data quality check (performed only on user-specified features)

        Parameters:
            features: DataFrame, feature data

        Returns:
            report: dict, quality check report
        """
        report = {
            'missing_values': int(features.isnull().sum().sum()),
            'zero_variance': [],
            'outliers': {},
            'selected_features': self.selected_columns,
            'feature_statistics': {}
        }

        # Check for zero-variance features
        for col in features.columns:
            std_val = features[col].std()
            if std_val < 1e-10:
                report['zero_variance'].append(col)

            # Record basic statistics for each feature
            report['feature_statistics'][col] = {
                'mean': float(features[col].mean()),
                'std': float(std_val),
                'min': float(features[col].min()),
                'max': float(features[col].max()),
                'missing': int(features[col].isnull().sum())
            }

        # Check for outliers (3-sigma rule)
        total_outliers = 0
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 1e-10:  # avoid division by zero
                outliers = ((features[col] < mean - 3*std) | (features[col] > mean + 3*std)).sum()
                if outliers > 0:
                    report['outliers'][col] = {
                        'count': int(outliers),
                        'percentage': float(outliers / len(features) * 100)
                    }
                    total_outliers += outliers

        print("[2/4] Data quality check (based on selected features)")
        print(f"      Total missing values: {report['missing_values']}")
        print(f"      Zero-variance features: {len(report['zero_variance'])}")
        if report['zero_variance']:
            print(f"      Zero-variance feature list: {report['zero_variance']}")
        print(f"      Total outliers: {total_outliers}")

        # Print missing value statistics per feature
        print("\n      Missing value statistics per feature:")
        for col in features.columns:
            missing_count = features[col].isnull().sum()
            if missing_count > 0:
                print(f"        {col}: {missing_count} missing values ({missing_count/len(features)*100:.1f}%)")

        return report

    def preprocess(self, features, handle_outliers=True, sigma_threshold=2.0):
        """
        Data preprocessing

        Parameters:
            features: DataFrame, feature data
            handle_outliers: bool, whether to handle outliers
            sigma_threshold: float, sigma threshold for outlier handling

        Returns:
            features_scaled: ndarray, standardized features
        """
        # 1. Handle missing values
        print("[3/4] Data preprocessing")
        print("      - Handling missing values (mean imputation)...")
        features_imputed = self.imputer.fit_transform(features)
        features_df = pd.DataFrame(features_imputed, columns=features.columns)

        # 2. Handle outliers (sigma clipping)
        if handle_outliers:
            print(f"      - Handling outliers ({sigma_threshold}-sigma clipping)...")
            for col in features_df.columns:
                mean = features_df[col].mean()
                std = features_df[col].std()
                if std > 1e-10:  # avoid division by zero
                    lower = mean - sigma_threshold * std
                    upper = mean + sigma_threshold * std
                    features_df[col] = features_df[col].clip(lower, upper)

        # 3. Feature standardization (Z-score)
        print("      - Feature standardization (Z-score)...")
        features_scaled = self.scaler.fit_transform(features_df)

        # Print post-standardization statistics
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        print("\n      Post-standardization statistics:")
        print(f"        Overall mean: {features_scaled.mean():.2e}")
        print(f"        Overall variance: {features_scaled.var():.2e}")

        print("\n      Per-feature post-standardization statistics:")
        for i, col in enumerate(features.columns):
            col_mean = features_scaled_df[col].mean()
            col_std = features_scaled_df[col].std()
            print(f"        {col}: mean={col_mean:.3f}, std={col_std:.3f}")

        return features_scaled

    def fit_transform(self, file_path, handle_outliers=True, sigma_threshold=2.0):
        """
        Complete preprocessing pipeline

        Parameters:
            file_path: path to the CSV file
            handle_outliers: bool, whether to handle outliers
            sigma_threshold: float, sigma threshold for outlier handling

        Returns:
            features_scaled: ndarray, standardized features
            quality_report: dict, quality check report
        """
        # Load and select data
        features = self.load_and_select_data(file_path)

        # Quality check
        quality_report = self.check_data_quality(features)

        # Preprocessing
        features_scaled = self.preprocess(features, handle_outliers, sigma_threshold)

        print(f"\n[OK] Data preprocessing complete")
        print(f"      Input features: {len(self.selected_columns)}")
        print(f"      Output shape: {features_scaled.shape}")

        return features_scaled, quality_report


def load_and_preprocess_manual(file_path, feature_columns, handle_outliers=True, sigma_threshold=2.0):
    """
    Convenience function for manual feature selection data loading and preprocessing

    Parameters:
        file_path: path to the CSV file
        feature_columns: list, user-specified feature column name list
        handle_outliers: bool, whether to handle outliers
        sigma_threshold: float, sigma threshold for outlier handling

    Returns:
        features: ndarray, standardized features
        preprocessor: ManualFeaturePreprocessor, preprocessor object
        report: dict, quality check report
    """
    preprocessor = ManualFeaturePreprocessor(feature_columns)
    features, report = preprocessor.fit_transform(file_path, handle_outliers, sigma_threshold)

    return features, preprocessor, report


if __name__ == "__main__":
    # Test example
    file_path = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"

    # User manually specifies features to use
    selected_features = ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']

    print("="*80)
    print("Manual Feature Selection Data Preprocessor Test")
    print("="*80)
    print(f"Specified features: {selected_features}")
    print("="*80)

    try:
        features, preprocessor, report = load_and_preprocess_manual(
            file_path=file_path,
            feature_columns=selected_features,
            handle_outliers=True,
            sigma_threshold=2.0
        )

        print("\n" + "="*80)
        print("Preprocessing results:")
        print(f"  Feature data shape: {features.shape}")
        print(f"  Features used: {preprocessor.selected_columns}")
        print("="*80)

    except ValueError as e:
        print(f"Error: {e}")
        print("Please check that the specified feature column names exist in the data.")
