import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler, MinMaxScaler


def apply_robust_scaler(df1, df2, feature_list):
    """
    Fits RobustScaler on df1 and applies the transformation to both df1 and df2.
    Plots KDE visualizations for feature comparisons.
    """
    df1 = df1[feature_list]
    df2 = df2[feature_list]

    scaler = RobustScaler()
    scaled_df1 = scaler.fit_transform(df1)  # Fit & transform df1
    scaled_df2 = scaler.transform(df2)  # Transform df2 with df1's scaling

    scaled_df1 = pd.DataFrame(scaled_df1, columns=feature_list)
    scaled_df2 = pd.DataFrame(scaled_df2, columns=feature_list)

    for column in feature_list:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(scaled_df1[column], label="Dataset 1 (RobustScaled)", fill=True, alpha=0.5)
        sns.kdeplot(scaled_df2[column], label="Dataset 2 (RobustScaled)", fill=True, alpha=0.5)
        plt.title(f"Comparison of {column} (RobustScaler - Fit on Dataset 1)")
        plt.legend()
        plt.show()

    return scaled_df1, scaled_df2


def apply_quantile_transformer(df1, df2, feature_list):
    """
    Fits QuantileTransformer on df1 and applies the transformation to both df1 and df2.
    Plots KDE visualizations for feature comparisons.
    """
    df1 = df1[feature_list]
    df2 = df2[feature_list]

    transformer = QuantileTransformer(output_distribution="normal", random_state=42)
    transformed_df1 = transformer.fit_transform(df1)  # Fit & transform df1
    transformed_df2 = transformer.transform(df2)  # Transform df2 with df1's transformation

    transformed_df1 = pd.DataFrame(transformed_df1, columns=feature_list)
    transformed_df2 = pd.DataFrame(transformed_df2, columns=feature_list)

    for column in feature_list:
        print(transformed_df1[column].mean(), transformed_df2[column].mean())
        plt.figure(figsize=(8, 4))
        sns.kdeplot(transformed_df1[column], label="Dataset 1 (Quantile Transformed)", fill=True, alpha=0.5)
        sns.kdeplot(transformed_df2[column], label="Dataset 2 (Quantile Transformed)", fill=True, alpha=0.5)
        plt.title(f"Comparison of {column} (QuantileTransformer - Fit on Dataset 1)")
        plt.legend()
        plt.show()

    return transformed_df1, transformed_df2


def apply_standard_scaler(df1, df2, feature_list):
    """
    Fits StandardScaler on df1 and applies the transformation to both df1 and df2.
    Plots KDE visualizations for feature comparisons.
    """
    df1 = df1[feature_list]
    df2 = df2[feature_list]

    scaler = StandardScaler()
    scaled_df1 = scaler.fit_transform(df1)  # Fit & transform df1
    scaled_df2 = scaler.transform(df2)  # Transform df2 with df1's scaling

    scaled_df1 = pd.DataFrame(scaled_df1, columns=feature_list)
    scaled_df2 = pd.DataFrame(scaled_df2, columns=feature_list)

    for column in feature_list:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(scaled_df1[column], label="Dataset 1 (StandardScaled)", fill=True, alpha=0.5)
        sns.kdeplot(scaled_df2[column], label="Dataset 2 (StandardScaled)", fill=True, alpha=0.5)
        plt.title(f"Comparison of {column} (StandardScaler - Fit on Dataset 1)")
        plt.legend()
        plt.show()

    return scaled_df1, scaled_df2


def apply_minmax_scaler(df1, df2, feature_list):
    """
    Fits MinMaxScaler on df1 and applies the transformation to both df1 and df2.
    Plots KDE visualizations for feature comparisons.
    """
    df1 = df1[feature_list]
    df2 = df2[feature_list]

    scaler = MinMaxScaler()
    scaled_df1 = scaler.fit_transform(df1)  # Fit & transform df1
    scaled_df2 = scaler.transform(df2)  # Transform df2 with df1's scaling

    scaled_df1 = pd.DataFrame(scaled_df1, columns=feature_list)
    scaled_df2 = pd.DataFrame(scaled_df2, columns=feature_list)

    for column in feature_list:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(scaled_df1[column], label="Dataset 1 (MinMaxScaled)", fill=True, alpha=0.5)
        sns.kdeplot(scaled_df2[column], label="Dataset 2 (MinMaxScaled)", fill=True, alpha=0.5)
        plt.title(f"Comparison of {column} (MinMaxScaler - Fit on Dataset 1)")
        plt.legend()
        plt.show()

    return scaled_df1, scaled_df2
