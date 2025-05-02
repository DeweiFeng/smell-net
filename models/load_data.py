import pandas as pd
import torch
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import ingredient_to_category


def subtract_first_row(df):
    return df - df.iloc[0]


def load_sensor_data(training_path, testing_path, categories=None, real_time_testing_path=None):
    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    min_len = float("inf")  # Track minimum length across all series

    # Helper: subtract first row
    def subtract_first_row(df):
        return df - df.iloc[0]

    # Walk through the training directory
    for folder_name in os.listdir(training_path):
        folder_path = os.path.join(training_path, folder_name)
        if categories is None or ingredient_to_category[folder_name] in categories:
            if os.path.isdir(folder_path):  # Make sure it's a folder
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        cur_path = os.path.join(folder_path, filename)
                        df = pd.read_csv(cur_path)
                        df = subtract_first_row(df)
                        training_data[folder_name].append(df)
                        min_len = min(min_len, df.shape[0])  # Update minimum length

    for folder_name in os.listdir(testing_path):
        folder_path = os.path.join(testing_path, folder_name)

        if ingredient_to_category[folder_name] in ["Nuts"]:
            if os.path.isdir(folder_path):  # Make sure it's a folder
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        cur_path = os.path.join(folder_path, filename)
                        df = pd.read_csv(cur_path)
                        df = subtract_first_row(df)
                        testing_data[folder_name].append(df)
                        min_len = min(min_len, df.shape[0])  # Update minimum length
    
    if real_time_testing_path:
        real_time_testing_data = defaultdict(list)
        for folder_name in os.listdir(real_time_testing_path):
            folder_path = os.path.join(real_time_testing_path, folder_name)

            if categories is None or ingredient_to_category[folder_name] in categories:
                if os.path.isdir(folder_path):  # Make sure it's a folder
                    for filename in os.listdir(folder_path):
                        if filename.endswith(".csv"):
                            cur_path = os.path.join(folder_path, filename)
                            df = pd.read_csv(cur_path)
                            df = subtract_first_row(df)
                            real_time_testing_data[folder_name].append(df)
                            min_len = min(min_len, df.shape[0])  # Update minimum length
        return training_data, testing_data, real_time_testing_data, min_len
    else:
        return training_data, testing_data, min_len


def load_gcms_data(path, le=None):
    df = pd.read_csv(path)

    feature_cols = df.columns[1:]
    label_col = df.columns[0]

    # Extract features and labels
    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    return X_scaled, y_encoded, le, scaler


def prepare_transformer_tensors(data, min_len):
    X = []
    y = []

    for ingredient, dfs in data.items():
        for df in dfs:
            X.append(df.iloc[:min_len].values)  # Truncate to min_len
            y.append(ingredient)

    X = np.stack(X)  # shape: (N_samples, min_len, 12)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return data_loader, le


def prepare_data_gradient(data, dropped_columns=None, period_len=50, trim_len=10, le=None):
    X = []
    y = []

    for ingredient, dfs in data.items():
        for df in dfs:
            df = df.copy()

            # Drop specified columns (safely)
            if dropped_columns:
                df.drop(
                    columns=[col for col in dropped_columns if col in df.columns],
                    inplace=True,
                )

            # Compute gradient (difference)
            diff_data = df.diff(periods=period_len)
            diff_data = diff_data.iloc[
                period_len:
            ]  # Drop first `period_len` rows with NaNs

            # Trim first and last `trim_len` rows if enough data
            if diff_data.shape[0] > 2 * trim_len:
                diff_data = diff_data.iloc[trim_len:-trim_len]

            # Keep only sensor columns
            sensor_cols = [
                col
                for col in diff_data.columns
                if (dropped_columns is None) or (col not in dropped_columns)
            ]

            # Remove rows where all sensors are zero
            diff_data = diff_data[~(diff_data[sensor_cols] == 0).all(axis=1)]

            if diff_data.shape[0] > 0:
                X.append(diff_data[sensor_cols].values)
                y.extend([ingredient] * diff_data.shape[0])

    X_concat = np.concatenate(X, axis=0)  # shape: (total_rows, num_features)

    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    X_tensor = torch.tensor(X_concat, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return data_loader, le


def filter_outliers(group):
    numerical_columns = group.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        Q1 = group[col].quantile(0.2)
        Q3 = group[col].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        group = group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
    return group


def select_median_representative(group, n=1):
    median_values = group.median()  # Calculate the median of each feature
    distances = np.linalg.norm(group - median_values, axis=1)  # Distance to median
    group["distance"] = distances  # Add distances as a temporary column
    closest_rows = group.nsmallest(n, "distance").drop(
        columns="distance"
    )  # Get n closest rows
    return closest_rows


def process_data_regular(data, le=None, batch_size=32, dropped_columns=None):
    X = []
    y = []

    for ingredient, dfs in data.items():
        for df in dfs:
            df = df.copy()

            # Drop unwanted columns if specified
            if dropped_columns:
                df.drop(
                    columns=[col for col in dropped_columns if col in df.columns],
                    inplace=True,
                )

            # Remove rows where all values are 0
            df = df[~(df == 0).all(axis=1)]

            if df.shape[0] > 0:
                X.append(df.values)
                y.extend([ingredient] * len(df))

    # Concatenate all data
    X_concat = np.concatenate(X, axis=0)  # shape: (total_rows, num_features)

    # Encode labels

    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    X_tensor = torch.tensor(X_concat, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader, le
