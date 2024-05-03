import numpy as np

normalise = lambda X: (X - np.mean(X, axis=0, keepdims=True)) / np.std(
    X, axis=0, keepdims=True
)  # Normalise an (n,p) numpy array to mean 0, variance 1.
clip = lambda x, x_min=-1, x_max=1: np.where(
    np.where(x < x_min, x_min, x) > x_max, x_max, np.where(x < x_min, x_min, x)
)  # Clip an array to values between x_min and x_max.


def normalize_columns(data):
    # Calculate mean and standard deviation for each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - mean) / (std)

    return normalized_data, mean, std


def normalize_columns_minmax(data):
    # Calculate minimum and maximum for each column
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - min) / (max - min)

    return normalized_data, min, max


def normalize_columns_log(data):
    log_transformed_data = np.log(data + 1)

    return log_transformed_data
