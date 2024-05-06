import numpy as np
from sklearn.linear_model import ElasticNet
from joblib import dump


def clip(x, x_min=-1, x_max=1):
    """
    Clips the input vector `x` to be within the range `[x_min, x_max]`.

    Args:
        x (numpy.ndarray): Input vector to be clipped.
        x_min (float, optional): Minimum value for clipping. Defaults to -1.
        x_max (float, optional): Maximum value for clipping. Defaults to 1.

    Returns:
        numpy.ndarray: Clipped vector.
    """
    clipped_x = np.where(x < x_min, x_min, x)
    clipped_x = np.where(clipped_x > x_max, x_max, clipped_x)
    return clipped_x


def normalize_columns(data):
    """
    Normalize each column of the input data separately using z-score normalization (standardization).

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The normalized data array where each column has been standardized.
        numpy.ndarray: The mean value of each column used for normalization.
        numpy.ndarray: The standard deviation of each column used for normalization.
    """

    # Calculate mean and standard deviation for each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - mean) / (std)

    return normalized_data, mean, std


def normalize_columns_minmax(data):
    """
    Normalize each column of the input data separately using min-max normalization.

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The normalized data array where each column has been scaled to the range [0, 1].
        numpy.ndarray: The minimum value of each column used for normalization.
        numpy.ndarray: The maximum value of each column used for normalization.
    """

    # Calculate minimum and maximum for each column
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)

    # Normalize each column separately
    normalized_data = (data - min) / (max - min)

    return normalized_data, min, max


def normalize_columns_log(data):
    """
    Perform log transformation on the input data.

    Args:
        data (numpy.ndarray): The input data array where each column represents a feature and each row represents a sample.

    Returns:
        numpy.ndarray: The log-transformed data array.
    """

    log_transformed_data = np.log(data + 1)

    return log_transformed_data


def train_elastic_net(
    dataframe, features, alpha, l1_ratio, val_fold, prediction, model_path_name
):
    """
    Train an ElasticNet regression model using specified hyperparameters and dataset splits.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the dataset.
        features (list): The list of feature column names.
        alpha (float): The alpha parameter for the ElasticNet model.
        l1_ratio (float): The l1_ratio parameter for the ElasticNet model.
        val_fold (int): The fold number to be used as the validation set.
        prediction (str): The type of prediction to perform ("P" for Pleasantness or "E" for Eventfulness).
        model_path_name (str): The file path to save the trained model.

    Returns:
        None

    Notes:
        - The function fits an ElasticNet regression model to the training data and saves the model to the specified path.
        - It evaluates the model's performance on training, validation, and test sets, and prints the mean squared error
          (MSE) and mean absolute error (MAE) for each fold.
    """
    # Define your ElasticNet model with specific hyperparameters
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, selection="random")

    # Extract dataframes
    df_train = dataframe[
        (dataframe["info.fold"] != val_fold) & (dataframe["info.fold"] > 0)
    ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
    df_val = dataframe[dataframe["info.fold"] == val_fold]
    df_test = (
        dataframe[dataframe["info.fold"] == 0]
        .groupby(["info.soundscape", "info.masker", "info.smr"])
        .mean()
    )  # For the test set, the same 48 stimuli were shown to all participants so we take the mean of their ratings as the ground truth

    # Get ground-truth labels
    if prediction == "P":
        print("Predicting Pleasantess")
        Y_train = df_train["info.P_ground_truth"].values
        Y_val = df_val["info.P_ground_truth"].values
        Y_test = df_test["info.P_ground_truth"].values
    elif prediction == "E":
        print("Predicting Eventfulness")
        Y_train = df_train["info.E_ground_truth"].values
        Y_val = df_val["info.E_ground_truth"].values
        Y_test = df_test["info.E_ground_truth"].values

    # Get features
    X_train = df_train[features].values
    X_val = df_val[features].values
    X_test = df_test[features].values

    # Fit model
    X_LR = model.fit(X_train, Y_train)

    # Save model to given path
    dump(model, model_path_name)

    # Get MSEs
    MSE_train = np.mean((clip(X_LR.predict(X_train)) - Y_train) ** 2)
    MSE_val = np.mean((clip(X_LR.predict(X_val)) - Y_val) ** 2)
    MSE_test = np.mean((clip(X_LR.predict(X_test)) - Y_test) ** 2)
    MAE_train = np.mean(np.abs(clip(X_LR.predict(X_train)) - Y_train))
    MAE_val = np.mean(np.abs(clip(X_LR.predict(X_val)) - Y_val))
    MAE_test = np.mean(np.abs(clip(X_LR.predict(X_test)) - Y_test))

    print("     |    Mean squared error    |        Mean  error       |")
    print("Fold |--------+--------+--------|--------+--------+--------|")
    print("     | Train  |   Val  |  Test  | Train  |   Val  |  Test  | ")
    print("-----+--------+--------+--------+--------+--------+---------")
    print(
        f"Result| {(MSE_train):.4f} | {(MSE_val):.4f} | {(MSE_test):.4f} | {(MAE_train):.4f} | {(MAE_val):.4f} | {(MAE_test):.4f} |"
    )


def test_model(model, dataframe, features, prediction):

    X_test = dataframe[features].values
    if prediction == "P":
        Y_test = dataframe["info.P_ground_truth"].values
    elif prediction == "E":
        Y_test = dataframe["info.E_ground_truth"].values

    # Get MSEs
    MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
    MAE_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))

    # Print metrics
    print("|   MSE   |   MAE     |")
    print("|---------+-----------|")
    print(f"|  {MSE_test:.4f} |   {MAE_test:.4f}  |")
    print()
