from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def rebalanced(X_train, y_train, imbalance_threshold=0.5):
    """
    Balanced the data if necessary
    Parameters:
        X_train: Training set
        y_train: Labels set associated to the training set
        imbalance_threshold: says if the data is balanced
    :return:
        X_train: Training set balanced
        y_train: Labels associated to training set balanced
    """
    class_counts = y_train.value_counts()
    if class_counts.min() / class_counts.max() < imbalance_threshold:  # Imbalance threshold
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def datasetPreprocessing(dataset, target_column, test_size=0.2, validation_size=0.1):
    """
    Preprocess the dataset for better training and testing performance:
    1) Split features and labels
    2) Substitute Nan value with median of the column
    3) Oversampling of minority classes in case of imbalance.
    4) Feature normalization
    5) Splits the data in training, validation and test sets
    Parameters:
        dataset (pd.DataFrame): Dataset to be preprocessed.
        target_column (str): Name of the target column.
        test_size (float): Percentage of data to reserve for the test set (default 0.2).
        validation_size (float): Percentage of data to reserve for the validation set (default 0.1).

    Returns:
        tuple: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # Split features and labels:
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Remove rows with Nan value in target column
    mask = y.notnull()
    X = X[mask]
    y = y[mask]

    #Substitute Nan values with median of the column
    if X.isnull().sum().any():
        X.fillna(X.median(), inplace=True)

    #Features Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #Splits in training, validation and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    validation_ratio = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=42)

    # Oversampling with SMOTE
    X_train, y_train = rebalanced(X_train, y_train)

    return X_train, X_valid, X_test, y_train.values.reshape(1, -1), y_valid.values.reshape(1, -1), y_test.values.reshape(1, -1)

