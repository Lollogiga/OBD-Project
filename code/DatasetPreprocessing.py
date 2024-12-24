import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def preprocess_data(df, label_column, test_size=0.2, validation_size=0.1):
    """
    Preprocessa i dati:
    1) Separazione delle feature e della label.
    2) Sostituzione dei valori NaN con la media della colonna.
    3) Oversampling delle classi minoritarie in caso di sbilanciamento.
    4) Normalizzazione delle feature.
    5) Suddivisione dei dati in training, validation e test set.

    Parameters:
        df (pd.DataFrame): Il DataFrame contenente i dati.
        label_column (str): Nome della colonna target (label).
        test_size (float): Percentuale dei dati da riservare al test set (default 0.2).
        validation_size (float): Percentuale dei dati da riservare al validation set (default 0.1).

    Returns:
        tuple: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # 1) Separazione delle feature e della label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # 2) Sostituzione dei valori NaN con la media della colonna
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)

    # 3) Oversampling in caso di sbilanciamento delle classi
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.max() > class_counts.min() * 2:
        # Ripeti le classi minoritarie per bilanciare il dataset
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        majority_data = X[y == majority_class]
        minority_data = X[y == minority_class]

        # Oversampling della classe minoritaria
        minority_data_resampled = resample(minority_data, replace=True,
                                           n_samples=len(majority_data), random_state=42)

        # Ricombina il dataset
        X = pd.concat([majority_data, minority_data_resampled])
        y = pd.concat([y[y == majority_class], y[y == minority_class].sample(len(majority_data), replace=True)])
        print("Oversampling eseguito.")

    # 4) Normalizzazione delle feature
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # 5) Suddivisione dei dati in training, validation e test set
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=42)
    validation_ratio = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=42)

    # Mantieni il reshape per y_train, y_valid e y_test
    y_train = y_train.values.reshape(1, -1)
    y_valid = y_valid.values.reshape(1, -1)
    y_test = y_test.values.reshape(1, -1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
