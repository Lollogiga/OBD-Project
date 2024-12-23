import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def preprocess_data(df, label_column, test_size = 0.2, validation_size = 0.1):
    """
    Preprocessa i dati:
    1) Separiamo features e labels.
    2) Sostituisce i valori NaN con la media della colonna.
    3) Esegue oversampling in caso di sbilanciamento delle classi.
    4) Normalizza i dati.
    5) Separiamo il dataset in training set, validation set, test set.

    Parameters:
        df (pandas.DataFrame): Il DataFrame contenente i dati.
        label_column (str): Il nome della colonna della label (target).
        training

    Returns:
        X_train, X_test, y_train, y_test: I dati preprocessati, separati in training e test set.
        test_size (float): La percentuale di dati da usare per il test set (default 0.2).
        validation_size (float): La percentuale di dati da usare per il validation set (default 0.1).
    """
    # 1) Dividi la label (y) dalle feature (X)
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # 2) Sostituisce i valori NaN con la media della rispettiva colonna
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)

    # 3) Esegui oversampling in caso di sbilanciamento delle classi
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.max() > class_counts.min() * 2:
        # Esegui oversampling solo se il bilanciamento delle classi è eccessivamente sbilanciato
        X_resampled, y_resampled = resample(X, y,
                                            replace=True,  # Campiona con sostituzione
                                            n_samples=len(X),  # Campiona per avere la stessa quantità
                                            random_state=42)
        print("Oversampling eseguito.")
        X, y = X_resampled, y_resampled

    # 4) Normalizza i dati
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # 5) Separazione del dataset in training, validation e test set
    # Prima dividi in training + test
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=42)

    # Poi prendi una porzione di training per la validazione
    validation_ratio = validation_size / (1 - test_size)  # Calcola la percentuale di validazione sul training
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=42)

    y_train = y_train.values.reshape(1,-1)
    y_valid = y_valid.values.reshape(1,-1)
    y_test = y_test.values.reshape(1,-1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

