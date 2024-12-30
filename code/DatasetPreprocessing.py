from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def rebalance(X_train, y_train, imbalance_threshold=0.5):
    """
    Effettua il rebalance dei dati di training usando SMOTE in caso di sbilanciamento.
    Parameters:
        X_train: Dati di training.
        y_train: Label di training.
        imbalance_threshold (float): Soglia per rilevare sbilanciamento (default 0.5).
    Returns:
        X_train, y_train: Dati bilanciati.
    """
    class_counts = y_train.value_counts()
    if class_counts.min() / class_counts.max() < imbalance_threshold:  # Soglia di sbilanciamento
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def datasetPreprocessing(dataset, target_column, test_size=0.2, validation_size=0.1):
    """
    Preprocessa i dati:
    1) Separazione delle feature e della label.
    2) Sostituzione dei valori NaN con la mediana della colonna.
    3) Oversampling delle classi minoritarie in caso di sbilanciamento.
    4) Normalizzazione delle feature.
    5) Suddivisione dei dati in training, validation e test set.

    Parameters:
        dataset (pd.DataFrame): Il DataFrame contenente i dati.
        target_column (str): Nome della colonna target (label).
        test_size (float): Percentuale dei dati da riservare al test set (default 0.2).
        validation_size (float): Percentuale dei dati da riservare al validation set (default 0.1).

    Returns:
        tuple: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # Separazione delle feature e della label
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Rimozione delle righe con valori NaN nella colonna target
    mask = y.notnull()
    X = X[mask]
    y = y[mask]

    # Sostituzione dei valori NaN con la mediana
    if X.isnull().sum().any():
        X.fillna(X.median(), inplace=True)

    # Standardizzazione delle feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Suddivisione in training, test e validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    validation_ratio = validation_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=42)

    # Oversampling con SMOTE
    X_train, y_train = rebalance(X_train, y_train)

    return X_train, X_valid, X_test, y_train.values.reshape(1, -1), y_valid.values.reshape(1, -1), y_test.values.reshape(1, -1)

