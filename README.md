# Implementazione di una Rete Neurale
## Introduzione:
In questo progetto viene implementata una rete neurale in Python per problemi di classificazione binaria.
## Prerequisiti:
Al fine di eseguire il modello è necessario scaricare le seguenti librerie:
- numpy
- matplotlib
- pandas
- scikit-learn
- imbalanced-learn

Per utenti Windows è possibile scaricare le seguenti librerie tramite:
 ```bash
    $ python .\code\Start.py
 ```
Altrimenti è possibile scaricare le librerie tramite <code> requirements.txt </code>, eseguedo il seguente comando:
 ```bash
    $ pip install -r requirements.txt 
 ```
**Importante**:

Se l'esecuzione del comando genera il seguente errore: <code> externally-managed-environment </code>, procedere come segue:
```bash
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt 
```
## Configurazione del modello
Tutte le costanti principali utilizzate dal modello possono essere modificate nel file <code>.\code\constant.py</code>. Di seguito sono riportate le variabili e il loro significato:
- **NUM_EPOCHS**: Numero di epoche per il training del modello.
- **BATCH_SIZE**: Dimensione del batch.
- **LEARNING_RATE**: Tasso di apprendimento.
- **MOMENTUM_BOOL**: Abilita o disabilita l'uso del momentum nell'aggiornamento dei parametri.
- **DECAY_BOOL**: Abilita o disabilita passo con diminishing stepsize.
- **HIDDEN_LAYERS**: Lista contenente il numero di neuroni per ciascun livello nascosto.
- **LAMBDA_L1_LIST**: Lista di valori di regolarizzazione L1 per cross-validation.
- **LAMBDA_L2_LIST**: Lista di valori di regolarizzazione L2 per cross-validation.
## Esecuzione del modello
Dopo aver configurato i parametri, è possibile avviare il modello eseguendo il file principale:
- Windows:
  ```bash
     $ cd .\code
     $ python main.py
  ```
- Linux:
  ```bash
     $ cd code/
     $ python main.py
  ```
  
