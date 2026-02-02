import numpy as np 



def split_data(data, drift_labels, true_labels, n_train_samples=1000, random_state=42):
      
    data = np.array(data)
    drift_labels = np.array(drift_labels)
    true_labels = np.array(true_labels)
    
    # Utiliser les n premiers échantillons pour l'entraînement
    X_train = data[:n_train_samples]
    X_test = data[n_train_samples:]
    
    y_drift_train = drift_labels[:n_train_samples]
    y_drift_test = drift_labels[n_train_samples:]
    
    y_true_train = true_labels[:n_train_samples]
    y_true_test = true_labels[n_train_samples:]
    
    return X_train, X_test, y_drift_train, y_drift_test, y_true_train, y_true_test