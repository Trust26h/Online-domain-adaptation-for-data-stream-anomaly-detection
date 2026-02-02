import pandas as pd 
import numpy as np 
from collections import deque

from wassertein import WassersteinDriftDetector
from offline import OfflineMROT



class OnlineMROTAD : 
    
    def __init__(self, 
                 mrot_params = {}, # paramètres de l'algorithme MROT
                 window_size=200, # taille de la fenêtre glissante pour les données en ligne
                 n_history = 10, # nombre de fenêtres historiques à conserver
                 m_barycenter = 5, # nombre d'anciennnes fenetres pour le barycentre de Wasserstein
                 theta_validation=0.80, # seuil de validation pour la mise à jour du modèle du modèle MROT
                 tau_anomaly = 0.75, # seuil de détection d'anomalies
                 K_retrain=5, # nombre de fenetres utilisées pour la reformation de MROT
                 data_online = None,
                 y_true_online = None,
                 data_offline = None,
                 y_true_offline = None,
                 drift_threshold = 0.5,
                 ):
        self.mrot_params = mrot_params
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.n_history = n_history
        self.m_barycenter = m_barycenter
        self.theta_validation = theta_validation
        self.tau_anomaly = tau_anomaly
        self.K_retrain = K_retrain
        self.threshold = drift_threshold
        self.auc_scores = []  # liste pour stocker les scores AUC au fil du temps
        
        self.wassertein = WassersteinDriftDetector(window_size=window_size,
                                              num_history_windows=self.n_history, 
                                              m_barycenter=2)
        
        self.online_data = data_online
        self.online_labels = y_true_online  
        self.offline_data = data_offline
        self.offline_labels = y_true_offline
        
        self.mrot_model = OfflineMROT(mrot_params=self.mrot_params)
        # Initial training on offline data
        self.fit_initial_model()
        
    
    def fit_initial_model(self):
        self.mrot_model.train_mrot_offline(self.offline_data)
        score =  self.mrot_model.predict(self.offline_data)
        print(f"Anomaly scores for initial data computed. AUC Score: {self.mrot_model.auc_score(self.offline_labels, score)}")
        
        
        
    def online_vesion_1_auc_sliding_windows(self):
        """"
        Implemente la détection de dérive en ligne avec des fenêtres glissantes
        """
        data_window = deque(maxlen=self.window_size)
        data_test_window = deque(maxlen=self.window_size)
        score_list = []
        drift_detected_list = []
        performance_threshold = self.theta_validation  # seuil de performance pour la détection de dérive
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            data_test_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                # Convertir les fenêtres en tableaux numpy pour le traitement
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                data_test_array = np.array(data_test_window)
                
                drift_detected = False
                
                if len(set(data_test_window)) > 1:
                    scores = self.mrot_model.predict(data_df)
                    auc_score = self.mrot_model.auc_score(data_test_array, scores)
                    print(f"AUC Score for current window: {auc_score:.4f}")
                    score_list.append(auc_score)
                    
                    if auc_score < performance_threshold:
                        drift_detected = True
                        print(f" PERFORMANCE DRIFT! AUC below threshold: {auc_score:.4f}")
                else:
                    pass
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:                    
                    retrain_df = pd.DataFrame(data_array, columns=self.online_data.columns)
                    retrain_df['label'] = data_test_array
                    # mise jour de MROT
                    self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1]) 
                    print(" Model retrained successfully!")
                
        return score_list, drift_detected_list
        
    
    def online_vesion_1_auc_remove_previous_batch(self):
        """
        Implemente la détection de dérive en ligne avec des fenêtres glissantes mais en supprimant les anciennes fenêtres après chaque itération.
        """
        data_window = deque(maxlen=self.window_size)
        data_test_window = deque(maxlen=self.window_size)
        score_list = []
        drift_detected_list = []
        performance_threshold = self.theta_validation  # seuil de performance pour la détection de dérive
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            data_test_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                # Convertir les fenêtres en tableaux numpy pour le traitement
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                data_test_array = np.array(data_test_window)
                
                drift_detected = False
                
                if len(set(data_test_window)) > 1:
                    scores = self.mrot_model.predict(data_df)
                    auc_score = self.mrot_model.auc_score(data_test_array, scores)
                    print(f"AUC Score for current window: {auc_score:.4f}")
                    score_list.append(auc_score)
                    
                    if auc_score < performance_threshold:
                        drift_detected = True
                        print(f" PERFORMANCE DRIFT! AUC below threshold: {auc_score:.4f}")
                        
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:                    
                    retrain_df = pd.DataFrame(data_array, columns=self.online_data.columns)
                    retrain_df['label'] = data_test_array
                    self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1]) 
                
                data_window.clear()
                data_test_window.clear()
                
        return score_list, drift_detected_list
        
    def online_version_fix_windows(self):
        data_window = deque(maxlen=self.window_size)
        data_test_window = deque(maxlen=self.window_size)
        previous_window = None
        score_list = []
        drift_detected_list = []
        performance_threshold = self.theta_validation  # seuil de performance pour la détection de dérive
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            data_test_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                # Convertir les fenêtres en tableaux numpy pour le traitement
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                data_test_array = np.array(data_test_window)
                
                drift_detected = False
                if previous_window is not None:
                   pass
                
                # 2. Vérification de la performance
                if len(set(data_test_window)) > 1:
                    scores = self.mrot_model.predict(data_df)
                    auc_score = self.mrot_model.auc_score(data_test_array, scores)
                    print(f"AUC Score for current window: {auc_score:.4f}")
                    score_list.append(auc_score)
                    
                    if auc_score < performance_threshold:
                        drift_detected = True
                        print(f" PERFORMANCE DRIFT! AUC below threshold: {auc_score:.4f}")
                else:
                    #print("Skipping AUC calculation: Only one class present in data_test_window.")
                    #auc_score = None
                    #score_list.append(auc_score)
                    pass
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:                    
                    retrain_df = pd.DataFrame(data_array, columns=self.online_data.columns)
                    retrain_df['label'] = data_test_array
                    
                    # Réentraîner le modèle
                    self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1]) 
                    print(" Model retrained successfully!")
                
                previous_window = np.array(data_window).copy()
                #data_window.clear()
                #data_test_window.clear()
                
        return score_list, drift_detected_list
    

    
    def online_wassertein(self):
        data_window = deque(maxlen=3)
        data_test_window = deque(maxlen=3)
        previous_window = None
        score_list = []
        drift_detected_list = []
        performance_threshold = self.theta_validation  # seuil de performance pour la détection de dérive
        
        feature_columns = self.online_data.columns.tolist()
        compteur_window = 0
        data_set_windows = []
        data_test_set_windows = []
        for index in range(len(self.online_data)):
            
            
            
            data_set_windows.append(self.online_data.iloc[index].values)
            data_test_set_windows.append(self.online_labels[index])
            compteur_window += 1
            
            if compteur_window == self.window_size:
                
                # Convertir les fenêtres en tableaux numpy pour le traitement
                data_array = np.array(data_set_windows)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                data_test_array = np.array(data_test_set_windows)
                
                data_window.append(data_set_windows)
                data_test_window.append(data_test_set_windows)
                
                drift_detected = False
                if previous_window is not None:
                   pass
                
                # 2. Vérification de la performance
                if len(set(data_test_array)) > 1:
                    print("len(data_df):", len(data_df))
                    print("len(data_test_array):", len(data_test_array))
                    

                    scores = self.mrot_model.predict(data_df)
                    print("len(scores):", len(scores))
                    auc_score = self.mrot_model.auc_score(data_test_array, scores)
                    print(f"AUC Score for current window: {auc_score:.4f}")
                    score_list.append(auc_score)
                    
                    if auc_score < performance_threshold:
                        drift_detected = True
                        print(f" AUC below threshold: {auc_score:.4f}")
                else:
                    #print("Skipping AUC calculation: Only one class present in data_test_window.")
                    #auc_score = None
                    #score_list.append(auc_score)
                    pass
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:                    
                    retrain_df = pd.DataFrame(data_array, columns=self.online_data.columns)
                    retrain_df['label'] = data_test_array
                    
                    # Réentraîner le modèle
                    self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1]) 
                    print(" Model retrained successfully!")
                
                previous_window = np.array(data_window).copy()
                data_set_windows = []
                data_set_windows =[]
                data_test_set_windows = [] 
                data_test_array = []
                compteur_window = 0
                
        return score_list, drift_detected_list
    

    def online_wassertein2(self):
    
        data_window = deque(maxlen=3)
        data_test_window = deque(maxlen=3)
        previous_window = None
        score_list = []
        drift_detected_list = []
    
        feature_columns = self.online_data.columns.tolist()
        wasserstein_score = []
    
        data_set_windows = [None] * self.window_size
        data_test_set_windows = [None] * self.window_size
    
    # Pré-conversion en numpy pour accès rapide (simulation streaming)
        online_data_values = self.online_data.values
        online_labels_values = (self.online_labels if isinstance(self.online_labels, np.ndarray) 
                       else np.array(self.online_labels))
        n_samples = len(self.online_data)
    
        compteur_window = 0
    
        for index in range(n_samples):
            data_set_windows[compteur_window] = online_data_values[index]
            data_test_set_windows[compteur_window] = online_labels_values[index]
            compteur_window += 1
        
            if compteur_window == self.window_size:
                
                data_array = np.array(data_set_windows)
                data_test_array = np.array(data_test_set_windows)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                data_window.append(data_set_windows.copy())
                data_test_window.append(data_test_set_windows.copy())
            
                drift_detected = False
                drift_score = 0.0  # Initialisation du score
            
                # Première fenêtre 
                if previous_window is None:
                    previous_window = data_array.copy()
                    self.wassertein.add_windows_batch(data_array.copy())
                
                    # Calculer AUC pour la première fenêtre
                    n_unique_classes = len(np.unique(data_test_array))
                    if n_unique_classes > 1:
                        scores = self.mrot_model.predict(data_df)
                        auc_score = self.mrot_model.auc_score(data_test_array, scores)
                        score_list.append(auc_score)
                    else:
                        score_list.append(np.nan)  # Pas assez de classes
                
                    wasserstein_score.append(0.0)  
                    drift_detected_list.append(False)
                    compteur_window = 0
                    continue
            
                # Attendre d'avoir assez de fenêtres historiques
                if self.wassertein.get_size_historical_windows() < self.n_history:
                    self.wassertein.add_windows_batch(data_array.copy())
                    previous_window = data_array.copy()
                
                # Calculer AUC
                    n_unique_classes = len(np.unique(data_test_array))
                    if n_unique_classes > 1:
                        scores = self.mrot_model.predict(data_df)
                        auc_score = self.mrot_model.auc_score(data_test_array, scores)
                        score_list.append(auc_score)
                    else:
                        score_list.append(np.nan)
                
                    wasserstein_score.append(0.0)  # Pas encore de détection
                    drift_detected_list.append(False)
                    compteur_window = 0
                    continue
            
            # Détection de drift avec historique complet
                barycenter = self.wassertein.compute_historical_barycenter()
                drift_score = self.wassertein.compute_wasserstein_barycenter(
                    barycenter, data_array, self.window_size
             )
                wasserstein_score.append(drift_score)
            
            # Calculer AUC
                n_unique_classes = len(np.unique(data_test_array))
                if n_unique_classes > 1:
                    print(f"Window: len(data)={len(data_df)}, len(labels)={len(data_test_array)}")
                    scores = self.mrot_model.predict(data_df)
                    auc_score = self.mrot_model.auc_score(data_test_array, scores)
                    score_list.append(auc_score)
                else:
                    score_list.append(np.nan)
            
            # Vérifier le drift
                print(drift_score)
                print(self.threshold)
                if drift_score > self.threshold:
                    drift_detected = True
                    print(f"Wasserstein Drift detected! Distance: {drift_score:.4f}")
                
                # Réentraîner le modèle
                    retrain_df = data_df.copy()
                    retrain_df['label'] = data_test_array
                    self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1])

                #self.wassertein.add_windows_batch(data_array.copy())
                
                else:
                # Pas de drift : mettre à jour l'historique et le seuil
                    self.wassertein.add_windows_batch(data_array.copy())
                    self.wassertein.drift_distances_history.append(drift_score)
                    self.threshold = self.wassertein.adaptative_threshold()
            
                drift_detected_list.append(drift_detected)
                previous_window = data_array.copy()
                compteur_window = 0 
    
        if compteur_window > 0:
            print(f"Warning: {compteur_window} échantillons restants non traités (fenêtre incomplète)")
        # Option: les traiter quand même avec une fenêtre plus petite
        # ou les ignorer (comportement actuel)
    
        return score_list, drift_detected_list, wasserstein_score
        
    