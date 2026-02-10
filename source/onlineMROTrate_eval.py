import pandas as pd 
import numpy as np 
from collections import deque

from wassertein import WassersteinDriftDetector
from offline import OfflineMROT
from metrics import Metrics

class OnlineMROTADrate:
    """
    Online MROT Anomaly Detection with Adaptive Learning Procedure
    
    This class implements online anomaly detection following the three-step procedure:
    1. Predict: Make predictions using the current model
    2. Diagnose: Evaluate performance and detect drift
    3. Update: Retrain the model when necessary
    """
    
    def __init__(self, 
                 mrot_params={},  # paramètres de l'algorithme MROT
                 window_size=200,  # taille de la fenêtre
                 n_history=10,  # nombre de fenêtres historiques à conserver
                 m_barycenter=5,  # nombre d'anciennes fenêtres pour le barycentre de Wasserstein
                 tau_anomaly=0.75,  # seuil de détection d'anomalies
                 K_retrain=5,  # nombre de fenêtres utilisées pour la reformation de MROT
                 data_online=None,
                 y_true_online=None,
                 data_offline=None,
                 y_true_offline=None,
                 ):
        # Model parameters
        self.mrot_params = mrot_params
        self.tau_anomaly = tau_anomaly
        self.K_retrain = K_retrain
        
        # Window parameters
        self.window_size = window_size
        self.n_history = n_history
        self.m_barycenter = m_barycenter
        
        # Data
        self.online_data = data_online
        self.online_labels = y_true_online  
        self.offline_data = data_offline
        self.offline_labels = y_true_offline
        
        # pour generer les derives de performance
        self.previous_auc = None
        self.auc_drop_threshold = 0.5  # Seuil de chute d'AUC pour détecter une dérive de performance
        
        self.threshold = tau_anomaly
        
        # Drift detection
        self.wassertein = WassersteinDriftDetector(
            window_size=window_size,
            num_history_windows=self.n_history, 
            m_barycenter=2
        )
        
        self.mrot_model = OfflineMROT(mrot_params=self.mrot_params)
        
        self.auc_scores = []
        
        # Initial training on offline data
        self.fit_initial_model()
        
    
    def fit_initial_model(self):
        self.mrot_model.train_mrot_offline(self.offline_data)
        score = self.mrot_model.predict(self.offline_data)
        auc = self.mrot_model.auc_score(self.offline_labels, score)
        print(f"Anomaly scores for initial data computed. AUC Score: {auc}")
    
    
    def _predict(self, data_df):
        """
        Step 1: PREDICT
        Fait une prédiction avec le modèle courant ψ_t
        
        Args:
            data_df: DataFrame contenant les données à prédire
            
        Returns:
            scores: Scores d'anomalie prédits
        """
        return self.mrot_model.predict(data_df)
    
    
    def anomaly_rate(self, y_true, y_pred):
        """
        Calcule le taux d'anomalies détectées dans la fenêtre courante.
        
        Returns:
            anomaly_rate: Taux d'anomalies détectées
        """
        
        predicted_anomalies = np.sum(y_pred > self.tau_anomaly)
        total_samples = len(y_true)
        
        anomaly_rate = predicted_anomalies / total_samples
        return anomaly_rate
    
    def _diagnose(self, y_true, y_pred):
        """
        Version alternative de DIAGNOSE qui utilise le taux d'anomalies détectées.
        Détecte une dérive si le taux d'anomalies dépasse un seuil.
        
        Returns:
            anomaly_rate: Taux d'anomalies détectées
            drift_detected: Boolean indiquant si une dérive est détectée
        """
        anomaly_rate = self.anomaly_rate(y_true, y_pred)
        drift_detected = anomaly_rate > self.threshold
        
        if drift_detected:
            print(f"  DRIFT détecté ! Taux d'anomalies: {anomaly_rate:.3f} (seuil: {self.threshold})")
        
        return anomaly_rate, drift_detected
    
    def _diagnose2(self, y_true, y_pred):

        if not hasattr(self, 'window'):
            self.window = {'y_true': deque(maxlen=1000), 'y_pred': deque(maxlen=1000)}
    
        self.window['y_true'].extend(y_true)
        self.window['y_pred'].extend(y_pred)
    
        window_y_true = np.array(self.window['y_true'])
        window_y_pred = np.array(self.window['y_pred'])
    
        # Calculer AUC seulement si les deux classes sont présentes
        unique_classes = len(set(window_y_true))
    
        if unique_classes > 1 and len(window_y_true) >= self.window_size:
            auc_score = self.mrot_model.auc_score(window_y_true, window_y_pred)
            drift_detected = auc_score < self.theta_validation
            return auc_score, drift_detected
    
        return None, False
    
    
    def _update(self, data_array, labels_array, feature_columns):
        """
        Step 3: UPDATE
        Met à jour le modèle ψ_t → ψ_{t+1}
        """
        retrain_df = pd.DataFrame(data_array, columns=feature_columns)
        retrain_df['label'] = labels_array
        
        self.mrot_model.train_mrot_offline(retrain_df.iloc[:, :-1])
        print(" Model retrained successfully!")
    

    def online_sliding_window(self):
        """
        SLIDING WINDOW (Fenêtre glissante)
        
        Fenêtre qui se déplace d'un pas à la fois (S_slide = 1).
        """
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        
        metrics = Metrics()
        
        score_list = []
        drift_detected_list = []
        auc_score_list = []
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                scores = self._predict(data_df)
                metrics.update(labels_array, scores)
                
                if len(set(labels_window)) > 1:
                    auc_score = self.mrot_model.auc_score(labels_window, scores)
                    #print(f"AUC Score for current window: {auc_score:.4f}")
                    auc_score_list.append(auc_score)
                
                
                anomaly_rate, drift_detected = self._diagnose(labels_array, scores)
                
                if anomaly_rate is not None:
                    score_list.append(anomaly_rate)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
        
        return score_list, drift_detected_list, metrics.get_auc_scores(), auc_score_list
    
    def online_sliding_window(self, stride=1):
    
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        metrics = Metrics()
    
        score_list = []
        drift_detected_list = []
        auc_score_list = []
    
        feature_columns = self.online_data.columns.tolist()
        
        
    
        index = 0
        while index < len(self.online_data):
        
            end_index = min(index + self.window_size, len(self.online_data))
        
       
            if stride >= self.window_size:
                data_window.clear()
                labels_window.clear()
        
        
            for i in range(index, end_index):
                data_window.append(self.online_data.iloc[i].values)
                labels_window.append(self.online_labels[i])
        
            if len(data_window) == self.window_size:
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
            
                scores = self._predict(data_df)
                metrics.update(labels_array, scores)
                if len(set(labels_window)) > 1:
                    auc_score = self.mrot_model.auc_score(labels_window, scores)
                    #print(f"AUC Score for current window: {auc_score:.4f}")
                    auc_score_list.append(auc_score)
            
                anomaly_rate, drift_detected = self._diagnose(labels_array, scores)
            
                if anomaly_rate is not None:
                    score_list.append(anomaly_rate)
            
                drift_detected_list.append(drift_detected)
            
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
        
        
            index += stride
    
        return score_list, drift_detected_list, metrics.get_auc_scores(), auc_score_list
    
    def online_tumbling_window(self):
        """
        TUMBLING WINDOW (Fenêtre basculante / Non-overlapping)
        
        Fenêtres successives de longueur égale sans chevauchement.
        Chaque observation appartient à une seule fenêtre.
        b_{j} = e_{i} pour S_i et S_j consécutifs.
        """
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        auc_score_list = []
        score_list = []
        drift_detected_list = []
        
        feature_columns = self.online_data.columns.tolist()
        metrics = Metrics()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                scores = self._predict(data_df)
                metrics.update(labels_array, scores)
                if len(set(labels_window)) > 1:
                    auc_score = self.mrot_model.auc_score(labels_window, scores)
                    #print(f"AUC Score for current window: {auc_score:.4f}")
                    auc_score_list.append(auc_score)
                
                
                anomaly_rate, drift_detected = self._diagnose(labels_array, scores)
                
                #self.mrot_model.auc_score(labels_array, scores)
                if anomaly_rate is not None:
                    score_list.append(anomaly_rate)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
                
                data_window.clear()
                labels_window.clear()
        
        return score_list, drift_detected_list, metrics.get_auc_scores(), auc_score_list
    
    
    def online_tumbling_window_domain_adaption(self):
        """
        TUMBLING WINDOW (Fenêtre basculante / Non-overlapping)
        
        Fenêtres successives de longueur égale sans chevauchement.
        Chaque observation appartient à une seule fenêtre.
        b_{j} = e_{i} pour S_i et S_j consécutifs.
        """
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        auc_score_list = []
        score_list = []
        drift_detected_list = []
        previous_window_data = None
        
        feature_columns = self.online_data.columns.tolist()
        metrics = Metrics()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                scores = self._predict(data_df)
                metrics.update(labels_array, scores)
                if len(set(labels_window)) > 1:
                    auc_score = self.mrot_model.auc_score(labels_window, scores)
                    #print(f"AUC Score for current window: {auc_score:.4f}")
                    auc_score_list.append(auc_score)
                
                
                anomaly_rate, drift_detected = self._diagnose(labels_array, scores)
                
                #self.mrot_model.auc_score(labels_array, scores)
                if anomaly_rate is not None:
                    score_list.append(anomaly_rate)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    if previous_window_data is not None:
                        sample_to_retrain, transport_loss = self.domain_adaption(previous_window_data, data_array)
                        print(f"Transport loss for domain adaptation: {transport_loss:.4f}")
                        self._update(sample_to_retrain, labels_array, feature_columns)
                    else:
                        self._update(data_array, labels_array, feature_columns)
                else:
                    previous_window_data = data_array.copy()
                    #self.wassertein.add_windows_batch(previous_window_data)
                
                data_window.clear()
                labels_window.clear()
        
        return score_list, drift_detected_list, metrics.get_auc_scores(), auc_score_list
    
    
    
    
    
    def online_sliding_window_with_domain_adaptation_(self, stride=2):
    
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        metrics = Metrics()
    
        score_list = []
        drift_detected_list = []
        auc_score_list = []
    
        feature_columns = self.online_data.columns.tolist()
        
        
    
        index = 0
        while index < len(self.online_data):
        
            end_index = min(index + self.window_size, len(self.online_data))
        
       
            if stride >= self.window_size:
                data_window.clear()
                labels_window.clear()
        
        
            for i in range(index, end_index):
                data_window.append(self.online_data.iloc[i].values)
                labels_window.append(self.online_labels[i])
        
            if len(data_window) == self.window_size:
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
            
                scores = self._predict(data_df)
                metrics.update(labels_array, scores)
                if len(set(labels_window)) > 1:
                    auc_score = self.mrot_model.auc_score(labels_window, scores)
                    #print(f"AUC Score for current window: {auc_score:.4f}")
                    auc_score_list.append(auc_score)
            
                anomaly_rate, drift_detected = self._diagnose(labels_array, scores)
            
                if anomaly_rate is not None:
                    score_list.append(anomaly_rate)
            
                drift_detected_list.append(drift_detected)
            
                if drift_detected:
                    sample_to_retrain, transport_loss = self.wassertein.domain_adaption(self.wassertein.compute_historical_barycenter(), data_array)
                    self._update(sample_to_retrain, labels_array, feature_columns)
        
        
            index += stride
    
        return score_list, drift_detected_list, metrics.get_auc_scores(), auc_score_list