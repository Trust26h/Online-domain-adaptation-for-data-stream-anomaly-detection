import pandas as pd 
import numpy as np 
from collections import deque

from wassertein import WassersteinDriftDetector
from offline import OfflineMROT


class OnlineMROTAD:
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
                 theta_validation=0.80,  # seuil de validation pour la mise à jour du modèle
                 tau_anomaly=0.75,  # seuil de détection d'anomalies
                 K_retrain=5,  # nombre de fenêtres utilisées pour la reformation de MROT
                 data_online=None,
                 y_true_online=None,
                 data_offline=None,
                 y_true_offline=None,
                 drift_threshold=0.5,
                 ):
        # Model parameters
        self.mrot_params = mrot_params
        self.theta_validation = theta_validation
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
        
        # Drift detection
        self.threshold = drift_threshold
        self.wassertein = WassersteinDriftDetector(
            window_size=window_size,
            num_history_windows=self.n_history, 
            m_barycenter=2
        )
        
        # Model
        self.mrot_model = OfflineMROT(mrot_params=self.mrot_params)
        
        # Metrics tracking
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
    
    
    def _diagnose__(self, y_true, y_pred):
        """
        Step 2: DIAGNOSE
        Évalue la performance et détecte la dérive
        
        Args:
            y_true: Labels réels
            y_pred: Scores prédits
            
        Returns:
            auc_score: Score AUC
            drift_detected: Boolean indiquant si une dérive est détectée
        """
        if len(set(y_true)) <= 1:
            return None, False
        
        # Calculer l'AUC
        auc_score = self.mrot_model.auc_score(y_true, y_pred)
        #print(f"AUC Score for current window: {auc_score:.4f}")
        
        drift_detected = auc_score < self.theta_validation
        
        if drift_detected:
            print(f" PERFORMANCE DRIFT! AUC below threshold: {auc_score:.4f}")
        
        return auc_score, drift_detected
    
    
    def _diagnose(self, y_true, y_pred):

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
        
        score_list = []
        drift_detected_list = []
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            # Attendre que la fenêtre soit pleine
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                scores = self._predict(data_df)
                
                # Step 2: DIAGNOSE
                auc_score, drift_detected = self._diagnose(labels_array, scores)
                
                if auc_score is not None:
                    score_list.append(auc_score)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
        
        return score_list, drift_detected_list
    
    
    def online_tumbling_window(self):
        """
        TUMBLING WINDOW (Fenêtre basculante / Non-overlapping)
        
        Fenêtres successives de longueur égale sans chevauchement.
        Chaque observation appartient à une seule fenêtre.
        b_{j} = e_{i} pour S_i et S_j consécutifs.
        """
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        
        score_list = []
        drift_detected_list = []
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                scores = self._predict(data_df)
                
                auc_score, drift_detected = self._diagnose(labels_array, scores)
                
                if auc_score is not None:
                    score_list.append(auc_score)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
                
                data_window.clear()
                labels_window.clear()
        
        return score_list, drift_detected_list
    
    
    def online_fixed_window(self):
        """
        FIXED-LENGTH WINDOW (Fenêtre de longueur fixe)
        
        Toutes les fenêtres ont une durée identique (window_size).
        Cette implémentation utilise une approche sliding avec fenêtres fixes.
        """
        data_window = deque(maxlen=self.window_size)
        labels_window = deque(maxlen=self.window_size)
        previous_window = None
        
        score_list = []
        drift_detected_list = []
        
        feature_columns = self.online_data.columns.tolist()

        for index in range(len(self.online_data)):
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            if len(data_window) == data_window.maxlen:
                
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                if previous_window is not None:
                    
                    pass
                
                scores = self._predict(data_df)
                
                auc_score, drift_detected = self._diagnose(labels_array, scores)
                
                if auc_score is not None:
                    score_list.append(auc_score)
                
                drift_detected_list.append(drift_detected)
                
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
                
                previous_window = data_array.copy()
        
        return score_list, drift_detected_list
    
    
    def online_adaptive_window(self, min_window_size=50, max_window_size=500):
        """
        ADAPTIVE WINDOW (Fenêtre adaptative)
        
        La taille de la fenêtre s'ajuste automatiquement selon le taux de changement.
        Inspiré de ADWIN: agrandit la fenêtre quand stable, réduit quand il y a du changement.
        
        Procédure adaptative:
        1. PREDICT: Calculer les scores d'anomalie
        2. DIAGNOSE: Évaluer l'AUC et détecter la dérive
        3. UPDATE: Réentraîner si dérive détectée et ajuster la taille de fenêtre
        
        Args:
            min_window_size: Taille minimale de la fenêtre
            max_window_size: Taille maximale de la fenêtre
        """
        # Initialiser avec la taille minimale
        current_window_size = self.window_size
        data_window = deque(maxlen=current_window_size)
        labels_window = deque(maxlen=current_window_size)
        
        score_list = []
        drift_detected_list = []
        window_sizes = []
        
        feature_columns = self.online_data.columns.tolist()
        consecutive_stable_windows = 0

        for index in range(len(self.online_data)):
            # Ajouter la nouvelle observation
            data_window.append(self.online_data.iloc[index].values)
            labels_window.append(self.online_labels[index])
            
            # Attendre que la fenêtre soit pleine
            if len(data_window) == current_window_size:
                
                # Convertir en arrays
                data_array = np.array(data_window)
                data_df = pd.DataFrame(data_array, columns=feature_columns)
                labels_array = np.array(labels_window)
                
                # Step 1: PREDICT
                scores = self._predict(data_df)
                
                # Step 2: DIAGNOSE
                auc_score, drift_detected = self._diagnose(labels_array, scores)
                
                if auc_score is not None:
                    score_list.append(auc_score)
                
                drift_detected_list.append(drift_detected)
                window_sizes.append(current_window_size)
                
                # Step 3: UPDATE et ADAPTATION DE LA TAILLE
                if drift_detected:
                    self._update(data_array, labels_array, feature_columns)
                    
                    # RÉDUIRE la fenêtre en cas de dérive
                    current_window_size = max(
                        min_window_size, 
                        int(current_window_size * 0.8)
                    )
                    print(f" Window size reduced to: {current_window_size}")
                    consecutive_stable_windows = 0
                    
                    # Recréer la fenêtre avec la nouvelle taille
                    data_window = deque(data_window, maxlen=current_window_size)
                    labels_window = deque(labels_window, maxlen=current_window_size)
                else:
                    # AGRANDIR la fenêtre si stable
                    consecutive_stable_windows += 1
                    
                    if consecutive_stable_windows >= 3:  # Stable pendant 3 fenêtres
                        current_window_size = min(
                            max_window_size, 
                            int(current_window_size * 1.2)
                        )
                        print(f" Window size increased to: {current_window_size}")
                        consecutive_stable_windows = 0
                        
                        # Recréer la fenêtre avec la nouvelle taille
                        data_window = deque(data_window, maxlen=current_window_size)
                        labels_window = deque(labels_window, maxlen=current_window_size)
        
        return score_list, drift_detected_list, window_sizes
    