from sklearn.metrics import roc_auc_score
from mrot import MassRepulsiveOptimalTransport


class OfflineMROT:
    # Cette classe implémente la version hors ligne de l'algorithme MROT pour la détection d'anomalies.
    def __init__(self, mrot_params={}):
        self.mrot_params = mrot_params
        self.mrot_offline = MassRepulsiveOptimalTransport()
        
    
    def train_mrot_offline(self, data):
        self.mrot_offline.fit(data)
    
    
    def predict(self, data):
        scores = self.mrot_offline.predict(data)
        return scores
    
    def auc_score(self, y_true, y_scores):
        return roc_auc_score(y_true, y_scores)
    
    