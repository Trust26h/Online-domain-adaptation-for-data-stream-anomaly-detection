from sklearn.metrics import roc_auc_score

class Metrics:
    def __init__(self):
        
        self.y_true = []
        self.y_scores = []

    def update(self, y_true, score):
        for yt, sc in zip(y_true, score):
            self.y_true.append(yt)
            self.y_scores.append(sc)
    

    def get_auc_scores(self):
        return roc_auc_score(self.y_true, self.y_scores)    
