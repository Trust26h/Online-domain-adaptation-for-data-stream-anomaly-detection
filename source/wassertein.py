from typing import List, Optional
from collections import deque
import numpy as np 
import ot


class WassersteinDriftDetector:
    
    def __init__(self,  window_size: int , num_history_windows: int , m_barycenter: int, drift_threshold: float=0.2,  k_sensitivity :float= 2.0, 
                 reg_entropy: float = 0.4 ,
                 weights_windows: Optional[List[np.ndarray]] = None):
        
        
        self.window_size = window_size
        self.num_history_windows = num_history_windows
        self.m_barycenter = m_barycenter
        self.k_sensitivity = k_sensitivity
        self.reg_entropy = reg_entropy
        self.threshold = drift_threshold
         
         # poids pour chaque fenêtre historique pour calculer le barycentre de Wasserstein
        
        if weights_windows is None:
            # si je n'ai pas de poids, j'initialise des poids uniformes 
            self.weights_windows = [np.ones(self.window_size) / self.window_size for _ in range(self.num_history_windows)]
        else:
            self.weights_windows = np.array(weights_windows)
            assert np.isclose(np.sum(self.weights_windows, axis=1), 1.0)
            
        
        self.historical_windows = deque(maxlen=num_history_windows)
        
        # Historique des distances de drift (pour calculer μ_d et σ_d)
        self.drift_distances_history = []
        self.normal_periods_history = []  # Indicateur si la période était normale
        
         # Cache pour stocker le dernier barycentre calculé
        self._cached_barycenter = None
        self._cache_valid = False
        
        
            
    def domain_adaption(self, source, target):
        n_source = source.shape[0]
        n_target = target.shape[0]
        
        ps = np.ones(n_source) / n_source
        pt = np.ones(n_target) / n_target
        
        M = ot.dist(source, target, metric='euclidean')
                
        #P = ot.sinkhorn(ps, pt, M, self.reg_entropy, numItermax=1000)

        #transported_source = n_source * P @ target
        P = ot.emd(ps, pt, M)

        transported_source = n_source * P @ target
        transport_loss = np.sum(P * M)
    
        return transported_source, transport_loss
    
    
    def compute_wasserstein_barycenter_slice(self, barycenter, current_window):
        """Calcule la distance avec Sliced Wasserstein (plus rapide et robuste)."""
    
        try:
            # Sliced Wasserstein est beaucoup plus rapide et ne nécessite pas de convergence
            # je veux tester cette approche plus tard apres bonne comprehension de slice
            distance = ot.sliced_wasserstein_distance(
                barycenter, 
            current_window,
            n_projections=50  # Nombre de projections aléatoires
         )
            return distance
        except Exception as e:
            print(f"Sliced Wasserstein failed: {e}")
            return 0.0
    
    def compute_wasserstein_barycenter(self, window1: np.ndarray, window2: np.ndarray , windows_size: int) -> float:
        
        if window1.shape[0] != windows_size or window2.shape[0] != windows_size:
            raise ValueError("Input windows must have the specified window_size.")
        
        p_s = np.ones((windows_size,)) / windows_size
        p_t = np.ones((windows_size,)) / windows_size # proba du domaine source
        
        
        #M = ot.dist(window1, window2, metric='euclidean') 
        
        #M = M**2 # matrice de 2 wassersteins
        
        M = ot.dist(window1, window2, metric='euclidean')
    
        wasserstein_dist = ot.emd2(p_s, p_t, M)
        #coupling_matrix = ot.sinkhorn(p_s, p_t, M, self.reg_entropy, numItermax=1000)
        
        return wasserstein_dist
    
    def add_windows_batch(self, window: np.ndarray):
        self.historical_windows.append(window)
        
    def get_size_historical_windows(self):
        return len(self.historical_windows)
    
    
    def adaptative_threshold(self):
        if len(self.drift_distances_history) < self.num_history_windows:
            return self.threshold  
        
        mu_d = np.mean(self.drift_distances_history)
        sigma_d = np.std(self.drift_distances_history)
        
        threshold = mu_d + self.k_sensitivity * sigma_d
        return threshold
    
    
    def compute_historical_barycenter_old(self, use_cache: bool = True) -> Optional[np.ndarray]:

        if self.get_size_historical_windows() < self.m_barycenter:
            return None
    
        if use_cache and self._cache_valid and self._cached_barycenter is not None:
            return self._cached_barycenter
    

        selected_windows = list(self.historical_windows)[-self.m_barycenter:]
        n_windows = len(selected_windows)
        
        # Shape: (n_windows, window_size, n_features)
        windows_array = np.array(selected_windows)
    
        if windows_array.ndim == 3:
            # Aplatir chaque fenêtre : (window_size * n_features,)
            n_features = windows_array.shape[2]
            flattened_windows = [w.flatten() for w in selected_windows]
            windows_array = np.array(flattened_windows).T  # Shape: (window_size*n_features, n_windows)
        else:
           
            windows_array = windows_array.T  # Shape: (window_size, n_windows)
    
        weights = np.ones(n_windows) / n_windows 
    
        n_points = windows_array.shape[0]
        points_indices = np.arange(n_points).reshape(-1, 1)
        M = ot.dist(points_indices, points_indices, metric='euclidean')
        try:
            barycenter_flat = ot.bregman.barycenter_sinkhorn(
                A=windows_array,          
            M=M,                       
            reg=self.reg_entropy,       # Régularisation entropique
            weights=weights,           
            numItermax=3000,           
            verbose=False
            )
        
            # Reformater le barycentre dans la forme originale
            original_shape = selected_windows[0].shape  
            if len(original_shape) > 1:
                barycenter = barycenter_flat.reshape(original_shape)
            else:
                barycenter = barycenter_flat
        
            # Mettre en cache
            self._cached_barycenter = barycenter
            self._cache_valid = True
        
            return barycenter
        
        except Exception as e:
            print(f" Erreur lors du calcul du barycentre: {e}")
            return None
        
        
    def compute_historical_barycenter(self, use_cache: bool = True) -> Optional[np.ndarray]:

        if self.get_size_historical_windows() < self.m_barycenter:
            return None

        if use_cache and self._cache_valid and self._cached_barycenter is not None:
            return self._cached_barycenter

        selected_windows = list(self.historical_windows)[-self.m_barycenter:]
        n_windows = len(selected_windows)

        # Chaque fenêtre devient un nuage de points dans R^{n_features}
        measures_locations = []
        measures_weights = []

        for w in selected_windows:
            w = np.asarray(w)

        # Cas univarié -> (window_size, 1)
            if w.ndim == 1:
                w = w[:, None]

            n_points = w.shape[0]

            measures_locations.append(w)
            measures_weights.append(
                np.ones(n_points) / n_points
            )

        weights = np.ones(n_windows) / n_windows

        try:
        # Initialisation du support barycentrique
            X_init = measures_locations[0].copy()

            barycenter = ot.lp.free_support_barycenter(
                measures_locations=measures_locations,
                measures_weights=measures_weights,
                X_init=X_init,
                weights=weights,
                numItermax=200,
                stopThr=1e-7
            )

            barycenter = barycenter.reshape(selected_windows[0].shape)

            self._cached_barycenter = barycenter
            self._cache_valid = True

            return barycenter

        except Exception as e:
            print(f"Erreur lors du calcul du barycentre OT (support libre): {e}")
        return None

    
    
    def compute_barycenter_previous_windows(self):
        if self.get_size_historical_windows() < self.num_history_windows:
            raise ValueError("Not enough historical windows to compute barycenter.")

            
            