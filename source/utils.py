import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns # type: ignore


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




def plot_auc_over_time(auc_lists, figsize=(16, 6), 
                       title="AUC scores over time in OnlineMROTAD",
                       xlabel="Window Index", ylabel="AUC Score",
                       add_threshold=None, save_path=None):
    """
    Trace l'évolution des scores AUC au fil du temps avec options avancées.
    
    Parameters:
    -----------
    auc_lists : list or array-like
        Liste des scores AUC pour chaque fenêtre
    figsize : tuple, default=(16, 6)
        Taille de la figure
    title : str
        Titre du graphique
    xlabel : str
        Label de l'axe X
    ylabel : str
        Label de l'axe Y
    add_threshold : float, optional
        Ajoute une ligne horizontale de seuil (ex: 0.5 pour AUC aléatoire)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot(x=range(len(auc_lists)), y=auc_lists, ax=ax, linewidth=2)
    
    if add_threshold is not None:
        ax.axhline(y=add_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({add_threshold})', alpha=0.7)
        ax.legend()
    
    mean_auc = np.mean(auc_lists)
    std_auc = np.std(auc_lists)
    ax.text(0.02, 0.98, f'Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax


def plot_auc_with_drift(auc_lists, drift_indicators, xlabel ="time", ylabel="auc", figsize=(16, 6), 
                        title="AUC Scores Over Time with Concept Drift Detection"):
    """
    Trace l'évolution des scores AUC avec mise en évidence des zones de drift.
    
    Parameters:
    -----------
    auc_lists : list or array-like
        Liste des scores AUC pour chaque fenêtre
    drift_indicators : list or array-like of bool
        Indicateurs de drift (True = drift détecté, False = pas de drift)
    figsize : tuple, default=(16, 6)
        Taille de la figure
    title : str
        Titre du graphique
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convertir en arrays numpy
    auc_lists = np.array(auc_lists)
    drift_indicators = np.array(drift_indicators)
    
    # Tracer la courbe AUC
    ax.plot(range(len(auc_lists)), auc_lists, linewidth=2, label='AUC Score', color='blue')
    
    # Mettre en évidence les zones de drift avec des bandes colorées
    for i in range(len(drift_indicators)):
        if drift_indicators[i]:
            ax.axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
    
    # Ajouter une légende pour les zones de drift
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label=xlabel),
        Patch(facecolor='red', alpha=0.3, label='Concept Drift Detected')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax