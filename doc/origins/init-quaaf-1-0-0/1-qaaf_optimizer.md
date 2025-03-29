# Arborescence des fonctions pour QAAF 1.0.0

Voici l'arborescence des fonctions pour QAAF 1.0.0, avec les éléments nouveaux, modifiés et supprimés par rapport à la version 0.9.9 :

```
qaaf/
├── metrics/ 
│   ├── calculator.py (modifié: paramètres optimisés)
│   ├── analyzer.py (inchangé)
│   ├── optimizer.py (NOUVEAU: remplace et améliore l'ancien optimizer.py)
│   └── pattern_detector.py (inchangé)
├── market/
│   ├── phase_analyzer.py (inchangé)
│   └── intensity_detector.py (inchangé)
├── allocation/
│   ├── adaptive_allocator.py (modifié: intégration avec le nouveau moteur d'optimisation)
│   └── amplitude_calculator.py (inchangé)
├── transaction/
│   ├── fees_evaluator.py (inchangé)
│   └── rebalance_optimizer.py (modifié: utilise l'approche de grid search efficiente)
├── validation/ (NOUVEAU module)
│   ├── out_of_sample_validator.py (NOUVEAU)
│   └── robustness_tester.py (NOUVEAU)
└── core/
    ├── qaaf_core.py (modifié: intégration du nouveau moteur d'optimisation)
    └── visualizer.py (modifié: nouveaux graphiques pour l'analyse des résultats d'optimisation)
```

## Changements radicaux par rapport à la version 0.9.9

### 1. Module d'optimisation (optimizer.py) - NOUVEAU

Voici le code complet du nouveau module d'optimisation qui diffère radicalement de la version précédente :

```python
"""
QAAF Optimizer - Version 1.0.0
------------------------------
Module d'optimisation efficiente par grid search inspiré de l'approche paxg_btc_first_stable
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from datetime import datetime
from tqdm import tqdm  # Pour les barres de progression

logger = logging.getLogger(__name__)

class QAAFOptimizer:
    """
    Optimiseur avancé pour QAAF, inspiré de l'approche grid search efficiente
    """
    
    def __init__(self, 
                data: Dict[str, pd.DataFrame],
                metrics_calculator,
                market_phase_analyzer,
                adaptive_allocator,
                backtester,
                initial_capital: float = 30000.0):
        """
        Initialise l'optimiseur QAAF
        
        Args:
            data: Dictionnaire des données ('BTC', 'PAXG', 'PAXG/BTC')
            metrics_calculator: Instance du calculateur de métriques
            market_phase_analyzer: Instance de l'analyseur de phases de marché
            adaptive_allocator: Instance de l'allocateur adaptatif
            backtester: Instance du backtester
            initial_capital: Capital initial pour les tests
        """
        self.data = data
        self.metrics_calculator = metrics_calculator
        self.market_phase_analyzer = market_phase_analyzer
        self.adaptive_allocator = adaptive_allocator
        self.backtester = backtester
        self.initial_capital = initial_capital
        
        # Analyse des phases de marché (calcul unique)
        self.market_phases = self.market_phase_analyzer.identify_market_phases(data['BTC'])
        
        # Stockage des résultats d'optimisation
        self.optimization_results = []
        self.best_combinations = {}
        self.metrics_importance = {}
        
        # Profils prédéfinis
        self.profiles = self._get_optimization_profiles()
    
    def _get_optimization_profiles(self) -> Dict:
        """
        Définit les profils d'optimisation prédéfinis
        
        Returns:
            Dictionnaire des profils d'optimisation
        """
        return {
            'max_return': {
                'description': 'Maximisation du rendement total',
                'score_weights': {
                    'total_return': 1.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'constraints': {
                    'min_return': None,
                    'max_drawdown': None,
                    'min_sharpe': None
                }
            },
            'min_drawdown': {
                'description': 'Minimisation du drawdown maximum',
                'score_weights': {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0
                },
                'constraints': {
                    'min_return': None,
                    'max_drawdown': None,
                    'min_sharpe': None
                }
            },
            'balanced': {
                'description': 'Équilibre rendement/risque',
                'score_weights': {
                    'total_return': 0.4,
                    'sharpe_ratio': 30,
                    'max_drawdown': 0.3
                },
                'constraints': {
                    'min_return': 50.0,  # Au moins 50% de rendement
                    'max_drawdown': -50.0,  # Maximum -50% de drawdown
                    'min_sharpe': 0.5   # Sharpe minimal de 0.5
                }
            },
            'safe': {
                'description': 'Rendement acceptable avec risque minimal',
                'score_weights': {
                    'total_return': 0.2,
                    'sharpe_ratio': 10,
                    'max_drawdown': 0.6
                },
                'constraints': {
                    'min_return': 30.0,  # Au moins 30% de rendement
                    'max_drawdown': -30.0,  # Maximum -30% de drawdown
                    'min_sharpe': 0.7   # Sharpe minimal de 0.7
                }
            },
            'max_sharpe': {
                'description': 'Maximisation du ratio de Sharpe',
                'score_weights': {
                    'total_return': 0.0,
                    'sharpe_ratio': 1.0,
                    'max_drawdown': 0.0
                },
                'constraints': {
                    'min_return': None,
                    'max_drawdown': None,
                    'min_sharpe': None
                }
            },
            'max_efficiency': {
                'description': 'Maximisation du ratio rendement/drawdown',
                'score_weights': {
                    'total_return': 0.5,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.5
                },
                'constraints': {
                    'min_return': None,
                    'max_drawdown': None,
                    'min_sharpe': None
                }
            }
        }
    
    def define_parameter_grid(self) -> Dict:
        """
        Définit la grille de paramètres à optimiser
        
        Returns:
            Dictionnaire des paramètres et leurs plages de valeurs
        """
        return {
            # Paramètres des métriques
            'volatility_window': [20, 30, 40, 50, 60],
            'spectral_window': [30, 45, 60, 75, 90],
            'min_periods': [15, 20, 25],
            
            # Paramètres des poids de métriques
            'vol_ratio_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'bound_coherence_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'alpha_stability_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'spectral_score_weight': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            
            # Paramètres de l'allocateur
            'min_btc_allocation': [0.1, 0.2, 0.3, 0.4],
            'max_btc_allocation': [0.6, 0.7, 0.8, 0.9],
            'sensitivity': [0.8, 1.0, 1.2, 1.5],
            
            # Paramètres du backtester
            'rebalance_threshold': [0.01, 0.03, 0.05, 0.07, 0.1],
            'observation_period': [3, 5, 7, 10]
        }
    
    def calculate_score(self, metrics: Dict, profile: str = 'balanced') -> float:
        """
        Calcule un score composite basé sur plusieurs métriques et un profil donné
        
        Args:
            metrics: Dictionnaire des métriques de performance
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Score composite
        """
        if profile not in self.profiles:
            logger.warning(f"Profil {profile} non trouvé, utilisation du profil 'balanced' par défaut")
            profile = 'balanced'
        
        weights = self.profiles[profile]['score_weights']
        
        # Inversion du signe pour le drawdown (plus négatif = moins bon)
        drawdown_term = weights.get('max_drawdown', 0) * (-metrics['max_drawdown'])
        
        return (
            weights.get('total_return', 0) * metrics['total_return'] +  # Rendement
            weights.get('sharpe_ratio', 0) * metrics['sharpe_ratio'] +  # Sharpe
            drawdown_term  # Drawdown (négatif)
        )
    
    def meets_constraints(self, metrics: Dict, profile: str = 'balanced') -> bool:
        """
        Vérifie si les métriques respectent les contraintes du profil
        
        Args:
            metrics: Dictionnaire des métriques de performance
            profile: Profil d'optimisation à utiliser
            
        Returns:
            True si les contraintes sont respectées, False sinon
        """
        if profile not in self.profiles:
            return True  # Par défaut, pas de contraintes
        
        constraints = self.profiles[profile]['constraints']
        
        # Vérification du rendement minimal
        if constraints['min_return'] is not None and metrics['total_return'] < constraints['min_return']:
            return False
        
        # Vérification du drawdown maximal
        if constraints['max_drawdown'] is not None and metrics['max_drawdown'] < constraints['max_drawdown']:
            return False
        
        # Vérification du Sharpe minimal
        if constraints['min_sharpe'] is not None and metrics['sharpe_ratio'] < constraints['min_sharpe']:
            return False
        
        return True
    
    def _generate_parameter_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Génère toutes les combinaisons de paramètres possibles
        
        Args:
            param_grid: Grille de paramètres
            
        Returns:
            Liste de dictionnaires de paramètres
        """
        # Extraction des clés et valeurs
        keys = list(param_grid.keys())
        values_list = list(param_grid.values())
        
        # Génération des combinaisons
        combinations = []
        for values in itertools.product(*values_list):
            combinations.append(dict(zip(keys, values)))
        
        return combinations
    
    def _is_valid_parameter_combination(self, params: Dict) -> bool:
        """
        Vérifie si une combinaison de paramètres est valide
        
        Args:
            params: Dictionnaire de paramètres
            
        Returns:
            True si la combinaison est valide, False sinon
        """
        # Vérification des bornes d'allocation
        if params['min_btc_allocation'] >= params['max_btc_allocation']:
            return False
        
        # Vérification des poids (somme > 0)
        weight_sum = (
            params['vol_ratio_weight'] + 
            params['bound_coherence_weight'] + 
            params['alpha_stability_weight'] + 
            params['spectral_score_weight']
        )
        
        if weight_sum <= 0:
            return False
        
        return True
    
    def _normalize_weights(self, weights: Dict) -> Dict:
        """
        Normalise les poids pour que leur somme soit égale à 1
        
        Args:
            weights: Dictionnaire des poids
            
        Returns:
            Dictionnaire des poids normalisés
        """
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _evaluate_parameter_combination(self, params: Dict, profile: str = 'balanced') -> Dict:
        """
        Évalue une combinaison de paramètres
        
        Args:
            params: Dictionnaire de paramètres
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Dictionnaire des résultats d'évaluation
        """
        # Configuration du calculateur de métriques
        self.metrics_calculator.volatility_window = params['volatility_window']
        self.metrics_calculator.spectral_window = params['spectral_window']
        self.metrics_calculator.min_periods = params['min_periods']
        
        # Calcul des métriques
        metrics = self.metrics_calculator.calculate_metrics(self.data)
        
        # Configuration des poids pour le score composite
        weights = {
            'vol_ratio': params['vol_ratio_weight'],
            'bound_coherence': params['bound_coherence_weight'],
            'alpha_stability': params['alpha_stability_weight'],
            'spectral_score': params['spectral_score_weight']
        }
        
        # Normalisation des poids
        normalized_weights = self._normalize_weights(weights)
        
        # Calcul du score composite
        normalized_metrics = self.metrics_calculator.normalize_metrics(metrics)
        composite_score = pd.Series(0.0, index=normalized_metrics[list(normalized_metrics.keys())[0]].index)
        
        for metric, weight in normalized_weights.items():
            composite_score += weight * normalized_metrics[metric]
        
        # Configuration de l'allocateur adaptatif
        self.adaptive_allocator.min_btc_allocation = params['min_btc_allocation']
        self.adaptive_allocator.max_btc_allocation = params['max_btc_allocation']
        self.adaptive_allocator.sensitivity = params['sensitivity']
        self.adaptive_allocator.observation_period = params['observation_period']
        
        # Calcul des allocations adaptatives
        allocations = self.adaptive_allocator.calculate_adaptive_allocation(
            composite_score,
            self.market_phases
        )
        
        # Configuration du backtester
        self.backtester.rebalance_threshold = params['rebalance_threshold']
        
        # Exécution du backtest
        portfolio_value, performance_metrics = self.backtester.run_backtest(
            self.data['BTC'],
            self.data['PAXG'],
            allocations
        )
        
        # Vérification des contraintes
        if not self.meets_constraints(performance_metrics, profile):
            # Si les contraintes ne sont pas respectées, retourner un score très faible
            score = float('-inf')
        else:
            # Calcul du score global
            score = self.calculate_score(performance_metrics, profile)
        
        return {
            'params': params.copy(),
            'normalized_weights': normalized_weights,
            'metrics': performance_metrics,
            'portfolio_values': portfolio_value,
            'allocations': allocations,
            'score': score
        }
    
    def run_optimization(self, profile: str = 'balanced', max_combinations: int = 10000) -> List[Dict]:
        """
        Exécute l'optimisation par grid search efficiente
        
        Args:
            profile: Profil d'optimisation à utiliser
            max_combinations: Nombre maximal de combinaisons à tester
            
        Returns:
            Liste des résultats d'optimisation triés
        """
        logger.info(f"Démarrage de l'optimisation avec le profil {profile}")
        
        # Définition de la grille de paramètres
        param_grid = self.define_parameter_grid()
        
        # Génération de toutes les combinaisons possibles
        all_combinations = self._generate_parameter_combinations(param_grid)
        
        # Filtrage des combinaisons invalides
        valid_combinations = [c for c in all_combinations if self._is_valid_parameter_combination(c)]
        
        # Limitation du nombre de combinaisons
        combinations_to_test = valid_combinations[:max_combinations]
        
        logger.info(f"Nombre total de combinaisons possibles: {len(all_combinations):,}")
        logger.info(f"Nombre de combinaisons valides: {len(valid_combinations):,}")
        logger.info(f"Nombre de combinaisons à tester: {len(combinations_to_test):,}")
        
        # Initialisation des résultats
        self.optimization_results = []
        
        # Boucle d'optimisation avec barre de progression
        for params in tqdm(combinations_to_test, desc=f"Optimisation ({profile})"):
            # Test de la combinaison
            result = self._evaluate_parameter_combination(params, profile)
            
            # Si le score n'est pas -inf (contraintes respectées), ajouter aux résultats
            if not np.isinf(result['score']):
                self.optimization_results.append(result)
        
        # Tri des résultats par score
        self.optimization_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Stockage des meilleures combinaisons pour ce profil
        if self.optimization_results:
            self.best_combinations[profile] = self.optimization_results[0]
            
            # Analyse de l'importance des métriques
            self._analyze_metrics_importance(profile)
        
        logger.info(f"Optimisation terminée pour le profil {profile}: {len(self.optimization_results)} résultats valides")
        
        return self.optimization_results
    
    def _analyze_metrics_importance(self, profile: str) -> None:
        """
        Analyse l'importance des métriques dans les meilleurs résultats
        
        Args:
            profile: Profil d'optimisation
        """
        if not self.optimization_results:
            return
        
        # Sélection des 100 meilleurs résultats ou moins
        top_results = self.optimization_results[:min(100, len(self.optimization_results))]
        
        # Initialisation du dictionnaire d'importance
        importance = {
            'vol_ratio': [],
            'bound_coherence': [],
            'alpha_stability': [],
            'spectral_score': []
        }
        
        # Collecte des poids normalisés
        for result in top_results:
            weights = result['normalized_weights']
            for metric in importance:
                importance[metric].append(weights.get(metric, 0))
        
        # Calcul des statistiques
        metrics_stats = {}
        for metric, values in importance.items():
            metrics_stats[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Stockage pour ce profil
        self.metrics_importance[profile] = metrics_stats
    
    def run_profile_optimization(self, profiles: List[str] = None, max_combinations: int = 10000) -> Dict:
        """
        Exécute l'optimisation pour plusieurs profils
        
        Args:
            profiles: Liste des profils à tester (None = tous les profils)
            max_combinations: Nombre maximal de combinaisons par profil
            
        Returns:
            Dictionnaire des meilleurs résultats par profil
        """
        # Si aucun profil n'est spécifié, utiliser tous les profils disponibles
        if profiles is None:
            profiles = list(self.profiles.keys())
        
        logger.info(f"Optimisation pour les profils: {', '.join(profiles)}")
        
        # Exécution de l'optimisation pour chaque profil
        for profile in profiles:
            if profile not in self.profiles:
                logger.warning(f"Profil {profile} non trouvé, ignoré")
                continue
            
            self.run_optimization(profile, max_combinations)
        
        # Retour des meilleurs résultats par profil
        return self.best_combinations
    
    def plot_optimization_results(self, profile: str = None) -> None:
        """
        Visualise les résultats de l'optimisation
        
        Args:
            profile: Profil d'optimisation (None = tous les profils)
        """
        if not self.optimization_results and not self.best_combinations:
            logger.warning("Aucun résultat d'optimisation disponible")
            return
        
        # Si un profil est spécifié, visualiser uniquement ce profil
        if profile is not None:
            if profile not in self.best_combinations:
                logger.warning(f"Aucun résultat pour le profil {profile}")
                return
            
            self._plot_profile_results(profile)
        else:
            # Visualiser tous les profils avec des résultats
            for profile in self.best_combinations:
                self._plot_profile_results(profile)
    
    def _plot_profile_results(self, profile: str) -> None:
        """
        Visualise les résultats d'un profil d'optimisation
        
        Args:
            profile: Profil d'optimisation
        """
        if profile not in self.best_combinations:
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        
        # Création de la figure
        fig = plt.figure(figsize=(15, 12))
        
        # Layout de la figure
        gs = fig.add_gridspec(3, 3)
        
        # Récupération du meilleur résultat
        best_result = self.best_combinations[profile]
        
        # 1. Graphique de la performance
        ax1 = fig.add_subplot(gs[0, :])
        
        # Performance normalisée
        portfolio_values = best_result['portfolio_values']
        start_value = portfolio_values.iloc[0]
        norm_portfolio = portfolio_values / start_value
        
        # Comparaison avec BTC et PAXG
        common_index = norm_portfolio.index
        btc_values = self.data['BTC']['close'].loc[common_index]
        paxg_values = self.data['PAXG']['close'].loc[common_index]
        
        norm_btc = btc_values / btc_values.iloc[0]
        norm_paxg = paxg_values / paxg_values.iloc[0]
        
        # Tracé des performances
        ax1.plot(norm_portfolio.index, norm_portfolio, 'b-', linewidth=2, label='QAAF')
        ax1.plot(norm_btc.index, norm_btc, 'g--', linewidth=1.5, label='BTC')
        ax1.plot(norm_paxg.index, norm_paxg, 'r--', linewidth=1.5, label='PAXG')
        
        ax1.set_title(f'Performance du portefeuille optimisé ({profile})')
        ax1.set_ylabel('Performance normalisée')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Graphique des allocations
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(best_result['allocations'], 'b-', linewidth=1.5)
        ax2.set_title('Allocation BTC')
        ax2.set_ylabel('Allocation (%)')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        # 3. Diagramme des poids des métriques
        ax3 = fig.add_subplot(gs[2, 0])
        
        # Extraction des poids
        weights = best_result['normalized_weights']
        metrics = list(weights.keys())
        values = list(weights.values())
        
        # Tracé du diagramme
        bars = ax3.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Ajout des valeurs
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax3.set_title('Poids des métriques')
        ax3.set_ylabel('Poids')
        ax3.set_ylim(0, 1)
        ax3.grid(True, axis='y')
        
        # 4. Tableau des métriques de performance
        ax4 = fig.add_subplot(gs[2, 1:])
        ax4.axis('off')
        
        # Extraction des métriques
        performance = best_result['metrics']
        metrics_table = {
            'Métrique': ['Rendement total', 'Drawdown max', 'Volatilité', 'Ratio de Sharpe', 'Ratio R/D'],
            'Valeur': [
                f"{performance['total_return']:.2f}%",
                f"{performance['max_drawdown']:.2f}%",
                f"{performance['volatility']:.2f}%",
                f"{performance['sharpe_ratio']:.2f}",
                f"{-performance['total_return']/performance['max_drawdown']:.2f}" if performance['max_drawdown'] != 0 else "N/A"
            ]
        }
        
        # Création du tableau
        table = ax4.table(
            cellText=list(zip(*metrics_table.values())),
            colLabels=None,
            loc='center',
            cellLoc='center'
        )
        
        # Formatage du tableau
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        ax4.set_title('Métriques de performance')
        
        # Ajout d'informations sur les paramètres
        plt.figtext(0.5, 0.01, f"Paramètres: Seuil={best_result['params']['rebalance_threshold']}, "
                             f"Min={best_result['params']['min_btc_allocation']}, "
                             f"Max={best_result['params']['max_btc_allocation']}, "
                             f"Sensibilité={best_result['params']['sensitivity']}", 
                  ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        plt.show()
    
    def plot_metrics_importance(self) -> None:
        """
        Visualise l'importance des métriques pour chaque profil
        """
        if not self.metrics_importance:
            logger.warning("Aucune donnée d'importance des métriques disponible")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        
        # Nombre de profils
        n_profiles = len(self.metrics_importance)
        
        # Création de la figure
        fig, axes = plt.subplots(1, n_profiles, figsize=(15, 5))
        
        # Si un seul profil, convertir en liste
        if n_profiles == 1:
            axes = [axes]
        
        # Métrique à visualiser
        metrics = ['vol_ratio', 'bound_coherence', 'alpha_stability', 'spectral_score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Pour chaque profil
        for i, (profile, importance) in enumerate(self.metrics_importance.items()):
            ax = axes[i]
            
            # Extraction des valeurs moyennes
            values = [importance[metric]['mean'] for metric in metrics]
            
            # Tracé du diagramme
            bars = ax.bar(metrics, values, color=colors)
            
            # Ajout des valeurs
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            ax.set_title(f'Profil: {profile}')
            ax.set_ylabel('Importance moyenne')
            ax.set_ylim(0, 1)
            ax.set_xticklabels(metrics, rotation=45)
            ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_profiles_comparison(self) -> None:
        """
        Compare les performances des différents profils
        """
        if not self.best_combinations:
            logger.warning("Aucun résultat de profil disponible")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        
        # Création de la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Préparation des données
        profiles = []
        returns = []
        drawdowns = []
        sharpes = []
        
        for profile, result in self.best_combinations.items():
            metrics = result['metrics']
            
            profiles.append(profile)
            returns.append(metrics['total_return'])
            drawdowns.append(metrics['max_drawdown'])
            sharpes.append(metrics['sharpe_ratio'])
        
        # 1. Comparaison rendement et drawdown
        ax1.bar(profiles, returns, color='g', alpha=0.7, label='Rendement (%)')
        
        # Axe secondaire pour le drawdown
        ax1b = ax1.twinx()
        ax1b.bar(profiles, drawdowns, color='r', alpha=0.5, label='Drawdown (%)')
        
        # Ajout des légendes
        ax1.set_title('Rendement vs Drawdown par profil')
        ax1.set_ylabel('Rendement (%)', color='g')
        ax1b.set_ylabel('Drawdown (%)', color='r')
        
       # Lignes de référence
       ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
       ax1.axhline(y=100, color='g', linestyle='--', alpha=0.3)
       ax1b.axhline(y=-30, color='r', linestyle='--', alpha=0.3)
       
       # 2. Ratio de Sharpe
       ax2.bar(profiles, sharpes, color='b', alpha=0.7)
       ax2.set_title('Ratio de Sharpe par profil')
       ax2.set_ylabel('Ratio de Sharpe')
       
       # Ligne de référence
       ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
       
       plt.tight_layout()
       plt.show()
   
   def export_results(self, filename: str = None) -> None:
       """
       Exporte les résultats d'optimisation dans un fichier JSON
       
       Args:
           filename: Nom du fichier (par défaut: qaaf_optimization_YYYYMMDD.json)
       """
       if not self.best_combinations:
           logger.warning("Aucun résultat à exporter")
           return
       
       # Génération du nom de fichier par défaut si non fourni
       if filename is None:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           filename = f"qaaf_optimization_{timestamp}.json"
       
       # Préparation des résultats pour la sauvegarde
       export_data = {}
       
       for profile, result in self.best_combinations.items():
           # Extraction des informations essentielles
           export_data[profile] = {
               'params': result['params'],
               'normalized_weights': result['normalized_weights'],
               'metrics': result['metrics'],
               # Conversion des séries temporelles en dict
               'portfolio_value_final': result['portfolio_values'].iloc[-1],
               'portfolio_value_max': result['portfolio_values'].max(),
               'portfolio_value_min': result['portfolio_values'].min(),
               'score': result['score']
           }
       
       # Ajout des méta-informations
       export_data['meta'] = {
           'timestamp': datetime.now().isoformat(),
           'version': '1.0.0',
           'data_range': {
               'start': self.data['BTC'].index[0].isoformat(),
               'end': self.data['BTC'].index[-1].isoformat()
           },
           'num_combinations_tested': len(self.optimization_results) if hasattr(self, 'optimization_results') else 0
       }
       
       # Sauvegarde au format JSON
       import json
       with open(filename, 'w') as f:
           json.dump(export_data, f, indent=2, default=str)
       
       logger.info(f"Résultats exportés dans {filename}")
   
    def generate_recommendation_report(self) -> str:
       """
       Génère un rapport de recommandation basé sur les résultats d'optimisation
       
       Returns:
           Rapport formaté en texte
       """
       if not self.best_combinations:
           return "Aucun résultat d'optimisation disponible pour générer des recommandations."
       
       # Entête du rapport
       report = "# Rapport d'Optimisation QAAF 1.0.0\n\n"
       report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
       report += f"Période d'analyse: {self.data['BTC'].index[0].strftime('%Y-%m-%d')} à {self.data['BTC'].index[-1].strftime('%Y-%m-%d')}\n\n"
       
       # Meilleures configurations par profil
       report += "## Meilleures Configurations par Profil\n\n"
       
       for profile, result in self.best_combinations.items():
           perf = result['metrics']
           params = result['params']
           weights = result['normalized_weights']
           
           report += f"### Profil: {profile}\n\n"
           report += f"**Performance:**\n"
           report += f"- Rendement total: {perf['total_return']:.2f}%\n"
           report += f"- Drawdown maximum: {perf['max_drawdown']:.2f}%\n"
           report += f"- Ratio de Sharpe: {perf['sharpe_ratio']:.2f}\n"
           report += f"- Volatilité: {perf['volatility']:.2f}%\n"
           report += f"- Ratio Rendement/Drawdown: {-perf['total_return']/perf['max_drawdown']:.2f}\n\n"
           
           report += f"**Poids des métriques:**\n"
           for metric, weight in weights.items():
               report += f"- {metric}: {weight:.2f}\n"
           report += "\n"
           
           report += f"**Paramètres optimaux:**\n"
           report += f"- Fenêtre de volatilité: {params['volatility_window']}\n"
           report += f"- Fenêtre spectrale: {params['spectral_window']}\n"
           report += f"- Allocation BTC min/max: {params['min_btc_allocation']:.2f}/{params['max_btc_allocation']:.2f}\n"
           report += f"- Seuil de rebalancement: {params['rebalance_threshold']:.2f}\n"
           report += f"- Sensibilité: {params['sensitivity']:.2f}\n"
           report += f"- Période d'observation: {params['observation_period']} jours\n\n"
           
       # Analyse inter-profils
       report += "## Analyse Comparative\n\n"
       
       # Profil le plus performant
       best_return_profile = max(self.best_combinations.items(), key=lambda x: x[1]['metrics']['total_return'])[0]
       # Profil le plus stable (meilleur ratio retour/drawdown)
       best_stability_profile = max(self.best_combinations.items(), key=lambda x: -x[1]['metrics']['total_return']/x[1]['metrics']['max_drawdown'] if x[1]['metrics']['max_drawdown'] < 0 else 0)[0]
       # Profil le plus sûr (drawdown minimal en valeur absolue)
       best_safety_profile = max(self.best_combinations.items(), key=lambda x: x[1]['metrics']['max_drawdown'])[0]
       
       report += f"- Profil le plus performant: **{best_return_profile}**\n"
       report += f"- Profil le plus stable: **{best_stability_profile}**\n"
       report += f"- Profil le plus sûr: **{best_safety_profile}**\n\n"
       
       # Recommandations générales
       report += "## Recommandations\n\n"
       
       # Analyse des poids dominants
       dominant_metrics = {}
       for profile, result in self.best_combinations.items():
           weights = result['normalized_weights']
           # Trouver la métrique dominante
           dominant_metric = max(weights.items(), key=lambda x: x[1])[0]
           if dominant_metric not in dominant_metrics:
               dominant_metrics[dominant_metric] = 0
           dominant_metrics[dominant_metric] += 1
       
       overall_dominant = max(dominant_metrics.items(), key=lambda x: x[1])[0]
       
       report += f"1. La métrique **{overall_dominant}** est dominante dans la plupart des profils optimaux.\n"
       
       # Recommandation sur la sensibilité
       avg_sensitivity = sum(result['params']['sensitivity'] for result in self.best_combinations.values()) / len(self.best_combinations)
       
       if avg_sensitivity > 1.2:
           report += "2. Une sensibilité élevée (>1.2) a été identifiée comme optimale, suggérant l'utilité d'une réaction plus forte aux signaux.\n"
       elif avg_sensitivity < 0.9:
           report += "2. Une sensibilité faible (<0.9) a été identifiée comme optimale, suggérant l'utilité d'une approche plus conservatrice.\n"
       else:
           report += "2. Une sensibilité modérée (0.9-1.2) a été identifiée comme optimale, suggérant un équilibre approprié dans la réaction aux signaux.\n"
       
       # Recommandation sur le seuil de rebalancement
       avg_threshold = sum(result['params']['rebalance_threshold'] for result in self.best_combinations.values()) / len(self.best_combinations)
       
       if avg_threshold <= 0.03:
           report += "3. Un seuil de rebalancement bas (≤3%) a été identifié comme optimal, suggérant des ajustements plus fréquents.\n"
       elif avg_threshold >= 0.07:
           report += "3. Un seuil de rebalancement élevé (≥7%) a été identifié comme optimal, suggérant des ajustements moins fréquents.\n"
       else:
           report += "3. Un seuil de rebalancement modéré (3-7%) a été identifié comme optimal, suggérant un équilibre coût/réactivité approprié.\n"
       
       return report