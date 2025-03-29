"""
QAAF Robustness Tester - Version 1.0.0
-------------------------------------
Module de test de robustesse pour évaluer la stabilité des stratégies QAAF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RobustnessTester:
    """
    Testeur de robustesse pour QAAF
    
    Cette classe permet d'évaluer la robustesse des stratégies QAAF
    en utilisant des approches comme la validation croisée temporelle
    et les tests de stress.
    """
    
    def __init__(self, 
                qaaf_core,
                data: Dict[str, pd.DataFrame]):
        """
        Initialise le testeur de robustesse
        
        Args:
            qaaf_core: Instance du core QAAF
            data: Dictionnaire des données ('BTC', 'PAXG', 'PAXG/BTC')
        """
        self.qaaf_core = qaaf_core
        self.data = data
        self.results = {}
    
    def run_time_series_cross_validation(self, 
                                        n_splits: int = 5, 
                                        test_size: int = 90,  # en jours
                                        gap: int = 0,  # en jours
                                        profile: str = 'balanced') -> Dict:
        """
        Exécute une validation croisée temporelle
        
        Args:
            n_splits: Nombre de divisions temporelles
            test_size: Taille de l'ensemble de test (en jours)
            gap: Écart entre l'entraînement et le test (en jours)
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Dictionnaire des résultats de validation croisée
        """
        logger.info(f"Démarrage de la validation croisée temporelle avec {n_splits} divisions")
        
        # Obtenir les dates communes à tous les DataFrames
        common_dates = self._get_common_dates()
        
        # Tri des dates
        common_dates = sorted(common_dates)
        
        # Définition des splits temporels
        splits = []
        test_size_pd = pd.Timedelta(days=test_size)
        gap_pd = pd.Timedelta(days=gap)
        
        # Calcul de l'intervalle total disponible
        total_span = common_dates[-1] - common_dates[0]
        
        # Calcul de la taille de chaque segment
        segment_size = total_span / n_splits
        
        for i in range(n_splits):
            # Calcul des dates de début et fin du test
            test_end_idx = len(common_dates) - 1 - i * int(len(common_dates) / n_splits)
            test_end = common_dates[test_end_idx]
            test_start = test_end - test_size_pd
            
            # Calcul de la date de fin d'entraînement
            train_end = test_start - gap_pd
            
            splits.append({
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
        
        # Sauvegarde de l'instance QAAF originale
        original_qaaf = self.qaaf_core
        
        # Préparation des résultats
        cv_results = []
        
        try:
            # Pour chaque split
            for i, split in enumerate(splits):
                logger.info(f"Validation croisée {i+1}/{n_splits}: "
                          f"Train jusqu'à {split['train_end']}, "
                          f"Test de {split['test_start']} à {split['test_end']}")
                
                # Préparation des ensembles d'entraînement et de test
                train_data = {}
                test_data = {}
                
                for asset, df in self.data.items():
                    train_mask = df.index <= split['train_end']
                    test_mask = (df.index >= split['test_start']) & (df.index <= split['test_end'])
                    
                    train_data[asset] = df[train_mask].copy()
                    test_data[asset] = df[test_mask].copy()
                
                # 1. Entraînement
                self.qaaf_core.data = train_data
                train_results = self._run_training_phase(profile)
                
                # Vérification si l'entraînement a réussi
                if train_results['best_params'] is None:
                    logger.warning(f"L'entraînement a échoué pour le split {i+1}")
                    continue
                
                # 2. Test
                self.qaaf_core.data = test_data
                test_results = self._run_testing_phase(train_results['best_params'], profile)
                
                # Stockage des résultats
                cv_results.append({
                    'split': i+1,
                    'train_end': split['train_end'],
                    'test_start': split['test_start'],
                    'test_end': split['test_end'],
                    'train': train_results,
                    'test': test_results
                })
        
        finally:
            # Restauration de l'instance QAAF originale
            self.qaaf_core = original_qaaf
        
        # Analyse des résultats
        self.results['cv'] = self._analyze_cv_results(cv_results)
        
        return self.results
    
    def _get_common_dates(self) -> List[pd.Timestamp]:
        """
        Obtient les dates communes à tous les DataFrames
        
        Returns:
            Liste des dates communes
        """
        if not self.data:
            return []
        
        # Initialisation avec les dates du premier DataFrame
        first_key = list(self.data.keys())[0]
        common_dates = set(self.data[first_key].index)
        
        # Intersection avec les dates des autres DataFrames
        for asset, df in self.data.items():
            if asset != first_key:
                common_dates &= set(df.index)
        
        return list(common_dates)
    
    def _run_training_phase(self, profile: str) -> Dict:
        """
        Exécute la phase d'entraînement
        
        Args:
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Dictionnaire des résultats d'entraînement
        """
        # Analyse des phases de marché
        self.qaaf_core.analyze_market_phases()
        
        # Calcul des métriques
        self.qaaf_core.calculate_metrics()
        
        # Optimisation des métriques
        optimization_results = self.qaaf_core.run_metrics_optimization(profile=profile)
        
        # Calcul du score composite avec les poids optimisés
        if profile in optimization_results['best_combinations']:
            best_combo = optimization_results['best_combinations'][profile]
            
            # Extraction des paramètres optimaux
            best_params = {
                'weights': best_combo['normalized_weights'],
                'min_btc_allocation': best_combo['params']['min_btc_allocation'],
                'max_btc_allocation': best_combo['params']['max_btc_allocation'],
                'rebalance_threshold': best_combo['params']['rebalance_threshold'],
                'sensitivity': best_combo['params']['sensitivity'],
                'observation_period': best_combo['params']['observation_period'],
                'volatility_window': best_combo['params']['volatility_window'],
                'spectral_window': best_combo['params']['spectral_window'],
                'min_periods': best_combo['params']['min_periods']
            }
            
            # Calcul du score composite avec les poids optimisés
            self.qaaf_core.calculate_composite_score(best_params['weights'])
            
            # Configuration de l'allocateur
            self.qaaf_core.adaptive_allocator.min_btc_allocation = best_params['min_btc_allocation']
            self.qaaf_core.adaptive_allocator.max_btc_allocation = best_params['max_btc_allocation']
            self.qaaf_core.adaptive_allocator.sensitivity = best_params['sensitivity']
            self.qaaf_core.adaptive_allocator.observation_period = best_params['observation_period']
            
            # Calcul des allocations
            self.qaaf_core.calculate_adaptive_allocations()
            
            # Configuration du backtester
            self.qaaf_core.backtester.rebalance_threshold = best_params['rebalance_threshold']
            
            # Exécution du backtest
            self.qaaf_core.run_backtest()
            
            # Retour des résultats
            return {
                'best_params': best_params,
                'optimization_results': optimization_results,
                'performance': self.qaaf_core.results['metrics'],
                'portfolio_values': self.qaaf_core.performance,
                'allocations': self.qaaf_core.allocations
            }
        else:
            logger.warning(f"Aucun résultat pour le profil {profile}")
            return {'best_params': None, 'optimization_results': optimization_results}
    
    def _run_testing_phase(self, best_params: Dict, profile: str) -> Dict:
        """
        Exécute la phase de test avec les paramètres optimaux
        
        Args:
            best_params: Paramètres optimaux
            profile: Profil d'optimisation
            
        Returns:
            Dictionnaire des résultats de test
        """
        if best_params is None:
            logger.warning("Aucun paramètre optimal fourni pour le test")
            return None
        
        # Configuration du calculateur de métriques
        self.qaaf_core.metrics_calculator.volatility_window = best_params['volatility_window']
        self.qaaf_core.metrics_calculator.spectral_window = best_params['spectral_window']
        self.qaaf_core.metrics_calculator.min_periods = best_params['min_periods']
        
        # Analyse des phases de marché
        self.qaaf_core.analyze_market_phases()
        
        # Calcul des métriques
        self.qaaf_core.calculate_metrics()
        
        # Calcul du score composite avec les poids optimisés
        self.qaaf_core.calculate_composite_score(best_params['weights'])
        
        # Configuration de l'allocateur
        self.qaaf_core.adaptive_allocator.min_btc_allocation = best_params['min_btc_allocation']
        self.qaaf_core.adaptive_allocator.max_btc_allocation = best_params['max_btc_allocation']
        self.qaaf_core.adaptive_allocator.sensitivity = best_params['sensitivity']
        self.qaaf_core.adaptive_allocator.observation_period = best_params['observation_period']
        
        # Calcul des allocations
        self.qaaf_core.calculate_adaptive_allocations()
        
        # Configuration du backtester
        self.qaaf_core.backtester.rebalance_threshold = best_params['rebalance_threshold']
        
        # Exécution du backtest
        self.qaaf_core.run_backtest()
        
        # Retour des résultats
        return {
            'performance': self.qaaf_core.results['metrics'],
            'portfolio_values': self.qaaf_core.performance,
            'allocations': self.qaaf_core.allocations
        }
    
    def _analyze_cv_results(self, cv_results: List[Dict]) -> Dict:
        """
        Analyse les résultats de validation croisée
        
        Args:
            cv_results: Liste des résultats de validation croisée
            
        Returns:
            Dictionnaire des résultats d'analyse
        """
        if not cv_results:
            return {}
        
        # Extraction des performances
        train_performances = []
        test_performances = []
        
        for result in cv_results:
            train_performances.append(result['train']['performance'])
            test_performances.append(result['test']['performance'])
        
        # Calcul des moyennes et écarts-types
        train_mean = {}
        train_std = {}
        test_mean = {}
        test_std = {}
        
        # Métriques à analyser
        metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'volatility']
        
        for metric in metrics:
            train_values = [perf[metric] for perf in train_performances]
            test_values = [perf[metric] for perf in test_performances]
            
            train_mean[metric] = np.mean(train_values)
            train_std[metric] = np.std(train_values)
            test_mean[metric] = np.mean(test_values)
            test_std[metric] = np.std(test_values)
        
        # Analyse des paramètres optimaux
        param_stability = self._analyze_parameter_stability(cv_results)
        
        return {
            'splits': len(cv_results),
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'param_stability': param_stability,
            'consistency_ratio': test_mean['total_return'] / train_mean['total_return'] if train_mean['total_return'] != 0 else float('inf'),
            'risk_ratio': test_mean['max_drawdown'] / train_mean['max_drawdown'] if train_mean['max_drawdown'] != 0 else float('inf'),
        }
    
    def _analyze_parameter_stability(self, cv_results: List[Dict]) -> Dict:
        """
        Analyse la stabilité des paramètres optimaux
        
        Args:
            cv_results: Liste des résultats de validation croisée
            
        Returns:
            Dictionnaire des analyses de stabilité
        """
        # Extraction des poids des métriques
        metric_weights = {
            'vol_ratio': [],
            'bound_coherence': [],
            'alpha_stability': [],
            'spectral_score': []
        }
        
        # Extraction des paramètres
        parameters = {
            'min_btc_allocation': [],
            'max_btc_allocation': [],
            'rebalance_threshold': [],
            'sensitivity': [],
            'observation_period': [],
            'volatility_window': [],
            'spectral_window': [],
            'min_periods': []
        }
        
        for result in cv_results:
            if result['train']['best_params'] is None:
                continue
                
            # Poids des métriques
            weights = result['train']['best_params']['weights']
            for metric in metric_weights:
                metric_weights[metric].append(weights.get(metric, 0))
            
            # Autres paramètres
            for param in parameters:
                parameters[param].append(result['train']['best_params'][param])
        
        # Calcul des statistiques pour les poids
        weight_stats = {}
        for metric, values in metric_weights.items():
            if not values:
                continue
                
            weight_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0  # Coefficient de variation
            }
        
        # Calcul des statistiques pour les paramètres
        param_stats = {}
        for param, values in parameters.items():
            if not values:
                continue
                
            param_stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0  # Coefficient de variation
            }
        
        return {
            'weight_stats': weight_stats,
            'param_stats': param_stats
        }
    
    def run_stress_test(self, 
                      scenarios: List[str] = ['bull_market', 'bear_market', 'consolidation'],
                      profile: str = 'balanced') -> Dict:
        """
        Exécute un test de stress sur différents scénarios de marché
        
        Args:
            scenarios: Liste des scénarios à tester
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Dictionnaire des résultats de test de stress
        """
        logger.info(f"Démarrage du test de stress pour les scénarios: {', '.join(scenarios)}")
        
        # Définition des périodes de scénario
        scenario_periods = self._define_scenario_periods()
        
        # Sauvegarde de l'instance QAAF originale
        original_qaaf = self.qaaf_core
        
        # Préparation des résultats
        stress_results = {}
        
        try:
            # 1. Entraînement sur toutes les données
            logger.info("Entraînement sur toutes les données")
            
            # Phase d'entraînement et d'optimisation
            self.qaaf_core.data = self.data
            train_results = self._run_training_phase(profile)
            
            if train_results['best_params'] is None:
                logger.warning("L'entraînement a échoué")
                return {}
            
            # 2. Test sur chaque scénario
            for scenario in scenarios:
                if scenario not in scenario_periods:
                    logger.warning(f"Scénario {scenario} non défini")
                    continue
                
                logger.info(f"Test sur le scénario: {scenario}")
                
                # Extraction des dates du scénario
                start_date = scenario_periods[scenario]['start']
                end_date = scenario_periods[scenario]['end']
                
                # Préparation des données du scénario
                scenario_data = {}
                
                for asset, df in self.data.items():
                    scenario_mask = (df.index >= start_date) & (df.index <= end_date)
                    scenario_data[asset] = df[scenario_mask].copy()
                
                # Phase de test
                self.qaaf_core.data = scenario_data
                scenario_results = self._run_testing_phase(train_results['best_params'], profile)
                
                # Stockage des résultats
                stress_results[scenario] = {
                    'period': f"{start_date} à {end_date}",
                    'performance': scenario_results['performance'],
                    'portfolio_values': scenario_results['portfolio_values'],
                    'allocations': scenario_results['allocations']
                }
            
            # Stockage des résultats
            self.results['stress'] = {
                'training': train_results,
                'scenarios': stress_results
            }
            
            return self.results
            
        finally:
            # Restauration de l'instance QAAF originale
            self.qaaf_core = original_qaaf
    
    def _define_scenario_periods(self) -> Dict:
        """
        Définit les périodes correspondant à différents scénarios de marché
        
        Returns:
            Dictionnaire des périodes par scénario
        """
        # Cette fonction devrait idéalement utiliser des points de retournement
        # détectés automatiquement, mais pour simplifier, nous utilisons des
        # périodes prédéfinies basées sur l'historique BTC
        
        # Obtention des dates communes
        common_dates = self._get_common_dates()
        start_date = min(common_dates)
        end_date = max(common_dates)
        
        # Définition des périodes de scénario (ces dates sont approximatives)
        return {
            'bull_market': {
                'start': pd.Timestamp('2020-10-01'),
                'end': pd.Timestamp('2021-04-30')
            },
            'bear_market': {
                'start': pd.Timestamp('2022-01-01'),
                'end': pd.Timestamp('2022-06-30')
            },
            'consolidation': {
                'start': pd.Timestamp('2023-01-01'),
                'end': pd.Timestamp('2023-06-30')
            },
            'recovery': {
                'start': pd.Timestamp('2023-07-01'),
                'end': pd.Timestamp('2024-02-17')
            }
        }
    
    def print_stress_test_summary(self) -> None:
        """
        Affiche un résumé des résultats du test de stress
        """
        if 'stress' not in self.results:
            logger.warning("Aucun résultat de test de stress disponible")
            return
        
        stress_results = self.results['stress']
        
        print("\n=== Résumé du Test de Stress ===\n")
        
        # Performance globale
        train_perf = stress_results['training']['performance']
        
        print("Performance Globale (Entraînement):")
        print(f"- Rendement total: {train_perf['total_return']:.2f}%")
        print(f"- Drawdown maximum: {train_perf['max_drawdown']:.2f}%")
        print(f"- Ratio de Sharpe: {train_perf['sharpe_ratio']:.2f}")
        
        # Performance par scénario
        print("\nPerformance par Scénario:")
        
        for scenario, results in stress_results['scenarios'].items():
            perf = results['performance']
            
            print(f"\n{scenario.upper()} ({results['period']}):")
            print(f"- Rendement total: {perf['total_return']:.2f}%")
            print(f"- Drawdown maximum: {perf['max_drawdown']:.2f}%")
            print(f"- Ratio de Sharpe: {perf['sharpe_ratio']:.2f}")
        
        # Analyse de robustesse
        print("\nAnalyse de Robustesse par Scénario:")
        
        for scenario, results in stress_results['scenarios'].items():
            perf = results['performance']
            
            # Ratio de rendement (scénario / entraînement)
            return_ratio = perf['total_return'] / train_perf['total_return'] if train_perf['total_return'] != 0 else float('inf')
            
            # Ratio de drawdown (scénario / entraînement)
            dd_ratio = perf['max_drawdown'] / train_perf['max_drawdown'] if train_perf['max_drawdown'] != 0 else float('inf')
            
            print(f"\n{scenario.upper()}:")
            print(f"- Ratio de rendement: {return_ratio:.2f}")
            print(f"- Ratio de drawdown: {dd_ratio:.2f}")
            
            # Classification de la robustesse
            robustness = "Bonne"
            if return_ratio < 0.5 or dd_ratio > 1.5:
                robustness = "Faible"
            elif return_ratio < 0.7 or dd_ratio > 1.3:
                robustness = "Moyenne"
            
            print(f"- Robustesse: {robustness}")
    
    def plot_stress_test_results(self) -> None:
        """
        Visualise les résultats du test de stress
        """
        if 'stress' not in self.results:
            logger.warning("Aucun résultat de test de stress disponible")
            return
        
        stress_results = self.results['stress']
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        
        # 1. Graphique des performances par scénario
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Liste des scénarios
        scenarios = list(stress_results['scenarios'].keys())
        
        # Préparation des données
        returns = [stress_results['scenarios'][s]['performance']['total_return'] for s in scenarios]
        drawdowns = [stress_results['scenarios'][s]['performance']['max_drawdown'] for s in scenarios]
        sharpes = [stress_results['scenarios'][s]['performance']['sharpe_ratio'] for s in scenarios]
        
        # Tracé des barres de rendement
        ax1.bar(scenarios, returns, alpha=0.7, color='green', label='Rendement (%)')
        
        # Axe secondaire pour le drawdown et le Sharpe
        ax2 = ax1.twinx()
        ax2.bar([s + 0.25 for s in range(len(scenarios))], drawdowns, width=0.25, alpha=0.7, color='red', label='Drawdown (%)')
        ax2.bar([s + 0.5 for s in range(len(scenarios))], sharpes, width=0.25, alpha=0.7, color='blue', label='Sharpe')
        
        # Configuration du graphique
        ax1.set_title('Performance par Scénario')
        ax1.set_xlabel('Scénario')
        ax1.set_ylabel('Rendement (%)', color='green')
        ax2.set_ylabel('Drawdown (%) / Sharpe', color='blue')
        
        # Légende
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 2. Graphique des allocations par scénario
        num_scenarios = len(scenarios)
        fig2, axes = plt.subplots(num_scenarios, 1, figsize=(12, 4 * num_scenarios))
        
        # Si un seul scénario, convertir en liste
        if num_scenarios == 1:
            axes = [axes]
        
        # Pour chaque scénario
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            
            # Extraction des allocations
            allocations = stress_results['scenarios'][scenario]['allocations']
            
            # Tracé des allocations
            ax.plot(allocations.index, allocations, 'b-')
            ax.set_title(f'Allocation BTC - {scenario.upper()}')
            ax.set_ylabel('Allocation (%)')
            ax.set_ylim(0, 1)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()