"""
QAAF Out-of-Sample Validator - Version 1.0.0
--------------------------------------------
Module de validation out-of-sample pour tester la robustesse des stratégies QAAF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OutOfSampleValidator:
    """
    Validateur Out-of-Sample pour QAAF
    
    Cette classe permet de tester la robustesse des stratégies QAAF
    en les entraînant sur une période et en les testant sur une autre.
    """
    
    def __init__(self, 
                qaaf_core,
                data: Dict[str, pd.DataFrame]):
        """
        Initialise le validateur Out-of-Sample
        
        Args:
            qaaf_core: Instance du core QAAF
            data: Dictionnaire des données ('BTC', 'PAXG', 'PAXG/BTC')
        """
        self.qaaf_core = qaaf_core
        self.data = data
        self.results = {}
    
    def split_data(self, 
                  test_ratio: float = 0.3, 
                  validation_ratio: float = 0.0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Divise les données en ensembles d'entraînement, validation et test
        
        Args:
            test_ratio: Proportion des données à utiliser pour le test
            validation_ratio: Proportion des données à utiliser pour la validation
            
        Returns:
            Dictionnaire des ensembles de données
        """
        logger.info(f"Division des données: Train={100*(1-test_ratio-validation_ratio):.0f}%, "
                  f"Validation={100*validation_ratio:.0f}%, Test={100*test_ratio:.0f}%")
        
        # Vérification des ratios
        if test_ratio + validation_ratio >= 1:
            raise ValueError("La somme des ratios test et validation doit être inférieure à 1")
        
        # Obtenir les dates communes à tous les DataFrames
        common_dates = self._get_common_dates()
        
        # Tri des dates
        common_dates = sorted(common_dates)
        
        # Nombre de points pour chaque ensemble
        n_total = len(common_dates)
        n_test = int(n_total * test_ratio)
        n_validation = int(n_total * validation_ratio)
        n_train = n_total - n_test - n_validation
        
        # Dates de séparation
        train_end_idx = n_train - 1
        validation_end_idx = n_train + n_validation - 1
        
        train_end_date = common_dates[train_end_idx]
        validation_end_date = common_dates[validation_end_idx] if validation_ratio > 0 else train_end_date
        
        logger.info(f"Dates de séparation: Train jusqu'à {train_end_date}, "
                  f"Validation jusqu'à {validation_end_date}, "
                  f"Test jusqu'à {common_dates[-1]}")
        
        # Création des sous-ensembles
        train_data = {}
        validation_data = {}
        test_data = {}
        
        for asset, df in self.data.items():
            # Conversion des dates en indices
            train_mask = df.index <= train_end_date
            validation_mask = (df.index > train_end_date) & (df.index <= validation_end_date)
            test_mask = df.index > validation_end_date
            
            train_data[asset] = df[train_mask].copy()
            validation_data[asset] = df[validation_mask].copy() if validation_ratio > 0 else pd.DataFrame()
            test_data[asset] = df[test_mask].copy()
        
        return {
            'train': train_data,
            'validation': validation_data if validation_ratio > 0 else None,
            'test': test_data,
            'split_dates': {
                'train_end': train_end_date,
                'validation_end': validation_end_date if validation_ratio > 0 else None
            }
        }
    
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
    
    def run_validation(self, 
                      test_ratio: float = 0.3,
                      validation_ratio: float = 0.0,
                      profile: str = 'balanced') -> Dict:
        """
        Exécute la validation out-of-sample
        
        Args:
            test_ratio: Proportion des données à utiliser pour le test
            validation_ratio: Proportion des données à utiliser pour la validation
            profile: Profil d'optimisation à utiliser
            
        Returns:
            Dictionnaire des résultats de validation
        """
        logger.info(f"Démarrage de la validation out-of-sample pour le profil {profile}")
        
        # Division des données
        split_data = self.split_data(test_ratio, validation_ratio)
        
        # Sauvegarde de l'instance QAAF originale
        original_qaaf = self.qaaf_core
        
        try:
            # 1. Entraînement sur l'ensemble d'entraînement
            logger.info("Entraînement sur l'ensemble d'entraînement")
            
            # Définir les données d'entraînement
            self.qaaf_core.data = split_data['train']
            
            # Phase d'entraînement et d'optimisation
            train_results = self._run_training_phase(profile)
            
            # 2. Validation (si applicable)
            validation_results = None
            if validation_ratio > 0 and split_data['validation'] is not None:
                logger.info("Validation de la meilleure configuration")
                
                # Définir les données de validation
                self.qaaf_core.data = split_data['validation']
                
                # Phase de validation
                validation_results = self._run_testing_phase(train_results['best_params'], profile)
            
            # 3. Test sur l'ensemble de test
            logger.info("Test sur l'ensemble de test")
            
            # Définir les données de test
            self.qaaf_core.data = split_data['test']
            
            # Phase de test
            test_results = self._run_testing_phase(train_results['best_params'], profile)
            
            # Stockage des résultats
            self.results = {
                'train': train_results,
                'validation': validation_results,
                'test': test_results,
                'split_dates': split_data['split_dates']
            }
            
            logger.info("Validation out-of-sample terminée")
            
            return self.results
            
        finally:
            # Restauration de l'instance QAAF originale
            self.qaaf_core = original_qaaf
    
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
    
    def print_validation_summary(self) -> None:
        """
        Affiche un résumé des résultats de validation
        """
        if not self.results:
            logger.warning("Aucun résultat de validation disponible")
            return
        
        print("\n=== Résumé de la Validation Out-of-Sample ===\n")
        
        # Informations sur la division des données
        train_end = self.results['split_dates']['train_end']
        validation_end = self.results['split_dates'].get('validation_end')
        
        print(f"Période d'entraînement: jusqu'au {train_end}")
        if validation_end:
            print(f"Période de validation: du {train_end} au {validation_end}")
        
        # Phase d'entraînement
        train_perf = self.results['train']['performance']
        
        print("\nPerformance en Entraînement:")
        print(f"- Rendement total: {train_perf['total_return']:.2f}%")
        print(f"- Drawdown maximum: {train_perf['max_drawdown']:.2f}%")
        print(f"- Ratio de Sharpe: {train_perf['sharpe_ratio']:.2f}")
        
        # Phase de validation (si applicable)
        if self.results.get('validation'):
            val_perf = self.results['validation']['performance']
            
            print("\nPerformance en Validation:")
            print(f"- Rendement total: {val_perf['total_return']:.2f}%")
            print(f"- Drawdown maximum: {val_perf['max_drawdown']:.2f}%")
            print(f"- Ratio de Sharpe: {val_perf['sharpe_ratio']:.2f}")
        
        # Phase de test
        test_perf = self.results['test']['performance']
        
        print("\nPerformance en Test:")
        print(f"- Rendement total: {test_perf['total_return']:.2f}%")
        print(f"- Drawdown maximum: {test_perf['max_drawdown']:.2f}%")
        print(f"- Ratio de Sharpe: {test_perf['sharpe_ratio']:.2f}")
        
        # Analyse de la robustesse
        print("\nAnalyse de Robustesse:")
        
        train_return = train_perf['total_return']
        test_return = test_perf['total_return']
        return_ratio = test_return / train_return if train_return != 0 else float('inf')
        
        train_dd = train_perf['max_drawdown']
        test_dd = test_perf['max_drawdown']
        dd_ratio = test_dd / train_dd if train_dd != 0 else float('inf')
        
        print(f"- Ratio de rendement (Test/Train): {return_ratio:.2f}")
        print(f"- Ratio de drawdown (Test/Train): {dd_ratio:.2f}")
        
        if return_ratio >= 0.7:
            print("✓ Robustesse du rendement: Bonne (≥70% du rendement d'entraînement)")
        elif return_ratio >= 0.5:
            print("△ Robustesse du rendement: Moyenne (50-70% du rendement d'entraînement)")
        else:
            print("✗ Robustesse du rendement: Faible (<50% du rendement d'entraînement)")
        
        if dd_ratio <= 1.3:
            print("✓ Robustesse du drawdown: Bonne (≤130% du drawdown d'entraînement)")
        elif dd_ratio <= 1.5:
            print("△ Robustesse du drawdown: Moyenne (130-150% du drawdown d'entraînement)")
        else:
            print("✗ Robustesse du drawdown: Faible (>150% du drawdown d'entraînement)")
    
    def plot_validation_results(self) -> None:
        """
        Visualise les résultats de validation
        """
        if not self.results:
            logger.warning("Aucun résultat de validation disponible")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        
        # Création des figures
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Extraction des données
        train_values = self.results['train']['portfolio_values']
        train_alloc = self.results['train']['allocations']
        
        test_values = self.results['test']['portfolio_values']
        test_alloc = self.results['test']['allocations']
        
        # Normalisation pour la comparaison
        train_start = train_values.iloc[0]
        test_start = test_values.iloc[0]
        
        norm_train = train_values / train_start
        norm_test = test_values / test_start
        
        # Phase de validation (si applicable)
        if self.results.get('validation'):
            val_values = self.results['validation']['portfolio_values']
            val_alloc = self.results['validation']['allocations']
            val_start = val_values.iloc[0]
            norm_val = val_values / val_start
        
        # 1. Graphique des performances
        # Entraînement
        ax1.plot(norm_train.index, norm_train, 'b-', label='Entraînement')
        
        # Validation (si applicable)
        if self.results.get('validation'):
            ax1.plot(norm_val.index, norm_val, 'g-', label='Validation')
        
        # Test
        ax1.plot(norm_test.index, norm_test, 'r-', label='Test')
        
        # Séparations
        ax1.axvline(x=self.results['split_dates']['train_end'], color='k', linestyle='--')
        if self.results['split_dates'].get('validation_end'):
            ax1.axvline(x=self.results['split_dates']['validation_end'], color='k', linestyle='--')
        
        ax1.set_title('Performance Out-of-Sample')
        ax1.set_ylabel('Performance normalisée')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Graphique des allocations
        # Entraînement
        ax2.plot(train_alloc.index, train_alloc, 'b-', label='Entraînement')
        
        # Validation (si applicable)
        if self.results.get('validation'):
            ax2.plot(val_alloc.index, val_alloc, 'g-', label='Validation')
        
        # Test
        ax2.plot(test_alloc.index, test_alloc, 'r-', label='Test')
        
        # Séparations
        ax2.axvline(x=self.results['split_dates']['train_end'], color='k', linestyle='--')
        if self.results['split_dates'].get('validation_end'):
            ax2.axvline(x=self.results['split_dates']['validation_end'], color='k', linestyle='--')
        
        ax2.set_title('Allocation BTC')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Allocation BTC (%)')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Comparaison des performances
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        # Extraction des métriques
        metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'volatility']
        labels = ['Rendement Total (%)', 'Drawdown Max (%)', 'Ratio de Sharpe', 'Volatilité (%)']
        
        train_metrics = [self.results['train']['performance'][m] for m in metrics]
        test_metrics = [self.results['test']['performance'][m] for m in metrics]
        
        # Données pour la validation (si applicable)
        val_metrics = None
        if self.results.get('validation'):
            val_metrics = [self.results['validation']['performance'][m] for m in metrics]
        
        # Configuration du graphique
        x = np.arange(len(metrics))
        width = 0.25
        
        # Tracé des barres
       rects1 = ax.bar(x - width, train_metrics, width, label='Entraînement', color='blue', alpha=0.7)
       
       if val_metrics:
           rects2 = ax.bar(x, val_metrics, width, label='Validation', color='green', alpha=0.7)
           rects3 = ax.bar(x + width, test_metrics, width, label='Test', color='red', alpha=0.7)
       else:
           rects2 = ax.bar(x + width, test_metrics, width, label='Test', color='red', alpha=0.7)
       
       # Configuration du graphique
       ax.set_ylabel('Valeur')
       ax.set_title('Comparaison des métriques de performance')
       ax.set_xticks(x)
       ax.set_xticklabels(labels)
       ax.legend()
       
       # Ajout des valeurs sur les barres
       def autolabel(rects):
           for rect in rects:
               height = rect.get_height()
               ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
       
       autolabel(rects1)
       if val_metrics:
           autolabel(rects2)
           autolabel(rects3)
       else:
           autolabel(rects2)
       
       fig2.tight_layout()
       plt.show()