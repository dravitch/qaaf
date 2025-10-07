"""
Module principal du framework QAAF
"""

import pandas as pd
import numpy as np
from typing import Dict,List,Optional,Tuple,Union
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt

from qaaf.metrics.calculator import MetricsCalculator

# Import des autres modules au fur et à mesure de leur création

logger=logging.getLogger (__name__)


class QAAFCore:
    """
    Classe principale du framework QAAF

    Combine les différents composants pour une expérience intégrée:
    - Chargement des données
    - Calcul des métriques
    - Optimisation
    - Backtest
    - Comparaison avec les benchmarks
    """

    def __init__ (self,
                  initial_capital: float = 30000.0,
                  trading_costs: float = 0.001,
                  start_date: str = '2020-01-01',
                  end_date: str = '2024-12-31',
                  allocation_min: float = 0.1,
                  allocation_max: float = 0.9,
                  use_gpu: bool = None):
        """
        Initialise le core QAAF

        Args:
            initial_capital: Capital initial pour le backtest
            trading_costs: Coûts de transaction (en % du montant)
            start_date: Date de début de l'analyse
            end_date: Date de fin de l'analyse
            allocation_min: Allocation minimale en BTC
            allocation_max: Allocation maximale en BTC
            use_gpu: Utiliser le GPU si disponible (None pour auto-détection)
        """
        self.initial_capital=initial_capital
        self.trading_costs=trading_costs
        self.start_date=start_date
        self.end_date=end_date
        self.use_gpu=use_gpu

        # Ici, ajoutez les autres initialisations (data_manager, market_phase_analyzer, etc.)
        # à mesure que les modules sont créés
        from qaaf.data.data_manager import DataManager
        from qaaf.metrics.calculator import MetricsCalculator
        from qaaf.market.phase_analyzer import MarketPhaseAnalyzer
        from qaaf.allocation.adaptive_allocator import AdaptiveAllocator
        from qaaf.transaction.fees_evaluator import TransactionFeesEvaluator
        from qaaf.transaction.backtester import QAAFBacktester

        # Initialisation des composants
        self.data_manager=DataManager ()
        self.metrics_calculator=MetricsCalculator (use_gpu=use_gpu)
        self.market_phase_analyzer=MarketPhaseAnalyzer ()
        self.adaptive_allocator=AdaptiveAllocator (
            min_btc_allocation=allocation_min,
            max_btc_allocation=allocation_max,
            neutral_allocation=0.5,
            sensitivity=1.0
        )
        self.fees_evaluator=TransactionFeesEvaluator (base_fee_rate=trading_costs)
        self.backtester=QAAFBacktester (
            initial_capital=initial_capital,
            fees_evaluator=self.fees_evaluator,
            rebalance_threshold=0.05
        )

        # Ces composants seront initialisés après le chargement des données
        self.optimizer=None
        self.validator=None
        self.robustness_tester=None

        # Stockage des résultats
        self.data=None
        self.metrics=None
        self.composite_score=None
        self.market_phases=None
        self.allocations=None
        self.performance=None
        self.results=None
        self.optimization_results=None
        self.validation_results=None
        self.robustness_results=None

    def load_data (self,start_date: Optional[str] = None,end_date: Optional[str] = None) -> Dict[str,pd.DataFrame]:
        """
        Charge les données nécessaires pour l'analyse

        Args:
            start_date: Date de début (optionnel, sinon utilise celle de l'initialisation)
            end_date: Date de fin (optionnel, sinon utilise celle de l'initialisation)

        Returns:
            Dictionnaire des DataFrames chargés
        """
        _start_date=start_date or self.start_date
        _end_date=end_date or self.end_date

        logger.info (f"Chargement des données de {_start_date} à {_end_date}")

        # Chargement des données via le DataManager
        self.data=self.data_manager.prepare_qaaf_data (_start_date,_end_date)

        # Initialisation des modules qui nécessitent les données
        from qaaf.optimization.grid_search import QAAFOptimizer
        from qaaf.validation.out_of_sample import OutOfSampleValidator
        from qaaf.validation.robustness import RobustnessTester

        self.optimizer=QAAFOptimizer (
            data=self.data,
            metrics_calculator=self.metrics_calculator,
            market_phase_analyzer=self.market_phase_analyzer,
            adaptive_allocator=self.adaptive_allocator,
            backtester=self.backtester,
            initial_capital=self.initial_capital
        )

        self.validator=OutOfSampleValidator (
            qaaf_core=self,
            data=self.data
        )

        self.robustness_tester=RobustnessTester (
            qaaf_core=self,
            data=self.data
        )

        return self.data

    def analyze_market_phases (self) -> pd.Series:
        """
        Analyse les phases de marché

        Returns:
            Série des phases de marché
        """
        if self.data is None:
            raise ValueError (
                "Aucune donnée chargée. Appelez load_data() ou affectez directement des données à qaaf.data.")

        logger.info ("Analyse des phases de marché")

        # Identification des phases de marché
        self.market_phases=self.market_phase_analyzer.identify_market_phases (self.data['BTC'])

        return self.market_phases

    def calculate_metrics (self) -> Dict[str,pd.Series]:
        """
        Calcule les métriques QAAF

        Returns:
            Dictionnaire des métriques calculées
        """
        if self.data is None:
            raise ValueError (
                "Aucune donnée chargée. Appelez load_data() ou affectez directement des données à qaaf.data.")

        logger.info ("Calcul des métriques QAAF")

        # Calcul des métriques via le MetricsCalculator
        self.metrics=self.metrics_calculator.calculate_metrics (self.data)

        return self.metrics

    def calculate_composite_score (self,weights: Optional[Dict[str,float]] = None) -> pd.Series:
        """
        Calcule le score composite

        Args:
            weights: Dictionnaire des poids pour chaque métrique

        Returns:
            Série du score composite
        """
        if self.metrics is None:
            raise ValueError ("Aucune métrique calculée. Appelez calculate_metrics() d'abord.")

        logger.info ("Calcul du score composite")

        # Poids par défaut si non fournis
        if weights is None:
            weights={
                'vol_ratio':0.3,
                'bound_coherence':0.3,
                'alpha_stability':0.2,
                'spectral_score':0.2
            }

        # Normalisation des métriques
        normalized_metrics=self.metrics_calculator.normalize_metrics (self.metrics)

        # Calcul du score composite
        self.composite_score=pd.Series (0.0,index=normalized_metrics[list (normalized_metrics.keys ())[0]].index)
        for name,series in normalized_metrics.items ():
            if name in weights:
                self.composite_score+=weights[name] * series

        return self.composite_score

    def calculate_adaptive_allocations (self) -> pd.Series:
        """
        Calcule les allocations adaptatives

        Returns:
            Série des allocations BTC
        """
        if self.composite_score is None:
            raise ValueError ("Aucun score composite calculé. Appelez calculate_composite_score() d'abord.")

        if self.market_phases is None:
            raise ValueError ("Aucune phase de marché identifiée. Appelez analyze_market_phases() d'abord.")

        logger.info ("Calcul des allocations adaptatives")

        # Calcul des allocations via l'AdaptiveAllocator
        self.allocations=self.adaptive_allocator.calculate_adaptive_allocation (
            self.composite_score,
            self.market_phases
        )

        return self.allocations

    def run_backtest (self) -> Dict:
        """
        Exécute le backtest

        Returns:
            Dictionnaire des résultats du backtest
        """
        if self.allocations is None:
            raise ValueError ("Aucune allocation calculée. Appelez calculate_adaptive_allocations() d'abord.")

        logger.info ("Exécution du backtest")

        # Exécution du backtest
        self.performance,metrics=self.backtester.run_backtest (
            self.data['BTC'],
            self.data['PAXG'],
            self.allocations
        )

        # Comparaison avec les benchmarks
        comparison=self.backtester.compare_with_benchmarks (metrics)

        # Stockage des résultats
        self.results={
            'metrics':metrics,
            'comparison':comparison
        }

        return self.results

    def optimize_rebalance_threshold (self,thresholds: List[float] = [0.01,0.03,0.05,0.1]) -> Dict:
        """
        Optimise le seuil de rebalancement

        Args:
            thresholds: Liste des seuils à tester

        Returns:
            Dictionnaire des résultats par seuil
        """
        if self.allocations is None:
            raise ValueError ("Aucune allocation calculée. Appelez calculate_adaptive_allocations() d'abord.")

        logger.info ("Optimisation du seuil de rebalancement")

        # Test des différents seuils
        results=self.backtester.run_multi_threshold_test (
            self.data['BTC'],
            self.data['PAXG'],
            self.allocations,
            thresholds
        )

        # Calcul du seuil optimal (meilleur compromis entre performance et frais)
        threshold_metrics=[]
        for threshold,result in results.items ():
            metrics=result['metrics']
            total_fees=result['total_fees']
            transaction_count=result['transaction_count']
            fee_drag=result['fee_drag']
            combined_score=result['combined_score']

            threshold_metrics.append ({
                'threshold':threshold,
                'total_return':metrics['total_return'],
                'sharpe_ratio':metrics['sharpe_ratio'],
                'max_drawdown':metrics['max_drawdown'],
                'total_fees':total_fees,
                'transaction_count':transaction_count,
                'fee_drag':fee_drag,
                'combined_score':combined_score
            })

        # Tri par score combiné
        sorted_metrics=sorted (threshold_metrics,key=lambda x:x['combined_score'],reverse=True)

        # Sélection du meilleur seuil
        optimal_threshold=sorted_metrics[0]['threshold']
        logger.info (f"Seuil optimal de rebalancement: {optimal_threshold:.1%}")

        return {
            'results':results,
            'metrics':threshold_metrics,
            'optimal_threshold':optimal_threshold
        }

    def run_metrics_optimization (self,profile: str = 'balanced',max_combinations: int = 10000) -> Dict:
        """
        Exécute l'optimisation des métriques et des poids

        Args:
            profile: Profil d'optimisation à utiliser
            max_combinations: Nombre maximal de combinaisons à tester

        Returns:
            Dictionnaire des résultats d'optimisation
        """
        if self.data is None:
            raise ValueError ("Aucune donnée chargée. Appelez load_data() d'abord.")

        if self.metrics is None:
            raise ValueError ("Aucune métrique calculée. Appelez calculate_metrics() d'abord.")

        logger.info (f"Exécution de l'optimisation des métriques avec le profil {profile}")

        # MODIFIÉ: Utilisation du nouveau QAAFOptimizer
        self.optimization_results=self.optimizer.run_optimization (profile,max_combinations)

        # Si disponible, mise à jour du score composite avec les poids optimaux
        if profile in self.optimizer.best_combinations:
            best_weights=self.optimizer.best_combinations[profile]['normalized_weights']
            self.calculate_composite_score (best_weights)

        return {
            'results':self.optimization_results,
            'best_combinations':self.optimizer.best_combinations
        }

    # NOUVELLES MÉTHODES pour la validation

    def run_out_of_sample_validation (self,test_ratio: float = 0.3,profile: str = 'balanced') -> Dict:
        """
        Exécute une validation out-of-sample

        Args:
            test_ratio: Proportion des données à utiliser pour le test
            profile: Profil d'optimisation à utiliser

        Returns:
            Dictionnaire des résultats de validation
        """
        if self.data is None:
            raise ValueError ("Aucune donnée chargée. Appelez load_data() d'abord.")

        logger.info (f"Exécution de la validation out-of-sample avec ratio de test {test_ratio}")

        # Exécution de la validation
        self.validation_results=self.validator.run_validation (test_ratio=test_ratio,profile=profile)

        # Affichage du résumé
        self.validator.print_validation_summary ()

        return self.validation_results

    def run_robustness_test (self,n_splits: int = 5,profile: str = 'balanced') -> Dict:
        """
        Exécute un test de robustesse via validation croisée temporelle

        Args:
            n_splits: Nombre de divisions temporelles
            profile: Profil d'optimisation à utiliser

        Returns:
            Dictionnaire des résultats de test de robustesse
        """
        if self.data is None:
            raise ValueError ("Aucune donnée chargée. Appelez load_data() d'abord.")

        logger.info (f"Exécution du test de robustesse avec {n_splits} divisions")

        # Exécution du test de robustesse
        self.robustness_results=self.robustness_tester.run_time_series_cross_validation (
            n_splits=n_splits,
            profile=profile
        )

        return self.robustness_results

    def run_stress_test (self,profile: str = 'balanced') -> Dict:
        """
        Exécute un test de stress sur différents scénarios de marché

        Args:
            profile: Profil d'optimisation à utiliser

        Returns:
            Dictionnaire des résultats de test de stress
        """
        if self.data is None:
            raise ValueError ("Aucune donnée chargée. Appelez load_data() d'abord.")

        logger.info ("Exécution du test de stress")

        # Exécution du test de stress
        stress_results=self.robustness_tester.run_stress_test (profile=profile)

        # Affichage du résumé
        self.robustness_tester.print_stress_test_summary ()

        return stress_results

    def print_summary (self) -> None:
        """Affiche un résumé des résultats de l'analyse QAAF."""
        if self.results is None:
            logger.warning ("Aucun résultat disponible. Appelez run_backtest() d'abord.")
            print ("\nAucun résultat à afficher.")
            return

        print ("\nRésumé des Résultats QAAF v1.0.0")
        print ("=" * 50)
        print (f"Date: {datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')}")
        print (f"Période analysée: {self.start_date} à {self.end_date}")
        print (f"Capital initial: ${self.initial_capital:,.2f}")

        # Métriques de performance
        metrics=self.results.get ('metrics',{})
        final_value=self.performance.iloc[-1] if self.performance is not None else 0
        print ("\nPerformance:")
        print (f"- Valeur finale: ${final_value:,.2f}")
        print (f"- Rendement total: {metrics.get ('total_return',0):.2f}%")
        print (f"- Volatilité: {metrics.get ('volatility',0):.2f}%")
        print (f"- Sharpe Ratio: {metrics.get ('sharpe_ratio',0):.2f}")
        print (f"- Drawdown maximum: {metrics.get ('max_drawdown',0):.2f}%")

        if metrics.get ('max_drawdown',0) != 0:
            rd_ratio=metrics.get ('total_return',0) / abs (metrics.get ('max_drawdown',0))
            print (f"- Ratio Rendement/Drawdown: {rd_ratio:.2f}")

        # Frais
        total_fees=self.fees_evaluator.get_total_fees ()
        fee_drag=(total_fees / final_value * 100) if final_value > 0 else 0
        print (f"\nFrais de transaction totaux: ${total_fees:,.2f}")
        print (f"Impact des frais sur la performance: {fee_drag:.2f}%")

        # Comparaison avec les benchmarks
        print ("\nComparaison avec les Benchmarks:")
        print ("-" * 50)
        comparison=self.results.get ('comparison',{})
        if comparison:
            for bench,values in comparison.items ():
                print (f"- {bench}: Rendement {values.get ('total_return',0):.2f}%, "
                       f"Sharpe {values.get ('sharpe_ratio',0):.2f}")
        else:
            print ("Aucune donnée de comparaison disponible.")

    def visualize_results (self) -> None:
        """
        Visualise les résultats de l'analyse
        """
        if self.performance is None or self.allocations is None:
            logger.warning ("Aucun résultat de backtest disponible. Appelez run_backtest() d'abord.")
            return

        logger.info ("Visualisation des résultats")

        if hasattr (self,'backtester') and callable (getattr (self.backtester,'plot_performance',None)):
            # Visualisation de la performance via le backtester
            self.backtester.plot_performance (
                self.performance,
                self.allocations,
                self.data['BTC'],
                self.data['PAXG']
            )
        else:
            # Visualisation basique en cas d'absence de backtester
            plt.figure (figsize=(15,10))

            # Graphique de performance
            plt.subplot (2,1,1)
            plt.plot (self.performance,'b-',label='QAAF Portfolio')
            plt.title ('Performance du Portefeuille')
            plt.legend ()
            plt.grid (True)

            # Graphique d'allocation
            plt.subplot (2,1,2)
            plt.plot (self.allocations,'g-',label='Allocation BTC')
            plt.title ('Allocation BTC')
            plt.ylim (0,1)
            plt.legend ()
            plt.grid (True)

            plt.tight_layout ()
            plt.show ()

    def run_full_analysis (self,
                           optimize_metrics: bool = True,
                           optimize_threshold: bool = True,
                           run_validation: bool = True,
                           run_robustness: bool = False,
                           profile: str = 'balanced') -> Dict:
        """
        Exécute l'analyse complète

        Args:
            optimize_metrics: Exécuter l'optimisation des métriques
            optimize_threshold: Exécuter l'optimisation du seuil de rebalancement
            run_validation: Exécuter la validation out-of-sample
            run_robustness: Exécuter les tests de robustesse
            profile: Profil d'optimisation à utiliser

        Returns:
            Dictionnaire des résultats
        """
        try:
            # Chargement des données
            self.load_data ()

            # Analyse des phases de marché
            self.analyze_market_phases ()

            # Calcul des métriques
            self.calculate_metrics ()

            # Optimisation des métriques (optionnel)
            metrics_results=None
            if optimize_metrics:
                logger.info ("Exécution de l'optimisation des métriques...")
                metrics_results=self.run_metrics_optimization (profile=profile)

                # Utilisation de la meilleure combinaison
                if profile in metrics_results.get ('best_combinations',{}):
                    best_combo=metrics_results['best_combinations'][profile]
                    logger.info (f"Utilisation de la meilleure combinaison (profil {profile})")

                    # Configuration selon les paramètres optimaux
                    self.configure_from_optimal_params (best_combo)
                else:
                    # Calcul du score composite avec les poids par défaut
                    self.calculate_composite_score ()
            else:
                # Calcul du score composite avec les poids par défaut
                self.calculate_composite_score ()

            # Calcul des allocations adaptatives
            self.calculate_adaptive_allocations ()

            # Optimisation du seuil de rebalancement (optionnel)
            threshold_results=None
            if optimize_threshold:
                logger.info ("Exécution de l'optimisation du seuil de rebalancement...")
                threshold_results=self.optimize_rebalance_threshold ()

                # Mise à jour du seuil de rebalancement
                if 'optimal_threshold' in threshold_results:
                    self.backtester.rebalance_threshold=threshold_results['optimal_threshold']

            # Exécution du backtest
            results=self.run_backtest ()

            # Validation out-of-sample (optionnel)
            validation_results=None
            if run_validation:
                logger.info ("Exécution de la validation out-of-sample...")
                validation_results=self.run_out_of_sample_validation (profile=profile)

            # Tests de robustesse (optionnel)
            robustness_results=None
            if run_robustness:
                logger.info ("Exécution des tests de robustesse...")
                robustness_results=self.run_robustness_test (profile=profile)

                logger.info ("Exécution des tests de stress...")
                stress_results=self.run_stress_test (profile=profile)

                # Combinaison des résultats de robustesse
                robustness_results={
                    'cross_validation':robustness_results,
                    'stress_test':stress_results
                }

            # Affichage du résumé
            self.print_summary ()

            # Visualisation des résultats
            try:
                self.visualize_results ()
            except Exception as viz_error:
                logger.error (f"Erreur lors de la visualisation: {str (viz_error)}")

            # Visualisation des résultats d'optimisation (si disponibles)
            if metrics_results is not None and hasattr (self,'optimizer'):
                try:
                    self.optimizer.plot_optimization_results (profile)
                    self.optimizer.plot_metrics_importance ()
                except Exception as opt_error:
                    logger.error (
                        f"Erreur lors de la visualisation des résultats d'optimisation: {str (opt_error)}")

            return {
                'results':results,
                'metrics_results':metrics_results,
                'threshold_results':threshold_results,
                'validation_results':validation_results,
                'robustness_results':robustness_results
            }
        except Exception as e:
            logger.error (f"Erreur lors de l'analyse complète: {str (e)}")
            # Retourner les résultats partiels s'ils existent
            return {
                'error':str (e),
                'performance':self.performance if hasattr (self,'performance') else None,
                'allocations':self.allocations if hasattr (self,'allocations') else None,
                'results':self.results if hasattr (self,'results') else None
            }

    def configure_from_optimal_params (self,optimal_config: Dict) -> None:
        """
        Configure les composants QAAF selon les paramètres optimaux

        Args:
            optimal_config: Configuration optimale
        """
        params=optimal_config['params']

        # Configuration du calculateur de métriques
        self.metrics_calculator.volatility_window=params['volatility_window']
        self.metrics_calculator.spectral_window=params['spectral_window']
        self.metrics_calculator.min_periods=params['min_periods']

        # Configuration de l'allocateur
        self.adaptive_allocator.min_btc_allocation=params['min_btc_allocation']
        self.adaptive_allocator.max_btc_allocation=params['max_btc_allocation']
        self.adaptive_allocator.sensitivity=params['sensitivity']
        self.adaptive_allocator.observation_period=params['observation_period']

        # Configuration du backtester
        self.backtester.rebalance_threshold=params['rebalance_threshold']

        logger.info ("Configuration des composants selon les paramètres optimaux terminée")

'''
    # def run_qaaf


    # Ajouter ici les méthodes principales:
    # - load_data
    # - analyze_market_phases
    # - calculate_metrics
    # - calculate_composite_score
    # - calculate_adaptive_allocations
    # - run_backtest
    # - optimize_rebalance_threshold
    # - run_metrics_optimization
    # - run_full_analysis
    # - print_summary
    # - visualize_results
'''
