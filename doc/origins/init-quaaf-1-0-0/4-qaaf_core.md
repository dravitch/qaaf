class QAAFCore:
    """
    Classe principale du framework QAAF
    
    Combine les différents composants pour une expérience intégrée :
    - Chargement des données
    - Calcul des métriques
    - Optimisation
    - Backtest
    - Comparaison avec les benchmarks
    """
    
    def __init__(self, 
                initial_capital: float = 30000.0,
                trading_costs: float = 0.001,
                start_date: str = '2020-01-01',
                end_date: str = '2024-02-25',
                allocation_min: float = 0.1,   # Nouveau: bornes d'allocation élargies
                allocation_max: float = 0.9):  # Nouveau: bornes d'allocation élargies
        """
        Initialise le core QAAF
        
        Args:
            initial_capital: Capital initial pour le backtest
            trading_costs: Coûts de transaction (en % du montant)
            start_date: Date de début de l'analyse
            end_date: Date de fin de l'analyse
            allocation_min: Allocation minimale en BTC
            allocation_max: Allocation maximale en BTC
        """
        self.initial_capital = initial_capital
        self.trading_costs = trading_costs
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialisation des composants
        self.data_manager = DataManager()
        self.metrics_calculator = MetricsCalculator()
        self.market_phase_analyzer = MarketPhaseAnalyzer()
        self.adaptive_allocator = AdaptiveAllocator(
            min_btc_allocation=allocation_min,
            max_btc_allocation=allocation_max,
            neutral_allocation=0.5,
            sensitivity=1.0
        )
        self.fees_evaluator = TransactionFeesEvaluator(base_fee_rate=trading_costs)
        self.backtester = QAAFBacktester(
            initial_capital=initial_capital,
            fees_evaluator=self.fees_evaluator,
            rebalance_threshold=0.05
        )
        
        # NOUVEAU: Ajout de l'optimiseur 1.0.0
        self.optimizer = None  # Sera initialisé après le chargement des données
        
        # NOUVEAU: Modules de validation
        self.validator = None  # Sera initialisé après le chargement des données
        self.robustness_tester = None  # Sera initialisé après le chargement des données
        
# Stockage des résultats
       self.data = None
       self.metrics = None
       self.composite_score = None
       self.market_phases = None
       self.allocations = None
       self.performance = None
       self.results = None
       self.optimization_results = None
       self.validation_results = None
       self.robustness_results = None
   
   def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
       """
       Charge les données nécessaires pour l'analyse
       
       Args:
           start_date: Date de début (optionnel, sinon utilise celle de l'initialisation)
           end_date: Date de fin (optionnel, sinon utilise celle de l'initialisation)
           
       Returns:
           Dictionnaire des DataFrames chargés
       """
       _start_date = start_date or self.start_date
       _end_date = end_date or self.end_date
       
       logger.info(f"Chargement des données de {_start_date} à {_end_date}")
       
       # Chargement des données via le DataManager
       self.data = self.data_manager.prepare_qaaf_data(_start_date, _end_date)
       
       # NOUVEAU: Initialisation des modules qui nécessitent les données
       self.optimizer = QAAFOptimizer(
           data=self.data,
           metrics_calculator=self.metrics_calculator,
           market_phase_analyzer=self.market_phase_analyzer,
           adaptive_allocator=self.adaptive_allocator,
           backtester=self.backtester,
           initial_capital=self.initial_capital
       )
       
       self.validator = OutOfSampleValidator(
           qaaf_core=self,
           data=self.data
       )
       
       self.robustness_tester = RobustnessTester(
           qaaf_core=self,
           data=self.data
       )
       
       return self.data
   
   # [...autres méthodes inchangées...]
   
   def run_metrics_optimization(self, profile: str = 'balanced', max_combinations: int = 10000) -> Dict:
       """
       Exécute l'optimisation des métriques et des poids
       
       Args:
           profile: Profil d'optimisation à utiliser
           max_combinations: Nombre maximal de combinaisons à tester
           
       Returns:
           Dictionnaire des résultats d'optimisation
       """
       if self.data is None:
           raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
       
       if self.metrics is None:
           raise ValueError("Aucune métrique calculée. Appelez calculate_metrics() d'abord.")
       
       logger.info(f"Exécution de l'optimisation des métriques avec le profil {profile}")
       
       # MODIFIÉ: Utilisation du nouveau QAAFOptimizer
       self.optimization_results = self.optimizer.run_optimization(profile, max_combinations)
       
       # Si disponible, mise à jour du score composite avec les poids optimaux
       if profile in self.optimizer.best_combinations:
           best_weights = self.optimizer.best_combinations[profile]['normalized_weights']
           self.calculate_composite_score(best_weights)
       
       return {
           'results': self.optimization_results,
           'best_combinations': self.optimizer.best_combinations
       }
   
   # NOUVELLES MÉTHODES pour la validation
   
   def run_out_of_sample_validation(self, test_ratio: float = 0.3, profile: str = 'balanced') -> Dict:
       """
       Exécute une validation out-of-sample
       
       Args:
           test_ratio: Proportion des données à utiliser pour le test
           profile: Profil d'optimisation à utiliser
           
       Returns:
           Dictionnaire des résultats de validation
       """
       if self.data is None:
           raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
       
       logger.info(f"Exécution de la validation out-of-sample avec ratio de test {test_ratio}")
       
       # Exécution de la validation
       self.validation_results = self.validator.run_validation(test_ratio=test_ratio, profile=profile)
       
       # Affichage du résumé
       self.validator.print_validation_summary()
       
       return self.validation_results
   
   def run_robustness_test(self, n_splits: int = 5, profile: str = 'balanced') -> Dict:
       """
       Exécute un test de robustesse via validation croisée temporelle
       
       Args:
           n_splits: Nombre de divisions temporelles
           profile: Profil d'optimisation à utiliser
           
       Returns:
           Dictionnaire des résultats de test de robustesse
       """
       if self.data is None:
           raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
       
       logger.info(f"Exécution du test de robustesse avec {n_splits} divisions")
       
       # Exécution du test de robustesse
       self.robustness_results = self.robustness_tester.run_time_series_cross_validation(
           n_splits=n_splits,
           profile=profile
       )
       
       return self.robustness_results
   
   def run_stress_test(self, profile: str = 'balanced') -> Dict:
       """
       Exécute un test de stress sur différents scénarios de marché
       
       Args:
           profile: Profil d'optimisation à utiliser
           
       Returns:
           Dictionnaire des résultats de test de stress
       """
       if self.data is None:
           raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
       
       logger.info("Exécution du test de stress")
       
       # Exécution du test de stress
       stress_results = self.robustness_tester.run_stress_test(profile=profile)
       
       # Affichage du résumé
       self.robustness_tester.print_stress_test_summary()
       
       return stress_results
   
   def run_full_analysis(self, 
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
       # Chargement des données
       self.load_data()
       
       # Analyse des phases de marché
       self.analyze_market_phases()
       
       # Calcul des métriques
       self.calculate_metrics()
       
       # Optimisation des métriques (optionnel)
       metrics_results = None
       if optimize_metrics:
           logger.info("Exécution de l'optimisation des métriques...")
           metrics_results = self.run_metrics_optimization(profile=profile)
           
           # Utilisation de la meilleure combinaison
           if profile in metrics_results['best_combinations']:
               best_combo = metrics_results['best_combinations'][profile]
               logger.info(f"Utilisation de la meilleure combinaison (profil {profile})")
               
               # Configuration selon les paramètres optimaux
               self.configure_from_optimal_params(best_combo)
           else:
               # Calcul du score composite avec les poids par défaut
               self.calculate_composite_score()
       else:
           # Calcul du score composite avec les poids par défaut
           self.calculate_composite_score()
       
       # Calcul des allocations adaptatives
       self.calculate_adaptive_allocations()
       
       # Optimisation du seuil de rebalancement (optionnel)
       threshold_results = None
       if optimize_threshold:
           logger.info("Exécution de l'optimisation du seuil de rebalancement...")
           threshold_results = self.optimize_rebalance_threshold()
           
           # Mise à jour du seuil de rebalancement
           if 'optimal_threshold' in threshold_results:
               self.backtester.rebalance_threshold = threshold_results['optimal_threshold']
       
       # Exécution du backtest
       results = self.run_backtest()
       
       # Validation out-of-sample (optionnel)
       validation_results = None
       if run_validation:
           logger.info("Exécution de la validation out-of-sample...")
           validation_results = self.run_out_of_sample_validation(profile=profile)
       
       # Tests de robustesse (optionnel)
       robustness_results = None
       if run_robustness:
           logger.info("Exécution des tests de robustesse...")
           robustness_results = self.run_robustness_test(profile=profile)
           
           logger.info("Exécution des tests de stress...")
           stress_results = self.run_stress_test(profile=profile)
           
           # Combinaison des résultats de robustesse
           robustness_results = {
               'cross_validation': robustness_results,
               'stress_test': stress_results
           }
       
       # Affichage du résumé
       self.print_summary()
       
       # Visualisation des résultats
       self.visualize_results()
       
       # Visualisation des résultats d'optimisation (si disponibles)
       if metrics_results is not None:
           self.optimizer.plot_optimization_results(profile)
           self.optimizer.plot_metrics_importance()
       
       return {
           'results': results,
           'metrics_results': metrics_results,
           'threshold_results': threshold_results,
           'validation_results': validation_results,
           'robustness_results': robustness_results
       }
   
   def configure_from_optimal_params(self, optimal_config: Dict) -> None:
       """
       Configure les composants QAAF selon les paramètres optimaux
       
       Args:
           optimal_config: Configuration optimale
       """
       params = optimal_config['params']
       
       # Configuration du calculateur de métriques
       self.metrics_calculator.volatility_window = params['volatility_window']
       self.metrics_calculator.spectral_window = params['spectral_window']
       self.metrics_calculator.min_periods = params['min_periods']
       
       # Configuration de l'allocateur
       self.adaptive_allocator.min_btc_allocation = params['min_btc_allocation']
       self.adaptive_allocator.max_btc_allocation = params['max_btc_allocation']
       self.adaptive_allocator.sensitivity = params['sensitivity']
       self.adaptive_allocator.observation_period = params['observation_period']
       
       # Configuration du backtester
       self.backtester.rebalance_threshold = params['rebalance_threshold']
       
       logger.info("Configuration des composants selon les paramètres optimaux terminée")