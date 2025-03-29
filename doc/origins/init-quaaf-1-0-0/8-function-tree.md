# Arborescence des fonctions de QAAF 1.0.0 (mise à jour)

Voici l'arborescence complète des fonctions pour QAAF 1.0.0, incluant les méthodes nouvellement ajoutées et les modifications :

```
qaaf/
├── metrics/
│   ├── calculator.py
│   │   ├── class MetricsCalculator
│   │   │   ├── __init__(volatility_window, spectral_window, min_periods)
│   │   │   ├── update_parameters(volatility_window, spectral_window, min_periods) [NOUVEAU]
│   │   │   ├── calculate_metrics(data, alpha)
│   │   │   ├── _calculate_volatility_ratio(paxg_btc_returns, btc_returns, paxg_returns)
│   │   │   ├── _calculate_bound_coherence(paxg_btc_data, btc_data, paxg_data)
│   │   │   ├── _calculate_alpha_stability(alpha)
│   │   │   ├── _calculate_spectral_score(paxg_btc_data, btc_data, paxg_data)
│   │   │   ├── _calculate_trend_component(paxg_btc_data, btc_data, paxg_data)
│   │   │   ├── _calculate_oscillation_component(paxg_btc_data, btc_data, paxg_data)
│   │   │   ├── _validate_metrics(metrics)
│   │   │   └── normalize_metrics(metrics)
│   │   │
│   ├── analyzer.py (inchangé)
│   ├── optimizer.py (remplacé par QAAFOptimizer)
│   └── pattern_detector.py (inchangé)
│
├── market/
│   ├── phase_analyzer.py
│   │   ├── class MarketPhaseAnalyzer
│   │   │   ├── __init__(short_window, long_window, volatility_window)
│   │   │   ├── identify_market_phases(btc_data)
│   │   │   └── analyze_metrics_by_phase(metrics, market_phases)
│   │   │
│   └── intensity_detector.py (inchangé)
│
├── allocation/
│   ├── adaptive_allocator.py
│   │   ├── class AdaptiveAllocator
│   │   │   ├── __init__(min_btc_allocation, max_btc_allocation, neutral_allocation, sensitivity)
│   │   │   ├── update_parameters(min_btc_allocation, max_btc_allocation, neutral_allocation, sensitivity, observation_period) [NOUVEAU]
│   │   │   ├── calculate_adaptive_allocation(composite_score, market_phases)
│   │   │   └── detect_intensity_peaks(composite_score, market_phases)
│   │   │
│   └── amplitude_calculator.py (inchangé)
│
├── transaction/
│   ├── fees_evaluator.py
│   │   ├── class TransactionFeesEvaluator
│   │   │   ├── __init__(base_fee_rate, fee_tiers, fixed_fee)
│   │   │   ├── calculate_fee(transaction_amount)
│   │   │   ├── record_transaction(date, amount, action)
│   │   │   ├── get_total_fees()
│   │   │   ├── get_fees_by_period(period)
│   │   │   ├── optimize_rebalance_frequency(portfolio_values, allocation_series, threshold_range)
│   │   │   ├── _calculate_combined_score(portfolio_values, fee_drag)
│   │   │   └── plot_fee_analysis(optimization_results)
│   │   │
│   └── rebalance_optimizer.py (modifié)
│
├── validation/ [NOUVEAU module]
│   ├── out_of_sample_validator.py [NOUVEAU]
│   │   ├── class OutOfSampleValidator
│   │   │   ├── __init__(qaaf_core, data)
│   │   │   ├── split_data(test_ratio, validation_ratio)
│   │   │   ├── _get_common_dates()
│   │   │   ├── run_validation(test_ratio, validation_ratio, profile)
│   │   │   ├── _run_training_phase(profile)
│   │   │   ├── _run_testing_phase(best_params, profile)
│   │   │   ├── print_validation_summary()
│   │   │   └── plot_validation_results()
│   │   │
│   └── robustness_tester.py [NOUVEAU]
│       ├── class RobustnessTester
│       │   ├── __init__(qaaf_core, data)
│       │   ├── run_time_series_cross_validation(n_splits, test_size, gap, profile)
│       │   ├── _get_common_dates()
│       │   ├── _run_training_phase(profile)
│       │   ├── _run_testing_phase(best_params, profile)
│       │   ├── _analyze_cv_results(cv_results)
│       │   ├── _analyze_parameter_stability(cv_results)
│       │   ├── run_stress_test(scenarios, profile)
│       │   ├── _define_scenario_periods()
│       │   ├── print_stress_test_summary()
│       │   └── plot_stress_test_results()
│       │
└── core/
    ├── qaaf_core.py (remanié)
    │   ├── class QAAFCore
    │   │   ├── __init__(initial_capital, trading_costs, start_date, end_date, allocation_min, allocation_max)
    │   │   ├── load_data(start_date, end_date)
    │   │   ├── analyze_market_phases()
    │   │   ├── calculate_metrics()
    │   │   ├── calculate_composite_score(weights)
    │   │   ├── calculate_adaptive_allocations()
    │   │   ├── run_backtest()
    │   │   ├── optimize_rebalance_threshold(thresholds)
    │   │   ├── run_metrics_optimization(profile, max_combinations)
    │   │   ├── run_out_of_sample_validation(test_ratio, profile) [NOUVEAU]
    │   │   ├── run_robustness_test(n_splits, profile) [NOUVEAU]
    │   │   ├── run_stress_test(profile) [NOUVEAU]
    │   │   ├── run_full_analysis(optimize_metrics, optimize_threshold, run_validation, run_robustness, profile)
    │   │   ├── configure_from_optimal_params(optimal_config) [NOUVEAU]
    │   │   ├── print_summary() [AJOUTÉ]
    │   │   ├── visualize_results() [AJOUTÉ]
    │   │   └── save_results(filename) [OPTIONNEL]
    │   │
    ├── visualizer.py (étendu)
    │   │
    └── run_qaaf.py [NOUVEAU]
        ├── run_qaaf(optimize_metrics, optimize_threshold, run_validation, profile, verbose)
        └── if __name__ == "__main__": (block principal)
```

### Modifications principales :

1. **Classes avec nouvelles méthodes d'update**:
   - `MetricsCalculator.update_parameters()`
   - `AdaptiveAllocator.update_parameters()`
   - `QAAFBacktester.update_parameters()`

2. **Nouvelles méthodes dans QAAFCore**:
   - `print_summary()` - Affiche un résumé des résultats d'analyse
   - `visualize_results()` - Visualise les résultats sous forme de graphiques
   - `configure_from_optimal_params()` - Configure les composants selon les paramètres optimaux
   - `run_out_of_sample_validation()` - Exécute une validation out-of-sample
   - `run_robustness_test()` - Exécute un test de robustesse
   - `run_stress_test()` - Exécute un test de stress sur différents scénarios de marché

3. **Nouveaux modules de validation**:
   - `OutOfSampleValidator` - Pour la validation train/test
   - `RobustnessTester` - Pour les tests de robustesse et tests de stress

4. **Nouvelle fonction principale**:
   - `run_qaaf()` - Fonction simplifiée pour exécuter QAAF dans Google Colab

Cette arborescence mise à jour reflète l'ajout des méthodes manquantes `print_summary()` et `visualize_results()` ainsi que l'organisation complète des fonctions dans QAAF 1.0.0.