Voici les classes attendues et leur ordre dans `qaaf_implementation_1.0.0.py`

Comme ordre idéal, je recommande de suivre cette séquence :
1. Imports et configuration
2. `StaticBenchmarks`
3. `TransactionFeesEvaluator`
4. `DataSource` (classe abstraite)
5. `YFinanceSource`
6. `DataManager`
7. `MarketPhaseAnalyzer` (à ajouter ici)
8. `MetricsCalculator`
9. `AdaptiveAllocator`
10. `QAAFBacktester`
11. `QAAFOptimizer`
12. `OutOfSampleValidator`
13. `RobustnessTester`
14. `QAAFCore`
15. La fonction `run_qaaf` et le bloc `if __name__ == "__main__"`
