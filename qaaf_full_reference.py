"""
QAAF (Quantitative Algorithmic Asset Framework) - Version 1.0.0
--------------------------------------------------------------
Retour aux fondamentaux pour la validation mathématique rigoureuse
Utilisation de benchmarks statiques pour la comparaison
Module d'optimisation efficiente par grid search inspiré de l'approche paxg_btc_first_stable

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

QAAF Optimizer - Version 1.0.0
------------------------------
"""
# Installation des dépendances
#pip install -q pandas numpy matplotlib seaborn yfinance tqdm

# Import des packages nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import itertools
import yfinance as yf
from abc import ABC, abstractmethod
from tqdm import tqdm  # Pour les barres de progression

# Configuration du logging et du style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class StaticBenchmarks:
    """
    Classe contenant les benchmarks statiques pour la période 2020-01-01 à 2024-12-31
    avec un investissement initial de $30,000 et des allocations 50/50 pour ALLOY
    """
    
    # Paramètres communs
    INITIAL_CAPITAL = 30000
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    
    # Benchmarks statiques (valeurs précalculées)
    BENCHMARKS = {
        'ALLOY_DCA': {
            'total_investment': 30000.00,
            'final_value': 80212.56,
            'total_return': 167.38,  # en pourcentage
            'max_drawdown': -49.36,  # en pourcentage
            'volatility': 59.46,     # en pourcentage annualisée
            'sharpe_ratio': 2.18
        },
        'ALLOY_BH': {
            'total_investment': 30000.00,
            'final_value': 218704.67,
            'total_return': 629.02,
            'max_drawdown': -68.66,
            'volatility': 40.83,
            'sharpe_ratio': 0.83
        },
        'BTC_DCA': {
            'total_investment': 30000.00,
            'final_value': 119332.61,
            'total_return': 297.78,
            'max_drawdown': -69.75,
            'volatility': 71.32,
            'sharpe_ratio': 2.01
        },
        'BTC_BH': {
            'total_investment': 30000.00,
            'final_value': 386004.03,
            'total_return': 1186.68,
            'max_drawdown': -76.63,
            'volatility': 53.27,
            'sharpe_ratio': 0.90
        },
        'PAXG_DCA': {
            'total_investment': 30000.00,
            'final_value': 41092.51,
            'total_return': 36.98,
            'max_drawdown': -7.58,
            'volatility': 49.43,
            'sharpe_ratio': 2.33
        },
        'PAXG_BH': {
            'total_investment': 30000.00,
            'final_value': 51405.31,
            'total_return': 71.35,
            'max_drawdown': -22.28,
            'volatility': 14.56,
            'sharpe_ratio': 0.45
        }
    }
    
    @classmethod
    def get_benchmark(cls, benchmark_name: str) -> Dict:
        """Récupère les données d'un benchmark spécifique"""
        if benchmark_name not in cls.BENCHMARKS:
            raise ValueError(f"Benchmark {benchmark_name} non trouvé")
        return cls.BENCHMARKS[benchmark_name]
    
    @classmethod
    def get_all_benchmarks(cls) -> Dict:
        """Récupère toutes les données de benchmark"""
        return cls.BENCHMARKS
    
    @classmethod
    def format_as_dataframe(cls) -> pd.DataFrame:
        """Convertit les benchmarks en DataFrame pour affichage"""
        data = []
        for strategy, metrics in cls.BENCHMARKS.items():
            row = {'Strategy': strategy}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.set_index('Strategy')


class TransactionFeesEvaluator:
    """
    Module d'évaluation des frais de transaction pour optimiser la fréquence de rebalancement
    """
    
    def __init__(self, 
                 base_fee_rate: float = 0.001,  # 0.1% par défaut (10 points de base)
                 fee_tiers: Optional[Dict[float, float]] = None,  # Niveaux de frais basés sur le volume
                 fixed_fee: float = 0.0):  # Frais fixe par transaction
        """
        Initialise l'évaluateur de frais
        
        Args:
            base_fee_rate: Taux de base des frais (en pourcentage)
            fee_tiers: Dictionnaire des niveaux de frais basés sur le volume
            fixed_fee: Frais fixe par transaction
        """
        self.base_fee_rate = base_fee_rate
        
        # Niveaux de frais par défaut si non fournis
        self.fee_tiers = fee_tiers or {
            0: base_fee_rate,  # 0.1% pour les transactions standards
            10000: 0.0008,     # 0.08% pour des volumes > $10,000
            50000: 0.0006,     # 0.06% pour des volumes > $50,000
            100000: 0.0004     # 0.04% pour des volumes > $100,000
        }
        
        self.fixed_fee = fixed_fee
        self.transaction_history = []
        
    def calculate_fee(self, transaction_amount: float) -> float:
        """
        Calcule les frais pour une transaction donnée
        
        Args:
            transaction_amount: Montant de la transaction
            
        Returns:
            Frais de transaction
        """
        # Détermination du niveau de frais approprié
        applicable_rate = self.base_fee_rate
        
        # Trouver le taux applicable en fonction du volume
        tier_thresholds = sorted(self.fee_tiers.keys())
        for threshold in reversed(tier_thresholds):
            if transaction_amount >= threshold:
                applicable_rate = self.fee_tiers[threshold]
                break
        
        # Calcul des frais
        percentage_fee = transaction_amount * applicable_rate
        total_fee = percentage_fee + self.fixed_fee
        
        return total_fee
    
    def record_transaction(self, 
                          date: pd.Timestamp, 
                          amount: float, 
                          action: str) -> None:
        """
        Enregistre une transaction dans l'historique
        
        Args:
            date: Date de la transaction
            amount: Montant de la transaction
            action: Type d'action (ex: 'rebalance', 'buy', 'sell')
        """
        fee = self.calculate_fee(amount)
        
        transaction = {
            'date': date,
            'amount': amount,
            'action': action,
            'fee': fee,
            'fee_rate': fee / amount if amount > 0 else 0
        }
        
        self.transaction_history.append(transaction)
    
    def get_total_fees(self) -> float:
        """
        Calcule le total des frais payés
        
        Returns:
            Total des frais
        """
        return sum(t['fee'] for t in self.transaction_history)
    
    def get_fees_by_period(self, period: str = 'M') -> pd.Series:
        """
        Calcule les frais par période
        
        Args:
            period: Période d'agrégation ('D'=jour, 'W'=semaine, 'M'=mois)
            
        Returns:
            Série des frais par période
        """
        if not self.transaction_history:
            return pd.Series()
        
        # Conversion en DataFrame
        df = pd.DataFrame(self.transaction_history)
        
        # Groupement par période
        grouped = df.set_index('date')['fee'].resample(period).sum()
        
        return grouped
    
    def optimize_rebalance_frequency(self, 
                                    portfolio_values: Dict[str, pd.Series], 
                                    allocation_series: Dict[str, pd.Series],
                                    threshold_range: List[float] = [0.01, 0.03, 0.05, 0.1]) -> Dict:
        """
        Optimise la fréquence de rebalancement en simulant différents seuils
        
        Args:
            portfolio_values: Dictionnaire des valeurs de portefeuille par stratégie
            allocation_series: Dictionnaire des séries d'allocation par stratégie
            threshold_range: Liste des seuils de rebalancement à tester
            
        Returns:
            Dictionnaire des résultats d'optimisation
        """
        results = {}
        
        for threshold in threshold_range:
            # Simulation des transactions pour ce seuil
            self.transaction_history = []  # Réinitialisation
            
            for strategy, allocations in allocation_series.items():
                portfolio_value = portfolio_values[strategy]
                # Calcul des changements d'allocation significatifs (> seuil)
                allocation_changes = allocations.diff().abs()
                rebalance_days = allocation_changes[allocation_changes > threshold].index
                
                # Simulation des transactions
                for date in rebalance_days:
                    # Estimation du montant de transaction (portion du portefeuille)
                    if date in portfolio_value.index:
                        transaction_amount = portfolio_value[date] * allocation_changes[date]
                        self.record_transaction(date, transaction_amount, f'rebalance_{strategy}')
            
            # Calcul des métriques
            total_fees = self.get_total_fees()
            transaction_count = len(self.transaction_history)
            
            # Calcul du "fee drag" (impact sur la performance)
            fee_drag = total_fees / portfolio_values[list(portfolio_values.keys())[0]].iloc[-1] * 100
            
            # Score combiné (performance - impact des frais)
            combined_score = self._calculate_combined_score(
                portfolio_values=portfolio_values,
                fee_drag=fee_drag
            )
            
            results[threshold] = {
                'total_fees': total_fees,
                'transaction_count': transaction_count,
                'average_fee': total_fees / transaction_count if transaction_count > 0 else 0,
                'fee_drag': fee_drag,
                'combined_score': combined_score
            }
        
        return results
    
    def _calculate_combined_score(self, portfolio_values: Dict[str, pd.Series], fee_drag: float) -> float:
        """
        Calcule le score combiné: performance - (impact des frais * 2)
        
        Args:
            portfolio_values: Dictionnaire des valeurs de portefeuille par stratégie
            fee_drag: Impact des frais sur la performance (en %)
            
        Returns:
            Score combiné
        """
        # Extraction de la performance du premier portefeuille
        first_portfolio = list(portfolio_values.keys())[0]
        performance = ((portfolio_values[first_portfolio].iloc[-1] / portfolio_values[first_portfolio].iloc[0]) - 1) * 100
        
        # Score combiné avec double pénalité pour les frais
        return performance - (fee_drag * 2)
    
    def plot_fee_analysis(self, optimization_results: Dict) -> None:
        """
        Visualise l'analyse des frais
        
        Args:
            optimization_results: Résultats de l'optimisation
        """
        thresholds = list(optimization_results.keys())
        total_fees = [results['total_fees'] for threshold, results in optimization_results.items()]
        transaction_counts = [results['transaction_count'] for threshold, results in optimization_results.items()]
        fee_drag = [results['fee_drag'] for threshold, results in optimization_results.items()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique 1: Frais totaux et nombre de transactions
        color1 = 'blue'
        ax1.bar(range(len(thresholds)), total_fees, alpha=0.7, color=color1, label='Frais Totaux')
        ax1.set_xticks(range(len(thresholds)))
        ax1.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax1.set_xlabel('Seuil de Rebalancement')
        ax1.set_ylabel('Frais Totaux ($)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Nombre de transactions sur l'axe secondaire
        ax1b = ax1.twinx()
        color2 = 'red'
        ax1b.plot(range(len(thresholds)), transaction_counts, 'o-', color=color2, label='Nombre de Transactions')
        ax1b.set_ylabel('Nombre de Transactions', color=color2)
        ax1b.tick_params(axis='y', labelcolor=color2)
        
        # Ajout de légendes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Graphique 2: Impact sur la performance (fee drag)
        ax2.bar(range(len(thresholds)), fee_drag, alpha=0.7, color='green')
        ax2.set_xticks(range(len(thresholds)))
        ax2.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax2.set_xlabel('Seuil de Rebalancement')
        ax2.set_ylabel('Drag de Performance (%)')
        
        plt.tight_layout()
        plt.show()

class DataSource(ABC):
    """Classe abstraite pour les sources de données"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère les données pour un symbole"""
        pass
        
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Valide les données récupérées"""
        pass

class YFinanceSource(DataSource):
    """Implémentation de DataSource pour Yahoo Finance"""
    
    def __init__(self):
        self.required_columns = {'open', 'high', 'low', 'close', 'volume'}
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère les données de Yahoo Finance"""
        try:
            logger.info(f"Chargement données {symbol} de {start_date} à {end_date}")
            
            # Téléchargement
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError(f"Aucune donnée pour {symbol}")
                
            # Standardisation
            df = self._standardize_data(df)
            
            # Validation
            if not self.validate_data(df):
                raise ValueError("Validation des données échouée")
                
            logger.info(f"✓ Données {symbol} chargées: {len(df)} points")
            return df
            
        except Exception as e:
            logger.error(f"Erreur chargement {symbol}: {str(e)}")
            raise
            
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise le format des données"""
        # Gestion du MultiIndex de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = pd.DataFrame({
                'open': df['Open'].iloc[:, 0],
                'high': df['High'].iloc[:, 0],
                'low': df['Low'].iloc[:, 0],
                'close': df['Close'].iloc[:, 0],
                'volume': df['Volume'].iloc[:, 0]
            })
        
        # Normalisation des noms de colonnes
        df.columns = df.columns.str.lower()
        
        return df
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Valide les données"""
        if df.empty:
            logger.error("DataFrame vide")
            return False
            
        # Vérification des colonnes
        missing = self.required_columns - set(df.columns)
        if missing:
            logger.error(f"Colonnes manquantes: {missing}")
            return False
            
        # Vérification des valeurs nulles
        if df.isnull().any().any():
            logger.warning("Valeurs manquantes détectées")
            
        return True
    
class DataManager:
    """Gestionnaire principal des données QAAF"""
    
    def __init__(self, data_source: Optional[DataSource] = None):
        self.data_source = data_source or YFinanceSource()
        self.data_cache = {}
        
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère et prépare les données pour QAAF"""
        try:
            # Vérification du cache
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Récupération des données
            df = self.data_source.fetch_data(symbol, start_date, end_date)
            
            # Mise en cache
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur récupération données: {str(e)}")
            raise
    
    def prepare_qaaf_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prépare toutes les données nécessaires pour l'analyse QAAF:
        - BTC-USD
        - PAXG-USD
        - PAXG/BTC (calculé)
        """
        # Récupération des données
        btc_data = self.get_data('BTC-USD', start_date, end_date)
        paxg_data = self.get_data('PAXG-USD', start_date, end_date)
        
        # Vérification de l'alignement des indices
        btc_data = btc_data.reindex(paxg_data.index, method='ffill')
        paxg_data = paxg_data.reindex(btc_data.index, method='ffill')
        
        # Calcul du ratio PAXG/BTC
        paxg_btc_ratio = paxg_data['close'] / btc_data['close']
        
        # Création d'un DataFrame complet pour PAXG/BTC
        paxg_btc_data = pd.DataFrame({
            'open': paxg_data['open'] / btc_data['open'],
            'high': paxg_data['high'] / btc_data['high'],
            'low': paxg_data['low'] / btc_data['low'],
            'close': paxg_btc_ratio,
            'volume': paxg_data['volume']  # Approximation
        })
        
        return {
            'BTC': btc_data,
            'PAXG': paxg_data,
            'PAXG/BTC': paxg_btc_data
        }

class MarketPhaseAnalyzer:
    """
    Analyseur des phases de marché pour QAAF
    """
    
    def __init__(self, 
                short_window: int = 20, 
                long_window: int = 50,
                volatility_window: int = 30):
        """
        Initialise l'analyseur des phases de marché
        
        Args:
            short_window: Fenêtre courte pour les moyennes mobiles
            long_window: Fenêtre longue pour les moyennes mobiles
            volatility_window: Fenêtre pour le calcul de volatilité
        """
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_window = volatility_window
    
    def identify_market_phases(self, btc_data: pd.DataFrame) -> pd.Series:
        """
        Identifie les phases de marché (haussier, baissier, consolidation)
        
        Args:
            btc_data: DataFrame des données BTC
            
        Returns:
            Série des phases de marché
        """
        # Extraction des prix de clôture
        close = btc_data['close']
        
        # Calcul des moyennes mobiles
        ma_short = close.rolling(window=self.short_window).mean()
        ma_long = close.rolling(window=self.long_window).mean()
        
        # Calcul du momentum (rendement sur la fenêtre courte)
        momentum = close.pct_change(periods=self.short_window)
        
        # Calcul de la volatilité
        volatility = close.pct_change().rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Initialisation des phases
        phases = pd.Series('consolidation', index=close.index)
        
        # Identification des phases haussières
        bullish_condition = ((ma_short > ma_long) & (momentum > 0.1)) | (momentum > 0.2)
        phases[bullish_condition] = 'bullish'
        
        # Identification des phases baissières
        bearish_condition = ((ma_short < ma_long) & (momentum < -0.1)) | (momentum < -0.2)
        phases[bearish_condition] = 'bearish'
        
        # Identification des phases de forte volatilité
        high_volatility = volatility > volatility.rolling(window=100).mean() * 1.5
        
        # Création d'une série combinée (phase_volatilité)
        combined_phases = phases.copy()
        for date in high_volatility.index:
            if high_volatility.loc[date]:
                combined_phases.loc[date] = f"{phases.loc[date]}_high_vol"
            else:
                combined_phases.loc[date] = f"{phases.loc[date]}_low_vol"
        
        return combined_phases
    
    def analyze_metrics_by_phase(self, 
                              metrics: Dict[str, pd.Series], 
                              market_phases: pd.Series) -> Dict:
        """
        Analyse les métriques par phase de marché
        
        Args:
            metrics: Dictionnaire des métriques
            market_phases: Série des phases de marché
            
        Returns:
            Dictionnaire d'analyse par phase
        """
        unique_phases = market_phases.unique()
        phase_analysis = {}
        
        for phase in unique_phases:
            phase_mask = market_phases == phase
            phase_data = {}
            
            for metric_name, metric_series in metrics.items():
                phase_values = metric_series[phase_mask]
                
                if len(phase_values) > 0:
                    phase_data[metric_name] = {
                        'mean': phase_values.mean(),
                        'std': phase_values.std(),
                        'min': phase_values.min(),
                        'max': phase_values.max(),
                        'median': phase_values.median()
                    }
            
            phase_analysis[phase] = phase_data
        
        return phase_analysis
    
class MetricsCalculator:
    """
    Calculateur des métriques QAAF
    
    Cette classe implémente les quatre métriques primaires du framework QAAF:
    1. Ratio de Volatilité
    2. Cohérence des Bornes
    3. Stabilité d'Alpha
    4. Score Spectral
    """
    
    def __init__(self, 
                volatility_window: int = 30, 
                spectral_window: int = 60,
                min_periods: int = 20):
        """
        Initialise le calculateur de métriques
        
        Args:
            volatility_window: Fenêtre pour le calcul de la volatilité
            spectral_window: Fenêtre pour le calcul des composantes spectrales
            min_periods: Nombre minimum de périodes pour les calculs
        """
        self.volatility_window = volatility_window
        self.spectral_window = spectral_window
        self.min_periods = min_periods
    
    # Modifications dans MetricsCalculator
    def update_parameters(self, 
                        volatility_window: Optional[int] = None,
                        spectral_window: Optional[int] = None,
                        min_periods: Optional[int] = None):
        """
        Met à jour les paramètres du calculateur
        
        Args:
            volatility_window: Fenêtre pour le calcul de la volatilité
            spectral_window: Fenêtre pour le calcul des composantes spectrales
            min_periods: Nombre minimum de périodes pour les calculs
        """
        if volatility_window is not None:
            self.volatility_window = volatility_window
        
        if spectral_window is not None:
            self.spectral_window = spectral_window
        
        if min_periods is not None:
            self.min_periods = min_periods
        
        logger.info(f"Paramètres du calculateur mis à jour: volatility_window={self.volatility_window}, "
                f"spectral_window={self.spectral_window}, min_periods={self.min_periods}")
    
    def calculate_metrics(self, data: Dict[str, pd.DataFrame], 
                         alpha: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Calcule toutes les métriques primaires QAAF
        
        Args:
            data: Dictionnaire contenant les DataFrames ('BTC', 'PAXG', 'PAXG/BTC')
            alpha: Série des allocations (optionnel)
            
        Returns:
            Dictionnaire avec les séries temporelles des métriques
        """
        # Extraction des données
        btc_data = data['BTC']
        paxg_data = data['PAXG']
        paxg_btc_data = data['PAXG/BTC']
        
        # Calcul des rendements
        btc_returns = btc_data['close'].pct_change().dropna()
        paxg_returns = paxg_data['close'].pct_change().dropna()
        paxg_btc_returns = paxg_btc_data['close'].pct_change().dropna()
        
        # Alignement des indices
        common_index = btc_returns.index.intersection(paxg_returns.index).intersection(paxg_btc_returns.index)
        btc_returns = btc_returns.loc[common_index]
        paxg_returns = paxg_returns.loc[common_index]
        paxg_btc_returns = paxg_btc_returns.loc[common_index]
        
        # 1. Ratio de Volatilité
        vol_ratio = self._calculate_volatility_ratio(paxg_btc_returns, btc_returns, paxg_returns)
        
        # 2. Cohérence des Bornes
        bound_coherence = self._calculate_bound_coherence(paxg_btc_data, btc_data, paxg_data)
        
        # 3. Stabilité d'Alpha (si alpha est fourni)
        if alpha is not None:
            alpha_stability = self._calculate_alpha_stability(alpha)
        else:
            # Créer un alpha statique de 0.5 pour démonstration
            alpha_static = pd.Series(0.5, index=common_index)
            alpha_stability = self._calculate_alpha_stability(alpha_static)
        
        # 4. Score Spectral
        spectral_score = self._calculate_spectral_score(paxg_btc_data, btc_data, paxg_data)
        
        # Résultats
        metrics = {
            'vol_ratio': vol_ratio,
            'bound_coherence': bound_coherence,
            'alpha_stability': alpha_stability,
            'spectral_score': spectral_score
        }
        
        # Validation des résultats (avec log de problèmes potentiels)
        self._validate_metrics(metrics)
        
        return metrics
    
    def _calculate_volatility_ratio(self, 
                                  paxg_btc_returns: pd.Series, 
                                  btc_returns: pd.Series, 
                                  paxg_returns: pd.Series) -> pd.Series:
        """
        Calcule le ratio de volatilité σ_C(t) / max(σ_A(t), σ_B(t))
        
        Une valeur proche de 0 est meilleure (volatilité faible par rapport aux actifs)
        Une valeur > 1 indique une volatilité plus élevée que les actifs sous-jacents
        """
        # Calcul des volatilités mobiles
        vol_paxg_btc = paxg_btc_returns.rolling(window=self.volatility_window, 
                                               min_periods=self.min_periods).std() * np.sqrt(252)
        vol_btc = btc_returns.rolling(window=self.volatility_window, 
                                     min_periods=self.min_periods).std() * np.sqrt(252)
        vol_paxg = paxg_returns.rolling(window=self.volatility_window, 
                                       min_periods=self.min_periods).std() * np.sqrt(252)
        
        # Calcul du maximum des volatilités sous-jacentes
        max_vol = pd.concat([vol_btc, vol_paxg], axis=1).max(axis=1)
        
        # Calcul du ratio (avec gestion des divisions par zéro)
        ratio = vol_paxg_btc / max_vol.replace(0, np.nan)
        
        # Limiter les valeurs pour éviter les extrêmes
        return ratio.clip(0.1, 10).fillna(1.0)
    
    def _calculate_bound_coherence(self, 
                                 paxg_btc_data: pd.DataFrame, 
                                 btc_data: pd.DataFrame, 
                                 paxg_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la cohérence des bornes P(min(A,B) ≤ C ≤ max(A,B))
        
        Une valeur proche de 1 est meilleure (prix entre les bornes naturelles)
        Une valeur proche de 0 indique un prix en dehors des bornes
        """
        # Extraction des séries de prix
        paxg_btc_prices = paxg_btc_data['close']
        btc_prices = btc_data['close']
        paxg_prices = paxg_data['close']
        
        # Normalisation pour comparaison (base 100)
        common_index = paxg_btc_prices.index.intersection(btc_prices.index).intersection(paxg_prices.index)
        start_date = common_index[0]
        
        norm_paxg_btc = paxg_btc_prices.loc[common_index] / paxg_btc_prices.loc[start_date] * 100
        norm_btc = btc_prices.loc[common_index] / btc_prices.loc[start_date] * 100
        norm_paxg = paxg_prices.loc[common_index] / paxg_prices.loc[start_date] * 100
        
        # Calcul des bornes
        min_bound = pd.concat([norm_btc, norm_paxg], axis=1).min(axis=1)
        max_bound = pd.concat([norm_btc, norm_paxg], axis=1).max(axis=1)
        
        # Vérification si le prix est dans les bornes
        in_bounds = (norm_paxg_btc >= min_bound) & (norm_paxg_btc <= max_bound)
        
        # Calcul de la cohérence sur une fenêtre mobile
        coherence = in_bounds.rolling(window=self.volatility_window, 
                                    min_periods=self.min_periods).mean()
        
        return coherence.fillna(0.5)
    
    def _calculate_alpha_stability(self, alpha: pd.Series) -> pd.Series:
        """
        Calcule la stabilité d'alpha -σ(α(t))
        
        Une valeur proche de 0 est meilleure (allocations stables)
        Une valeur très négative indique des changements fréquents d'allocation
        """
        # Calcul de la volatilité des allocations
        alpha_volatility = alpha.rolling(window=self.volatility_window, 
                                       min_periods=self.min_periods).std()
        
        # Inversion du signe (-σ) pour que les valeurs proches de 0 soient meilleures
        stability = -alpha_volatility
        
        # Normalisation entre 0 et 1
        normalized_stability = (stability - stability.min()) / (stability.max() - stability.min() + 1e-10)
        
        return normalized_stability.fillna(1.0)
    
    def _calculate_spectral_score(self, 
                               paxg_btc_data: pd.DataFrame, 
                               btc_data: pd.DataFrame, 
                               paxg_data: pd.DataFrame) -> pd.Series:
        """
        Calcule le score spectral (combinaison de tendance et d'oscillation)
        
        Une valeur proche de 1 indique un bon équilibre tendance/oscillation
        """
        # 1. Composante tendancielle (70%)
        trend_score = self._calculate_trend_component(paxg_btc_data, btc_data, paxg_data)
        
        # 2. Composante oscillatoire (30%)
        oscillation_score = self._calculate_oscillation_component(paxg_btc_data, btc_data, paxg_data)
        
        # Score combiné
        spectral_score = 0.7 * trend_score + 0.3 * oscillation_score
        
        return spectral_score.fillna(0.5)
    
    def _calculate_trend_component(self, 
                                paxg_btc_data: pd.DataFrame, 
                                btc_data: pd.DataFrame, 
                                paxg_data: pd.DataFrame) -> pd.Series:
        """Calcule la composante tendancielle du score spectral"""
        # Moyennes mobiles
        ma_paxg_btc = paxg_btc_data['close'].rolling(window=self.spectral_window, 
                                                   min_periods=self.min_periods).mean()
        ma_btc = btc_data['close'].rolling(window=self.spectral_window, 
                                         min_periods=self.min_periods).mean()
        ma_paxg = paxg_data['close'].rolling(window=self.spectral_window, 
                                           min_periods=self.min_periods).mean()
        
        # Synthèse des actifs sous-jacents
        ma_combined = (ma_btc + ma_paxg) / 2
        
        # Pour éviter des complications, nous retournons une mesure simplifiée
        # de la différence entre le ratio et la moyenne des actifs
        trend_diff = (ma_paxg_btc - ma_combined).abs()
        max_diff = ma_combined.max() - ma_combined.min()
        
        # Normalisation entre 0 et 1 (1 = bonne tendance, 0 = mauvaise)
        trend_score = 1 - (trend_diff / (max_diff + 1e-10)).clip(0, 1)
        
        return trend_score
    
    def _calculate_oscillation_component(self, 
                                      paxg_btc_data: pd.DataFrame, 
                                      btc_data: pd.DataFrame, 
                                      paxg_data: pd.DataFrame) -> pd.Series:
        """Calcule la composante oscillatoire du score spectral"""
        # Calcul des rendements
        returns_paxg_btc = paxg_btc_data['close'].pct_change()
        returns_btc = btc_data['close'].pct_change()
        returns_paxg = paxg_data['close'].pct_change()
        
        # Calcul des volatilités
        vol_paxg_btc = returns_paxg_btc.rolling(window=self.volatility_window, 
                                              min_periods=self.min_periods).std()
        vol_btc = returns_btc.rolling(window=self.volatility_window, 
                                    min_periods=self.min_periods).std()
        vol_paxg = returns_paxg.rolling(window=self.volatility_window, 
                                      min_periods=self.min_periods).std()
        
        # Score oscillatoire basé sur le rapport de volatilité
        max_vol = pd.concat([vol_btc, vol_paxg], axis=1).max(axis=1)
        vol_ratio = vol_paxg_btc / max_vol.replace(0, np.nan)
        
        # Normalisation (1 = bonne oscillation, 0 = mauvaise)
        # Une bonne oscillation a un ratio de volatilité optimal (ni trop élevé, ni trop faible)
        osc_score = 1 - (vol_ratio - 0.5).abs().clip(0, 0.5) * 2
        
        return osc_score.fillna(0.5)
    
    def _validate_metrics(self, metrics: Dict[str, pd.Series]) -> None:
        """Valide les métriques calculées et log les anomalies"""
        for name, series in metrics.items():
            # Vérification des NaN
            nan_count = series.isna().sum()
            if nan_count > 0:
                logger.warning(f"La métrique {name} contient {nan_count} valeurs NaN")
            
            # Vérification des valeurs extrêmes
            if name == 'vol_ratio':
                if (series > 5).any():
                    logger.warning(f"Valeurs de vol_ratio > 5 détectées")
            elif name == 'bound_coherence':
                if (series < 0.2).any():
                    logger.warning(f"Faible cohérence des bornes détectée (< 0.2)")
            
            # Vérification des sauts brusques
            diff = series.diff().abs()
            mean_diff = diff.mean()
            max_diff = diff.max()
            if max_diff > 5 * mean_diff:
                logger.warning(f"Sauts importants détectés dans {name}: max={max_diff}, mean={mean_diff}")
        
    def normalize_metrics(self, metrics: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Normalise les métriques entre 0 et 1 pour une comparaison équitable
        
        Args:
            metrics: Dictionnaire des métriques calculées
            
        Returns:
            Dictionnaire des métriques normalisées
        """
        normalized_metrics = {}
        
        for name, series in metrics.items():
            min_val = series.min()
            max_val = series.max()
            
            if max_val > min_val:
                normalized_metrics[name] = (series - min_val) / (max_val - min_val)
            else:
                normalized_metrics[name] = series - min_val
        
        return normalized_metrics


class AdaptiveAllocator:
    """
    Allocateur adaptatif pour QAAF
    
    Cette classe implémente l'allocation adaptative avec amplitude variable
    basée sur l'intensité des signaux du score composite.
    """
    
    def __init__(self, 
                min_btc_allocation: float = 0.3,
                max_btc_allocation: float = 0.7,
                neutral_allocation: float = 0.5,
                sensitivity: float = 1.0):
        """
        Initialise l'allocateur adaptatif
        
        Args:
            min_btc_allocation: Allocation minimale en BTC
            max_btc_allocation: Allocation maximale en BTC
            neutral_allocation: Allocation neutre (point d'équilibre)
            sensitivity: Sensibilité aux signaux (1.0 = standard)
        """
        self.min_btc_allocation = min_btc_allocation
        self.max_btc_allocation = max_btc_allocation
        self.neutral_allocation = neutral_allocation
        self.sensitivity = sensitivity
        
        # Variables de suivi
        self.last_signal_date = None
        self.observation_period = 10  # jours d'observation après un signal fort
        self.last_allocation = neutral_allocation
    
    # Modifications dans AdaptiveAllocator
    def update_parameters(self, 
                        min_btc_allocation: Optional[float] = None,
                        max_btc_allocation: Optional[float] = None,
                        neutral_allocation: Optional[float] = None,
                        sensitivity: Optional[float] = None,
                        observation_period: Optional[int] = None):
        """
        Met à jour les paramètres de l'allocateur
        
        Args:
            min_btc_allocation: Allocation minimale en BTC
            max_btc_allocation: Allocation maximale en BTC
            neutral_allocation: Allocation neutre (point d'équilibre)
            sensitivity: Sensibilité aux signaux
            observation_period: Période d'observation après un signal
        """
        if min_btc_allocation is not None:
            self.min_btc_allocation = min_btc_allocation
        
        if max_btc_allocation is not None:
            self.max_btc_allocation = max_btc_allocation
        
        if neutral_allocation is not None:
            self.neutral_allocation = neutral_allocation
        
        if sensitivity is not None:
            self.sensitivity = sensitivity
        
        if observation_period is not None:
            self.observation_period = observation_period
        
        logger.info(f"Paramètres de l'allocateur mis à jour: min={self.min_btc_allocation}, "
                f"max={self.max_btc_allocation}, sensitivity={self.sensitivity}")
    
    def calculate_adaptive_allocation(self, 
                                    composite_score: pd.Series,
                                    market_phases: pd.Series) -> pd.Series:
        """
        Calcule l'allocation adaptative avec amplitude variable
        
        Args:
            composite_score: Score composite calculé
            market_phases: Phases de marché identifiées
            
        Returns:
            Série temporelle des allocations BTC
        """
        # Normalisation du score composite
        normalized_score = (composite_score - composite_score.mean()) / composite_score.std()
        
        # Allocation par défaut (neutre)
        allocations = pd.Series(self.neutral_allocation, index=composite_score.index)
        
        # Paramètres d'amplitude par phase de marché
        amplitude_by_phase = {
            'bullish': 1.0,  # Amplitude normale en phase haussière
            'bearish': 1.5,  # Amplitude plus grande en phase baissière (réaction plus forte)
            'consolidation': 0.7,  # Amplitude réduite en consolidation
            'bullish_high_vol': 1.2,  # Amplitude ajustée pour volatilité élevée
            'bearish_high_vol': 1.8,  # Réaction très forte en baisse volatile
            'consolidation_high_vol': 0.9  # Consolidation volatile
        }
        
        # Paramètres par défaut si la phase n'est pas reconnue
        default_amplitude = 1.0
        
        # Initialisation des métriques de signal
        signal_points = []
        signal_strengths = []
        
        # Calcul de l'allocation pour chaque date
        for date in composite_score.index:
            # Récupération de la phase et du score normalisé
            if date in market_phases.index:
                phase = market_phases.loc[date]
                amplitude = amplitude_by_phase.get(phase, default_amplitude)
            else:
                amplitude = default_amplitude
            
            score = normalized_score.loc[date]
            
            # Détection des signaux forts (dépassement de seuil)
            signal_threshold = 1.5 * self.sensitivity
            
            if abs(score) > signal_threshold:
                # Signal fort détecté
                signal_points.append(date)
                signal_strengths.append(abs(score))
                
                # Détermination de l'amplitude adaptative
                # Plus le signal est fort, plus l'amplitude est grande
                signal_strength_factor = min(2.0, abs(score) / signal_threshold)
                adjusted_amplitude = amplitude * signal_strength_factor
                
                # Direction de l'allocation
                if score > 0:
                    # Signal positif = augmentation de l'allocation BTC
                    target_allocation = self.neutral_allocation + (self.max_btc_allocation - self.neutral_allocation) * adjusted_amplitude
                else:
                    # Signal négatif = diminution de l'allocation BTC
                    target_allocation = self.neutral_allocation - (self.neutral_allocation - self.min_btc_allocation) * adjusted_amplitude
                
                # Application de l'allocation avec contraintes
                allocations.loc[date] = max(self.min_btc_allocation, min(self.max_btc_allocation, target_allocation))
                
                # Mise à jour de l'état
                self.last_signal_date = date
                self.last_allocation = allocations.loc[date]
            else:
                # Pas de signal fort
                
                # Vérification de la période d'observation
                if self.last_signal_date is not None:
                    days_since_signal = (date - self.last_signal_date).days
                    
                    if days_since_signal < self.observation_period:
                        # Pendant la période d'observation, maintien de l'allocation
                        allocations.loc[date] = self.last_allocation
                    else:
                        # Retour progressif vers l'allocation neutre
                        # Plus on s'éloigne du signal, plus on se rapproche du neutre
                        recovery_factor = min(1.0, (days_since_signal - self.observation_period) / 10)
                        allocations.loc[date] = self.last_allocation + (self.neutral_allocation - self.last_allocation) * recovery_factor
                else:
                    # Pas de signal récent, allocation basée sur le score actuel avec amplitude réduite
                    reduced_amplitude = amplitude * 0.5  # Amplitude réduite hors des signaux forts
                    allocation_change = (self.max_btc_allocation - self.min_btc_allocation) * 0.5 * score * reduced_amplitude
                    allocations.loc[date] = self.neutral_allocation + allocation_change
                    allocations.loc[date] = max(self.min_btc_allocation, min(self.max_btc_allocation, allocations.loc[date]))
        
                    # Stats des signaux
                    if signal_points:
                        logger.info(f"Signaux forts détectés: {len(signal_points)}")
                        logger.info(f"Force moyenne des signaux: {np.mean(signal_strengths):.2f}")
                    else:
                        logger.info("Aucun signal fort détecté")
        
        return allocations
    
    def detect_intensity_peaks(self, 
                             composite_score: pd.Series, 
                             market_phases: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Détecte les pics d'intensité dans le score composite
        
        Args:
            composite_score: Score composite calculé
            market_phases: Phases de marché identifiées
            
        Returns:
            Tuple contenant (peaks, troughs, deviation)
        """
        # Calcul de l'écart-type mobile du score
        rolling_std = composite_score.rolling(window=30).std()
        
        # Détection des pics (écarts par rapport à la moyenne > X écarts-types)
        rolling_mean = composite_score.rolling(window=30).mean()
        deviation = (composite_score - rolling_mean) / rolling_std
        
        # Seuils d'intensité adaptatifs par phase
        thresholds = {
            'bullish': 1.8,
            'bearish': 1.5,  # Plus sensible
            'consolidation': 2.2,  # Moins sensible
            'bullish_high_vol': 1.6,
            'bearish_high_vol': 1.3,  # Encore plus sensible
            'consolidation_high_vol': 2.0
        }
        
        # Initialisation des séries de pics
        peaks = pd.Series(False, index=composite_score.index)
        troughs = pd.Series(False, index=composite_score.index)
        
        # Détection adaptative par phase
        for date in composite_score.index:
            if date in market_phases.index:
                phase = market_phases.loc[date]
                base_phase = phase.split('_')[0] if '_' in phase else phase
                threshold = thresholds.get(base_phase, 1.8) * self.sensitivity
                
                if deviation.loc[date] > threshold:
                    peaks.loc[date] = True
                elif deviation.loc[date] < -threshold:
                    troughs.loc[date] = True
        
        return peaks, troughs, deviation


class QAAFBacktester:
    """
    Backtest des stratégies QAAF avec prise en compte des frais de transaction
    
    Cette classe permet de tester la performance des métriques QAAF
    sur des données historiques et de comparer avec les benchmarks.
    """
    
    def __init__(self, 
                initial_capital: float = 30000.0,
                fees_evaluator: Optional[TransactionFeesEvaluator] = None,
                rebalance_threshold: float = 0.05):
        """
        Initialise le backtester
        
        Args:
            initial_capital: Capital initial
            fees_evaluator: Évaluateur des frais de transaction
            rebalance_threshold: Seuil de rééquilibrage
        """
        self.initial_capital = initial_capital
        self.fees_evaluator = fees_evaluator or TransactionFeesEvaluator()
        self.rebalance_threshold = rebalance_threshold
        
        # Performance et allocation tracking
        self.performance_history = None
        self.allocation_history = None
        self.transaction_history = []
    
    # Modifications dans QAAFBacktester
    def update_parameters(self, 
                        rebalance_threshold: Optional[float] = None):
        """
        Met à jour les paramètres du backtester
        
        Args:
            rebalance_threshold: Seuil de rééquilibrage
        """
        if rebalance_threshold is not None:
            self.rebalance_threshold = rebalance_threshold
        
        logger.info(f"Paramètres du backtester mis à jour: rebalance_threshold={self.rebalance_threshold}")
    
    def run_backtest(self, 
                   btc_data: pd.DataFrame, 
                   paxg_data: pd.DataFrame, 
                   allocations: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Exécute le backtest avec les allocations données
        
        Args:
            btc_data: DataFrame des données BTC
            paxg_data: DataFrame des données PAXG
            allocations: Série des allocations BTC
            
        Returns:
            Tuple contenant la performance du portefeuille et les métriques
        """
        # Alignement des données sur les allocations
        common_index = btc_data.index.intersection(paxg_data.index).intersection(allocations.index)
        btc_close = btc_data.loc[common_index, 'close']
        paxg_close = paxg_data.loc[common_index, 'close']
        alloc_btc = allocations.loc[common_index]
        
        # Initialisation du portefeuille
        portfolio_value = pd.Series(self.initial_capital, index=common_index)
        btc_units = (self.initial_capital * alloc_btc.iloc[0]) / btc_close.iloc[0]
        paxg_units = (self.initial_capital * (1 - alloc_btc.iloc[0])) / paxg_close.iloc[0]
        
        # Date du dernier rééquilibrage
        last_rebalance_date = common_index[0]
        
        # Réinitialisation de l'historique des transactions
        self.transaction_history = []
        self.fees_evaluator.transaction_history = []
        
        # Enregistrement de la transaction initiale
        initial_btc_amount = self.initial_capital * alloc_btc.iloc[0]
        initial_paxg_amount = self.initial_capital * (1 - alloc_btc.iloc[0])
        self.fees_evaluator.record_transaction(common_index[0], initial_btc_amount, 'initial_buy_btc')
        self.fees_evaluator.record_transaction(common_index[0], initial_paxg_amount, 'initial_buy_paxg')
        
        # Déduction des frais initiaux
        initial_fees = self.fees_evaluator.calculate_fee(initial_btc_amount) + self.fees_evaluator.calculate_fee(initial_paxg_amount)
        portfolio_value.iloc[0] -= initial_fees
        
        # Tracking des allocations réelles
        realized_allocations = pd.Series(alloc_btc.iloc[0], index=common_index)
        
        # Trading logic
        for i, date in enumerate(common_index[1:], 1):
            prev_date = common_index[i-1]
            
            # Mise à jour de la valeur des actifs
            btc_value = btc_units * btc_close.loc[date]
            paxg_value = paxg_units * paxg_close.loc[date]
            
            # Valeur totale avant rebalancement
            current_value = btc_value + paxg_value
            
            # Allocation actuelle
            current_btc_allocation = btc_value / current_value if current_value > 0 else 0.5
            realized_allocations.loc[date] = current_btc_allocation
            
            # Vérification du besoin de rééquilibrage
            allocation_diff = abs(current_btc_allocation - alloc_btc.loc[date])
            days_since_rebalance = (date - last_rebalance_date).days
            
            # Rééquilibrage si nécessaire (seuil ou périodique)
            if allocation_diff > self.rebalance_threshold or days_since_rebalance > 30:
                # Nouveaux montants cibles
                target_btc_allocation = alloc_btc.loc[date]
                target_btc_value = current_value * target_btc_allocation
                target_paxg_value = current_value * (1 - target_btc_allocation)
                
                # Montants à ajuster
                btc_adjustment = target_btc_value - btc_value
                
                # Simulation des frais de transaction
                transaction_amount = abs(btc_adjustment)
                transaction_fee = self.fees_evaluator.calculate_fee(transaction_amount)
                
                # Enregistrement de la transaction
                self.fees_evaluator.record_transaction(date, transaction_amount, 'rebalance')
                
                # Mise à jour après frais
                current_value -= transaction_fee
                
                # Recalcul des montants avec les frais déduits
                target_btc_value = current_value * target_btc_allocation
                target_paxg_value = current_value * (1 - target_btc_allocation)
                
                # Mise à jour des unités
                btc_units = target_btc_value / btc_close.loc[date]
                paxg_units = target_paxg_value / paxg_close.loc[date]
                
                # Mise à jour de la date du dernier rééquilibrage
                last_rebalance_date = date
                
                # Conservation de l'historique pour analyse
                self.transaction_history.append({
                    'date': date,
                    'current_allocation': current_btc_allocation,
                    'target_allocation': target_btc_allocation,
                    'portfolio_value': current_value,
                    'transaction_amount': transaction_amount,
                    'fee': transaction_fee
                })
            
            # Mise à jour de la valeur du portefeuille
            portfolio_value.loc[date] = btc_units * btc_close.loc[date] + paxg_units * paxg_close.loc[date]
        
        # Calcul des métriques
        metrics = self.calculate_metrics(portfolio_value, realized_allocations)
        
        # Sauvegarde des historiques
        self.performance_history = portfolio_value
        self.allocation_history = realized_allocations
        
        return portfolio_value, metrics
    
    def run_multi_threshold_test(self, 
                               btc_data: pd.DataFrame, 
                               paxg_data: pd.DataFrame, 
                               allocations: pd.Series,
                               thresholds: List[float] = [0.01, 0.03, 0.05, 0.1]) -> Dict:
        """
        Teste différents seuils de rebalancement pour évaluer l'impact des frais
        
        Args:
            btc_data: DataFrame des données BTC
            paxg_data: DataFrame des données PAXG
            allocations: Série des allocations BTC
            thresholds: Liste des seuils à tester
            
        Returns:
            Dictionnaire des résultats par seuil
        """
        results = {}
        
        for threshold in thresholds:
            logger.info(f"Test avec seuil de rebalancement: {threshold:.1%}")
            
            # Mise à jour du seuil
            self.rebalance_threshold = threshold
            
            # Exécution du backtest
            portfolio_value, metrics = self.run_backtest(btc_data, paxg_data, allocations)
            
            # Calcul des frais totaux
            total_fees = self.fees_evaluator.get_total_fees()
            transaction_count = len(self.fees_evaluator.transaction_history)
            
            # Calcul du drag de performance
            fee_drag = total_fees / portfolio_value.iloc[-1] * 100  # en pourcentage
            
            # Calcul du score combiné
            combined_score = metrics['total_return'] - (fee_drag * 2)
            
            # Stockage des résultats
            results[threshold] = {
                'portfolio_value': portfolio_value,
                'metrics': metrics,
                'total_fees': total_fees,
                'transaction_count': transaction_count,
                'fee_drag': fee_drag,
                'combined_score': combined_score
            }
        
        return results
    
    def calculate_metrics(self, portfolio_value: pd.Series, allocations: pd.Series) -> Dict:
        """
        Calcule les métriques de performance
        
        Args:
            portfolio_value: Série des valeurs du portefeuille
            allocations: Série des allocations réalisées
            
        Returns:
            Dictionnaire des métriques de performance
        """
        # Calcul des rendements
        returns = portfolio_value.pct_change().dropna()
        
        # Rendement total
        total_return = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1) * 100
        
        # Volatilité annualisée
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Calcul du ratio de Sharpe (avec un taux sans risque de 2%)
        risk_free_rate = 0.02
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Calcul du drawdown maximum
        cumulative_max = portfolio_value.cummax()
        drawdown = (portfolio_value - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Calcul du ratio rendement/drawdown (en valeur absolue)
        return_drawdown_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else total_return
        
        # Calcul de la stabilité des allocations
        allocation_volatility = allocations.diff().abs().mean() * 100
        
        # Frais totaux
        total_fees = self.fees_evaluator.get_total_fees()
        
        return {
            'total_investment': self.initial_capital,
            'final_value': portfolio_value.iloc[-1],
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'return_drawdown_ratio': return_drawdown_ratio,
            'allocation_volatility': allocation_volatility,
            'total_fees': total_fees,
            'fee_drag': total_fees / portfolio_value.iloc[-1] * 100
        }
       
    def compare_with_benchmarks(self, metrics: Dict) -> pd.DataFrame:
        """
        Compare les résultats avec les benchmarks
        
        Args:
            metrics: Métriques calculées pour la stratégie
            
        Returns:
            DataFrame de comparaison
        """
        # Récupération des benchmarks
        benchmarks_df = StaticBenchmarks.format_as_dataframe()
        
        # Création d'une Series à partir du dictionnaire de métriques
        metrics_series = pd.Series(metrics)
        
        # Ajout des résultats de la stratégie comme nouvelle ligne
        results = benchmarks_df.copy()
        results.loc['QAAF'] = metrics_series
    
        return results

    def plot_performance(self, 
                       portfolio_value: pd.Series, 
                       allocations: pd.Series,
                       btc_data: pd.DataFrame, 
                       paxg_data: pd.DataFrame) -> None:
        """
        Visualise la performance du backtest
        
        Args:
            portfolio_value: Série des valeurs du portefeuille
            allocations: Série des allocations
            btc_data: DataFrame BTC
            paxg_data: DataFrame PAXG
        """
        # Configuration
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Normalisation des séries pour la comparaison
        start_date = portfolio_value.index[0]
        norm_portfolio = portfolio_value / portfolio_value.iloc[0]
        
        # Création des séries normalisées pour BTC et PAXG
        common_index = norm_portfolio.index
        norm_btc = btc_data.loc[common_index, 'close'] / btc_data.loc[start_date, 'close']
        norm_paxg = paxg_data.loc[common_index, 'close'] / paxg_data.loc[start_date, 'close']
        
        # Graphique 1: Performance
        ax1.plot(norm_portfolio.index, norm_portfolio, 'b-', linewidth=2, label='QAAF')
        ax1.plot(norm_btc.index, norm_btc, 'g--', linewidth=1.5, label='BTC')
        ax1.plot(norm_paxg.index, norm_paxg, 'r--', linewidth=1.5, label='PAXG')
        
        # Alloy DCA et BH (approximation normalisée)
        alloy_dca_factor = StaticBenchmarks.BENCHMARKS['ALLOY_DCA']['final_value'] / StaticBenchmarks.BENCHMARKS['ALLOY_DCA']['total_investment']
        alloy_bh_factor = StaticBenchmarks.BENCHMARKS['ALLOY_BH']['final_value'] / StaticBenchmarks.BENCHMARKS['ALLOY_BH']['total_investment']
        
        # Création de séries linéaires simplifiées (pour référence)
        start_value = 1.0
        end_date = portfolio_value.index[-1]
        start_to_end_days = (end_date - start_date).days
        
        # ALLOY DCA (ligne de référence)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        alloy_dca_series = pd.Series(
            np.linspace(start_value, alloy_dca_factor, len(dates)),
            index=dates
        )
        
        # ALLOY BH (ligne de référence)
        alloy_bh_series = pd.Series(
            np.linspace(start_value, alloy_bh_factor, len(dates)),
            index=dates
        )
        
        # Ajouter au graphique avec une transparence pour indiquer que c'est approximatif
        ax1.plot(alloy_dca_series.index, alloy_dca_series, 'g-.', alpha=0.5, linewidth=1, label='ALLOY DCA (ref)')
        ax1.plot(alloy_bh_series.index, alloy_bh_series, 'r-.', alpha=0.5, linewidth=1, label='ALLOY BH (ref)')
        
        ax1.set_title('Performance Comparée (Base 100)')
        ax1.set_ylabel('Performance')
        ax1.legend()
        ax1.grid(True)
        
        # Graphique 2: Allocation BTC
        ax2.plot(allocations.index, allocations, 'b-', linewidth=1.5)
        ax2.set_title('Allocation BTC')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Allocation BTC (%)')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_fees_impact(self, results: Dict) -> None:
        """
        Visualise l'impact des frais sur la performance
        
        Args:
            results: Résultats du test multi-seuils
        """
        thresholds = list(results.keys())
        total_fees = [result['total_fees'] for threshold, result in results.items()]
        transaction_counts = [result['transaction_count'] for threshold, result in results.items()]
        returns = [result['metrics']['total_return'] for threshold, result in results.items()]
        fee_drags = [result['fee_drag'] for threshold, result in results.items()]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graphique 1: Frais totaux
        ax1.bar(range(len(thresholds)), total_fees, color='blue', alpha=0.7)
        ax1.set_xticks(range(len(thresholds)))
        ax1.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax1.set_xlabel('Seuil de Rebalancement')
        ax1.set_ylabel('Frais Totaux ($)')
        ax1.set_title('Frais Totaux par Seuil')
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Nombre de transactions
        ax2.bar(range(len(thresholds)), transaction_counts, color='red', alpha=0.7)
        ax2.set_xticks(range(len(thresholds)))
        ax2.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax2.set_xlabel('Seuil de Rebalancement')
        ax2.set_ylabel('Nombre de Transactions')
        ax2.set_title('Nombre de Transactions par Seuil')
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Rendement total
        ax3.bar(range(len(thresholds)), returns, color='green', alpha=0.7)
        ax3.set_xticks(range(len(thresholds)))
        ax3.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax3.set_xlabel('Seuil de Rebalancement')
        ax3.set_ylabel('Rendement Total (%)')
        ax3.set_title('Rendement Total par Seuil')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Drag de performance
        ax4.bar(range(len(thresholds)), fee_drags, color='purple', alpha=0.7)
        ax4.set_xticks(range(len(thresholds)))
        ax4.set_xticklabels([f'{t:.1%}' for t in thresholds])
        ax4.set_xlabel('Seuil de Rebalancement')
        ax4.set_ylabel('Drag de Performance (%)')
        ax4.set_title('Impact des Frais sur la Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        
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
        #return {
        #    # Paramètres des métriques
        #    'volatility_window': [20, 30, 40, 50, 60],
        #    'spectral_window': [30, 45, 60, 75, 90],
        #    'min_periods': [15, 20, 25],
        
        # Dans QAAFOptimizer.define_parameter_grid()
        # Réduisez les options pour limiter l'explosion combinatoire
        return {
            'volatility_window': [30],  # Au lieu de [20, 30, 40, 50, 60]
            'spectral_window': [60],    # Au lieu de [30, 45, 60, 75, 90]
            'min_periods': [20],        # Valeur unique
            # Réduire les autres options...
        
            # Paramètres des poids de métriques
            'vol_ratio_weight': [0.0, 0.1, 0.2], # 0.3, 0.4, 0.5, 0.6, 0.7],
            'bound_coherence_weight': [0.0, 0.1, 0.2], # 0.3, 0.4, 0.5, 0.6, 0.7],
            'alpha_stability_weight': [0.0, 0.1, 0.2], # 0.3, 0.4, 0.5, 0.6, 0.7],
            'spectral_score_weight': [0.0, 0.1, 0.2], # 0.3, 0.4, 0.5, 0.6, 0.7],
            
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
        
        # Et que les valeurs min_btc < max_btc
        if params['min_btc_allocation'] >= params['max_btc_allocation']:
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
        
        # Limiter intelligemment les combinaisons à tester
        np.random.seed(42)  # Pour la reproductibilité
        if len(valid_combinations) > max_combinations:
            combinations_to_test = np.random.choice(valid_combinations, size=max_combinations, replace=False)
        else:
            combinations_to_test = valid_combinations
        
        logger.info(f"Nombre total de combinaisons possibles: {len(all_combinations):,}")
        logger.info(f"Nombre de combinaisons valides: {len(valid_combinations):,}")
        logger.info(f"Nombre de combinaisons à tester: {len(combinations_to_test):,}")
        
        # Initialisation des résultats
        self.optimization_results = []
        
        # Boucle d'optimisation avec barre de progression
        for params in tqdm(combinations_to_test, desc=f"Optimisation ({profile})"):
            logger.debug(f"Test des paramètres: {params}") #Ajout d'un log.
            try:
                # Test de la combinaison
                result = self._evaluate_parameter_combination(params, profile)
                
                # Si le score n'est pas -inf (contraintes respectées), ajouter aux résultats
                if not np.isinf(result['score']):
                    self.optimization_results.append(result)
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation des paramètres {params}: {e}")
        
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
                'end': pd.Timestamp('2024-12-31')
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
                end_date: str = '2024-12-31',
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
    
    def analyze_market_phases(self) -> pd.Series:
        """
        Analyse les phases de marché
        
        Returns:
            Série des phases de marché
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
        
        logger.info("Analyse des phases de marché")
        
        # Identification des phases de marché
        self.market_phases = self.market_phase_analyzer.identify_market_phases(self.data['BTC'])
        
        return self.market_phases
    
    def calculate_metrics(self) -> Dict[str, pd.Series]:
        """
        Calcule les métriques QAAF
        
        Returns:
            Dictionnaire des métriques calculées
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
        
        logger.info("Calcul des métriques QAAF")
        
        # Calcul des métriques via le MetricsCalculator
        self.metrics = self.metrics_calculator.calculate_metrics(self.data)
        
        return self.metrics
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Calcule le score composite
        
        Args:
            weights: Dictionnaire des poids pour chaque métrique
            
        Returns:
            Série du score composite
        """
        if self.metrics is None:
            raise ValueError("Aucune métrique calculée. Appelez calculate_metrics() d'abord.")
        
        logger.info("Calcul du score composite")
        
        # Poids par défaut si non fournis
        if weights is None:
            weights = {
                'vol_ratio': 0.3,
                'bound_coherence': 0.3,
                'alpha_stability': 0.2,
                'spectral_score': 0.2
            }
        
        # Normalisation des métriques
        normalized_metrics = self.metrics_calculator.normalize_metrics(self.metrics)
        
        # Calcul du score composite
        self.composite_score = pd.Series(0.0, index=normalized_metrics[list(normalized_metrics.keys())[0]].index)
        for name, series in normalized_metrics.items():
            if name in weights:
                self.composite_score += weights[name] * series
        
        return self.composite_score
    
    def calculate_adaptive_allocations(self) -> pd.Series:
        """
        Calcule les allocations adaptatives
        
        Returns:
            Série des allocations BTC
        """
        if self.composite_score is None:
            raise ValueError("Aucun score composite calculé. Appelez calculate_composite_score() d'abord.")
        
        if self.market_phases is None:
            raise ValueError("Aucune phase de marché identifiée. Appelez analyze_market_phases() d'abord.")
        
        logger.info("Calcul des allocations adaptatives")
        
        # Calcul des allocations via l'AdaptiveAllocator
        self.allocations = self.adaptive_allocator.calculate_adaptive_allocation(
            self.composite_score,
            self.market_phases
        )
        
        return self.allocations
    
    def run_backtest(self) -> Dict:
        """
        Exécute le backtest
        
        Returns:
            Dictionnaire des résultats du backtest
        """
        if self.allocations is None:
            raise ValueError("Aucune allocation calculée. Appelez calculate_adaptive_allocations() d'abord.")
        
        logger.info("Exécution du backtest")
        
        # Exécution du backtest
        self.performance, metrics = self.backtester.run_backtest(
            self.data['BTC'],
            self.data['PAXG'],
            self.allocations
        )
        
        # Comparaison avec les benchmarks
        comparison = self.backtester.compare_with_benchmarks(metrics)
        
        # Stockage des résultats
        self.results = {
            'metrics': metrics,
            'comparison': comparison
        }
        
        return self.results
    
    def optimize_rebalance_threshold(self, thresholds: List[float] = [0.01, 0.03, 0.05, 0.1]) -> Dict:
        """
        Optimise le seuil de rebalancement
        
        Args:
            thresholds: Liste des seuils à tester
            
        Returns:
            Dictionnaire des résultats par seuil
        """
        if self.allocations is None:
            raise ValueError("Aucune allocation calculée. Appelez calculate_adaptive_allocations() d'abord.")
        
        logger.info("Optimisation du seuil de rebalancement")
        
        # Test des différents seuils
        results = self.backtester.run_multi_threshold_test(
            self.data['BTC'],
            self.data['PAXG'],
            self.allocations,
            thresholds
        )
        
        # Calcul du seuil optimal (meilleur compromis entre performance et frais)
        threshold_metrics = []
        for threshold, result in results.items():
            metrics = result['metrics']
            total_fees = result['total_fees']
            transaction_count = result['transaction_count']
            fee_drag = result['fee_drag']
            combined_score = result['combined_score']
            
            threshold_metrics.append({
                'threshold': threshold,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_fees': total_fees,
                'transaction_count': transaction_count,
                'fee_drag': fee_drag,
                'combined_score': combined_score
            })
        
        # Tri par score combiné
        sorted_metrics = sorted(threshold_metrics, key=lambda x: x['combined_score'], reverse=True)
        
        # Sélection du meilleur seuil
        optimal_threshold = sorted_metrics[0]['threshold']
        logger.info(f"Seuil optimal de rebalancement: {optimal_threshold:.1%}")
        
        return {
            'results': results,
            'metrics': threshold_metrics,
            'optimal_threshold': optimal_threshold
        }
    
    
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
    
    def print_summary(self) -> None:
        """Affiche un résumé des résultats de l'analyse QAAF."""
        if self.results is None:
            logger.warning("Aucun résultat disponible. Appelez run_backtest() d'abord.")
            print("\nAucun résultat à afficher.")
            return
        
        print("\nRésumé des Résultats QAAF v1.0.0")
        print("=" * 50)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Période analysée: {self.start_date} à {self.end_date}")
        print(f"Capital initial: ${self.initial_capital:,.2f}")
        
        # Métriques de performance
        metrics = self.results.get('metrics', {})
        final_value = self.performance.iloc[-1] if self.performance is not None else 0
        print("\nPerformance:")
        print(f"- Valeur finale: ${final_value:,.2f}")
        print(f"- Rendement total: {metrics.get('total_return', 0):.2f}%")
        print(f"- Volatilité: {metrics.get('volatility', 0):.2f}%")
        print(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"- Drawdown maximum: {metrics.get('max_drawdown', 0):.2f}%")
        
        if metrics.get('max_drawdown', 0) != 0:
            rd_ratio = metrics.get('total_return', 0) / abs(metrics.get('max_drawdown', 0))
            print(f"- Ratio Rendement/Drawdown: {rd_ratio:.2f}")
        
        # Frais
        total_fees = self.fees_evaluator.get_total_fees()
        fee_drag = (total_fees / final_value * 100) if final_value > 0 else 0
        print(f"\nFrais de transaction totaux: ${total_fees:,.2f}")
        print(f"Impact des frais sur la performance: {fee_drag:.2f}%")
        
        # Comparaison avec les benchmarks
        print("\nComparaison avec les Benchmarks:")
        print("-" * 50)
        comparison = self.results.get('comparison', {})
        if comparison:
            for bench, values in comparison.items():
                print(f"- {bench}: Rendement {values.get('total_return', 0):.2f}%, "
                    f"Sharpe {values.get('sharpe_ratio', 0):.2f}")
        else:
            print("Aucune donnée de comparaison disponible.")
            
    def visualize_results(self) -> None:
        """
        Visualise les résultats de l'analyse
        """
        if self.performance is None or self.allocations is None:
            logger.warning("Aucun résultat de backtest disponible. Appelez run_backtest() d'abord.")
            return
        
        logger.info("Visualisation des résultats")
        
        if hasattr(self, 'backtester') and callable(getattr(self.backtester, 'plot_performance', None)):
            # Visualisation de la performance via le backtester
            self.backtester.plot_performance(
                self.performance,
                self.allocations,
                self.data['BTC'],
                self.data['PAXG']
            )
        else:
            # Visualisation basique en cas d'absence de backtester
            plt.figure(figsize=(15, 10))
            
            # Graphique de performance
            plt.subplot(2, 1, 1)
            plt.plot(self.performance, 'b-', label='QAAF Portfolio')
            plt.title('Performance du Portefeuille')
            plt.legend()
            plt.grid(True)
            
            # Graphique d'allocation
            plt.subplot(2, 1, 2)
            plt.plot(self.allocations, 'g-', label='Allocation BTC')
            plt.title('Allocation BTC')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
                
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
        try:
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
                if profile in metrics_results.get('best_combinations', {}):
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
            try:
                self.visualize_results()
            except Exception as viz_error:
                logger.error(f"Erreur lors de la visualisation: {str(viz_error)}")
            
            # Visualisation des résultats d'optimisation (si disponibles)
            if metrics_results is not None and hasattr(self, 'optimizer'):
                try:
                    self.optimizer.plot_optimization_results(profile)
                    self.optimizer.plot_metrics_importance()
                except Exception as opt_error:
                    logger.error(f"Erreur lors de la visualisation des résultats d'optimisation: {str(opt_error)}")
            
            return {
                'results': results,
                'metrics_results': metrics_results,
                'threshold_results': threshold_results,
                'validation_results': validation_results,
                'robustness_results': robustness_results
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse complète: {str(e)}")
            # Retourner les résultats partiels s'ils existent
            return {
                'error': str(e),
                'performance': self.performance if hasattr(self, 'performance') else None,
                'allocations': self.allocations if hasattr(self, 'allocations') else None,
                'results': self.results if hasattr(self, 'results') else None
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


# Fonction pour exécuter dans Google Colab
def run_qaaf(optimize_metrics: bool = True, optimize_threshold: bool = True):
    """Exécute le framework QAAF dans Google Colab"""
    
    print("\n🔹 QAAF (Quantitative Algorithmic Asset Framework) - Version 1.0.8 🔹")
    print("=" * 70)
    print("Ajout d'un module d'optimisation avancé des métriques et des frais")
    print("Identification de combinaisons optimales selon différents profils de risque/rendement")

def run_qaaf(optimize_metrics: bool = True, 
             optimize_threshold: bool = True, 
             run_validation: bool = True,
             profile: str = 'balanced',
             verbose: bool = True):

    """
    Fonction principale d'exécution de QAAF 1.0.0 adaptée pour Google Colab
    """
    # Configuration du niveau de logging
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    print("\n�� QAAF (Quantitative Algorithmic Asset Framework) - Version 1.0.0 🔹")
    print("=" * 70)
    print("Framework avancé d'analyse et trading algorithmique avec moteur d'optimisation efficace")
    print("Intégration de validation out-of-sample et tests de robustesse")
    
    # Configuration
    initial_capital = 30000.0
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    trading_costs = 0.001  # 0.1% (10 points de base)
    
    print(f"\nConfiguration:")
    print(f"- Capital initial: ${initial_capital:,.2f}")
    print(f"- Période d'analyse: {start_date} à {end_date}")
    print(f"- Frais de transaction: {trading_costs:.2%}")
    print(f"- Profil d'optimisation: {profile}")
    print(f"- Optimisation des métriques: {'Oui' if optimize_metrics else 'Non'}")
    print(f"- Optimisation du seuil de rebalancement: {'Oui' if optimize_threshold else 'Non'}")
    print(f"- Validation out-of-sample: {'Oui' if run_validation else 'Non'}")
    
    # Initialisation
    qaaf = QAAFCore(
        initial_capital=initial_capital,
        trading_costs=trading_costs,
        start_date=start_date,
        end_date=end_date,
        allocation_min=0.1,  # Bornes d'allocation élargies
        allocation_max=0.9
    )
    
    # Exécution de l'analyse complète
    print("\n📊 Démarrage de l'analyse...\n")
    
    try:
        results = qaaf.run_full_analysis(
            optimize_metrics=optimize_metrics,
            optimize_threshold=optimize_threshold,
            run_validation=run_validation,
            profile=profile
        )
        
        # Génération et affichage d'un rapport de recommandation
        if optimize_metrics and qaaf.optimizer:
            print("\n📋 Rapport de recommandation:\n")
            recommendation_report = qaaf.optimizer.generate_recommendation_report()
            print(recommendation_report)
        
        print("\n✅ Analyse complétée avec succès!")
        return qaaf, results
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, None

# Point d'entrée principal (utiliser cette version dans Colab)
if __name__ == "__main__":
    # Version Economie de ressources    
    qaaf, results = run_qaaf(
        optimize_metrics=True,
        optimize_threshold=False,  # Désactiver pour économiser la mémoire
        run_validation=False,      # Désactiver la validation
        profile='balanced',
        verbose=False              # Réduire les logs
    )
'''
    # Version simplifiée qui fonctionne dans Colab
    qaaf, results = run_qaaf(
        optimize_metrics=True,
        optimize_threshold=True,
        run_validation=True,
        profile='balanced',
        verbose=True
    )
'''
    # Visualisation des résultats
    if qaaf is not None:
        qaaf.visualize_results()