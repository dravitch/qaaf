"""
qaaf/analysis/trades_extractor.py

Extraction des détails des transactions QAAF pour analyse approfondie.
Génère les fichiers JSON détaillés: trades_detail, allocations_calculated, allocations_executed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradesExtractor:
    """Extrait et analyse les transactions QAAF en détail"""
    
    def __init__(self, rebalance_threshold: float = 0.05):
        """
        Initialise l'extracteur de transactions
        
        Args:
            rebalance_threshold: Seuil de rebalancement (ex: 0.05 = 5%)
        """
        self.rebalance_threshold = rebalance_threshold
    
    def extract_from_backtest(
        self,
        allocations_calculated: pd.Series,
        portfolio_values: pd.Series,
        btc_prices: pd.Series,
        paxg_prices: pd.Series,
        btc_data: pd.DataFrame,
        paxg_data: pd.DataFrame
    ) -> Dict:
        """
        Extrait les détails des transactions depuis un backtest
        
        Args:
            allocations_calculated: Allocations recommandées par QAAF
            portfolio_values: Valeurs du portefeuille
            btc_prices: Prix BTC
            paxg_prices: Prix PAXG
            btc_data: DataFrame BTC complet
            paxg_data: DataFrame PAXG complet
        
        Returns:
            Dictionnaire avec trades_detail, allocations_calculated, allocations_executed
        """
        trades = []
        allocations_executed = []
        
        # Simulation du backtest pour extraire les trades
        current_btc_alloc = allocations_calculated.iloc[0]
        last_rebalance_date = allocations_calculated.index[0]
        
        for i, date in enumerate(allocations_calculated.index):
            target_alloc = allocations_calculated.loc[date]
            
            # Vérifier si rebalancement nécessaire
            allocation_diff = abs(current_btc_alloc - target_alloc)
            days_since_rebalance = (date - last_rebalance_date).days
            
            rebalanced = False
            
            if allocation_diff > self.rebalance_threshold or days_since_rebalance > 30:
                # Trade détecté
                portfolio_value = portfolio_values.loc[date] if date in portfolio_values.index else 0
                
                trade = {
                    'date': date.isoformat(),
                    'signal_type': 'rebalance',
                    'action': 'SELL_BTC_BUY_PAXG' if target_alloc < current_btc_alloc else 'BUY_BTC_SELL_PAXG',
                    'btc_allocation_before': float(current_btc_alloc),
                    'btc_allocation_after': float(target_alloc),
                    'paxg_allocation_before': float(1 - current_btc_alloc),
                    'paxg_allocation_after': float(1 - target_alloc),
                    'portfolio_value_before': float(portfolio_value),
                    'btc_price': float(btc_prices.loc[date]),
                    'paxg_price': float(paxg_prices.loc[date]),
                    'allocation_diff': float(allocation_diff),
                    'days_since_last_rebalance': days_since_rebalance
                }
                
                trades.append(trade)
                
                current_btc_alloc = target_alloc
                last_rebalance_date = date
                rebalanced = True
            
            # Allocation exécutée
            exec_alloc = {
                'date': date.isoformat(),
                'btc_allocation_actual': float(current_btc_alloc),
                'paxg_allocation_actual': float(1 - current_btc_alloc),
                'btc_allocation_target': float(target_alloc),
                'variance': float(abs(current_btc_alloc - target_alloc)),
                'rebalanced': rebalanced,
                'days_since_last_rebalance': days_since_rebalance
            }
            
            allocations_executed.append(exec_alloc)
        
        # Statistiques
        stats = {
            'total_trades': len(trades),
            'avg_days_between_trades': np.mean([t['days_since_last_rebalance'] for t in trades]) if trades else 0,
            'total_allocation_changes': sum(abs(t['btc_allocation_after'] - t['btc_allocation_before']) for t in trades)
        }
        
        return {
            'trades': trades,
            'allocations_executed': allocations_executed,
            'statistics': stats
        }
    
    def extract_allocation_signals(
        self,
        composite_score: pd.Series,
        allocations: pd.Series,
        vol_ratio: pd.Series,
        bound_coherence: pd.Series,
        alpha_stability: pd.Series,
        spectral_score: pd.Series
    ) -> List[Dict]:
        """
        Extrait les signaux d'allocation calculés par QAAF
        
        Returns:
            Liste des signaux avec contexte complet
        """
        signals = []
        
        for date in allocations.index:
            signal = {
                'date': date.isoformat(),
                'btc_allocation_target': float(allocations.loc[date]),
                'paxg_allocation_target': float(1 - allocations.loc[date]),
                'composite_score': float(composite_score.loc[date]) if date in composite_score.index else 0,
                'vol_ratio': float(vol_ratio.loc[date]) if date in vol_ratio.index else 0,
                'bound_coherence': float(bound_coherence.loc[date]) if date in bound_coherence.index else 0,
                'alpha_stability': float(alpha_stability.loc[date]) if date in alpha_stability.index else 0,
                'spectral_score': float(spectral_score.loc[date]) if date in spectral_score.index else 0
            }
            
            # Déterminer le signal
            if signal['btc_allocation_target'] > 0.6:
                signal['allocation_signal'] = 'AGGRESSIVE (high BTC)'
            elif signal['btc_allocation_target'] < 0.4:
                signal['allocation_signal'] = 'DEFENSIVE (high PAXG)'
            else:
                signal['allocation_signal'] = 'NEUTRAL'
            
            signals.append(signal)
        
        return signals
    
    def calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """Calcule les statistiques sur les trades"""
        
        if not trades:
            return {
                'total_trades': 0,
                'avg_days_between': 0,
                'avg_allocation_change': 0
            }
        
        return {
            'total_trades': len(trades),
            'buy_signals': sum(1 for t in trades if 'BUY_BTC' in t['action']),
            'sell_signals': sum(1 for t in trades if 'SELL_BTC' in t['action']),
            'avg_days_between': np.mean([t['days_since_last_rebalance'] for t in trades]),
            'avg_allocation_change': np.mean([t['allocation_diff'] for t in trades]),
            'max_allocation_change': max([t['allocation_diff'] for t in trades]),
            'min_allocation_change': min([t['allocation_diff'] for t in trades])
        }
    
    def identify_anomalies(
        self,
        trades: List[Dict],
        allocations_signals: List[Dict],
        qaaf_metrics_summary: Dict
    ) -> List[Dict]:
        """
        Identifie les anomalies dans les trades et allocations
        
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Anomalie 1: Allocation moyenne = 0
        if allocations_signals:
            avg_alloc = np.mean([s['btc_allocation_target'] for s in allocations_signals])
            
            if avg_alloc == 0:
                anomalies.append({
                    'type': 'CRITICAL',
                    'description': 'Allocation moyenne BTC = 0%',
                    'impact': 'Aucune allocation générée, algorithme non fonctionnel',
                    'suggested_fix': 'Vérifier adaptive_allocator.calculate_adaptive_allocation()'
                })
        
        # Anomalie 2: Pas de trades
        if len(trades) == 0:
            anomalies.append({
                'type': 'WARNING',
                'description': 'Aucun trade exécuté',
                'impact': 'Portfolio statique, pas de rebalancement',
                'suggested_fix': 'Vérifier rebalance_threshold ou allocation variance'
            })
        
        # Anomalie 3: Métriques QAAF très faibles
        for metric_name, metric_stats in qaaf_metrics_summary.items():
            if metric_stats['mean'] < 0.01:
                anomalies.append({
                    'type': 'WARNING',
                    'description': f'{metric_name} moyenne très faible (< 0.01)',
                    'impact': 'Score composite potentiellement affecté',
                    'suggested_fix': f'Vérifier calcul de {metric_name} dans MetricsCalculator'
                })
        
        return anomalies
    
    def generate_trade_report(
        self,
        period_name: str,
        trades: List[Dict],
        allocations_signals: List[Dict],
        statistics: Dict,
        anomalies: List[Dict]
    ) -> Dict:
        """Génère un rapport complet des trades"""
        
        return {
            'period': period_name,
            'summary': {
                'total_trades': len(trades),
                'total_signals': len(allocations_signals),
                'statistics': statistics,
                'anomalies_count': len(anomalies)
            },
            'trades': trades,
            'allocations_signals': allocations_signals,
            'anomalies': anomalies
        }