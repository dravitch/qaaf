"""
qaaf/benchmarks/benchmark_runner.py

Module d'exécution des benchmarks pour comparaison avec QAAF.
Implémente DCA et Buy & Hold sur BTC/PAXG.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Exécute les benchmarks DCA et Buy & Hold."""
    
    def __init__(self, initial_capital: float = 30000):
        """
        Initialise le runner de benchmarks.
        
        Args:
            initial_capital: Capital initial pour tous les benchmarks
        """
        self.initial_capital = initial_capital
        self.results = {}
        logger.info(f"BenchmarkRunner initialisé avec capital: ${initial_capital:,.2f}")
    
    def dca_50_50(self, btc_prices: pd.Series, paxg_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """
        DCA 50/50 BTC-PAXG avec rebalancing mensuel.
        
        Args:
            btc_prices: Série des prix BTC (index temporel)
            paxg_prices: Série des prix PAXG (index temporel)
            
        Returns:
            Tuple (portfolio_values, trades)
        """
        # Alignement mensuel
        btc_monthly = btc_prices.resample('ME').last().dropna()
        paxg_monthly = paxg_prices.resample('ME').last().dropna()
        
        # Index commun
        common_index = btc_monthly.index.intersection(paxg_monthly.index)
        btc_monthly = btc_monthly.loc[common_index]
        paxg_monthly = paxg_monthly.loc[common_index]
        
        periods = len(btc_monthly)
        investment_per_period = self.initial_capital / periods
        
        # Calcul des parts achetées à chaque période
        btc_invest_per_period = investment_per_period * 0.5
        paxg_invest_per_period = investment_per_period * 0.5
        
        btc_shares_per_period = btc_invest_per_period / btc_monthly
        paxg_shares_per_period = paxg_invest_per_period / paxg_monthly
        
        # Cumul des parts
        btc_cumulative_shares = btc_shares_per_period.cumsum()
        paxg_cumulative_shares = paxg_shares_per_period.cumsum()
        
        # CORRECTION: Interpoler pour avoir des valeurs journalières
        # On réindexe sur l'index complet des prix journaliers
        full_index = btc_prices.index.union(paxg_prices.index)
        full_index = full_index[(full_index >= common_index[0]) & (full_index <= common_index[-1])]
        
        # Forward fill des parts cumulées (les parts ne changent qu'aux dates d'achat)
        btc_shares_daily = btc_cumulative_shares.reindex(full_index, method='ffill').fillna(0)
        paxg_shares_daily = paxg_cumulative_shares.reindex(full_index, method='ffill').fillna(0)
        
        # Prix journaliers alignés
        btc_prices_aligned = btc_prices.reindex(full_index, method='ffill')
        paxg_prices_aligned = paxg_prices.reindex(full_index, method='ffill')
        
        # Valeur du portefeuille quotidienne
        portfolio_values = (btc_shares_daily * btc_prices_aligned + 
                          paxg_shares_daily * paxg_prices_aligned)
        
        # Génération des trades (mensuels)
        trades = []
        total_invested = 0
        for i, date in enumerate(btc_monthly.index):
            total_invested += investment_per_period
            trades.append({
                'date': date,
                'action': 'DCA_50_50',
                'btc_price': float(btc_monthly.iloc[i]),
                'paxg_price': float(paxg_monthly.iloc[i]),
                'btc_invested': float(btc_invest_per_period),
                'paxg_invested': float(paxg_invest_per_period),
                'btc_units_cumulative': float(btc_cumulative_shares.iloc[i]),
                'paxg_units_cumulative': float(paxg_cumulative_shares.iloc[i]),
                'total_invested': float(total_invested)
            })
        
        logger.info(f"DCA 50/50: {len(trades)} périodes, valeur finale ${portfolio_values.iloc[-1]:,.2f}")
        
        return portfolio_values, trades
    
    def _dca_allocation(self, btc_prices: pd.Series, paxg_prices: pd.Series, 
                       btc_ratio: float, name: str) -> Tuple[pd.Series, List[Dict]]:
        """
        DCA générique avec allocation BTC/PAXG configurable.
        
        Args:
            btc_prices: Série des prix BTC
            paxg_prices: Série des prix PAXG
            btc_ratio: Ratio d'allocation BTC (0-1)
            name: Nom de la stratégie
            
        Returns:
            Tuple (portfolio_values, trades)
        """
        btc_monthly = btc_prices.resample('ME').last().dropna()
        paxg_monthly = paxg_prices.resample('ME').last().dropna()
        
        common_index = btc_monthly.index.intersection(paxg_monthly.index)
        btc_monthly = btc_monthly.loc[common_index]
        paxg_monthly = paxg_monthly.loc[common_index]
        
        periods = len(btc_monthly)
        investment_per_period = self.initial_capital / periods
        
        # Calcul des parts achetées à chaque période
        btc_invest_per_period = investment_per_period * btc_ratio
        paxg_invest_per_period = investment_per_period * (1 - btc_ratio)
        
        btc_shares_per_period = btc_invest_per_period / btc_monthly
        paxg_shares_per_period = paxg_invest_per_period / paxg_monthly
        
        # Cumul des parts
        btc_cumulative_shares = btc_shares_per_period.cumsum()
        paxg_cumulative_shares = paxg_shares_per_period.cumsum()
        
        # Interpoler pour avoir des valeurs journalières
        full_index = btc_prices.index.union(paxg_prices.index)
        full_index = full_index[(full_index >= common_index[0]) & (full_index <= common_index[-1])]
        
        btc_shares_daily = btc_cumulative_shares.reindex(full_index, method='ffill').fillna(0)
        paxg_shares_daily = paxg_cumulative_shares.reindex(full_index, method='ffill').fillna(0)
        
        btc_prices_aligned = btc_prices.reindex(full_index, method='ffill')
        paxg_prices_aligned = paxg_prices.reindex(full_index, method='ffill')
        
        portfolio_values = (btc_shares_daily * btc_prices_aligned + 
                          paxg_shares_daily * paxg_prices_aligned)
        
        # Génération des trades
        trades = []
        total_invested = 0
        for i, date in enumerate(btc_monthly.index):
            total_invested += investment_per_period
            trades.append({
                'date': date,
                'action': name,
                'btc_price': float(btc_monthly.iloc[i]),
                'paxg_price': float(paxg_monthly.iloc[i]),
                'btc_invested': float(btc_invest_per_period),
                'paxg_invested': float(paxg_invest_per_period),
                'btc_units_cumulative': float(btc_cumulative_shares.iloc[i]),
                'paxg_units_cumulative': float(paxg_cumulative_shares.iloc[i]),
                'total_invested': float(total_invested)
            })
        
        logger.info(f"{name}: {len(trades)} périodes, valeur finale ${portfolio_values.iloc[-1]:,.2f}")
        
        return portfolio_values, trades
    
    def dca_50_50(self, btc_prices: pd.Series, paxg_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """DCA 50/50 BTC-PAXG (équilibre risque/refuge)."""
        return self._dca_allocation(btc_prices, paxg_prices, 0.5, 'DCA_50_50')
    
    def dca_60_40(self, btc_prices: pd.Series, paxg_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """DCA 60/40 BTC-PAXG (légèrement agressif)."""
        return self._dca_allocation(btc_prices, paxg_prices, 0.6, 'DCA_60_40')
    
    def dca_70_30(self, btc_prices: pd.Series, paxg_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """DCA 70/30 BTC-PAXG (agressif avec protection)."""
        return self._dca_allocation(btc_prices, paxg_prices, 0.7, 'DCA_70_30')
    
    def buy_hold_btc(self, btc_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """
        100% BTC Buy & Hold au premier prix.
        
        Args:
            btc_prices: Série des prix BTC
            
        Returns:
            Tuple (portfolio_values, trades)
        """
        first_price = btc_prices.iloc[0]
        
        if first_price == 0 or pd.isna(first_price):
            raise ValueError(f"Prix initial BTC invalide: {first_price}")
        
        btc_units = self.initial_capital / first_price
        portfolio_values = btc_units * btc_prices
        
        trades = [{
            'date': btc_prices.index[0],
            'action': 'BUY_HOLD_BTC',
            'btc_price': float(first_price),
            'btc_units': float(btc_units),
            'investment': self.initial_capital
        }]
        
        logger.info(f"Buy & Hold BTC: Valeur finale ${portfolio_values.iloc[-1]:,.2f}")
        
        return portfolio_values, trades
    
    def buy_hold_paxg(self, paxg_prices: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        """
        100% PAXG Buy & Hold au premier prix.
        
        Args:
            paxg_prices: Série des prix PAXG
            
        Returns:
            Tuple (portfolio_values, trades)
        """
        first_price = paxg_prices.iloc[0]
        
        if first_price == 0 or pd.isna(first_price):
            raise ValueError(f"Prix initial PAXG invalide: {first_price}")
        
        paxg_units = self.initial_capital / first_price
        portfolio_values = paxg_units * paxg_prices
        
        trades = [{
            'date': paxg_prices.index[0],
            'action': 'BUY_HOLD_PAXG',
            'paxg_price': float(first_price),
            'paxg_units': float(paxg_units),
            'investment': self.initial_capital
        }]
        
        logger.info(f"Buy & Hold PAXG: Valeur finale ${portfolio_values.iloc[-1]:,.2f}")
        
        return portfolio_values, trades
    
    def run_all_benchmarks(self, btc_prices: pd.Series, 
                           paxg_prices: pd.Series) -> Dict[str, Tuple[pd.Series, List[Dict]]]:
        """
        Exécute tous les benchmarks à la fois.
        
        Args:
            btc_prices: Série des prix BTC
            paxg_prices: Série des prix PAXG
            
        Returns:
            Dict avec résultats de chaque benchmark
        """
        logger.info("Exécution de tous les benchmarks...")
        
        results = {
            'DCA_50_50': self.dca_50_50(btc_prices, paxg_prices),
            'DCA_60_40': self.dca_60_40(btc_prices, paxg_prices),
            'BH_BTC': self.buy_hold_btc(btc_prices),
            'BH_PAXG': self.buy_hold_paxg(paxg_prices)
        }
        
        self.results = results
        logger.info(f"Tous les benchmarks exécutés ({len(results)} stratégies)")
        
        return results
    
    def calculate_metrics(self, portfolio_values: pd.Series, total_invested: float = None) -> Dict:
        """
        Calcule les 5 métriques de performance clés.
        
        Args:
            portfolio_values: Série de valeurs du portefeuille
            total_invested: Montant total investi (pour DCA, sinon utilise initial_capital)
            
        Returns:
            Dict avec métriques
        """
        if len(portfolio_values) < 2:
            raise ValueError("Série insuffisante pour calcul de métriques")
        
        returns = portfolio_values.pct_change().dropna()
        
        # CORRECTION: Pour DCA, comparer à total_invested; pour B&H, à initial_capital
        if total_invested is None:
            total_invested = self.initial_capital
        
        # 1. Total Return - basé sur le capital investi total
        total_return = ((portfolio_values.iloc[-1] / total_invested) - 1) * 100
        
        # 2. Max Drawdown - calculé sur les valeurs journalières
        peak = portfolio_values.expanding().max()
        drawdown = ((portfolio_values - peak) / peak * 100)
        max_drawdown = drawdown.min()
        
        # 3. Sharpe Ratio
        risk_free_rate = 0.02
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe = excess_returns / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        # 4. Calmar Ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 5. Volatility
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'volatility': volatility
        }
    
    def compare_benchmarks(self, btc_prices: pd.Series, 
                          paxg_prices: pd.Series) -> Dict[str, Dict]:
        """
        Compare tous les benchmarks et retourne les métriques.
        
        Args:
            btc_prices: Série des prix BTC
            paxg_prices: Série des prix PAXG
            
        Returns:
            Dict avec métriques pour chaque benchmark
        """
        benchmarks = self.run_all_benchmarks(btc_prices, paxg_prices)
        
        comparison = {}
        for name, (portfolio_values, trades) in benchmarks.items():
            # Pour DCA, passer le total investi; pour B&H, None (utilisera initial_capital)
            if name.startswith('DCA'):
                total_invested = self.initial_capital
            else:
                total_invested = None
            
            metrics = self.calculate_metrics(portfolio_values, total_invested)
            comparison[name] = {
                'metrics': metrics,
                'portfolio_values': portfolio_values,
                'trades': trades,
                'final_value': float(portfolio_values.iloc[-1])
            }
        
        return comparison