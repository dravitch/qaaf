"""
Script autonome pour analyser les allocations DCA BTC/PAXG.
Contient toutes les fonctions nécessaires.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Metrics:
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    volatility: float

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Télécharge les données de prix."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        prices = data['Close'].squeeze()
        
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices, index=data.index)
        
        return prices
    except Exception as e:
        print(f"Erreur téléchargement {ticker}: {e}")
        return None

def calculate_metrics(portfolio_values: pd.Series, total_invested: float) -> Metrics:
    """Calcule les métriques de performance."""
    if len(portfolio_values) < 2:
        raise ValueError("Série insuffisante")
    
    returns = portfolio_values.pct_change().dropna()
    
    # Total Return
    total_return = ((portfolio_values.iloc[-1] / total_invested) - 1) * 100
    
    # Max Drawdown
    peak = portfolio_values.expanding().max()
    drawdown = ((portfolio_values - peak) / peak * 100)
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    risk_free_rate = 0.02
    if len(returns) > 0 and returns.std() > 0:
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe = excess_returns / (returns.std() * np.sqrt(252))
    else:
        sharpe = 0
    
    # Calmar Ratio
    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
    
    return Metrics(
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        volatility=volatility
    )

def dca_allocation(btc_prices: pd.Series, paxg_prices: pd.Series, 
                   btc_ratio: float, initial_capital: float) -> Tuple[pd.Series, Metrics]:
    """
    Simule une stratégie DCA avec allocation configurable.
    
    Args:
        btc_prices: Prix BTC quotidiens
        paxg_prices: Prix PAXG quotidiens
        btc_ratio: Proportion BTC (0-1)
        initial_capital: Capital total à investir
    """
    # Resampling mensuel
    btc_monthly = btc_prices.resample('ME').last().dropna()
    paxg_monthly = paxg_prices.resample('ME').last().dropna()
    
    # Index commun
    common_index = btc_monthly.index.intersection(paxg_monthly.index)
    btc_monthly = btc_monthly.loc[common_index]
    paxg_monthly = paxg_monthly.loc[common_index]
    
    # Investissement par période
    periods = len(btc_monthly)
    investment_per_period = initial_capital / periods
    
    # Calcul des parts
    btc_invest = investment_per_period * btc_ratio
    paxg_invest = investment_per_period * (1 - btc_ratio)
    
    btc_shares_per_period = btc_invest / btc_monthly
    paxg_shares_per_period = paxg_invest / paxg_monthly
    
    btc_cumulative = btc_shares_per_period.cumsum()
    paxg_cumulative = paxg_shares_per_period.cumsum()
    
    # Interpolation journalière
    full_index = btc_prices.index.union(paxg_prices.index)
    full_index = full_index[(full_index >= common_index[0]) & (full_index <= common_index[-1])]
    
    btc_shares_daily = btc_cumulative.reindex(full_index, method='ffill').fillna(0)
    paxg_shares_daily = paxg_cumulative.reindex(full_index, method='ffill').fillna(0)
    
    btc_prices_aligned = btc_prices.reindex(full_index, method='ffill')
    paxg_prices_aligned = paxg_prices.reindex(full_index, method='ffill')
    
    # Valeur du portfolio
    portfolio_values = (btc_shares_daily * btc_prices_aligned + 
                       paxg_shares_daily * paxg_prices_aligned)
    
    # Métriques
    metrics = calculate_metrics(portfolio_values, initial_capital)
    
    return portfolio_values, metrics

def buy_hold_btc(btc_prices: pd.Series, initial_capital: float) -> Tuple[pd.Series, Metrics]:
    """Stratégie Buy & Hold 100% BTC."""
    first_price = btc_prices.iloc[0]
    btc_units = initial_capital / first_price
    portfolio_values = btc_units * btc_prices
    
    metrics = calculate_metrics(portfolio_values, initial_capital)
    return portfolio_values, metrics

def analyze_allocations(btc_prices: pd.Series, paxg_prices: pd.Series, 
                       initial_capital: float = 30000) -> Dict:
    """Analyse complète des allocations 50/50, 60/40, 70/30, 100/0."""
    
    print("\n" + "="*80)
    print("CALCUL DES STRATÉGIES DCA...")
    print("="*80)
    
    # Calculer toutes les stratégies
    strategies = {}
    
    for ratio, name in [(0.5, 'DCA 50/50'), (0.6, 'DCA 60/40'), (0.7, 'DCA 70/30')]:
        print(f"  • {name}...")
        portfolio, metrics = dca_allocation(btc_prices, paxg_prices, ratio, initial_capital)
        strategies[name] = {
            'portfolio': portfolio,
            'metrics': metrics,
            'btc_ratio': int(ratio * 100)
        }
    
    print(f"  • 100% BTC (Buy & Hold)...")
    portfolio, metrics = buy_hold_btc(btc_prices, initial_capital)
    strategies['100% BTC'] = {
        'portfolio': portfolio,
        'metrics': metrics,
        'btc_ratio': 100
    }
    
    return strategies

def plot_results(strategies: Dict, save_path: Optional[str] = None):
    """Génère les graphiques de comparaison."""
    
    # Extraire les métriques
    names = list(strategies.keys())
    returns = [strategies[n]['metrics'].total_return for n in names]
    drawdowns = [abs(strategies[n]['metrics'].max_drawdown) for n in names]
    sharpes = [strategies[n]['metrics'].sharpe_ratio for n in names]
    btc_ratios = [strategies[n]['btc_ratio'] for n in names]
    
    # Calculer Return/Drawdown ratios
    rd_ratios = [r / abs(d) if d != 0 else 0 for r, d in zip(returns, drawdowns)]
    
    # Créer la figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Frontière Efficiente
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(drawdowns, returns, s=300, alpha=0.6, c=btc_ratios, cmap='RdYlGn_r')
    
    for i, name in enumerate(names):
        ax1.annotate(name, (drawdowns[i], returns[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    ax1.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Frontière Efficiente: Rendement vs Risque', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Subplot 2: Return/Drawdown Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#c0392b']
    bars = ax2.bar(range(len(names)), rd_ratios, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([f"{r}% BTC" for r in btc_ratios], fontsize=11)
    ax2.set_ylabel('Return/Drawdown Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Efficience par Allocation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, ratio in zip(bars, rd_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 3: Evolution des portfolios
    ax3 = fig.add_subplot(gs[1, :])
    for i, (name, data) in enumerate(strategies.items()):
        ax3.plot(data['portfolio'].index, data['portfolio'].values, 
                label=name, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Valeur du Portfolio ($)', fontsize=12, fontweight='bold')
    ax3.set_title('Évolution des Portfolios dans le Temps', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: {save_path}")
    
    plt.show()

def print_summary(strategies: Dict):
    """Affiche le tableau récapitulatif."""
    
    print("\n" + "="*80)
    print("ANALYSE COMPARATIVE DES ALLOCATIONS BTC/PAXG")
    print("="*80)
    
    for name, data in strategies.items():
        metrics = data['metrics']
        rd_ratio = metrics.total_return / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
        
        print(f"\n{name}:")
        print(f"  • Return:            {metrics.total_return:>8.2f}%")
        print(f"  • Max Drawdown:      {metrics.max_drawdown:>8.2f}%")
        print(f"  • Sharpe Ratio:      {metrics.sharpe_ratio:>8.2f}")
        print(f"  • Volatility:        {metrics.volatility:>8.2f}%")
        print(f"  • Return/DD Ratio:   {rd_ratio:>8.2f}")
    
    print("="*80)
    
    # Trouver le meilleur ratio
    rd_ratios = {}
    for name, data in strategies.items():
        metrics = data['metrics']
        rd_ratios[name] = metrics.total_return / abs(metrics.max_drawdown) if metrics.max_drawdown != 0 else 0
    
    best = max(rd_ratios, key=rd_ratios.get)
    print(f"\n🏆 Meilleur Return/Drawdown Ratio: {best}")
    print(f"   Ratio: {rd_ratios[best]:.2f}")
    print("="*80)

def main():
    # Configuration
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    initial_capital = 30000
    
    print("="*80)
    print("ANALYSE DES ALLOCATIONS DCA BTC/PAXG")
    print("="*80)
    print(f"Période: {start_date} → {end_date}")
    print(f"Capital initial: ${initial_capital:,}")
    print("\nTéléchargement des données...")
    
    # Télécharger les données
    btc_prices = fetch_price_data('BTC-USD', start_date, end_date)
    paxg_prices = fetch_price_data('PAXG-USD', start_date, end_date)
    
    if btc_prices is None or paxg_prices is None:
        print("❌ Erreur: Impossible de télécharger les données")
        return
    
    print(f"✓ BTC: {len(btc_prices)} points de données")
    print(f"✓ PAXG: {len(paxg_prices)} points de données")
    
    # Analyser les allocations
    strategies = analyze_allocations(btc_prices, paxg_prices, initial_capital)
    
    # Afficher les résultats
    print_summary(strategies)
    
    # Générer les graphiques
    plot_results(strategies, save_path='allocation_frontier.png')

if __name__ == "__main__":
    main()