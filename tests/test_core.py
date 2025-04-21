import pytest
import pandas as pd
import numpy as np
from qaaf.core.qaaf_core import QAAFCore

# Fixture pour créer des données synthétiques pour les tests
@pytest.fixture
def synthetic_data ():
    dates=pd.date_range (start='2020-01-01',end='2020-12-31',freq='D')
    n=len (dates)

    # Données BTC
    btc_data=pd.DataFrame ({
        'open':np.random.rand (n) * 10000 + 5000,
        'high':np.random.rand (n) * 10000 + 7000,
        'low':np.random.rand (n) * 10000 + 3000,
        'close':np.random.rand (n) * 10000 + 6000,
        'volume':np.random.rand (n) * 1000000
    },index=dates)

    # Données PAXG
    paxg_data=pd.DataFrame ({
        'open':np.random.rand (n) * 100 + 1500,
        'high':np.random.rand (n) * 100 + 1600,
        'low':np.random.rand (n) * 100 + 1400,
        'close':np.random.rand (n) * 100 + 1550,
        'volume':np.random.rand (n) * 10000
    },index=dates)

    # Données PAXG/BTC
    paxg_btc_data=pd.DataFrame ({
        'open':paxg_data['open'] / btc_data['open'],
        'high':paxg_data['high'] / btc_data['high'],
        'low':paxg_data['low'] / btc_data['low'],
        'close':paxg_data['close'] / btc_data['close'],
        'volume':paxg_data['volume']
    },index=dates)

    return {
        'BTC':btc_data,
        'PAXG':paxg_data,
        'PAXG/BTC':paxg_btc_data
    }

# Test d'initialisation
def test_qaaf_core_init ():
    qaaf=QAAFCore ()
    assert qaaf is not None
    assert qaaf.initial_capital == 30000.0
    assert qaaf.trading_costs == 0.001


# Test d'assignation directe des données
def test_direct_data_assignment (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    assert qaaf.data is not None
    assert 'BTC' in qaaf.data
    assert 'PAXG' in qaaf.data
    assert 'PAXG/BTC' in qaaf.data


# Test d'analyse des phases de marché
def test_analyze_market_phases (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    phases=qaaf.analyze_market_phases ()
    assert phases is not None
    assert isinstance (phases,pd.Series)
    assert len (phases) == len (synthetic_data['BTC'])
    assert qaaf.market_phases is not None


# Test de calcul des métriques
def test_calculate_metrics (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    metrics=qaaf.calculate_metrics ()
    assert metrics is not None
    assert isinstance (metrics,dict)
    assert 'vol_ratio' in metrics
    assert 'bound_coherence' in metrics
    assert 'alpha_stability' in metrics
    assert 'spectral_score' in metrics


# Test de calcul du score composite
def test_calculate_composite_score (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    qaaf.calculate_metrics ()
    score=qaaf.calculate_composite_score ()
    assert score is not None
    assert isinstance (score,pd.Series)
    assert len (score) == len (synthetic_data['BTC'])


# Test de calcul des allocations adaptatives
def test_calculate_adaptive_allocations (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    qaaf.analyze_market_phases ()
    qaaf.calculate_metrics ()
    qaaf.calculate_composite_score ()
    allocations=qaaf.calculate_adaptive_allocations ()
    assert allocations is not None
    assert isinstance (allocations,pd.Series)
    assert len (allocations) == len (synthetic_data['BTC'])
    assert (allocations >= 0).all () and (allocations <= 1).all ()


# Test du workflow complet
def test_full_workflow (synthetic_data):
    qaaf=QAAFCore ()
    qaaf.data=synthetic_data
    qaaf.analyze_market_phases ()
    qaaf.calculate_metrics ()
    qaaf.calculate_composite_score ()
    qaaf.calculate_adaptive_allocations ()
    qaaf.run_backtest ()
    assert qaaf.performance is not None
    assert qaaf.results is not None