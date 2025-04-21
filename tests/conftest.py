import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def synthetic_data ():
    """Fixture qui génère des données synthétiques pour les tests"""
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


@pytest.fixture
def qaaf_core_instance ():
    """Fixture qui crée une instance QAAFCore correctement initialisée"""
    from qaaf.core.qaaf_core import QAAFCore
    return QAAFCore ()