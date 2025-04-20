mkdir - p
tests
cat > tests / test_initialization.py << EOF
import sys
import os

# Ajoutez le répertoire parent au path pour les imports
sys.path.append (os.path.dirname (os.path.dirname (os.path.abspath (__file__))))

from qaaf.core.qaaf_core import QAAFCore
import pandas as pd
import numpy as np


def test_initialization ():
    print ("Test d'initialisation")
    qaaf=QAAFCore ()
    print ("✓ QAAFCore initialisé avec succès")
    return qaaf


def test_direct_data_assignment ():
    print ("\nTest d'assignation directe des données")
    qaaf=QAAFCore ()

    # Création de données synthétiques
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

    # Assignation directe
    qaaf.data={
        'BTC':btc_data,
        'PAXG':paxg_data,
        'PAXG/BTC':paxg_btc_data
    }

    print ("✓ Données assignées directement")

    # Test des méthodes principales
    try:
        qaaf.analyze_market_phases ()
        print ("✓ analyze_market_phases() fonctionne")

        qaaf.calculate_metrics ()
        print ("✓ calculate_metrics() fonctionne")

        qaaf.calculate_composite_score ()
        print ("✓ calculate_composite_score() fonctionne")

        qaaf.calculate_adaptive_allocations ()
        print ("✓ calculate_adaptive_allocations() fonctionne")
    except Exception as e:
        print (f"✗ Erreur: {e}")

    return qaaf


if __name__ == "__main__":
    print ("=== Tests d'initialisation et fonctions de base de QAAF ===\n")
    qaaf1=test_initialization ()
    qaaf2=test_direct_data_assignment ()
    print ("\nTous les tests sont terminés!")
EOF