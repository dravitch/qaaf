# tests/test_qaaf_data.py
"""
Tests unitaires pour le module de données QAAF.
Commence par tester l'implémentation YFinance avant d'intégrer Dukascopy.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from unittest.mock import Mock,patch

# Import des classes à tester
from qaaf.data import YFinanceSource,DataManager


class TestYFinanceSource (unittest.TestCase):
    """Tests de la source de données Yahoo Finance"""

    def setUp (self):
        """Configuration initiale pour chaque test"""
        self.source=YFinanceSource ()

        # Données de test
        self.test_data=pd.DataFrame ({
            'Open':[100,101],
            'High':[105,106],
            'Low':[95,96],
            'Close':[102,103],
            'Volume':[1000,1100]
        },index=pd.date_range ('2024-01-01',periods=2))

        # MultiIndex comme retourné par yfinance
        multi_cols=pd.MultiIndex.from_product ([
            ['Open','High','Low','Close','Volume'],
            ['Adj']
        ])
        self.test_data_multi=pd.DataFrame (
            self.test_data.values,
            index=self.test_data.index,
            columns=multi_cols
        )

    def test_standardize_data (self):
        """Test de la standardisation des données"""
        # Test avec MultiIndex
        result=self.source._standardize_data (self.test_data_multi)
        self.assertTrue (all (col in result.columns for col in
                              ['open','high','low','close','volume']))

        # Test avec colonnes simples
        result=self.source._standardize_data (self.test_data)
        self.assertTrue (all (col in result.columns for col in
                              ['open','high','low','close','volume']))

    def test_validate_data (self):
        """Test de la validation des données"""
        # Test données valides
        df=self.source._standardize_data (self.test_data)
        self.assertTrue (self.source.validate_data (df))

        # Test données invalides (colonne manquante)
        df_invalid=df.drop ('close',axis=1)
        self.assertFalse (self.source.validate_data (df_invalid))

        # Test DataFrame vide
        self.assertFalse (self.source.validate_data (pd.DataFrame ()))

    @patch ('yfinance.download')
    def test_fetch_data (self,mock_download):
        """Test de la récupération des données"""
        mock_download.return_value=self.test_data

        df=self.source.fetch_data ('BTC-USD','2024-01-01','2024-01-02')
        self.assertIsInstance (df,pd.DataFrame)
        self.assertEqual (len (df),2)
        self.assertTrue (all (col in df.columns for col in
                              ['open','high','low','close','volume']))


class TestDataManager (unittest.TestCase):
    """Tests du gestionnaire de données QAAF"""

    def setUp (self):
        """Configuration initiale pour chaque test"""
        self.manager=DataManager ()

        # Données de test pour BTC
        self.btc_data=pd.DataFrame ({
            'close':[40000,41000]
        },index=pd.date_range ('2024-01-01',periods=2))

        # Données de test pour PAXG
        self.paxg_data=pd.DataFrame ({
            'close':[2000,2050]
        },index=pd.date_range ('2024-01-01',periods=2))

    def test_resample_data (self):
        """Test du rééchantillonnage des données"""
        # Création de données quotidiennes
        df=pd.DataFrame ({
            'open':[100] * 10,
            'high':[105] * 10,
            'low':[95] * 10,
            'close':[102] * 10,
            'volume':[1000] * 10
        },index=pd.date_range ('2024-01-01',periods=10))

        # Test rééchantillonnage hebdomadaire
        resampled=self.manager._resample_data (df,'W-MON')
        self.assertLess (len (resampled),len (df))
        self.assertTrue (all (col in resampled.columns for col in df.columns))

    def test_clear_cache (self):
        """Test du nettoyage du cache"""
        # Ajout de données au cache
        self.manager.data_cache['test_key']=pd.DataFrame ()
        self.assertEqual (len (self.manager.data_cache),1)

        # Nettoyage du cache
        self.manager.clear_cache ()
        self.assertEqual (len (self.manager.data_cache),0)

    @patch ('qaaf.data.YFinanceSource.fetch_data')
    def test_get_qaaf_data (self,mock_fetch):
        """Test de la préparation des données QAAF"""
        mock_fetch.side_effect=[self.btc_data,self.paxg_data]

        df=self.manager.get_qaaf_data ('2024-01-01','2024-01-02')

        # Vérifications
        self.assertIsInstance (df,pd.DataFrame)
        self.assertTrue ('BTC' in df.columns)
        self.assertTrue ('PAXG/BTC' in df.columns)
        self.assertEqual (len (df),2)

        # Vérification du ratio calculé
        expected_ratio=self.paxg_data['close'] / self.btc_data['close']
        pd.testing.assert_series_equal (df['PAXG/BTC'],expected_ratio)


def run_tests ():
    """Exécute la suite de tests"""
    unittest.main (argv=[''],verbosity=2,exit=False)


if __name__ == '__main__':
    run_tests ()

# docs/README.md
"""
QAAF - Module de Données
=======================

Le module de données QAAF fournit une interface unifiée pour :
- Récupérer les données de marché de différentes sources
- Standardiser et valider les données
- Préparer les données pour l'analyse QAAF

Installation
-----------
```bash
pip install -r requirements.txt
```

Structure
---------
- `data/` : Module principal de gestion des données
  - `__init__.py` : Classes de base et interfaces
  - `sources/` : Implémentations des sources de données
- `tests/` : Tests unitaires et d'intégration
- `docs/` : Documentation détaillée

Utilisation
----------
```python
from qaaf.data import DataManager

# Initialisation
data_manager = DataManager()

# Récupération des données
data = data_manager.get_qaaf_data(
    start_date='2020-01-01',
    end_date='2024-02-17'
)

print(data.head())
```

Format des Données
----------------
Les données sont retournées au format suivant :
- Index : DatetimeIndex (fréquence hebdomadaire)
- Colonnes :
  - BTC : Prix BTC-USD
  - PAXG/BTC : Ratio PAXG/BTC

Tests
-----
Pour exécuter les tests :
```bash
python -m unittest tests.test_qaaf_data
```
"""