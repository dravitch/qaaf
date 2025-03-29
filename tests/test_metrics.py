import unittest
import pandas as pd
import numpy as np
from qaaf.metrics.calculator import MetricsCalculator


class TestMetricsCalculator (unittest.TestCase):

    def setUp (self):
        """Préparation des données de test"""
        # Créer des données synthétiques pour les tests
        dates=pd.date_range (start='2020-01-01',periods=100)

        # BTC data
        btc_prices=np.linspace (10000,20000,100) + np.random.normal (0,500,100)
        self.btc_data=pd.DataFrame ({
            'open':btc_prices,
            'high':btc_prices * 1.01,
            'low':btc_prices * 0.99,
            'close':btc_prices,
            'volume':np.random.normal (1000,100,100)
        },index=dates)

        # PAXG data
        paxg_prices=np.linspace (1500,2000,100) + np.random.normal (0,50,100)
        self.paxg_data=pd.DataFrame ({
            'open':paxg_prices,
            'high':paxg_prices * 1.005,
            'low':paxg_prices * 0.995,
            'close':paxg_prices,
            'volume':np.random.normal (500,50,100)
        },index=dates)

        # PAXG/BTC data
        paxg_btc_ratio=paxg_prices / btc_prices
        self.paxg_btc_data=pd.DataFrame ({
            'open':self.paxg_data['open'] / self.btc_data['open'],
            'high':self.paxg_data['high'] / self.btc_data['high'],
            'low':self.paxg_data['low'] / self.btc_data['low'],
            'close':paxg_btc_ratio,
            'volume':self.paxg_data['volume']
        },index=dates)

        self.test_data={
            'BTC':self.btc_data,
            'PAXG':self.paxg_data,
            'PAXG/BTC':self.paxg_btc_data
        }

        # Initialisation du calculateur
        self.calculator=MetricsCalculator (
            volatility_window=20,
            spectral_window=30,
            min_periods=10,
            use_gpu=False  # Désactivé pour les tests
        )

    def test_metrics_calculation (self):
        """Test du calcul des métriques"""
        metrics=self.calculator.calculate_metrics (self.test_data)

        # Vérifier que toutes les métriques sont calculées
        self.assertIn ('vol_ratio',metrics)
        self.assertIn ('bound_coherence',metrics)
        self.assertIn ('alpha_stability',metrics)
        self.assertIn ('spectral_score',metrics)

        # Vérifier les dimensions des métriques
        for metric_name,metric_series in metrics.items ():
            # Les premières valeurs peuvent être NaN en raison de la fenêtre
            non_nan_values=metric_series.dropna ()
            self.assertTrue (len (non_nan_values) > 0,f"La métrique {metric_name} ne contient que des NaN")

    def test_volatility_ratio (self):
        """Test spécifique du ratio de volatilité"""
        # Test de l'implémentation spécifique
        btc_returns=self.btc_data['close'].pct_change ().dropna ()
        paxg_returns=self.paxg_data['close'].pct_change ().dropna ()
        paxg_btc_returns=self.paxg_btc_data['close'].pct_change ().dropna ()

        # Calcul du ratio de volatilité
        vol_ratio=self.calculator._calculate_volatility_ratio (
            paxg_btc_returns,btc_returns,paxg_returns
        )

        # Vérifications de base
        self.assertIsInstance (vol_ratio,pd.Series)
        self.assertTrue (vol_ratio.min () >= 0.1)  # Valeur minimale clippée à 0.1
        self.assertTrue (vol_ratio.max () <= 10.0)  # Valeur maximale clippée à 10.0

    def test_normalize_metrics (self):
        """Test de la normalisation des métriques"""
        # Création de métriques fictives pour tester
        metrics={
            'test_metric1':pd.Series ([1,2,3,4,5]),
            'test_metric2':pd.Series ([0,25,50,75,100])
        }

        normalized=self.calculator.normalize_metrics (metrics)

        # Vérifier que toutes les métriques sont normalisées entre 0 et 1
        for metric_name,metric_series in normalized.items ():
            self.assertGreaterEqual (metric_series.min (),0.0)
            self.assertLessEqual (metric_series.max (),1.0)


if __name__ == '__main__':
    unittest.main ()