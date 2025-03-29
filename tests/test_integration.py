import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from unittest.mock import patch, MagicMock

# Import des modules QAAF
from qaaf.data.data_manager import DataManager, YFinanceSource
from qaaf.metrics.calculator import MetricsCalculator
from qaaf.market.phase_analyzer import MarketPhaseAnalyzer
from qaaf.allocation.adaptive_allocator import AdaptiveAllocator
from qaaf.transaction.fees_evaluator import TransactionFeesEvaluator
from qaaf.core.qaaf_core import QAAFCore


class TestQAAFIntegration (unittest.TestCase):
    def setUp (self):
        """Préparation des données et des composants"""
        # Créer des données synthétiques
        self.dates=pd.date_range (start='2020-01-01',periods=100)

        # BTC data
        btc_prices=np.linspace (10000,20000,100) + np.random.normal (0,500,100)
        self.btc_data=pd.DataFrame ({
            'open':btc_prices,
            'high':btc_prices * 1.01,
            'low':btc_prices * 0.99,
            'close':btc_prices,
            'volume':np.random.normal (1000,100,100)
        },index=self.dates)

        # PAXG data
        paxg_prices=np.linspace (1500,2000,100) + np.random.normal (0,50,100)
        self.paxg_data=pd.DataFrame ({
            'open':paxg_prices,
            'high':paxg_prices * 1.005,
            'low':paxg_prices * 0.995,
            'close':paxg_prices,
            'volume':np.random.normal (500,50,100)
        },index=self.dates)

        # PAXG/BTC data
        paxg_btc_ratio=paxg_prices / btc_prices
        self.paxg_btc_data=pd.DataFrame ({
            'open':self.paxg_data['open'] / self.btc_data['open'],
            'high':self.paxg_data['high'] / self.btc_data['high'],
            'low':self.paxg_data['low'] / self.btc_data['low'],
            'close':paxg_btc_ratio,
            'volume':self.paxg_data['volume']
        },index=self.dates)

        self.test_data={
            'BTC':self.btc_data,
            'PAXG':self.paxg_data,
            'PAXG/BTC':self.paxg_btc_data
        }

        # Initialisation du calculateur
        self.calculator=MetricsCalculator (
            volatility_window=20,
            spectral_window=30,
            min_periods=10
        )

        # Création d'un mock du DataManager
        with patch ('qaaf.core.qaaf_core.DataManager') as mock_data_manager:
            mock_data_manager_instance=MagicMock ()
            mock_data_manager_instance.prepare_qaaf_data.return_value=self.test_data
            mock_data_manager.return_value=mock_data_manager_instance

            # Initialisation de QAAFCore
            self.qaaf=QAAFCore (
                initial_capital=10000.0,
                trading_costs=0.001,
                start_date='2020-01-01',
                end_date='2020-04-09'  # Approximativement 100 jours après le 1er janvier
            )

            # Chargement des données (remplacées par notre mock)
            self.qaaf.load_data ()

    def test_basic_workflow (self):
        """Test du flux de travail de base"""
        # 1. Analyse des phases de marché
        self.qaaf.analyze_market_phases ()
        self.assertIsNotNone (self.qaaf.market_phases)

        # 2. Calcul des métriques
        self.qaaf.calculate_metrics ()
        self.assertIsNotNone (self.qaaf.metrics)
        self.assertEqual (len (self.qaaf.metrics),4)  # 4 métriques de base

        # 3. Calcul du score composite
        weights={
            'vol_ratio':0.3,
            'bound_coherence':0.3,
            'alpha_stability':0.2,
            'spectral_score':0.2
        }
        self.qaaf.calculate_composite_score (weights)
        self.assertIsNotNone (self.qaaf.composite_score)

        # 4. Calcul des allocations adaptatives
        self.qaaf.calculate_adaptive_allocations ()
        self.assertIsNotNone (self.qaaf.allocations)

        # 5. Exécution du backtest
        self.qaaf.run_backtest ()
        self.assertIsNotNone (self.qaaf.results)
        self.assertIsNotNone (self.qaaf.performance)

        # Vérification des résultats de base
        self.assertIn ('metrics',self.qaaf.results)
        self.assertIn ('total_return',self.qaaf.results['metrics'])

    def test_optimization (self):
        """Test de l'optimisation des métriques"""
        # Test de l'optimisation (si implémentée)
        pass


class TestIntegration (unittest.TestCase):
    def setUp (self):
        # Générer des données synthétiques pour les tests
        dates=pd.date_range (start='2023-01-01',periods=100)

        # Tendance haussière puis baissière pour BTC
        btc_prices=np.concatenate ([
            np.linspace (20000,60000,50),  # Hausse
            np.linspace (60000,30000,50)  # Baisse
        ])

        # Tendance inverse puis similaire pour PAXG
        paxg_prices=np.concatenate ([
            np.linspace (2000,1800,50),  # Baisse
            np.linspace (1800,2200,50)  # Hausse
        ])

        # Ajouter du bruit
        np.random.seed (42)
        btc_noise=np.random.normal (0,1000,100)
        paxg_noise=np.random.normal (0,50,100)

        btc_prices+=btc_noise
        paxg_prices+=paxg_noise

        # Créer les DataFrames
        self.btc_data=pd.DataFrame ({
            'open':btc_prices * 0.99,
            'high':btc_prices * 1.02,
            'low':btc_prices * 0.98,
            'close':btc_prices,
            'volume':np.random.randint (1000,5000,100)
        },index=dates)

        self.paxg_data=pd.DataFrame ({
            'open':paxg_prices * 0.99,
            'high':paxg_prices * 1.01,
            'low':paxg_prices * 0.98,
            'close':paxg_prices,
            'volume':np.random.randint (500,2000,100)
        },index=dates)

        # Calculer le ratio PAXG/BTC
        paxg_btc_ratio=paxg_prices / btc_prices

        self.paxg_btc_data=pd.DataFrame ({
            'open':(paxg_prices * 0.99) / (btc_prices * 0.99),
            'high':(paxg_prices * 1.01) / (btc_prices * 1.02),
            'low':(paxg_prices * 0.98) / (btc_prices * 0.98),
            'close':paxg_btc_ratio,
            'volume':np.random.randint (300,1000,100)
        },index=dates)

        # Créer un dictionnaire de données comme celui retourné par DataManager
        self.test_data={
            'BTC':self.btc_data,
            'PAXG':self.paxg_data,
            'PAXG/BTC':self.paxg_btc_data
        }

        # Patch DataManager pour qu'il retourne nos données de test
        self.data_manager_patch=patch ('qaaf.core.qaaf_core.DataManager')
        self.mock_data_manager=self.data_manager_patch.start ()

        mock_dm_instance=MagicMock ()
        mock_dm_instance.prepare_qaaf_data.return_value=self.test_data
        self.mock_data_manager.return_value=mock_dm_instance

    def tearDown (self):
        self.data_manager_patch.stop ()

    def test_full_pipeline_integration (self):
        """Test de l'intégration complète du pipeline QAAF."""
        # Initialiser QAAFCore
        qaaf=QAAFCore (
            initial_capital=10000.0,
            trading_costs=0.001,
            start_date='2023-01-01',
            end_date='2023-04-10'
        )

        # Charger les données (notre mock sera utilisé)
        qaaf.load_data ()

if __name__ == '__main__':
    unittest.main ()