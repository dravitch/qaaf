import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch,MagicMock

from qaaf.market.phase_analyzer import MarketPhaseAnalyzer


class TestMarketPhaseAnalyzer (unittest.TestCase):
    def setUp (self):
        self.analyzer=MarketPhaseAnalyzer (
            short_window=5,
            long_window=10,
            volatility_window=5
        )

        # Créer des données de test
        # Une série temporelle simple allant de bullish à bearish puis consolidation
        dates=pd.date_range (start='2023-01-01',periods=40)

        # Génération d'une tendance haussière puis baissière
        prices=np.array ([
            # Tendance haussière
            *np.linspace (100,200,15),
            # Tendance baissière
            *np.linspace (200,100,15),
            # Consolidation
            *np.linspace (100,110,10)
        ])

        # Ajouter un peu de bruit
        np.random.seed (42)
        noise=np.random.normal (0,5,len (prices))
        prices+=noise

        # Créer un DataFrame comme celui retourné par l'API
        self.btc_data=pd.DataFrame ({
            'open':prices - 2,
            'high':prices + 5,
            'low':prices - 5,
            'close':prices,
            'volume':np.random.randint (1000,2000,len (prices))
        },index=dates)

    def test_identify_market_phases (self):
        # Identifier les phases de marché
        phases=self.analyzer.identify_market_phases (self.btc_data)

        # Vérifier que nous avons une phase pour chaque point de données
        self.assertEqual (len (phases),len (self.btc_data))

        # Vérifier que toutes les phases sont valides (dans notre liste de phases attendues)
        expected_phases=[
            'bullish_low_vol','bullish_high_vol',
            'bearish_low_vol','bearish_high_vol',
            'consolidation_low_vol','consolidation_high_vol'
        ]

        for phase in phases.unique ():
            self.assertIn (phase,expected_phases)

        # Vérifier que nous détectons au moins une phase haussière au début
        self.assertTrue (any (p.startswith ('bullish') for p in phases[:15]))

        # Vérifier que nous détectons au moins une phase baissière au milieu
        self.assertTrue (any (p.startswith ('bearish') for p in phases[15:30]))

        # Vérifier que nous détectons au moins une phase de consolidation à la fin
        self.assertTrue (any (p.startswith ('consolidation') for p in phases[30:]))

    def test_analyze_metrics_by_phase (self):
        # Identifier les phases de marché
        phases=self.analyzer.identify_market_phases (self.btc_data)

        # Créer quelques métriques fictives pour tester
        metrics={
            'metric1':pd.Series (np.random.random (len (self.btc_data)),index=self.btc_data.index),
            'metric2':pd.Series (np.random.random (len (self.btc_data)),index=self.btc_data.index)
        }

        # Analyser les métriques par phase
        phase_analysis=self.analyzer.analyze_metrics_by_phase (metrics,phases)

        # Vérifier que l'analyse contient une entrée pour chaque phase unique
        for phase in phases.unique ():
            self.assertIn (phase,phase_analysis)

            # Vérifier que chaque phase contient des statistiques pour chaque métrique
            for metric_name in metrics:
                self.assertIn (metric_name,phase_analysis[phase])

                # Vérifier que les statistiques attendues sont présentes
                stats=phase_analysis[phase][metric_name]
                self.assertIn ('mean',stats)
                self.assertIn ('std',stats)
                self.assertIn ('min',stats)
                self.assertIn ('max',stats)
                self.assertIn ('median',stats)

    def test_window_parameters (self):
        # Tester avec différentes tailles de fenêtre
        analyzer_small=MarketPhaseAnalyzer (short_window=3,long_window=6,volatility_window=3)
        analyzer_large=MarketPhaseAnalyzer (short_window=1