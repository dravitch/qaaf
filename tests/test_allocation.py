import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch,MagicMock

from qaaf.allocation.adaptive_allocator import AdaptiveAllocator


class TestAdaptiveAllocator (unittest.TestCase):
    def setUp (self):
        self.allocator=AdaptiveAllocator (
            min_btc_allocation=0.3,
            max_btc_allocation=0.7,
            neutral_allocation=0.5,
            sensitivity=1.0
        )

        # Créer des données de test
        dates=pd.date_range (start='2023-01-01',periods=100)

        # Score composite simulé - oscillant entre positif et négatif
        self.composite_score=pd.Series (
            np.sin (np.linspace (0,6 * np.pi,100)),
            index=dates
        )

        # Phases de marché simulées
        phases=[]
        for i in range (100):
            if i < 30:
                phases.append ('bullish_low_vol')
            elif i < 60:
                phases.append ('bearish_high_vol')
            else:
                phases.append ('consolidation_low_vol')

        self.market_phases=pd.Series (phases,index=dates)

    def test_calculate_adaptive_allocation (self):
        # Calculer les allocations adaptatives
        allocations=self.allocator.calculate_adaptive_allocation (
            self.composite_score,
            self.market_phases
        )

        # Vérifier que nous avons une allocation pour chaque point de données
        self.assertEqual (len (allocations),len (self.composite_score))

        # Vérifier que toutes les allocations sont dans les limites
        self.assertTrue (all (allocations >= self.allocator.min_btc_allocation))
        self.assertTrue (all (allocations <= self.allocator.max_btc_allocation))

        # Vérifier que les scores positifs élevés correspondent à des allocations plus élevées
        # et les scores négatifs élevés à des allocations plus faibles
        highest_scores=self.composite_score.nlargest (10).index
        lowest_scores=self.composite_score.nsmallest (10).index

        avg_high_allocation=allocations[highest_scores].mean ()
        avg_low_allocation=allocations[lowest_scores].mean ()

        self.assertGreater (avg_high_allocation,avg_low_allocation)

    def test_detect_intensity_peaks (self):
        # Détecter les pics d'intensité
        peaks,troughs,deviation=self.allocator.detect_intensity_peaks (
            self.composite_score,
            self.market_phases
        )

        # Vérifier que nous avons une valeur pour chaque point de données
        self.assertEqual (len (peaks),len (self.composite_score))
        self.assertEqual (len (troughs),len (self.composite_score))
        self.assertEqual (len (deviation),len (self.composite_score))

        # Vérifier que peaks et troughs sont des booléens
        self.assertTrue (all (isinstance (p,bool) for p in peaks))
        self.assertTrue (all (isinstance (t,bool) for t in troughs))

        # Vérifier que la somme des pics et creux est raisonnable
        # (pas trop de signaux, pas trop peu)
        total_signals=peaks.sum () + troughs.sum ()
        self.assertGreater (total_signals,0)
        self.assertLess (total_signals,len (self.composite_score) * 0.5)  # Pas plus de 50% de signaux

    def test_update_parameters (self):
        # Paramètres initiaux
        initial_min=self.allocator.min_btc_allocation
        initial_max=self.allocator.max_btc_allocation
        initial_neutral=self.allocator.neutral_allocation
        initial_sensitivity=self.allocator.sensitivity

        # Mettre à jour les paramètres
        new_min=0.2
        new_max=0.8
        new_neutral=0.4
        new_sensitivity=1.2
        new_observation_period=5

        self.allocator.update_parameters (
            min_btc_allocation=new_min,
            max_btc_allocation=new_max,
            neutral_allocation=new_neutral,
            sensitivity=new_sensitivity,
            observation_period=new_observation_period
        )

        # Vérifier que les paramètres ont été mis à jour
        self.assertEqual (self.allocator.min_btc_allocation,new_min)
        self.assertEqual (self.allocator.max_btc_allocation,new_max)
        self.assertEqual (self.allocator.neutral_allocation,new_neutral)
        self.assertEqual (self.allocator.sensitivity,new_sensitivity)
        self.assertEqual (self.allocator.observation_period,new_observation_period)

        # Mise à jour partielle
        self.allocator.update_parameters (min_btc_allocation=0.1)
        self.assertEqual (self.allocator.min_btc_allocation,0.1)
        self.assertEqual (self.allocator.max_btc_allocation,new_max)  # Inchangé

    def test_phase_sensitivity (self):
        # Tester que différentes phases de marché produisent des allocations différentes

        # Créer des données de test avec la même tendance mais des phases différentes
        dates=pd.date_range (start='2023-01-01',periods=30)
        score=pd.Series (np.linspace (-1,1,30),index=dates)

        # Trois phases différentes
        bullish_phases=pd.Series (['bullish_low_vol'] * 30,index=dates)
        bearish_phases=pd.Series (['bearish_high_vol'] * 30,index=dates)
        consolidation_phases=pd.Series (['consolidation_low_vol'] * 30,index=dates)

        # Calculer les allocations pour chaque phase
        bullish_allocations=self.allocator.calculate_adaptive_allocation (score,bullish_phases)
        bearish_allocations=self.allocator.calculate_adaptive_allocation (score,bearish_phases)
        consolidation_allocations=self.allocator.calculate_adaptive_allocation (score,consolidation_phases)

        # Les allocations devraient être différentes selon la phase
        # Par exemple, les phases bearish_high_vol devraient produire des réactions plus fortes
        # On peut vérifier en comparant les écart-types
        self.assertNotEqual (bullish_allocations.std (),bearish_allocations.std ())
        self.assertNotEqual (bullish_allocations.std (),consolidation_allocations.std ())
        self.assertNotEqual (bearish_allocations.std (),consolidation_allocations.std ())


if __name__ == '__main__':
    unittest.main ()