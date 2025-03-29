"""
Module d'allocation adaptative pour QAAF
"""

import pandas as pd
import numpy as np
from typing import Dict,List,Optional,Tuple
import logging

logger=logging.getLogger (__name__)


class AdaptiveAllocator:
    """
    Allocateur adaptatif pour QAAF

    Cette classe implémente l'allocation adaptative avec amplitude variable
    basée sur l'intensité des signaux du score composite.
    """

    def __init__ (self,
                  min_btc_allocation: float = 0.3,
                  max_btc_allocation: float = 0.7,
                  neutral_allocation: float = 0.5,
                  sensitivity: float = 1.0,
                  observation_period: int = 10):
        """
        Initialise l'allocateur adaptatif

        Args:
            min_btc_allocation: Allocation minimale en BTC
            max_btc_allocation: Allocation maximale en BTC
            neutral_allocation: Allocation neutre (point d'équilibre)
            sensitivity: Sensibilité aux signaux (1.0 = standard)
            observation_period: Jours d'observation après un signal fort
        """
        self.min_btc_allocation=min_btc_allocation
        self.max_btc_allocation=max_btc_allocation
        self.neutral_allocation=neutral_allocation
        self.sensitivity=sensitivity
        self.observation_period=observation_period

        # Variables de suivi
        self.last_signal_date=None
        self.last_allocation=neutral_allocation

    def update_parameters (self,
                           min_btc_allocation: Optional[float] = None,
                           max_btc_allocation: Optional[float] = None,
                           neutral_allocation: Optional[float] = None,
                           sensitivity: Optional[float] = None,
                           observation_period: Optional[int] = None):
        """
        Met à jour les paramètres de l'allocateur

        Args:
            min_btc_allocation: Allocation minimale en BTC
            max_btc_allocation: Allocation maximale en BTC
            neutral_allocation: Allocation neutre (point d'équilibre)
            sensitivity: Sensibilité aux signaux
            observation_period: Jours d'observation après un signal
        """
        if min_btc_allocation is not None:
            self.min_btc_allocation=min_btc_allocation

        if max_btc_allocation is not None:
            self.max_btc_allocation=max_btc_allocation

        if neutral_allocation is not None:
            self.neutral_allocation=neutral_allocation

        if sensitivity is not None:
            self.sensitivity=sensitivity

        if observation_period is not None:
            self.observation_period=observation_period

        logger.info (f"Paramètres de l'allocateur mis à jour: min={self.min_btc_allocation}, "
                     f"max={self.max_btc_allocation}, sensitivity={self.sensitivity}")

    def calculate_adaptive_allocation (self,
                                       composite_score: pd.Series,
                                       market_phases: pd.Series) -> pd.Series:
        """
        Calcule l'allocation adaptative avec amplitude variable

        Args:
            composite_score: Score composite calculé
            market_phases: Phases de marché identifiées

        Returns:
            Série temporelle des allocations BTC
        """
        # Normalisation du score composite
        normalized_score=(composite_score - composite_score.mean ()) / composite_score.std ()

        # Allocation par défaut (neutre)
        allocations=pd.Series (self.neutral_allocation,index=composite_score.index)

        # Paramètres d'amplitude par phase de marché
        amplitude_by_phase={
            'bullish_low_vol':1.0,  # Amplitude normale en phase haussière
            'bullish_high_vol':1.2,  # Amplitude ajustée pour volatilité élevée
            'bearish_low_vol':1.3,  # Plus réactive en phase baissière
            'bearish_high_vol':1.8,  # Réaction très forte en baisse volatile
            'consolidation_low_vol':0.7,  # Amplitude réduite en consolidation
            'consolidation_high_vol':0.9  # Consolidation volatile
        }

        # Paramètres par défaut si la phase n'est pas reconnue
        default_amplitude=1.0

        # Initialisation des métriques de signal
        signal_points=[]
        signal_strengths=[]

        # Calcul de l'allocation pour chaque date
        for date in composite_score.index:
            # Récupération de la phase et du score normalisé
            if date in market_phases.index:
                phase=market_phases.loc[date]
                amplitude=amplitude_by_phase.get (phase,default_amplitude)
            else:
                amplitude=default_amplitude

            score=normalized_score.loc[date]

            # Détection des signaux forts (dépassement de seuil)
            signal_threshold=1.5 * self.sensitivity

            if abs (score) > signal_threshold:
                # Signal fort détecté
                signal_points.append (date)
                signal_strengths.append (abs (score))

                # Détermination de l'amplitude adaptative
                # Plus le signal est fort, plus l'amplitude est grande
                signal_strength_factor=min (2.0,abs (score) / signal_threshold)
                adjusted_amplitude=amplitude * signal_strength_factor

                # Direction de l'allocation
                if score > 0:
                    # Signal positif = augmentation de l'allocation BTC
                    target_allocation=self.neutral_allocation + (
                                self.max_btc_allocation - self.neutral_allocation) * adjusted_amplitude
                else:
                    # Signal négatif = diminution de l'allocation BTC
                    target_allocation=self.neutral_allocation - (
                                self.neutral_allocation - self.min_btc_allocation) * adjusted_amplitude

                # Application de l'allocation avec contraintes
                allocations.loc[date]=max (self.min_btc_allocation,min (self.max_btc_allocation,target_allocation))

                # Mise à jour de l'état
                self.last_signal_date=date
                self.last_allocation=allocations.loc[date]
            else:
                # Pas de signal fort

                # Vérification de la période d'observation
                if self.last_signal_date is not None:
                    days_since_signal=(date - self.last_signal_date).days

                    if days_since_signal < self.observation_period:
                        # Pendant la période d'observation, maintien de l'allocation
                        allocations.loc[date]=self.last_allocation
                    else:
                        # Retour progressif vers l'allocation neutre
                        recovery_factor=min (1.0,(days_since_signal - self.observation_period) / 10)
                        allocations.loc[date]=self.last_allocation + (
                                    self.neutral_allocation - self.last_allocation) * recovery_factor
                else:
                    # Pas de signal récent, allocation basée sur le score actuel avec amplitude réduite
                    reduced_amplitude=amplitude * 0.5
                    allocation_change=(
                                                  self.max_btc_allocation - self.min_btc_allocation) * 0.5 * score * reduced_amplitude
                    allocations.loc[date]=self.neutral_allocation + allocation_change
                    allocations.loc[date]=max (self.min_btc_allocation,
                                               min (self.max_btc_allocation,allocations.loc[date]))

        # Affichage des statistiques des signaux
        if signal_points:
            logger.info (f"Signaux forts détectés: {len (signal_points)}")
            logger.info (f"Force moyenne des signaux: {np.mean (signal_strengths):.2f}")
        else:
            logger.info ("Aucun signal fort détecté")

        return allocations