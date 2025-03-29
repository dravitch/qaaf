"""
Module d'évaluation des frais de transaction pour QAAF
"""

import pandas as pd
import numpy as np
from typing import Dict,List,Optional
import logging

logger=logging.getLogger (__name__)


class TransactionFeesEvaluator:
    """
    Module d'évaluation des frais de transaction pour optimiser la fréquence de rebalancement
    """

    def __init__ (self,
                  base_fee_rate: float = 0.001,  # 0.1% par défaut (10 points de base)
                  fee_tiers: Optional[Dict[float,float]] = None,  # Niveaux de frais basés sur le volume
                  fixed_fee: float = 0.0):  # Frais fixe par transaction
        """
        Initialise l'évaluateur de frais

        Args:
            base_fee_rate: Taux de base des frais (en pourcentage)
            fee_tiers: Dictionnaire des niveaux de frais basés sur le volume
            fixed_fee: Frais fixe par transaction
        """
        self.base_fee_rate=base_fee_rate

        # Niveaux de frais par défaut si non fournis
        self.fee_tiers=fee_tiers or {
            0:base_fee_rate,  # 0.1% pour les transactions standards
            10000:0.0008,  # 0.08% pour des volumes > $10,000
            50000:0.0006,  # 0.06% pour des volumes > $50,000
            100000:0.0004  # 0.04% pour des volumes > $100,000
        }

        self.fixed_fee=fixed_fee
        self.transaction_history=[]

    def update_parameters (self,
                           base_fee_rate: Optional[float] = None,
                           fee_tiers: Optional[Dict[float,float]] = None,
                           fixed_fee: Optional[float] = None):
        """
        Met à jour les paramètres de l'évaluateur de frais

        Args:
            base_fee_rate: Taux de base des frais
            fee_tiers: Dictionnaire des niveaux de frais
            fixed_fee: Frais fixe par transaction
        """
        if base_fee_rate is not None:
            self.base_fee_rate=base_fee_rate
            # Mise à jour du taux de base dans les niveaux
            if 0 in self.fee_tiers:
                self.fee_tiers[0]=base_fee_rate

        if fee_tiers is not None:
            self.fee_tiers=fee_tiers

        if fixed_fee is not None:
            self.fixed_fee=fixed_fee

        logger.info (f"Paramètres de frais mis à jour: taux de base={self.base_fee_rate}, "
                     f"frais fixe={self.fixed_fee}")

    def calculate_fee (self,transaction_amount: float) -> float:
        """
        Calcule les frais pour une transaction donnée

        Args:
            transaction_amount: Montant de la transaction

        Returns:
            Frais de transaction
        """
        # Détermination du niveau de frais approprié
        applicable_rate=self.base_fee_rate

        # Trouver le taux applicable en fonction du volume
        tier_thresholds=sorted (self.fee_tiers.keys ())
        for threshold in reversed (tier_thresholds):
            if transaction_amount >= threshold:
                applicable_rate=self.fee_tiers[threshold]
                break

        # Calcul des frais
        percentage_fee=transaction_amount * applicable_rate
        total_fee=percentage_fee + self.fixed_fee

        return total_fee

    def record_transaction (self,
                            date: pd.Timestamp,
                            amount: float,
                            action: str) -> None:
        """
        Enregistre une transaction dans l'historique

        Args:
            date: Date de la transaction
            amount: Montant de la transaction
            action: Type d'action (ex: 'rebalance', 'buy', 'sell')
        """
        fee=self.calculate_fee (amount)

        transaction={
            'date':date,
            'amount':amount,
            'action':action,
            'fee':fee,
            'fee_rate':fee / amount if amount > 0 else 0
        }

        self.transaction_history.append (transaction)

    def get_total_fees (self) -> float:
        """
        Calcule le total des frais payés

        Returns:
            Total des frais
        """
        return sum (t['fee'] for t in self.transaction_history)

    def get_fees_by_period (self,period: str = 'M') -> pd.Series:
        """
        Calcule les frais par période

        Args:
            period: Période d'agrégation ('D'=jour, 'W'=semaine, 'M'=mois)

        Returns:
            Série des frais par période
        """
        if not self.transaction_history:
            return pd.Series ()

        # Conversion en DataFrame
        df=pd.DataFrame (self.transaction_history)

        # Groupement par période
        grouped=df.set_index ('date')['fee'].resample (period).sum ()

        return grouped