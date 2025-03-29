import unittest
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from unittest.mock import patch,MagicMock

from qaaf.transaction.fees_evaluator import TransactionFeesEvaluator


class TestTransactionFeesEvaluator (unittest.TestCase):
    def setUp (self):
        self.evaluator=TransactionFeesEvaluator (base_fee_rate=0.001)

        # Créer quelques dates pour les tests
        self.dates=pd.date_range (start='2023-01-01',periods=10)

        # Créer des séries pour les tests d'optimisation
        self.portfolio_values={
            'strategy1':pd.Series (
                np.linspace (100000,150000,10),
                index=self.dates
            )
        }

        self.allocation_series={
            'strategy1':pd.Series (
                np.linspace (0.3,0.7,10),
                index=self.dates
            )
        }

    def test_calculate_fee (self):
        # Test avec les taux par défaut
        self.assertEqual (self.evaluator.calculate_fee (1000),1.0)
        self.assertEqual (self.evaluator.calculate_fee (5000),5.0)

        # Test avec des taux personnalisés
        evaluator_custom=TransactionFeesEvaluator (
            base_fee_rate=0.002,
            fee_tiers={
                0:0.002,
                5000:0.001,
                20000:0.0005
            }
        )

        self.assertEqual (evaluator_custom.calculate_fee (1000),2.0)
        self.assertEqual (evaluator_custom.calculate_fee (10000),10.0)  # 10000 * 0.001
        self.assertEqual (evaluator_custom.calculate_fee (50000),25.0)  # 50000 * 0.0005

    def test_record_transaction (self):
        # Enregistrer quelques transactions
        self.evaluator.record_transaction (self.dates[0],1000,'buy')
        self.evaluator.record_transaction (self.dates[1],2000,'sell')

        # Vérifier que les transactions ont été enregistrées
        self.assertEqual (len (self.evaluator.transaction_history),2)

        # Vérifier les détails de la première transaction
        first_transaction=self.evaluator.transaction_history[0]
        self.assertEqual (first_transaction['date'],self.dates[0])
        self.assertEqual (first_transaction['amount'],1000)
        self.assertEqual (first_transaction['action'],'buy')
        self.assertEqual (first_transaction['fee'],1.0)

        # Vérifier les frais de la deuxième transaction
        second_transaction=self.evaluator.transaction_history[1]
        self.assertEqual (second_transaction['fee'],2.0)

    def test_get_total_fees (self):
        # Sans transactions
        self.assertEqual (self.evaluator.get_total_fees (),0)

        # Avec des transactions
        self.evaluator.record_transaction (self.dates[0],1000,'buy')
        self.evaluator.record_transaction (self.dates[1],2000,'sell')

        expected_total=1.0 + 2.0
        self.assertEqual (self.evaluator.get_total_fees (),expected_total)

    def test_get_fees_by_period (self):
        # Enregistrer des transactions à différentes dates
        self.evaluator.record_transaction (pd.Timestamp ('2023-01-01'),1000,'buy')
        self.evaluator.record_transaction (pd.Timestamp ('2023-01-02'),2000,'sell')
        self.evaluator.record_transaction (pd.Timestamp ('2023-02-01'),3000,'buy')

        # Obtenir les frais par mois
        fees_by_month=self.evaluator.get_fees_by_period ('M')

        # Vérifier que nous avons des frais pour janvier et février
        self.assertEqual (len (fees_by_month),2)

        # Vérifier les montants
        january_fees=fees_by_month.loc['2023-01-31']
        february_fees=fees_by_month.loc['2023-02-28']

        self.assertEqual (january_fees,1.0 + 2.0)
        self.assertEqual (february_fees,3.0)

    def test_optimize_rebalance_frequency (self):
        # Tester l'optimisation avec différents seuils
        results=self.evaluator.optimize_rebalance_frequency (
            self.portfolio_values,
            self.allocation_series,
            threshold_range=[0.01,0.05,0.1]
        )

        # Vérifier que nous avons des résultats pour chaque seuil
        self.assertEqual (len (results),3)

        # Vérifier que chaque résultat contient les métriques attendues
        for threshold,metrics in results.items ():
            self.assertIn ('total_fees',metrics)
            self.assertIn ('transaction_count',metrics)
            self.assertIn ('average_fee',metrics)
            self.assertIn ('fee_drag',metrics)
            self.assertIn ('combined_score',metrics)

            # Vérifier que le nombre de transactions diminue avec l'augmentation du seuil
            if threshold > 0.01:
                self.assertLessEqual (
                    metrics['transaction_count'],
                    results[0.01]['transaction_count']
                )

    def test_calculate_combined_score (self):
        # Tester la fonction de score combiné
        portfolio_values={
            'strategy1':pd.Series ([100000,150000])  # 50% de rendement
        }

        # Avec un fee_drag de 1%
        score1=self.evaluator._calculate_combined_score (portfolio_values,1.0)
        # Avec un fee_drag de 2%
        score2=self.evaluator._calculate_combined_score (portfolio_values,2.0)

        # Le score devrait être inférieur avec un fee_drag plus élevé
        self.assertLess (score2,score1)

        # Calcul manuel du score
        expected_score1=50 - (1.0 * 2)  # Rendement - (fee_drag * 2)
        self.assertEqual (score1,expected_score1)


if __name__ == '__main__':
    unittest.main ()