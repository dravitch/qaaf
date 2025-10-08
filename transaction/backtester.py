import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import traceback
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


class QAAFBacktester:
    """
    Backtest des stratégies QAAF avec prise en compte des frais de transaction.
    """

    def __init__(
        self,
        initial_capital: float = 30000.0,
        fees_evaluator=None,
        rebalance_threshold: float = 0.05,
    ):
        """
        Initialise le backtester
        """
        self.initial_capital = initial_capital
        self.fees_evaluator = fees_evaluator
        self.rebalance_threshold = rebalance_threshold

        # Performance et allocation tracking
        self.performance_history = None
        self.allocation_history = None
        self.transaction_history = []
        
        # Tracking des erreurs
        self.last_error = None
        self.last_error_traceback = None

    def update_parameters(self, rebalance_threshold: Optional[float] = None):
        """
        Met à jour les paramètres du backtester
        """
        if rebalance_threshold is not None:
            self.rebalance_threshold = rebalance_threshold

        logger.info(f"Paramètres du backtester mis à jour: rebalance_threshold={self.rebalance_threshold}")

    def run_backtest(
        self,
        btc_data: pd.DataFrame,
        paxg_data: pd.DataFrame,
        allocations: pd.Series,
    ) -> Tuple[pd.Series, Dict]:
        """
        Exécute le backtest avec les allocations données

        Args:
            btc_data: DataFrame des données BTC (doit contenir colonne 'close')
            paxg_data: DataFrame des données PAXG (doit contenir colonne 'close')
            allocations: Série des allocations BTC (index temporel)

        Returns:
            Tuple contenant la série de performance du portefeuille et le dict de métriques
        """
        try:
            # Alignement des indices
            common_index = btc_data.index.intersection(paxg_data.index).intersection(allocations.index)

            if len(common_index) == 0:
                raise ValueError("common_index vide dans run_backtest: vérifier indices BTC/PAXG/allocations")

            # Extraire et aligner les séries de prix, remplir petits trous
            # CORRECTION: Utiliser ffill() et bfill() au lieu de fillna(method=...)
            btc_close = btc_data.loc[common_index, "close"].ffill().bfill()
            paxg_close = paxg_data.loc[common_index, "close"].ffill().bfill()

            # Réindexer allocations dans une nouvelle variable (ne pas écraser paramètre)
            alloc_reindexed = allocations.reindex(common_index).ffill().bfill()
            if alloc_reindexed.isna().all():
                # Fallback neutre si allocations complètement manquantes
                alloc_reindexed = pd.Series(0.5, index=common_index)

            alloc_btc = alloc_reindexed.loc[common_index]

            # Protections contre prix initiaux invalides
            if btc_close.empty or paxg_close.empty:
                raise ValueError("Series de prix vides après alignement dans run_backtest")

            if btc_close.iloc[0] == 0 or pd.isna(btc_close.iloc[0]) or paxg_close.iloc[0] == 0 or pd.isna(
                paxg_close.iloc[0]
            ):
                raise ValueError("Prix initiaux invalides dans run_backtest: vérifie btc_close/paxg_close au premier indice")

            # Initialisation du portefeuille
            portfolio_value = pd.Series(self.initial_capital, index=common_index, dtype=float)


            # Calcul des unités initiales (protéger contre division par zéro)
            first_alloc = float(alloc_btc.iloc[0]) if len(alloc_btc) > 0 else 0.5
            btc_price0 = btc_close.iloc[0]
            paxg_price0 = paxg_close.iloc[0]

            btc_units = (self.initial_capital * first_alloc) / btc_price0
            paxg_units = (self.initial_capital * (1 - first_alloc)) / paxg_price0

            if not np.isfinite(btc_units) or not np.isfinite(paxg_units):
                raise ValueError(f"Units invalides: btc_units={btc_units}, paxg_units={paxg_units}")

            # Date du dernier rééquilibrage
            last_rebalance_date = common_index[0]

            # Réinitialisation de l'historique des transactions
            self.transaction_history = []
            if self.fees_evaluator is not None:
                self.fees_evaluator.transaction_history = []

            # Enregistrement de la transaction initiale (si fees_evaluator disponible)
            initial_btc_amount = self.initial_capital * first_alloc
            initial_paxg_amount = self.initial_capital * (1 - first_alloc)
            if self.fees_evaluator is not None:
                try:
                    self.fees_evaluator.record_transaction(common_index[0], initial_btc_amount, "initial_buy_btc")
                    self.fees_evaluator.record_transaction(common_index[0], initial_paxg_amount, "initial_buy_paxg")
                    initial_fees = self.fees_evaluator.calculate_fee(initial_btc_amount) + self.fees_evaluator.calculate_fee(
                        initial_paxg_amount
                    )
                except Exception:
                    initial_fees = 0.0
            else:
                initial_fees = 0.0

            portfolio_value.iloc[0] -= initial_fees

            # Tracking des allocations réalisées
            realized_allocations = pd.Series(first_alloc, index=common_index)

            # Boucle temporelle du backtest
            for i, date in enumerate(common_index[1:], 1):
                prev_date = common_index[i - 1]

                # Valeur des actifs au jour courant
                btc_value = btc_units * btc_close.loc[date]
                paxg_value = paxg_units * paxg_close.loc[date]

                # Valeur totale avant rééquilibrage
                current_value = btc_value + paxg_value

                # Allocation actuelle réalisée
                current_btc_allocation = btc_value / current_value if current_value > 0 else 0.5
                realized_allocations.loc[date] = current_btc_allocation

                # Allocation cible fournie par la série d'allocations
                target_allocation = float(alloc_btc.loc[date]) if date in alloc_btc.index else float(first_alloc)

                # Vérification du besoin de rééquilibrage
                allocation_diff = abs(current_btc_allocation - target_allocation)
                days_since_rebalance = (date - last_rebalance_date).days

                if allocation_diff > self.rebalance_threshold or days_since_rebalance > 30:
                    # Montants cibles
                    target_btc_value = current_value * target_allocation
                    target_paxg_value = current_value * (1 - target_allocation)

                    # Ajustement nécessaire en valeur
                    btc_adjustment = target_btc_value - btc_value
                    transaction_amount = abs(btc_adjustment)

                    # Calcul des frais (si disponible)
                    if self.fees_evaluator is not None:
                        transaction_fee = self.fees_evaluator.calculate_fee(transaction_amount)
                        self.fees_evaluator.record_transaction(date, transaction_amount, "rebalance")
                    else:
                        transaction_fee = 0.0

                    # Appliquer frais puis recalculer cibles
                    current_value_after_fees = current_value - transaction_fee
                    target_btc_value = current_value_after_fees * target_allocation
                    target_paxg_value = current_value_after_fees * (1 - target_allocation)

                    # Mise à jour des unités (protéger contre division par zéro)
                    btc_units = target_btc_value / btc_close.loc[date] if btc_close.loc[date] != 0 else btc_units
                    paxg_units = target_paxg_value / paxg_close.loc[date] if paxg_close.loc[date] != 0 else paxg_units

                    last_rebalance_date = date

                    # Enregistrement
                    self.transaction_history.append(
                        {
                            "date": date,
                            "current_allocation": current_btc_allocation,
                            "target_allocation": target_allocation,
                            "portfolio_value": current_value_after_fees,
                            "transaction_amount": transaction_amount,
                            "fee": transaction_fee,
                        }
                    )

                # Mise à jour de la valeur du portefeuille
                portfolio_value.loc[date] = btc_units * btc_close.loc[date] + paxg_units * paxg_close.loc[date]

            # Calcul des métriques finales
            metrics = self.calculate_metrics(portfolio_value, realized_allocations)

            # Sauvegarde des historiques
            self.performance_history = portfolio_value
            self.allocation_history = realized_allocations

            return portfolio_value, metrics
            
        except Exception as e:
            # Capturer l'erreur complète
            self.last_error = e
            self.last_error_traceback = traceback.format_exc()
            
            # Logger l'erreur complète
            logger.error("=" * 80)
            logger.error("❌ ERREUR DANS RUN_BACKTEST")
            logger.error("=" * 80)
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            logger.error("\nTraceback complet:")
            logger.error(self.last_error_traceback)
            logger.error("=" * 80)
            
            # Afficher aussi sur stdout pour visibilité
            print("\n" + "=" * 80, file=sys.stderr)
            print("❌ ERREUR CRITIQUE DANS LE BACKTEST", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Type: {type(e).__name__}", file=sys.stderr)
            print(f"Message: {str(e)}", file=sys.stderr)
            print("\nTraceback:", file=sys.stderr)
            print(self.last_error_traceback, file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            # Re-lever l'exception
            raise

    def run_multi_threshold_test(
        self,
        btc_data: pd.DataFrame,
        paxg_data: pd.DataFrame,
        allocations: pd.Series,
        thresholds: List[float] = [0.01, 0.03, 0.05, 0.1],
    ) -> Dict:
        """
        Teste différents seuils de rebalancement pour évaluer l'impact des frais
        """
        try:
            results = {}

            for threshold in thresholds:
                logger.info(f"Test avec seuil de rebalancement: {threshold:.1%}")

                # Mise à jour du seuil
                original_threshold = self.rebalance_threshold
                self.rebalance_threshold = threshold

                # Exécution du backtest
                portfolio_value, metrics = self.run_backtest(btc_data, paxg_data, allocations)

                # Calcul des frais totaux
                total_fees = self.fees_evaluator.get_total_fees() if self.fees_evaluator is not None else 0.0
                transaction_count = len(self.fees_evaluator.transaction_history) if self.fees_evaluator is not None else 0

                # Calcul du drag de performance (protection division)
                final_val = portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 1.0
                fee_drag = total_fees / final_val * 100 if final_val != 0 else 0.0

                # Score combiné
                combined_score = metrics["total_return"] - (fee_drag * 2)

                results[threshold] = {
                    "portfolio_value": portfolio_value,
                    "metrics": metrics,
                    "total_fees": total_fees,
                    "transaction_count": transaction_count,
                    "fee_drag": fee_drag,
                    "combined_score": combined_score,
                }

                # restaurer seuil
                self.rebalance_threshold = original_threshold

            return results
            
        except Exception as e:
            # Capturer et afficher l'erreur
            self.last_error = e
            self.last_error_traceback = traceback.format_exc()
            
            logger.error("=" * 80)
            logger.error("❌ ERREUR DANS RUN_MULTI_THRESHOLD_TEST")
            logger.error("=" * 80)
            logger.error(f"Type: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            logger.error("\nTraceback:")
            logger.error(self.last_error_traceback)
            logger.error("=" * 80)
            
            print("\n" + "=" * 80, file=sys.stderr)
            print("❌ ERREUR DANS MULTI_THRESHOLD_TEST", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(self.last_error_traceback, file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            raise

    def calculate_metrics(self, portfolio_value: pd.Series, allocations: pd.Series) -> Dict:
        """
        Calcule les métriques de performance
        """
        try:
            if portfolio_value is None or len(portfolio_value) < 2:
                return {
                    "total_investment": self.initial_capital,
                    "final_value": portfolio_value.iloc[-1] if len(portfolio_value) > 0 else self.initial_capital,
                    "total_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "return_drawdown_ratio": 0.0,
                    "allocation_volatility": 0.0,
                    "total_fees": self.fees_evaluator.get_total_fees() if self.fees_evaluator is not None else 0.0,
                    "fee_drag": 0.0,
                }

            returns = portfolio_value.pct_change().dropna()
            total_return = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else 0.0

            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate if not returns.empty else 0.0
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

            cumulative_max = portfolio_value.cummax()
            drawdown = (portfolio_value - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0.0

            return_drawdown_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else total_return

            allocation_volatility = allocations.diff().abs().mean() * 100 if not allocations.empty else 0.0

            total_fees = self.fees_evaluator.get_total_fees() if self.fees_evaluator is not None else 0.0
            fee_drag = total_fees / (portfolio_value.iloc[-1] if portfolio_value.iloc[-1] != 0 else 1.0) * 100

            return {
                "total_investment": self.initial_capital,
                "final_value": portfolio_value.iloc[-1],
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "return_drawdown_ratio": return_drawdown_ratio,
                "allocation_volatility": allocation_volatility,
                "total_fees": total_fees,
                "fee_drag": fee_drag,
            }
            
        except Exception as e:
            # Capturer l'erreur dans calculate_metrics
            self.last_error = e
            self.last_error_traceback = traceback.format_exc()
            
            logger.error("=" * 80)
            logger.error("❌ ERREUR DANS CALCULATE_METRICS")
            logger.error("=" * 80)
            logger.error(f"Type: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            logger.error("\nTraceback:")
            logger.error(self.last_error_traceback)
            logger.error("=" * 80)
            
            print("\n" + "=" * 80, file=sys.stderr)
            print("❌ ERREUR DANS CALCULATE_METRICS", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(self.last_error_traceback, file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            raise

    def compare_with_benchmarks(self, metrics: Dict) -> pd.DataFrame:
        """
        Compare les résultats avec les benchmarks.
        Placeholder: retourne DataFrame vide si StaticBenchmarks non disponible.

        try:
            from qaaf.transaction.backtester import StaticBenchmarks  # attempt local reference if available
            benchmarks_df = StaticBenchmarks.format_as_dataframe()
            metrics_series = pd.Series(metrics)
            results = benchmarks_df.copy()
            results.loc["QAAF"] = metrics_series
            return results

        except Exception as e:
            logger.warning(f"StaticBenchmarks non disponible: {e}")
            # fallback: DataFrame minimal
            return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).set_index("metric")
        """

        try:
            from qaaf.transaction.static_benchmarks import StaticBenchmarks
            benchmarks_df = StaticBenchmarks.format_as_dataframe()
        except Exception:
            logger.debug("StaticBenchmarks not available, returning minimal benchmarks")
            benchmarks_df = pd.DataFrame({"metric": [], "value": []}).set_index("metric")



    
    def get_last_error_info(self) -> Dict:
        """
        Retourne les informations sur la dernière erreur
        """
        return {
            "error": self.last_error,
            "error_type": type(self.last_error).__name__ if self.last_error else None,
            "error_message": str(self.last_error) if self.last_error else None,
            "traceback": self.last_error_traceback
        }