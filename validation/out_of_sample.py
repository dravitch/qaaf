"""
Module de validation out-of-sample pour QAAF.
Permet d'évaluer la robustesse des paramètres optimisés sur des données non utilisées
pour l'optimisation.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict,List,Tuple,Any,Optional,Callable
from datetime import datetime

# Configuration du logging
logger=logging.getLogger (__name__)


class OutOfSampleValidator:
    """
    Validateur out-of-sample pour QAAF.
    Évalue la performance des paramètres optimisés sur des données hors échantillon.
    """

    def __init__ (self,
                  data: Dict[str,pd.DataFrame],
                  backtest_function: Callable,
                  metric_keys: List[str] = None):
        """
        Initialise le validateur out-of-sample.

        Args:
            data: Dictionnaire des données pour la validation (BTC, PAXG, PAXG/BTC)
            backtest_function: Fonction de backtest prenant des données et des paramètres
            metric_keys: Liste des métriques à analyser pour la validation
        """
        self.data=data
        self.backtest_function=backtest_function
        self.metric_keys=metric_keys or ['total_return','max_drawdown','sharpe_ratio','volatility']
        self.results={}

    def split_data (self,
                    train_ratio: float = 0.7,
                    validation_ratio: float = 0.0,
                    test_ratio: float = 0.3) -> Dict[str,Dict[str,pd.DataFrame]]:
        """
        Divise les données en ensembles d'entraînement, validation et test.

        Args:
            train_ratio: Proportion des données pour l'entraînement
            validation_ratio: Proportion des données pour la validation (0 pour n'utiliser que train/test)
            test_ratio: Proportion des données pour le test

        Returns:
            Dictionnaire des ensembles de données
        """
        # Vérification que les ratios somment à 1
        total_ratio=train_ratio + validation_ratio + test_ratio
        if abs (total_ratio - 1.0) > 1e-10:
            logger.warning (f"Les ratios somment à {total_ratio}, normalisation appliquée")
            train_ratio/=total_ratio
            validation_ratio/=total_ratio
            test_ratio/=total_ratio

        # Récupération des dates communes
        common_dates=self._get_common_dates ()

        # Calcul des indices pour la division
        n_total=len (common_dates)
        n_train=int (n_total * train_ratio)
        n_validation=int (n_total * validation_ratio)

        train_end_idx=n_train - 1
        validation_end_idx=n_train + n_validation - 1

        train_end_date=common_dates[train_end_idx]
        validation_end_date=common_dates[validation_end_idx] if validation_ratio > 0 else train_end_date

        # Division des données
        result={}

        # Ensemble d'entraînement
        train_data={}
        for asset,df in self.data.items ():
            train_mask=df.index <= train_end_date
            train_data[asset]=df[train_mask].copy ()
        result['train']=train_data

        # Ensemble de validation (si demandé)
        if validation_ratio > 0:
            validation_data={}
            for asset,df in self.data.items ():
                validation_mask=(df.index > train_end_date) & (df.index <= validation_end_date)
                validation_data[asset]=df[validation_mask].copy ()
            result['validation']=validation_data

        # Ensemble de test
        test_data={}
        for asset,df in self.data.items ():
            test_mask=df.index > validation_end_date
            test_data[asset]=df[test_mask].copy ()
        result['test']=test_data

        # Stockage des résultats
        self.data_splits=result

        # Logging
        logger.info (f"Division des données: Train jusqu'au {train_end_date.strftime ('%Y-%m-%d')}")
        if validation_ratio > 0:
            logger.info (
                f"Validation du {train_end_date.strftime ('%Y-%m-%d')} au {validation_end_date.strftime ('%Y-%m-%d')}")
        logger.info (f"Test du {validation_end_date.strftime ('%Y-%m-%d')} à la fin")

        return result

    def _get_common_dates (self) -> List[pd.Timestamp]:
        """
        Récupère les dates communes à tous les DataFrames.

        Returns:
            Liste des dates communes
        """
        if not self.data:
            return []

        # Initialisation avec les dates du premier DataFrame
        first_key=next (iter (self.data))
        common_dates=set (self.data[first_key].index)

        # Intersection avec les dates des autres DataFrames
        for asset,df in self.data.items ():
            if asset != first_key:
                common_dates&=set (df.index)

        return sorted (list (common_dates))

    def run_validation (self,
                        train_params: Dict[str,Any],
                        train_ratio: float = 0.7,
                        test_ratio: float = 0.3) -> Dict:
        """
        Exécute la validation out-of-sample avec des paramètres donnés.

        Args:
            train_params: Paramètres optimisés sur les données d'entraînement
            train_ratio: Proportion des données pour l'entraînement
            test_ratio: Proportion des données pour le test

        Returns:
            Résultats de la validation
        """
        # Division des données
        self.split_data (train_ratio=train_ratio,test_ratio=test_ratio)

        # Exécution du backtest sur les données d'entraînement
        logger.info ("Exécution du backtest sur les données d'entraînement...")
        train_results=self.backtest_function (self.data_splits['train'],train_params)

        # Exécution du backtest sur les données de test
        logger.info ("Exécution du backtest sur les données de test...")
        test_results=self.backtest_function (self.data_splits['test'],train_params)

        # Analyse des résultats
        validation_analysis=self.analyze_validation_results (train_results,test_results)

        # Stockage des résultats
        self.results={
            'train':train_results,
            'test':test_results,
            'analysis':validation_analysis
        }

        return self.results

    def analyze_validation_results (self,
                                    train_results: Dict,
                                    test_results: Dict) -> Dict:
        """
        Analyse les résultats de la validation out-of-sample.

        Args:
            train_results: Résultats sur les données d'entraînement
            test_results: Résultats sur les données de test

        Returns:
            Analyse des résultats
        """
        analysis={}

        # Pour chaque métrique d'intérêt
        for metric in self.metric_keys:
            if metric not in train_results or metric not in test_results:
                continue

            train_value=train_results[metric]
            test_value=test_results[metric]

            # Calcul du ratio test/train ou train/test selon la métrique
            if metric == 'max_drawdown':
                # Pour le drawdown (valeur négative), le ratio est inversé
                ratio=train_value / test_value if test_value != 0 else float ('inf')
            else:
                ratio=test_value / train_value if train_value != 0 else float ('inf')

            # Stockage des résultats
            analysis[metric]={
                'train':train_value,
                'test':test_value,
                'ratio':ratio,
                'consistency':self._evaluate_consistency (ratio,metric)
            }

        # Score global de robustesse
        robustness_scores=[
            analysis[m]['consistency']
            for m in analysis.keys ()
            if 'consistency' in analysis[m]
        ]

        analysis['overall_robustness']=np.mean (robustness_scores) if robustness_scores else 0.0

        return analysis

    def _evaluate_consistency (self,ratio: float,metric: str) -> float:
        """
        Évalue la consistance entre les résultats train et test.

        Args:
            ratio: Ratio des valeurs test/train
            metric: Nom de la métrique évaluée

        Returns:
            Score de consistance (0-1)
        """
        # Pour le rendement, volatilité, Sharpe
        if metric in ['total_return','sharpe_ratio']:
            # Idéalement ratio proche de 1 (performance similaire)
            if ratio > 1.5 or ratio < 0.5:
                return 0.0  # Très inconsistant
            elif ratio > 1.2 or ratio < 0.8:
                return 0.5  # Modérément inconsistant
            else:
                return 1.0  # Consistant

        # Pour le drawdown (déjà traité avec ratio inversé)
        elif metric == 'max_drawdown':
            if ratio > 1.5 or ratio < 0.5:
                return 0.0  # Très inconsistant
            elif ratio > 1.2 or ratio < 0.8:
                return 0.5  # Modérément inconsistant
            else:
                return 1.0  # Consistant

        # Pour d'autres métriques
        else:
            # Approche générique
            return max (0.0,min (1.0,1.0 - abs (ratio - 1.0)))

    def plot_validation_results (self) -> None:
        """
        Affiche une visualisation des résultats de validation.
        """
        if not self.results:
            logger.warning ("Aucun résultat de validation disponible à visualiser")
            return

        # Extraction des données de performance
        train_perf=self.results['train'].get ('portfolio_values',pd.Series ())
        test_perf=self.results['test'].get ('portfolio_values',pd.Series ())

        # Si les données de performances ne sont pas disponibles
        if train_perf.empty or test_perf.empty:
            logger.warning ("Données de performance insuffisantes pour la visualisation")
            return

        # Normalisation pour la comparaison
        norm_train=train_perf / train_perf.iloc[0]
        norm_test=test_perf / test_perf.iloc[0]

        # Création de la figure
        plt.figure (figsize=(15,10))

        # Plot des performances
        plt.subplot (2,1,1)
        plt.plot (norm_train.index,norm_train,'b-',label='Entraînement')
        plt.plot (norm_test.index,norm_test,'r-',label='Test')
        plt.title ('Performance Out-of-Sample')
        plt.ylabel ('Performance normalisée')
        plt.legend ()
        plt.grid (True)

        # Plot des métriques de validation
        plt.subplot (2,1,2)
        metrics=list (self.results['analysis'].keys ())
        metrics=[m for m in metrics if m != 'overall_robustness']

        if metrics:
            # Barres groupées pour train et test
            x=np.arange (len (metrics))
            width=0.35

            train_values=[self.results['analysis'][m]['train'] for m in metrics]
            test_values=[self.results['analysis'][m]['test'] for m in metrics]

            # Ajustement des valeurs pour l'affichage (drawdown est négatif)
            for i,metric in enumerate (metrics):
                if metric == 'max_drawdown':
                    train_values[i]=-train_values[i]
                    test_values[i]=-test_values[i]

            plt.bar (x - width / 2,train_values,width,label='Entraînement')
            plt.bar (x + width / 2,test_values,width,label='Test')

            plt.xlabel ('Métrique')
            plt.ylabel ('Valeur')
            plt.title ('Comparaison des métriques')
            plt.xticks (x,metrics)
            plt.legend ()
            plt.grid (True,axis='y')

            # Ajout des valeurs sur les barres
            for i,v in enumerate (train_values):
                plt.text (i - width / 2,v + 0.02,f'{v:.2f}',ha='center')
            for i,v in enumerate (test_values):
                plt.text (i + width / 2,v + 0.02,f'{v:.2f}',ha='center')

        plt.tight_layout ()
        plt.show ()

    def generate_validation_report (self) -> str:
        """
        Génère un rapport textuel des résultats de validation.

        Returns:
            Rapport formaté
        """
        if not self.results:
            return "Aucun résultat de validation disponible."

        report="# Rapport de Validation Out-of-Sample\n\n"
        report+=f"Date: {datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')}\n\n"

        # Données générales
        train_start=min (self.data_splits['train']['BTC'].index)
        train_end=max (self.data_splits['train']['BTC'].index)
        test_start=min (self.data_splits['test']['BTC'].index)
        test_end=max (self.data_splits['test']['BTC'].index)

        report+=f"## Périodes d'analyse\n"
        report+=f"- **Entraînement:** {train_start.strftime ('%Y-%m-%d')} - {train_end.strftime ('%Y-%m-%d')}\n"
        report+=f"- **Test:** {test_start.strftime ('%Y-%m-%d')} - {test_end.strftime ('%Y-%m-%d')}\n\n"

        # Résultats des métriques
        report+="## Comparaison des métriques\n\n"
        report+="| Métrique | Entraînement | Test | Ratio | Consistance |\n"
        report+="|----------|--------------|------|-------|------------|\n"

        for metric in self.metric_keys:
            if metric not in self.results['analysis']:
                continue

            analysis=self.results['analysis'][metric]
            train_val=analysis['train']
            test_val=analysis['test']
            ratio=analysis['ratio']
            consistency=analysis['consistency']

            # Formatage des valeurs
            train_str=f"{train_val:.2f}" if isinstance (train_val,(int,float)) else str (train_val)
            test_str=f"{test_val:.2f}" if isinstance (test_val,(int,float)) else str (test_val)
            ratio_str=f"{ratio:.2f}" if isinstance (ratio,(int,float)) else str (ratio)

            # Évaluation de la consistance
            if consistency >= 0.8:
                cons_str="✅ Excellente"
            elif consistency >= 0.5:
                cons_str="⚠️ Modérée"
            else:
                cons_str="❌ Faible"

            report+=f"| {metric} | {train_str} | {test_str} | {ratio_str} | {cons_str} |\n"

        # Score global de robustesse
        robustness=self.results['analysis'].get ('overall_robustness',0.0)
        report+=f"\n## Évaluation globale\n\n"
        report+=f"Score de robustesse: {robustness:.2f}/1.00\n\n"

        if robustness >= 0.8:
            report+="📊 **Analyse**: Les paramètres montrent une excellente robustesse sur les données de test.\n"
        elif robustness >= 0.5:
            report+="📊 **Analyse**: Les paramètres montrent une robustesse modérée, acceptable pour un déploiement prudent.\n"
        else:
            report+="📊 **Analyse**: Les paramètres montrent une faible robustesse, suggérant un possible surajustement aux données d'entraînement.\n"

        return report

    def validate_strategy (backtest_function: Callable,
                           data: Dict[str,pd.DataFrame],
                           params: Dict[str,Any],
                           train_ratio: float = 0.7,
                           test_ratio: float = 0.3) -> Dict:
        """
        Fonction utilitaire pour valider rapidement une stratégie.

        Args:
            backtest_function: Fonction de backtest
            data: Données de marché
            params: Paramètres à valider
            train_ratio: Proportion des données pour l'entraînement
            test_ratio: Proportion des données pour le test

        Returns:
            Résultats de validation
        """
        validator=OutOfSampleValidator (data,backtest_function)
        return validator.run_validation (params,train_ratio,test_ratio)