"""
Module de tests de robustesse pour QAAF.
Permet d'évaluer la stabilité et la performance des stratégies
dans diverses conditions de marché et avec différentes configurations.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict,List,Tuple,Any,Optional,Callable
from datetime import datetime,timedelta
from tqdm import tqdm

# Configuration du logging
logger=logging.getLogger (__name__)


class RobustnessTester:
    """
    Testeur de robustesse pour QAAF.
    Évalue la stabilité et la performance des stratégies dans diverses conditions.
    """

    def __init__ (self,
                  data: Dict[str,pd.DataFrame],
                  backtest_function: Callable,
                  metric_keys: List[str] = None):
        """
        Initialise le testeur de robustesse.

        Args:
            data: Dictionnaire des données pour les tests (BTC, PAXG, PAXG/BTC)
            backtest_function: Fonction de backtest prenant des données et des paramètres
            metric_keys: Liste des métriques à analyser pour les tests
        """
        self.data=data
        self.backtest_function=backtest_function
        self.metric_keys=metric_keys or ['total_return','max_drawdown','sharpe_ratio','volatility']
        self.results={}

    def run_parameter_sensitivity (self,
                                   base_params: Dict[str,Any],
                                   sensitivity_range: Dict[str,List[Any]] = None,
                                   n_variations: int = 5) -> Dict:
        """
        Effectue une analyse de sensibilité sur les paramètres.

        Args:
            base_params: Paramètres de base pour la stratégie
            sensitivity_range: Plages de variation pour chaque paramètre
            n_variations: Nombre de variations à tester par paramètre

        Returns:
            Résultats de l'analyse de sensibilité
        """
        logger.info ("Analyse de sensibilité des paramètres...")

        # Si aucune plage de sensibilité n'est fournie, en créer une par défaut
        if sensitivity_range is None:
            sensitivity_range=self._create_default_sensitivity_range (base_params)

        # Création des variations de paramètres
        parameter_sets=self._generate_parameter_variations (base_params,sensitivity_range,n_variations)

        # Test de chaque ensemble de paramètres
        sensitivity_results={}

        for param_name,param_values in sensitivity_range.items ():
            sensitivity_results[param_name]=[]

            for param_value in param_values:
                # Création d'un ensemble de paramètres avec cette variation
                test_params=base_params.copy ()
                test_params[param_name]=param_value

                # Exécution du backtest
                try:
                    results=self.backtest_function (self.data,test_params)

                    # Extraction des métriques d'intérêt
                    metrics={
                        k:results[k] for k in self.metric_keys if k in results
                    }

                    sensitivity_results[param_name].append ({
                        'param_value':param_value,
                        'metrics':metrics
                    })
                except Exception as e:
                    logger.warning (f"Erreur lors du test avec {param_name}={param_value}: {str (e)}")

        # Calcul des mesures de sensibilité
        parameter_sensitivity=self._analyze_parameter_sensitivity (sensitivity_results)

        # Stockage des résultats
        self.results['sensitivity']={
            'raw_results':sensitivity_results,
            'parameter_sensitivity':parameter_sensitivity
        }

        return self.results['sensitivity']

    def _create_default_sensitivity_range (self,
                                           base_params: Dict[str,Any]) -> Dict[str,List[Any]]:
        """
        Crée une plage de sensibilité par défaut pour les paramètres.

        Args:
            base_params: Paramètres de base

        Returns:
            Plages de variation pour chaque paramètre
        """
        sensitivity_range={}

        for param_name,param_value in base_params.items ():
            # Pour les paramètres numériques
            if isinstance (param_value,(int,float)):
                # Valeurs entières
                if isinstance (param_value,int):
                    if param_name in ['volatility_window','spectral_window','min_periods','observation_period']:
                        # Fenêtres et périodes: variations plus importantes
                        min_val=max (5,int (param_value * 0.5))
                        max_val=int (param_value * 1.5)
                        sensitivity_range[param_name]=list (
                            range (min_val,max_val + 1,max (1,(max_val - min_val) // 5)))
                    else:
                        # Autres entiers: variations plus petites
                        min_val=max (1,param_value - 2)
                        max_val=param_value + 2
                        sensitivity_range[param_name]=list (range (min_val,max_val + 1))

                # Valeurs flottantes
                else:
                    if param_name in ['vol_ratio_weight','bound_coherence_weight','alpha_stability_weight',
                                      'spectral_score_weight']:
                        # Poids des métriques (0-1)
                        sensitivity_range[param_name]=[
                            round (max (0.0,min (1.0,param_value * (1 + factor))),2)
                            for factor in [-0.4,-0.2,0,0.2,0.4]
                        ]
                    elif param_name in ['min_btc_allocation','max_btc_allocation']:
                        # Allocations (0-1)
                        if param_name == 'min_btc_allocation':
                            min_val=max (0.1,param_value - 0.1)
                            max_val=min (0.5,param_value + 0.1)
                        else:  # max_btc_allocation
                            min_val=max (0.5,param_value - 0.1)
                            max_val=min (0.9,param_value + 0.1)
                        sensitivity_range[param_name]=[
                            round (val,2) for val in np.linspace (min_val,max_val,5)
                        ]
                    else:
                        # Autres flottants
                        sensitivity_range[param_name]=[
                            round (param_value * (1 + factor),3)
                            for factor in [-0.3,-0.15,0,0.15,0.3]
                        ]

            # Pour les paramètres non numériques (non pris en charge pour le moment)
            else:
                sensitivity_range[param_name]=[param_value]

        return sensitivity_range

    def _generate_parameter_variations (self,
                                        base_params: Dict[str,Any],
                                        sensitivity_range: Dict[str,List[Any]],
                                        n_variations: int) -> List[Dict[str,Any]]:
        """
        Génère des variations de paramètres pour les tests.

        Args:
            base_params: Paramètres de base
            sensitivity_range: Plages de variation pour chaque paramètre
            n_variations: Nombre de variations à générer

        Returns:
            Liste de dictionnaires de paramètres variés
        """
        parameter_sets=[base_params.copy ()]

        # Pour chaque paramètre à varier
        for param_name,param_values in sensitivity_range.items ():
            if param_name not in base_params:
                continue

            # Sélection des valeurs à tester
            if len (param_values) <= n_variations:
                test_values=param_values
            else:
                # Échantillonnage régulier
                indices=np.linspace (0,len (param_values) - 1,n_variations).astype (int)
                test_values=[param_values[i] for i in indices]

            # Ajout des ensembles de paramètres variés
            for value in test_values:
                if value == base_params[param_name]:
                    continue  # Éviter de dupliquer le cas de base

                variation=base_params.copy ()
                variation[param_name]=value
                parameter_sets.append (variation)

        return parameter_sets

    def _analyze_parameter_sensitivity (self,
                                        sensitivity_results: Dict[str,List[Dict]]) -> Dict[str,Dict]:
        """
        Analyse la sensibilité des paramètres à partir des résultats des tests.

        Args:
            sensitivity_results: Résultats des tests de sensibilité

        Returns:
            Analyse de la sensibilité par paramètre
        """
        parameter_sensitivity={}

        for param_name,param_results in sensitivity_results.items ():
            if not param_results:
                continue

            # Extraction des valeurs et métriques pour ce paramètre
            param_values=[r['param_value'] for r in param_results]

            # Analyse par métrique
            metric_sensitivity={}

            for metric in self.metric_keys:
                metric_values=[r['metrics'].get (metric,np.nan) for r in param_results]

                # Calcul des métriques de sensibilité uniquement si toutes les valeurs sont disponibles
                if not any (np.isnan (metric_values)):
                    # Calcul de la variation
                    min_val=min (metric_values)
                    max_val=max (metric_values)
                    range_val=max_val - min_val

                    # Calcul de la variation relative
                    if abs (np.mean (metric_values)) > 1e-10:
                        relative_range=range_val / abs (np.mean (metric_values))
                    else:
                        relative_range=0.0

                    # Calcul de la corrélation entre le paramètre et la métrique
                    if len (param_values) > 1 and len (set (param_values)) > 1:
                        correlation=np.corrcoef (param_values,metric_values)[0,1]
                    else:
                        correlation=0.0

                    # Évaluation de la sensibilité
                    sensitivity_score=min (1.0,abs (relative_range))

                    metric_sensitivity[metric]={
                        'min':min_val,
                        'max':max_val,
                        'range':range_val,
                        'relative_range':relative_range,
                        'correlation':correlation,
                        'sensitivity_score':sensitivity_score
                    }

            # Calcul du score global de sensibilité pour ce paramètre
            if metric_sensitivity:
                # Moyenne des scores de sensibilité individuels
                overall_score=np.mean ([
                    ms['sensitivity_score']
                    for ms in metric_sensitivity.values ()
                    if 'sensitivity_score' in ms
                ])

                parameter_sensitivity[param_name]={
                    'metrics':metric_sensitivity,
                    'overall_score':overall_score
                }

        return parameter_sensitivity

    def run_monte_carlo_simulation (self,
                                    params: Dict[str,Any],
                                    n_simulations: int = 100,
                                    perturbation_factors: Dict[str,float] = None) -> Dict:
        """
        Effectue des simulations Monte Carlo pour évaluer la robustesse.

        Args:
            params: Paramètres à tester
            n_simulations: Nombre de simulations à effectuer
            perturbation_factors: Facteurs de perturbation pour chaque paramètre

        Returns:
            Résultats des simulations Monte Carlo
        """
        logger.info (f"Exécution de {n_simulations} simulations Monte Carlo...")

        # Si aucun facteur de perturbation n'est fourni, utiliser des valeurs par défaut
        if perturbation_factors is None:
            perturbation_factors=self._default_perturbation_factors (params)

        # Résultats des simulations
        simulation_results=[]

        # Exécution des simulations
        for i in tqdm (range (n_simulations),desc="Simulations Monte Carlo"):
            # Génération de paramètres perturbés
            perturbed_params=self._perturb_parameters (params,perturbation_factors)

            # Exécution du backtest
            try:
                results=self.backtest_function (self.data,perturbed_params)

                # Extraction des métriques d'intérêt
                metrics={
                    k:results[k] for k in self.metric_keys if k in results
                }

                simulation_results.append ({
                    'params':perturbed_params,
                    'metrics':metrics
                })
            except Exception as e:
                logger.warning (f"Erreur lors de la simulation {i}: {str (e)}")

        # Analyse des résultats
        mc_analysis=self._analyze_monte_carlo_results (simulation_results)

        # Stockage des résultats
        self.results['monte_carlo']={
            'simulations':simulation_results,
            'analysis':mc_analysis
        }

        return self.results['monte_carlo']

    def _default_perturbation_factors (self,params: Dict[str,Any]) -> Dict[str,float]:
        """
        Crée des facteurs de perturbation par défaut pour les paramètres.

        Args:
            params: Paramètres à perturber

        Returns:
            Facteurs de perturbation pour chaque paramètre
        """
        perturbation_factors={}

        for param_name,param_value in params.items ():
            # Pour les paramètres numériques
            if isinstance (param_value,(int,float)):
                if param_name in ['volatility_window','spectral_window','min_periods','observation_period']:
                    # Fenêtres et périodes
                    perturbation_factors[param_name]=0.2
                elif param_name in ['vol_ratio_weight','bound_coherence_weight','alpha_stability_weight',
                                    'spectral_score_weight',
                                    'min_btc_allocation','max_btc_allocation']:
                    # Poids et allocations
                    perturbation_factors[param_name]=0.1
                else:
                    # Autres paramètres
                    perturbation_factors[param_name]=0.15

        return perturbation_factors

    def _perturb_parameters (self,
                             params: Dict[str,Any],
                             perturbation_factors: Dict[str,float]) -> Dict[str,Any]:
        """
        Perturbe les paramètres pour les simulations Monte Carlo.

        Args:
            params: Paramètres à perturber
            perturbation_factors: Facteurs de perturbation pour chaque paramètre

        Returns:
            Paramètres perturbés
        """
        perturbed_params={}

        for param_name,param_value in params.items ():
            # Pour les paramètres numériques
            if isinstance (param_value,(int,float)):
                factor=perturbation_factors.get (param_name,0.1)

                # Génération d'une perturbation aléatoire
                perturbation=np.random.normal (0,factor)

                if isinstance (param_value,int):
                    # Paramètres entiers
                    perturbed_value=max (1,int (round (param_value * (1 + perturbation))))
                else:
                    # Paramètres flottants
                    perturbed_value=param_value * (1 + perturbation)

                    # Contraintes spécifiques
                    if param_name in ['vol_ratio_weight','bound_coherence_weight','alpha_stability_weight',
                                      'spectral_score_weight',
                                      'min_btc_allocation','max_btc_allocation']:
                        perturbed_value=max (0.0,min (1.0,perturbed_value))

                perturbed_params[param_name]=perturbed_value
            else:
                # Paramètres non numériques (inchangés)
                perturbed_params[param_name]=param_value

        # Vérification des contraintes
        if ('min_btc_allocation' in perturbed_params and
                'max_btc_allocation' in perturbed_params and
                perturbed_params['min_btc_allocation'] >= perturbed_params['max_btc_allocation']):
            # Correction des bornes d'allocation
            mid=(perturbed_params['min_btc_allocation'] + perturbed_params['max_btc_allocation']) / 2
            perturbed_params['min_btc_allocation']=mid - 0.1
            perturbed_params['max_btc_allocation']=mid + 0.1

        return perturbed_params

    def _analyze_monte_carlo_results (self,simulation_results: List[Dict]) -> Dict:
        """
        Analyse les résultats des simulations Monte Carlo.

        Args:
            simulation_results: Résultats des simulations

        Returns:
            Analyse des résultats
        """
        if not simulation_results:
            return {}

        analysis={}

        # Analyse des métriques
        for metric in self.metric_keys:
            metric_values=[r['metrics'].get (metric,np.nan) for r in simulation_results if metric in r['metrics']]

            if metric_values and not all (np.isnan (metric_values)):
                # Filtre des valeurs valides
                valid_values=[v for v in metric_values if not np.isnan (v)]

                if valid_values:
                    # Calcul des statistiques
                    mean_val=np.mean (valid_values)
                    median_val=np.median (valid_values)
                    std_val=np.std (valid_values)
                    min_val=np.min (valid_values)
                    max_val=np.max (valid_values)

                    # Calcul des percentiles
                    percentiles=np.percentile (valid_values,[5,25,50,75,95])

                    # Probabilité de performance négative (pour le rendement)
                    if metric == 'total_return':
                        prob_negative=np.mean ([v < 0 for v in valid_values])
                    else:
                        prob_negative=None

                    analysis[metric]={
                        'mean':mean_val,
                        'median':median_val,
                        'std':std_val,
                        'min':min_val,
                        'max':max_val,
                        'percentiles':{
                            '5%':percentiles[0],
                            '25%':percentiles[1],
                            '50%':percentiles[2],
                            '75%':percentiles[3],
                            '95%':percentiles[4]
                        },
                        'prob_negative':prob_negative
                    }

        # Analyse des paramètres
        param_analysis={}

        # Récupération de tous les noms de paramètres
        param_names=set ()
        for sim in simulation_results:
            param_names.update (sim['params'].keys ())

        # Analyse de chaque paramètre
        for param_name in param_names:
            param_values=[r['params'].get (param_name) for r in simulation_results if param_name in r['params']]

            if param_values and all (isinstance (v,(int,float)) for v in param_values):
                # Calcul des statistiques
                mean_val=np.mean (param_values)
                median_val=np.median (param_values)
                std_val=np.std (param_values)

                param_analysis[param_name]={
                    'mean':mean_val,
                    'median':median_val,
                    'std':std_val,
                    'cv':std_val / mean_val if mean_val != 0 else 0  # Coefficient de variation
                }

        # Analyse des corrélations paramètres-métriques
        correlation_analysis={}

        for param_name in param_names:
            param_values=[r['params'].get (param_name) for r in simulation_results if param_name in r['params']]

            if param_values and all (isinstance (v,(int,float)) for v in param_values):
                param_corr={}

                for metric in self.metric_keys:
                    metric_values=[r['metrics'].get (metric) for r in simulation_results if metric in r['metrics']]

                    if (len (param_values) == len (metric_values) and
                            len (param_values) > 1 and
                            not all (np.isnan (metric_values))):

                        # Filtre des valeurs valides
                        valid_indices=[i for i,v in enumerate (metric_values) if not np.isnan (v)]
                        valid_params=[param_values[i] for i in valid_indices]
                        valid_metrics=[metric_values[i] for i in valid_indices]

                        if len (valid_params) > 1 and len (set (valid_params)) > 1:
                            # Calcul de la corrélation
                            correlation=np.corrcoef (valid_params,valid_metrics)[0,1]
                            param_corr[metric]=correlation

                if param_corr:
                    correlation_analysis[param_name]=param_corr

        return {
            'metrics':analysis,
            'parameters':param_analysis,
            'correlations':correlation_analysis
        }

    def run_time_series_cross_validation (self,
                                          params: Dict[str,Any],
                                          n_splits: int = 5,
                                          test_size: int = 60) -> Dict:
        """
        Effectue une validation croisée temporelle.

        Args:
            params: Paramètres à tester
            n_splits: Nombre de divisions temporelles
            test_size: Taille de l'ensemble de test (en jours)

        Returns:
            Résultats de la validation croisée
        """
        logger.info (f"Exécution de la validation croisée temporelle avec {n_splits} divisions...")

        # Récupération des dates communes
        common_dates=self._get_common_dates ()

        # Tri des dates
        common_dates=sorted (common_dates)

        # Définition des splits temporels
        splits=[]
        test_size_pd=pd.Timedelta (days=test_size)

        # Calcul de l'intervalle total disponible
        total_span=common_dates[-1] - common_dates[0]

        # Calcul de la taille de chaque segment
        segment_size=total_span / n_splits

        for i in range (n_splits):
            # Calcul des dates de début et fin du test
            test_end_idx=len (common_dates) - 1 - i * int (len (common_dates) / n_splits)
            test_end=common_dates[min (test_end_idx,len (common_dates) - 1)]
            test_start=test_end - test_size_pd

            # Calcul de la date de fin d'entraînement
            train_end=test_start - pd.Timedelta (days=1)

            splits.append ({
                'train_end':train_end,
                'test_start':test_start,
                'test_end':test_end
            })

        # Préparation des résultats
        cv_results=[]

        # Pour chaque split
        for i,split in enumerate (splits):
            logger.info (f"Validation croisée {i + 1}/{n_splits}: "
                         f"Train jusqu'à {split['train_end'].strftime ('%Y-%m-%d')}, "
                         f"Test de {split['test_start'].strftime ('%Y-%m-%d')} à {split['test_end'].strftime ('%Y-%m-%d')}")

            # Préparation des ensembles d'entraînement et de test
            train_data={}
            test_data={}

            for asset,df in self.data.items ():
                train_mask=df.index <= split['train_end']
                test_mask=(df.index >= split['test_start']) & (df.index <= split['test_end'])

                train_data[asset]=df[train_mask].copy ()
                test_data[asset]=df[test_mask].copy ()

            # Exécution du backtest sur les données d'entraînement et de test
            try:
                train_results=self.backtest_function (train_data,params)
                test_results=self.backtest_function (test_data,params)

                # Extraction des métriques d'intérêt
                train_metrics={k:train_results.get (k) for k in self.metric_keys if k in train_results}
                test_metrics={k:test_results.get (k) for k in self.metric_keys if k in test_results}

                cv_results.append ({
                    'split':i + 1,
                    'train_end':split['train_end'],
                    'test_start':split['test_start'],
                    'test_end':split['test_end'],
                    'train_metrics':train_metrics,
                    'test_metrics':test_metrics
                })
            except Exception as e:
                logger.warning (f"Erreur lors de la validation croisée {i + 1}: {str (e)}")

        # Analyse des résultats
        cv_analysis=self._analyze_cv_results (cv_results)

        # Stockage des résultats
        self.results['cross_validation']={
            'splits':cv_results,
            'analysis':cv_analysis
        }

        return self.results['cross_validation']

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

    def _analyze_cv_results (self,cv_results: List[Dict]) -> Dict:
        """
        Analyse les résultats de la validation croisée.

        Args:
            cv_results: Résultats de la validation croisée

        Returns:
            Analyse des résultats
        """
        if not cv_results:
            return {}

        analysis={'metrics':{}}

        # Pour chaque métrique
        train_values=[r['train_metrics'].get (metric) for r in cv_results if metric in r['train_metrics']]
        test_values=[r['test_metrics'].get (metric) for r in cv_results if metric in r['test_metrics']]

        if train_values and test_values and len (train_values) == len (test_values):
                # Calcul des statistiques pour les données d'entraînement
            train_mean=np.mean (train_values)
            train_std=np.std (train_values)
            train_min=np.min (train_values)
            train_max=np.max (train_values)

            # Calcul des statistiques pour les données de test
            test_mean=np.mean (test_values)
            test_std=np.std (test_values)
            test_min=np.min (test_values)
            test_max=np.max (test_values)

            # Calcul des ratios test/train
            if train_mean != 0:
                ratio=test_mean / train_mean
            else:
                ratio=float ('inf')

            # Calcul de la stabilité (coefficient de variation)
            train_cv=train_std / abs (train_mean) if train_mean != 0 else float ('inf')
            test_cv=test_std / abs (test_mean) if test_mean != 0 else float ('inf')

            analysis['metrics'][metric]={
                'train':{
                    'mean':train_mean,
                    'std':train_std,
                    'cv':train_cv,
                    'min':train_min,
                    'max':train_max
                },
                'test':{
                    'mean':test_mean,
                    'std':test_std,
                    'cv':test_cv,
                    'min':test_min,
                    'max':test_max
                },
                'ratio':ratio,
                'stability':1.0 / (1.0 + test_cv) if test_cv < float ('inf') else 0.0
            }

def _analyze_cv_results(self, cv_results: List[Dict]) -> Dict:
    """
    Analyse les résultats de la validation croisée.

    Args:
        cv_results: Résultats de la validation croisée

    Returns:
        Analyse des résultats
    """
    if not cv_results:
        return {}

    analysis = {'metrics': {}}

    # Pour chaque métrique
    for metric in self.metric_keys:
        train_values = [
            r['train_metrics'].get(metric) 
            for r in cv_results 
            if 'train_metrics' in r and metric in r['train_metrics']
        ]
        test_values = [
            r['test_metrics'].get(metric) 
            for r in cv_results 
            if 'test_metrics' in r and metric in r['test_metrics']
        ]

        if train_values and test_values and len(train_values) == len(test_values):
            # Calcul des statistiques pour les données d'entraînement
            train_mean = np.mean(train_values)
            train_std = np.std(train_values)
            train_min = np.min(train_values)
            train_max = np.max(train_values)

            # Calcul des statistiques pour les données de test
            test_mean = np.mean(test_values)
            test_std = np.std(test_values)
            test_min = np.min(test_values)
            test_max = np.max(test_values)

            # Calcul des ratios test/train
            if train_mean != 0:
                ratio = test_mean / train_mean
            else:
                ratio = float('inf')

            # Calcul de la stabilité (coefficient de variation)
            train_cv = train_std / abs(train_mean) if train_mean != 0 else float('inf')
            test_cv = test_std / abs(test_mean) if test_mean != 0 else float('inf')

            analysis['metrics'][metric] = {
                'train': {
                    'mean': train_mean,
                    'std': train_std,
                    'cv': train_cv,
                    'min': train_min,
                    'max': train_max
                },
                'test': {
                    'mean': test_mean,
                    'std': test_std,
                    'cv': test_cv,
                    'min': test_min,
                    'max': test_max
                },
                'ratio': ratio,
                'stability': 1.0 / (1.0 + test_cv) if test_cv < float('inf') else 0.0
            }

    # Calcul du score global de robustesse
    stability_scores = [
        analysis['metrics'][m]['stability']
        for m in analysis['metrics']
        if 'stability' in analysis['metrics'][m]
    ]

    # Pondération des ratios test/train
    ratio_scores = []
    for metric in analysis['metrics']:
        if 'ratio' in analysis['metrics'][metric]:
            ratio = analysis['metrics'][metric]['ratio']

            # Pour le rendement, volatilité, Sharpe
            if metric in ['total_return', 'sharpe_ratio']:
                # Idéalement ratio proche de 1 (performance similaire)
                score = 1.0 - min(1.0, abs(ratio - 1.0))

            # Pour le drawdown (négatif)
            elif metric == 'max_drawdown':
                # Idéalement drawdown de test pas pire que celui d'entraînement
                score = 1.0 - min(1.0, max(0.0, ratio - 1.0))

            # Pour d'autres métriques
            else:
                # Approche générique
                score = 1.0 - min(1.0, abs(ratio - 1.0))

            ratio_scores.append(score)

    # Score global
    analysis['overall_robustness'] = (
        np.mean(stability_scores) * 0.5 + np.mean(ratio_scores) * 0.5
        if stability_scores and ratio_scores
        else 0.0
    )

    return analysis


def run_stress_tests (self,
                      params: Dict[str,Any],
                      scenarios: Dict[str,Dict[str,str]] = None) -> Dict:
    """
    Effectue des tests de stress selon différents scénarios de marché.

    Args:
        params: Paramètres à tester
        scenarios: Dictionnaire des périodes de scénarios (nom: {start, end})

    Returns:
        Résultats des tests de stress
    """
    logger.info ("Exécution des tests de stress...")

    # Si aucun scénario n'est fourni, définir des scénarios par défaut
    if scenarios is None:
        scenarios=self._define_default_scenarios ()

    # Résultats des tests
    stress_results={}

    # Pour chaque scénario
    for scenario_name,period in scenarios.items ():
        logger.info (f"Test du scénario '{scenario_name}': {period['start']} à {period['end']}")

        # Extraction des données pour ce scénario
        start_date=pd.Timestamp (period['start'])
        end_date=pd.Timestamp (period['end'])

        scenario_data={}
        for asset,df in self.data.items ():
            scenario_mask=(df.index >= start_date) & (df.index <= end_date)
            scenario_data[asset]=df[scenario_mask].copy ()

        # Vérification que nous avons suffisamment de données
        first_key=next (iter (scenario_data))
        if len (scenario_data[first_key]) < 10:
            logger.warning (f"Données insuffisantes pour le scénario '{scenario_name}', ignoré")
            continue

        # Exécution du backtest
        try:
            results=self.backtest_function (scenario_data,params)

            # Extraction des métriques d'intérêt
            metrics={k:results.get (k) for k in self.metric_keys if k in results}

            stress_results[scenario_name]={
                'period':f"{start_date.strftime ('%Y-%m-%d')} à {end_date.strftime ('%Y-%m-%d')}",
                'metrics':metrics
            }
        except Exception as e:
            logger.warning (f"Erreur lors du test du scénario '{scenario_name}': {str (e)}")

    # Analyse des résultats
    stress_analysis=self._analyze_stress_results (stress_results)

    # Stockage des résultats
    self.results['stress_tests']={
        'scenarios':stress_results,
        'analysis':stress_analysis
    }

    return self.results['stress_tests']


def _define_default_scenarios (self) -> Dict[str,Dict[str,str]]:
    """
    Définit des scénarios de marché par défaut pour les tests de stress.

    Returns:
        Dictionnaire des périodes de scénarios
    """
    # Récupération des dates communes
    common_dates=self._get_common_dates ()

    # Si nous n'avons pas suffisamment de données
    if len (common_dates) < 365:
        logger.warning ("Données insuffisantes pour définir des scénarios par défaut")
        return {}

    # Conversion en datetime pour faciliter les calculs
    start_date=common_dates[0].to_pydatetime ()
    end_date=common_dates[-1].to_pydatetime ()

    # Durée totale en jours
    total_days=(end_date - start_date).days

    # Division en quartiles approximatifs
    q1_end=start_date + timedelta (days=total_days // 4)
    q2_end=start_date + timedelta (days=total_days // 2)
    q3_end=start_date + timedelta (days=3 * total_days // 4)

    # Définition des scénarios génériques
    return {
        'early_period':{
            'start':start_date.strftime ('%Y-%m-%d'),
            'end':q1_end.strftime ('%Y-%m-%d')
        },
        'mid_period_1':{
            'start':q1_end.strftime ('%Y-%m-%d'),
            'end':q2_end.strftime ('%Y-%m-%d')
        },
        'mid_period_2':{
            'start':q2_end.strftime ('%Y-%m-%d'),
            'end':q3_end.strftime ('%Y-%m-%d')
        },
        'recent_period':{
            'start':q3_end.strftime ('%Y-%m-%d'),
            'end':end_date.strftime ('%Y-%m-%d')
        },
        'full_period':{
            'start':start_date.strftime ('%Y-%m-%d'),
            'end':end_date.strftime ('%Y-%m-%d')
        }
    }


def _analyze_stress_results (self,stress_results: Dict[str,Dict]) -> Dict:
    """
    Analyse les résultats des tests de stress.

    Args:
        stress_results: Résultats des tests de stress

    Returns:
        Analyse des résultats
    """
    if not stress_results:
        return {}

    analysis={'metrics':{}}

    # Pour chaque métrique
    for metric in self.metric_keys:
        # Collecte des valeurs pour cette métrique dans tous les scénarios
        values={}
        for scenario,result in stress_results.items ():
            if 'metrics' in result and metric in result['metrics']:
                values[scenario]=result['metrics'][metric]

        if values:
            # Calcul des statistiques
            min_scenario=min (values.items (),key=lambda x:x[1])[0]
            max_scenario=max (values.items (),key=lambda x:x[1])[0]
            mean_val=np.mean (list (values.values ()))
            std_val=np.std (list (values.values ()))

            # Calcul du coefficient de variation
            cv=std_val / abs (mean_val) if mean_val != 0 else float ('inf')

            # Score de robustesse pour cette métrique
            robustness_score=1.0 / (1.0 + cv) if cv < float ('inf') else 0.0

            analysis['metrics'][metric]={
                'min':{
                    'scenario':min_scenario,
                    'value':values[min_scenario]
                },
                'max':{
                    'scenario':max_scenario,
                    'value':values[max_scenario]
                },
                'mean':mean_val,
                'std':std_val,
                'cv':cv,
                'robustness_score':robustness_score
            }

    # Score global de robustesse
    robustness_scores=[
        analysis['metrics'][m]['robustness_score']
        for m in analysis['metrics']
        if 'robustness_score' in analysis['metrics'][m]
    ]

    analysis['overall_robustness']=np.mean (robustness_scores) if robustness_scores else 0.0

    return analysis


def plot_sensitivity_analysis (self) -> None:
    """
    Affiche une visualisation de l'analyse de sensibilité.
    """
    if 'sensitivity' not in self.results:
        logger.warning ("Aucun résultat d'analyse de sensibilité disponible à visualiser")
        return

    sensitivity_results=self.results['sensitivity']

    if 'parameter_sensitivity' not in sensitivity_results:
        logger.warning ("Analyse de sensibilité incomplète")
        return

    parameter_sensitivity=sensitivity_results['parameter_sensitivity']

    # Tri des paramètres par score global de sensibilité
    sorted_params=sorted (
        parameter_sensitivity.items (),
        key=lambda x:x[1]['overall_score'],
        reverse=True
    )

    # Extraction des noms de paramètres et des scores
    param_names=[p[0] for p in sorted_params]
    sensitivity_scores=[p[1]['overall_score'] for p in sorted_params]

    # Création de la figure
    plt.figure (figsize=(12,8))

    # Barplot des scores de sensibilité
    plt.barh (param_names,sensitivity_scores,color='skyblue')
    plt.xlabel ('Score de sensibilité')
    plt.title ('Sensibilité des paramètres')
    plt.xlim (0,1)
    plt.grid (True,axis='x',alpha=0.3)

    # Ajout des valeurs
    for i,score in enumerate (sensitivity_scores):
        plt.text (score + 0.01,i,f'{score:.2f}',va='center')

    plt.tight_layout ()
    plt.show ()

    # Si nous avons des détails sur les métriques individuelles
    for param_name,param_data in sorted_params[:5]:  # Afficher seulement les 5 plus sensibles
        if 'metrics' in param_data:
            # Extraire les sensibilités par métrique
            metric_names=list (param_data['metrics'].keys ())
            metric_scores=[param_data['metrics'][m]['sensitivity_score'] for m in metric_names]

            plt.figure (figsize=(10,6))
            plt.barh (metric_names,metric_scores,color='salmon')
            plt.xlabel ('Score de sensibilité')
            plt.title (f'Sensibilité de {param_name} par métrique')
            plt.xlim (0,1)
            plt.grid (True,axis='x',alpha=0.3)

            # Ajout des valeurs
            for i,score in enumerate (metric_scores):
                plt.text (score + 0.01,i,f'{score:.2f}',va='center')

            plt.tight_layout ()
            plt.show ()


def plot_monte_carlo_results (self) -> None:
    """
    Affiche une visualisation des résultats des simulations Monte Carlo.
    """
    if 'monte_carlo' not in self.results:
        logger.warning ("Aucun résultat de simulation Monte Carlo disponible à visualiser")
        return

    mc_results=self.results['monte_carlo']

    if 'analysis' not in mc_results or 'metrics' not in mc_results['analysis']:
        logger.warning ("Analyse Monte Carlo incomplète")
        return

    metrics_analysis=mc_results['analysis']['metrics']

    # Création d'un tableau pour les statistiques
    plt.figure (figsize=(12,len (metrics_analysis) * 2))

    for i,(metric,stats) in enumerate (metrics_analysis.items ()):
        # Récupération des valeurs pour cette métrique
        values=[sim['metrics'].get (metric) for sim in mc_results['simulations']
                if metric in sim['metrics'] and not np.isnan (sim['metrics'][metric])]

        if not values:
            continue

        # Création d'un sous-plot pour l'histogramme
        plt.subplot (len (metrics_analysis),1,i + 1)

        # Tracé de l'histogramme
        n,bins,patches=plt.hist (values,bins=30,alpha=0.7,color='skyblue')

        # Ajout des lignes pour les percentiles
        percentiles=stats['percentiles']
        plt.axvline (percentiles['5%'],color='r',linestyle='--',alpha=0.5,label='5%')
        plt.axvline (percentiles['25%'],color='orange',linestyle='--',alpha=0.5,label='25%')
        plt.axvline (percentiles['50%'],color='g',linestyle='-',alpha=0.7,label='Médiane')
        plt.axvline (percentiles['75%'],color='orange',linestyle='--',alpha=0.5,label='75%')
        plt.axvline (percentiles['95%'],color='r',linestyle='--',alpha=0.5,label='95%')

        plt.title (f'Distribution de {metric}')
        plt.xlabel ('Valeur')
        plt.ylabel ('Fréquence')
        plt.legend ()

        # Ajout de statistiques en texte
        text=(f"Moyenne: {stats['mean']:.2f}\n"
              f"Médiane: {stats['median']:.2f}\n"
              f"Écart-type: {stats['std']:.2f}\n"
              f"Min: {stats['min']:.2f}\n"
              f"Max: {stats['max']:.2f}")

        if 'prob_negative' in stats and stats['prob_negative'] is not None:
            text+=f"\nProb. négative: {stats['prob_negative']:.1%}"

        plt.text (0.02,0.95,text,transform=plt.gca ().transAxes,
                  verticalalignment='top',bbox=dict (boxstyle='round',alpha=0.1))

    plt.tight_layout ()
    plt.show ()

    # Graphique des corrélations
    if 'correlations' in mc_results['analysis']:
        correlations=mc_results['analysis']['correlations']

        if correlations:
            # Création d'une matrice de corrélation
            param_names=list (correlations.keys ())
            metric_names=list (self.metric_keys)

            corr_matrix=np.zeros ((len (param_names),len (metric_names)))

            for i,param in enumerate (param_names):
                for j,metric in enumerate (metric_names):
                    if metric in correlations[param]:
                        corr_matrix[i,j]=correlations[param][metric]

            # Création de la figure
            plt.figure (figsize=(12,8))
            plt.imshow (corr_matrix,cmap='coolwarm',vmin=-1,vmax=1)

            # Configuration des ticks
            plt.xticks (np.arange (len (metric_names)),metric_names,rotation=45)
            plt.yticks (np.arange (len (param_names)),param_names)

            # Ajout des valeurs
            for i in range (len (param_names)):
                for j in range (len (metric_names)):
                    plt.text (j,i,f"{corr_matrix[i,j]:.2f}",
                              ha="center",va="center",
                              color="black" if abs (corr_matrix[i,j]) < 0.5 else "white")

            plt.colorbar (label='Corrélation')
            plt.title ('Corrélation entre paramètres et métriques')
            plt.tight_layout ()
            plt.show ()


def generate_robustness_report (self) -> str:
    """
    Génère un rapport complet de l'analyse de robustesse.

    Returns:
        Rapport formaté
    """
    report="# Rapport d'Analyse de Robustesse QAAF\n\n"
    report+=f"Date: {datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')}\n\n"

    # Synthèse des résultats disponibles
    available_tests=[]
    if 'sensitivity' in self.results:
        available_tests.append ("Analyse de sensibilité")
    if 'monte_carlo' in self.results:
        available_tests.append ("Simulations Monte Carlo")
    if 'cross_validation' in self.results:
        available_tests.append ("Validation croisée temporelle")
    if 'stress_tests' in self.results:
        available_tests.append ("Tests de stress")

    report+=f"Tests effectués: {', '.join (available_tests)}\n\n"

    # Score global de robustesse
    robustness_scores=[]

    if 'sensitivity' in self.results and 'parameter_sensitivity' in self.results['sensitivity']:
        # Calculer un score de robustesse inversé (moins sensible = plus robuste)
        param_scores=[1.0 - param['overall_score']
                      for param in self.results['sensitivity']['parameter_sensitivity'].values ()]
        if param_scores:
            sensitivity_robustness=np.mean (param_scores)
            robustness_scores.append (sensitivity_robustness)

            report+=f"## Analyse de Sensibilité\n\n"
            report+=f"Score de robustesse: {sensitivity_robustness:.2f}/1.00\n\n"

            # Top 3 des paramètres les plus sensibles
            sorted_params=sorted (
                self.results['sensitivity']['parameter_sensitivity'].items (),
                key=lambda x:x[1]['overall_score'],
                reverse=True
            )

            if sorted_params:
                report+="Paramètres les plus sensibles:\n"
                for i,(param,data) in enumerate (sorted_params[:3],1):
                    report+=f"{i}. **{param}**: Score de sensibilité {data['overall_score']:.2f}\n"
                report+="\n"

    if 'monte_carlo' in self.results and 'analysis' in self.results['monte_carlo']:
        mc_analysis=self.results['monte_carlo']['analysis']

        report+=f"## Simulations Monte Carlo\n\n"

        if 'metrics' in mc_analysis:
            for metric,stats in mc_analysis['metrics'].items ():
                report+=f"### {metric}\n\n"
                report+=f"- Moyenne: {stats['mean']:.2f}\n"
                report+=f"- Médiane: {stats['median']:.2f}\n"
                report+=f"- Écart-type: {stats['std']:.2f}\n"
                report+=f"- Min/Max: {stats['min']:.2f}/{stats['max']:.2f}\n"

                if 'percentiles' in stats:
                    report+=f"- Intervalle de confiance 90%: [{stats['percentiles']['5%']:.2f}, {stats['percentiles']['95%']:.2f}]\n"

                if 'prob_negative' in stats and stats['prob_negative'] is not None:
                    report+=f"- Probabilité de rendement négatif: {stats['prob_negative']:.1%}\n"

                report+="\n"

            # Calcul d'un score de robustesse basé sur la variabilité
            cv_scores=[]
            for stats in mc_analysis['metrics'].values ():
                if 'mean' in stats and 'std' in stats and stats['mean'] != 0:
                    cv=stats['std'] / abs (stats['mean'])
                    cv_scores.append (1.0 / (1.0 + cv))

            if cv_scores:
                mc_robustness=np.mean (cv_scores)
                robustness_scores.append (mc_robustness)
                report+=f"Score de robustesse Monte Carlo: {mc_robustness:.2f}/1.00\n\n"

    if 'cross_validation' in self.results and 'analysis' in self.results['cross_validation']:
        cv_analysis=self.results['cross_validation']['analysis']

        report+=f"## Validation Croisée Temporelle\n\n"

        if 'overall_robustness' in cv_analysis:
            cv_robustness=cv_analysis['overall_robustness']
            robustness_scores.append (cv_robustness)
            report+=f"Score de robustesse: {cv_robustness:.2f}/1.00\n\n"

        if 'metrics' in cv_analysis:
            for metric,stats in cv_analysis['metrics'].items ():
                report+=f"### {metric}\n\n"
                report+=f"- Train (moyenne ± écart-type): {stats['train']['mean']:.2f} ± {stats['train']['std']:.2f}\n"
                report+=f"- Test (moyenne ± écart-type): {stats['test']['mean']:.2f} ± {stats['test']['std']:.2f}\n"
                report+=f"- Ratio Test/Train: {stats['ratio']:.2f}\n"

                if 'stability' in stats:
                    if stats['stability'] >= 0.8:
                        stability_msg="✅ Excellente"
                    elif stats['stability'] >= 0.5:
                        stability_msg="⚠️ Modérée"
                    else:
                        stability_msg="❌ Faible"

                    report+=f"- Stabilité: {stats['stability']:.2f} ({stability_msg})\n"

                report+="\n"

    if 'stress_tests' in self.results and 'analysis' in self.results['stress_tests']:
        stress_analysis=self.results['stress_tests']['analysis']

        report+=f"## Tests de Stress\n\n"

        if 'overall_robustness' in stress_analysis:
            stress_robustness=stress_analysis['overall_robustness']
            robustness_scores.append (stress_robustness)
            report+=f"Score de robustesse: {stress_robustness:.2f}/1.00\n\n"

        if 'metrics' in stress_analysis:
            for metric,stats in stress_analysis['metrics'].items ():
                report+=f"### {metric}\n\n"
                report+=f"- Moyenne: {stats['mean']:.2f}\n"
                report+=f"- Écart-type: {stats['std']:.2f}\n"
                report+=f"- Meilleur scénario: {stats['max']['scenario']} ({stats['max']['value']:.2f})\n"
                report+=f"- Pire scénario: {stats['min']['scenario']} ({stats['min']['value']:.2f})\n"

                if 'robustness_score' in stats:
                    if stats['robustness_score'] >= 0.8:
                        robustness_msg="✅ Excellente"
                    elif stats['robustness_score'] >= 0.5:
                        robustness_msg="⚠️ Modérée"
                    else:
                        robustness_msg="❌ Faible"

                    report+=f"- Robustesse: {stats['robustness_score']:.2f} ({robustness_msg})\n"

                report+="\n"

    # Score global de robustesse
    if robustness_scores:
        overall_robustness=np.mean (robustness_scores)

        report+=f"## Évaluation Globale\n\n"
        report+=f"Score de robustesse global: {overall_robustness:.2f}/1.00\n\n"

        if overall_robustness >= 0.8:
            report+="📊 **Analyse**: La stratégie montre une excellente robustesse à travers tous les tests.\n"
        elif overall_robustness >= 0.6:
            report+="📊 **Analyse**: La stratégie montre une bonne robustesse, adaptée pour un déploiement en production.\n"
        elif overall_robustness >= 0.4:
            report+="📊 **Analyse**: La stratégie montre une robustesse modérée, acceptable pour un déploiement prudent avec supervision.\n"
        else:
            report+="📊 **Analyse**: La stratégie montre une faible robustesse, suggérant un risque élevé en conditions réelles.\n"

    return report


def test_strategy_robustness (backtest_function: Callable,
                              data: Dict[str,pd.DataFrame],
                              params: Dict[str,Any],
                              run_sensitivity: bool = True,
                              run_monte_carlo: bool = True,
                              run_cross_validation: bool = True,
                              run_stress_tests: bool = True,
                              n_monte_carlo: int = 100,
                              n_cv_splits: int = 5) -> Dict:
    """
    Fonction utilitaire pour tester rapidement la robustesse d'une stratégie.

    Args:
        backtest_function: Fonction de backtest
        data: Données de marché
        params: Paramètres à tester
        run_sensitivity: Exécuter l'analyse de sensibilité
        run_monte_carlo: Exécuter les simulations Monte Carlo
        run_cross_validation: Exécuter la validation croisée
        run_stress_tests: Exécuter les tests de stress
        n_monte_carlo: Nombre de simulations Monte Carlo
        n_cv_splits: Nombre de divisions pour la validation croisée

    Returns:
        Résultats de l'analyse de robustesse
    """
    tester=RobustnessTester (data,backtest_function)
    results={}

    if run_sensitivity:
        results['sensitivity']=tester.run_parameter_sensitivity (params)

    if run_monte_carlo:
        results['monte_carlo']=tester.run_monte_carlo_simulation (params,n_monte_carlo)

    if run_cross_validation:
        results['cross_validation']=tester.run_time_series_cross_validation (params,n_cv_splits)

    if run_stress_tests:
        results['stress_tests']=tester.run_stress_tests (params)

    # Génération du rapport
    report=tester.generate_robustness_report ()
    results['report']=report

    return results