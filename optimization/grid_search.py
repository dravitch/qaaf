"""
Module d'optimisation par grid search pour QAAF.
Explore systématiquement l'espace des paramètres pour identifier les combinaisons optimales.
"""

import pandas as pd
import numpy as np
import itertools
import logging
from typing import Dict,List,Tuple,Callable,Any,Optional
from tqdm import tqdm

# Configuration du logging
logger=logging.getLogger (__name__)


class GridSearchOptimizer:
    """
    Optimiseur par grid search pour QAAF.
    Explore l'espace des paramètres pour trouver les combinaisons optimales
    selon différents profils d'optimisation.
    """

    def __init__ (self,
                  objective_function: Callable,
                  param_grid: Dict[str,List[Any]],
                  constraints: Optional[Dict[str,Callable]] = None,
                  max_combinations: int = 10000):
        """
        Initialise l'optimiseur par grid search.

        Args:
            objective_function: Fonction à optimiser, prend un dict de paramètres et renvoie un score
            param_grid: Dictionnaire des paramètres à explorer avec leurs valeurs possibles
            constraints: Dictionnaire des fonctions de contrainte (renvoient True si valide)
            max_combinations: Nombre maximal de combinaisons à tester
        """
        self.objective_function=objective_function
        self.param_grid=param_grid
        self.constraints=constraints or {}
        self.max_combinations=max_combinations
        self.results=[]

    def _generate_combinations (self) -> List[Dict[str,Any]]:
        """
        Génère toutes les combinaisons possibles des paramètres.

        Returns:
            Liste de dictionnaires de paramètres
        """
        param_names=list (self.param_grid.keys ())
        param_values=list (self.param_grid.values ())

        combinations=[]
        for values in itertools.product (*param_values):
            combination=dict (zip (param_names,values))
            combinations.append (combination)

        return combinations

    def _filter_combinations (self,all_combinations: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        Filtre les combinaisons selon les contraintes définies.

        Args:
            all_combinations: Liste de toutes les combinaisons à filtrer

        Returns:
            Liste des combinaisons valides
        """
        valid_combinations=[]

        for combination in all_combinations:
            # Vérifier chaque contrainte
            valid=True
            for constraint_name,constraint_func in self.constraints.items ():
                if not constraint_func (combination):
                    valid=False
                    break

            if valid:
                valid_combinations.append (combination)

        return valid_combinations

    def optimize (self,verbose: bool = True) -> Dict:
        """
        Exécute l'optimisation par grid search.

        Args:
            verbose: Si True, affiche une barre de progression

        Returns:
            Dictionnaire contenant les résultats d'optimisation
        """
        logger.info ("Génération des combinaisons de paramètres...")
        all_combinations=self._generate_combinations ()
        logger.info (f"Nombre total de combinaisons: {len (all_combinations):,}")

        # Filtrage selon les contraintes
        logger.info ("Filtrage des combinaisons selon les contraintes...")
        valid_combinations=self._filter_combinations (all_combinations)
        logger.info (f"Nombre de combinaisons valides: {len (valid_combinations):,}")

        # Limitation du nombre de combinaisons
        if len (valid_combinations) > self.max_combinations:
            logger.warning (f"Limitation à {self.max_combinations:,} combinaisons sur {len (valid_combinations):,}")
            np.random.shuffle (valid_combinations)  # Mélange aléatoire
            combinations_to_test=valid_combinations[:self.max_combinations]
        else:
            combinations_to_test=valid_combinations

        # Test des combinaisons
        logger.info ("Test des combinaisons...")
        self.results=[]

        iterator=tqdm (combinations_to_test) if verbose else combinations_to_test
        for params in iterator:
            try:
                score=self.objective_function (params)
                self.results.append ({
                    'params':params,
                    'score':score
                })
            except Exception as e:
                logger.warning (f"Erreur lors de l'évaluation des paramètres {params}: {str (e)}")

        # Tri des résultats par score
        self.results.sort (key=lambda x:x.get ('score',float ('-inf')),reverse=True)

        # Préparation des résultats
        return {
            'best_params':self.results[0]['params'] if self.results else None,
            'best_score':self.results[0]['score'] if self.results else None,
            'all_results':self.results,
            'tested_combinations':len (self.results),
            'valid_combinations':len (valid_combinations),
            'total_combinations':len (all_combinations)
        }

    def get_top_n (self,n: int = 10) -> List[Dict]:
        """
        Récupère les N meilleures combinaisons trouvées.

        Args:
            n: Nombre de combinaisons à retourner

        Returns:
            Liste des N meilleures combinaisons
        """
        return self.results[:min (n,len (self.results))]

    def analyze_parameter_importance (self) -> Dict[str,Dict[str,float]]:
        """
        Analyse l'importance relative de chaque paramètre dans les résultats.

        Returns:
            Dictionnaire des statistiques d'importance par paramètre
        """
        if not self.results:
            return {}

        # Limiter à 100 meilleurs résultats pour l'analyse
        top_results=self.results[:min (100,len (self.results))]

        # Initialisation des statistiques
        param_stats={}

        # Pour chaque paramètre
        for param_name in self.param_grid.keys ():
            # Collecter les valeurs du paramètre parmi les meilleurs résultats
            param_values=[result['params'].get (param_name) for result in top_results]

            # Si le paramètre est numérique
            if all (isinstance (v,(int,float)) for v in param_values if v is not None):
                param_stats[param_name]={
                    'mean':np.mean (param_values),
                    'std':np.std (param_values),
                    'min':np.min (param_values),
                    'max':np.max (param_values),
                    'importance':self._calculate_numeric_importance (param_name,param_values,top_results)
                }
            else:
                # Pour les paramètres catégoriels
                value_counts={}
                for value in param_values:
                    value_counts[value]=value_counts.get (value,0) + 1

                param_stats[param_name]={
                    'most_common':max (value_counts.items (),key=lambda x:x[1])[0],
                    'value_distribution':{k:v / len (param_values) for k,v in value_counts.items ()},
                    'importance':self._calculate_categorical_importance (param_name,param_values,top_results)
                }

        return param_stats

    def _calculate_numeric_importance (self,
                                       param_name: str,
                                       values: List[float],
                                       results: List[Dict]) -> float:
        """
        Calcule l'importance d'un paramètre numérique.

        Args:
            param_name: Nom du paramètre
            values: Valeurs du paramètre dans les meilleurs résultats
            results: Liste des meilleurs résultats

        Returns:
            Score d'importance (0-1)
        """
        # Si toutes les valeurs sont identiques, le paramètre n'a pas d'influence
        if len (set (values)) <= 1:
            return 0.0

        # Calculer la corrélation entre valeurs du paramètre et scores
        scores=[result['score'] for result in results]

        try:
            correlation=np.abs (np.corrcoef (values,scores)[0,1])
            return min (1.0,max (0.0,correlation))
        except:
            return 0.0

    def _calculate_categorical_importance (self,
                                           param_name: str,
                                           values: List[Any],
                                           results: List[Dict]) -> float:
        """
        Calcule l'importance d'un paramètre catégoriel.

        Args:
            param_name: Nom du paramètre
            values: Valeurs du paramètre dans les meilleurs résultats
            results: Liste des meilleurs résultats

        Returns:
            Score d'importance (0-1)
        """
        # Si toutes les valeurs sont identiques, le paramètre n'a pas d'influence
        if len (set (values)) <= 1:
            return 0.0

        # Pour les paramètres catégoriels, nous analysons la variance des scores par catégorie
        score_by_category={}

        for i,value in enumerate (values):
            if value not in score_by_category:
                score_by_category[value]=[]

            score_by_category[value].append (results[i]['score'])

        # Calculer la variance inter-catégories vs intra-catégories
        category_means=[np.mean (scores) for scores in score_by_category.values () if scores]
        overall_mean=np.mean ([result['score'] for result in results])

        if len (category_means) <= 1:
            return 0.0

        # Variance inter-catégories (pondérée par le nombre d'échantillons)
        between_variance=sum (len (score_by_category[cat]) * (mean - overall_mean) ** 2
                              for cat,mean in zip (score_by_category.keys (),category_means)) / len (values)

        # Variance totale
        total_variance=np.var ([result['score'] for result in results])

        if total_variance == 0:
            return 0.0

        # Ratio de la variance expliquée
        importance=between_variance / total_variance

        return min (1.0,max (0.0,importance))


def get_optimization_profiles () -> Dict[str,Dict]:
    """
    Définit les profils d'optimisation standards pour QAAF.

    Returns:
        Dictionnaire des profils d'optimisation
    """
    return {
        'max_return':{
            'description':'Maximisation du rendement total',
            'score_formula':lambda metrics:metrics['total_return'],
            'constraints':{
                'min_sharpe':lambda metrics:metrics['sharpe_ratio'] >= 0.5,
                'max_drawdown':lambda metrics:metrics['max_drawdown'] >= -70.0
            }
        },
        'balanced':{
            'description':'Équilibre rendement/risque',
            'score_formula':lambda metrics:(
                    0.4 * metrics['total_return'] / 1000 +  # Normalisation
                    0.3 * metrics['sharpe_ratio'] +
                    0.3 * (-metrics['max_drawdown']) / 50  # Normalisation
            ),
            'constraints':{
                'min_return':lambda metrics:metrics['total_return'] >= 50.0,
                'max_drawdown':lambda metrics:metrics['max_drawdown'] >= -50.0
            }
        },
        'min_drawdown':{
            'description':'Minimisation du risque de perte',
            'score_formula':lambda metrics:-metrics['max_drawdown'],
            'constraints':{
                'min_return':lambda metrics:metrics['total_return'] >= 20.0
            }
        },
        'max_sharpe':{
            'description':'Maximisation du ratio rendement/risque',
            'score_formula':lambda metrics:metrics['sharpe_ratio'],
            'constraints':{}
        }
    }


def define_standard_param_grid (memory_constraint: Optional[str] = None) -> Dict[str,List[Any]]:
    """
    Définit la grille paramétrique standard pour l'optimisation QAAF.

    Args:
        memory_constraint: Contrainte mémoire ('low', 'very_low', None)

    Returns:
        Grille de paramètres pour l'optimisation
    """
    if memory_constraint == 'very_low':
        # Configuration minimale pour les environnements très contraints
        return {
            'volatility_window':[30],
            'spectral_window':[60],
            'min_periods':[20],
            'vol_ratio_weight':[0.0,0.3],
            'bound_coherence_weight':[0.3,0.7],
            'alpha_stability_weight':[0.0,0.3],
            'spectral_score_weight':[0.0,0.3],
            'min_btc_allocation':[0.2],
            'max_btc_allocation':[0.8],
            'sensitivity':[1.0],
            'rebalance_threshold':[0.05],
            'observation_period':[7]
        }
    elif memory_constraint == 'low':
        # Configuration restreinte pour environnements à mémoire limitée
        return {
            'volatility_window':[30],
            'spectral_window':[60],
            'min_periods':[20],
            'vol_ratio_weight':[0.0,0.3,0.6],
            'bound_coherence_weight':[0.0,0.3,0.6],
            'alpha_stability_weight':[0.0,0.3,0.6],
            'spectral_score_weight':[0.0,0.3,0.6],
            'min_btc_allocation':[0.2,0.4],
            'max_btc_allocation':[0.6,0.8],
            'sensitivity':[1.0],
            'rebalance_threshold':[0.03,0.05],
            'observation_period':[7]
        }
    else:
        # Configuration standard pour environnements sans contraintes
        return {
            'volatility_window':[20,30,40],
            'spectral_window':[40,60,80],
            'min_periods':[15,20],
            'vol_ratio_weight':[0.0,0.2,0.4,0.6,0.8],
            'bound_coherence_weight':[0.0,0.2,0.4,0.6,0.8],
            'alpha_stability_weight':[0.0,0.2,0.4,0.6,0.8],
            'spectral_score_weight':[0.0,0.2,0.4,0.6,0.8],
            'min_btc_allocation':[0.1,0.2,0.3,0.4],
            'max_btc_allocation':[0.6,0.7,0.8,0.9],
            'sensitivity':[0.8,1.0,1.2],
            'rebalance_threshold':[0.03,0.05,0.07,0.10],
            'observation_period':[3,5,7,10]
        }


def define_param_constraints () -> Dict[str,Callable]:
    """
    Définit les contraintes standard pour filtrer les combinaisons de paramètres.

    Returns:
        Dictionnaire des fonctions de contrainte
    """
    return {
        'valid_allocation_range':lambda params:params['min_btc_allocation'] < params['max_btc_allocation'],
        'valid_metric_weights':lambda params:(
                params['vol_ratio_weight'] +
                params['bound_coherence_weight'] +
                params['alpha_stability_weight'] +
                params['spectral_score_weight'] > 0
        ),
        'valid_windows':lambda params:params['volatility_window'] <= params['spectral_window']
    }