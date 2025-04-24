# Documentation Complète du Projet QAAF v1.0.0

## Table des matières

1. [Introduction](#introduction)
2. [Évolution du projet](#évolution-du-projet)
3. [Architecture et modules](#architecture-et-modules)
4. [Métriques fondamentales](#métriques-fondamentales)
5. [Moteur d'optimisation par grid search](#moteur-doptimisation-par-grid-search)
6. [Profils d'optimisation](#profils-doptimisation)
7. [Modèle d'allocation adaptative](#modèle-dallocation-adaptative)
8. [Validation et robustesse](#validation-et-robustesse)
9. [Gestion des ressources](#gestion-des-ressources)
10. [Installation et utilisation](#installation-et-utilisation)
11. [Perspectives d'évolution](#perspectives-dévolution)

## Introduction

Le Quantitative Algorithmic Asset Framework (QAAF) est un framework mathématique et quantitatif conçu pour analyser et optimiser les stratégies d'allocation entre paires d'actifs, avec un focus particulier sur le couple PAXG/BTC. La version 1.0.0 du framework représente la première version stable et complète, offrant une base solide pour l'optimisation d'allocations dynamiques entre actifs complémentaires.

QAAF 1.0.0 s'appuie sur l'analyse des relations intrinsèques entre actifs plutôt que sur la prédiction de prix futurs, ce qui lui confère une robustesse particulière face aux conditions changeantes du marché. Le framework offre une approche systématique pour créer des "alliages financiers" - des combinaisons d'actifs dont les propriétés combinées surpassent celles des actifs individuels en termes de ratio rendement/risque.

**IMPORTANT** : Cette version 1.0.0 est un modèle expérimental destiné à être amélioré et enrichi par la communauté. Elle a été optimisée pour tirer parti des capacités GPU lorsqu'elles sont disponibles, améliorant significativement les performances de calcul pour les opérations vectorielles intensives.

## Évolution du projet

### Chronologie des versions

| Version | Date | Principales améliorations |
|---------|------|---------------------------|
| 0.5.0   | Q4 2023 | Concepts initiaux et première implémentation des métriques |
| 0.8.0   | Q1 2024 | Moteur d'optimisation préliminaire et backtests basiques |
| 0.9.0   | Q1 2024 | Profils d'optimisation et analyse des phases de marché |
| 1.0.0   | Q2 2024 | Architecture modulaire complète, optimisation par grid search et validation approfondie |

### Améliorations clés de la version 1.0.0

1. **Architecture modulaire complète**
   - Séparation claire des composants fonctionnels
   - Interfaces standardisées entre modules
   - Flexibilité pour les évolutions futures

2. **Optimisation par grid search avancée**
   - Exploration exhaustive de l'espace des paramètres
   - Filtrage intelligent des combinaisons selon les profils
   - Évaluation multidimensionnelle des performances

3. **Allocation adaptative sophistiquée**
   - Ajustement dynamique selon les phases de marché
   - Amplitude variable selon l'intensité des signaux
   - Périodes d'observation post-signal pour éviter les transactions excessives

4. **Validation approfondie**
   - Tests sur la période complète 2020-2024
   - Validation out-of-sample et tests de robustesse
   - Comparaison avec benchmarks statiques

## Architecture et modules

### Structure modulaire

QAAF 1.0.0 est construit autour d'une architecture modulaire qui favorise la réutilisation et l'extensibilité :

```
qaaf/
├── metrics/
│   ├── __init__.py
│   ├── calculator.py       # Calcul des 4 métriques fondamentales
│   └── enhanced_metrics.py # Métriques avancées (Sortino, Calmar) (en attente v2)
├── market/
│   ├── __init__.py
│   └── phase_analyzer.py   # Analyse des phases de marché
├── allocation/
│   ├── __init__.py
│   └── adaptive_allocator.py # Allocation adaptative
│   └── amplitude_calculator.py ???
├── transaction/
│   ├── __init__.py
│   └── fees_evaluator.py   # Évaluation des frais
│   └── rebalance_optimizer.py ???
├── optimization/
│   ├── __init__.py
│   ├── grid_search.py      # Optimisation par grid search
│   └── bayesian_opt.py     # Optimisation bayésienne   (en attente v2)
├── validation/
│   ├── __init__.py
│   ├── out_of_sample.py    # Validation out-of-sample
│   └── robustness.py       # Tests de robustesse
├── data/
│   ├── __init__.py
│   ├── data_manager.py     # Gestion des données
│   └── synthetic_data.py   # Générateur de données synthétiques pour tests
├── utils/
│   ├── __init__.py
│   ├── gpu_utils.py        # Utilitaires pour GPU
│   └── visualizer.py       # Visualisations
├── core/
│   ├── __init__.py
│   └── qaaf_core.py        # Orchestration des composants
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_market_phase.py
│   ├── test_allocation.py
│   ├── test_transaction.py
│   ├── test_data_manager.py
│   └── test_integration.py
└── docs/
    ├── architecture.md     # Documentation de l'architecture
    ├── metrics.md          # Documentation des métriques
    └── results/            # Résultats des tests

```

### Classes principales

| Classe | Rôle principal | Caractéristiques notables |
|--------|----------------|---------------------------|
| `QAAFCore` | Orchestration des composants | Point d'entrée principal, gestion du flux de travail complet |
| `MetricsCalculator` | Calcul des métriques | Implémentation des 4 métriques fondamentales avec paramètres configurables |
| `MarketPhaseAnalyzer` | Analyse des phases de marché | Détection de 6 phases combinant tendance et volatilité |
| `AdaptiveAllocator` | Allocation dynamique | Ajustement des allocations selon le score composite et la phase |
| `QAAFOptimizer` | Optimisation des paramètres | Exploration par grid search des combinaisons de paramètres |
| `TransactionFeesEvaluator` | Évaluation des frais | Simulation des frais et optimisation du seuil de rebalancement |
| `OutOfSampleValidator` | Validation train/test | Vérification de la généralisation des paramètres optimaux |

## Métriques fondamentales

QAAF 1.0.0 s'appuie sur quatre métriques fondamentales qui capturent différentes dimensions de la relation entre les actifs :

### 1. Ratio de Volatilité (vol_ratio)

Le ratio de volatilité compare la volatilité du ratio PAXG/BTC à la volatilité maximale des actifs sous-jacents :

```
vol_ratio = σ(PAXG/BTC) / max(σ(BTC), σ(PAXG))
```

Cette métrique identifie les périodes où le comportement relationnel devient instable ou présente des opportunités spécifiques. Un ratio proche de 0 indique une stabilité relationnelle, tandis qu'un ratio > 1 signale une volatilité accrue de la relation.

### 2. Cohérence des Bornes (bound_coherence)

Cette métrique évalue dans quelle mesure le ratio PAXG/BTC reste dans les limites naturelles définies par les performances individuelles des actifs :

```
bound_coherence = P(min(BTC, PAXG) ≤ PAXG/BTC ≤ max(BTC, PAXG))
```

Une valeur proche de 1 indique une relation stable et prévisible entre les actifs. Cette métrique s'est révélée particulièrement importante, représentant 50% du poids optimal dans la configuration la plus performante.

### 3. Stabilité d'Alpha (alpha_stability)

La stabilité d'alpha mesure la constance des allocations optimales dans le temps :

```
alpha_stability = -σ(α(t))
```

Une faible stabilité suggère des conditions de marché changeantes nécessitant des ajustements plus fréquents. Dans la version 1.0.0, cette métrique s'est avérée moins significative pour la performance optimale.

### 4. Score Spectral (spectral_score)

Le score spectral analyse l'équilibre entre les composantes tendancielles (70%) et oscillatoires (30%) dans le comportement du ratio des actifs :

```
spectral_score = 0.7 * trend_component + 0.3 * oscillation_component
```

Cette métrique a montré une importance considérable, représentant 50% du poids optimal dans la configuration la plus performante.

## Moteur d'optimisation par grid search

### Approche d'optimisation exhaustive

QAAF 1.0.0 utilise une approche d'optimisation par grid search qui explore systématiquement l'espace des paramètres :

1. **Définition de la grille paramétrique**
   - Fenêtres temporelles (volatilité, spectre)
   - Poids des métriques (0% à 100%)
   - Paramètres d'allocation (min/max, sensibilité)
   - Seuils de rebalancement (1% à 10%)

2. **Filtrage des combinaisons**
   - Élimination des combinaisons invalides (ex: somme des poids ≠ 100%)
   - Filtrage selon les contraintes spécifiques du profil

3. **Évaluation des performances**
   - Backtest complet de chaque combinaison
   - Calcul des métriques de performance (rendement, drawdown, Sharpe)
   - Score combiné selon les priorités du profil

4. **Sélection de la configuration optimale**
   - Tri des résultats selon le score
   - Sélection de la meilleure combinaison
   - Analyse des facteurs clés de succès

### Résultats d'optimisation

La combinaison optimale identifiée pour le profil balanced présente les caractéristiques suivantes :

| Paramètre | Valeur optimale |
|-----------|-----------------|
| bound_coherence_weight | 0.50 |
| spectral_score_weight | 0.50 |
| vol_ratio_weight | 0.00 |
| alpha_stability_weight | 0.00 |
| volatility_window | 30 |
| spectral_window | 60 |
| min_btc_allocation | 0.10 |
| max_btc_allocation | 0.90 |
| rebalance_threshold | 0.07 |
| sensitivity | 0.80 |
| observation_period | 3 |

Cette configuration a généré une performance exceptionnelle avec un rendement total de 919.48% et un ratio rendement/drawdown de 19.42.

## Profils d'optimisation

QAAF 1.0.0 propose plusieurs profils d'optimisation prédéfinis pour s'adapter aux différents objectifs d'investissement :

### Profil Balanced

```python
'balanced': {
    'description': 'Équilibre rendement/risque',
    'score_weights': {
        'total_return': 0.4,
        'sharpe_ratio': 30,
        'max_drawdown': 0.3
    },
    'constraints': {
        'min_return': 50.0,
        'max_drawdown': -50.0,
        'min_sharpe': 0.5
    }
}
```

Ce profil vise un équilibre entre rendement et risque, avec des contraintes modérées sur le drawdown maximum et le rendement minimal.

### Profil Max Return

```python
'max_return': {
    'description': 'Maximisation du rendement total',
    'score_weights': {
        'total_return': 1.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0
    },
    'constraints': {
        'min_return': None,
        'max_drawdown': None,
        'min_sharpe': None
    }
}
```

Ce profil prioritise exclusivement le rendement total, sans contraintes sur le risque.

### Autres profils disponibles

- **Safe** : Priorité à la limitation du risque
- **Max Sharpe** : Maximisation du ratio rendement/volatilité
- **Max Efficiency** : Optimisation du ratio rendement/drawdown

## Modèle d'allocation adaptative

### Allocation dynamique selon les signaux

Le modèle d'allocation adaptative ajuste les proportions d'actifs en fonction de plusieurs facteurs :

1. **Score composite normalisé**
   ```python
   normalized_score = (composite_score - composite_score.mean()) / composite_score.std()
   ```

2. **Amplitude adaptée à la phase de marché**
   ```python
   amplitude = amplitude_by_phase.get(market_phase, default_amplitude)
   ```

3. **Intensité du signal**
   ```python
   signal_strength_factor = min(2.0, abs(score) / signal_threshold)
   adjusted_amplitude = amplitude * signal_strength_factor
   ```

4. **Direction de l'allocation**
   ```python
   if score > 0:
       # Signal positif = augmentation de l'allocation BTC
       target_allocation = neutral_allocation + (max_allocation - neutral_allocation) * adjusted_amplitude
   else:
       # Signal négatif = diminution de l'allocation BTC
       target_allocation = neutral_allocation - (neutral_allocation - min_allocation) * adjusted_amplitude
   ```

### Périodes d'observation

Pour éviter les transactions excessives, le modèle maintient l'allocation après un signal fort :

```python
if days_since_signal < observation_period:
    # Maintien de l'allocation pendant la période d'observation
    allocation = last_allocation
else:
    # Retour progressif vers l'allocation neutre
    recovery_factor = min(1.0, (days_since_signal - observation_period) / 10)
    allocation = last_allocation + (neutral_allocation - last_allocation) * recovery_factor
```

## Validation et robustesse

### Validation out-of-sample

QAAF 1.0.0 implémente une validation out-of-sample rigoureuse :

1. **Division des données**
   - 70% pour l'entraînement (optimisation des paramètres)
   - 30% pour le test (validation des paramètres optimaux)

2. **Méthodologie**
   - Optimisation complète sur l'ensemble d'entraînement
   - Application des paramètres optimaux sur l'ensemble de test
   - Comparaison des performances train/test

3. **Métriques de robustesse**
   - Ratio de consistance (performance test / performance train)
   - Stabilité des paramètres optimaux
   - Détection du surapprentissage

### Tests de stress

Des scénarios de marché spécifiques sont utilisés pour évaluer la robustesse de la stratégie :

- **Bull Market** : Période haussière forte (2020-10 à 2021-04)
- **Bear Market** : Période baissière prolongée (2022-01 à 2022-06)
- **Consolidation** : Période de faible directionnalité (2023-01 à 2023-06)
- **Recovery** : Période de reprise graduelle (2023-07 à 2024-12)

## Gestion des ressources

### Optimisations computationnelles

QAAF 1.0.0 inclut plusieurs optimisations pour améliorer l'efficacité des calculs :

1. **Filtrage intelligent des combinaisons**
   ```python
   if params['min_btc_allocation'] >= params['max_btc_allocation']:
       return False  # Combinaison invalide
   
   weight_sum = (
       params['vol_ratio_weight'] + 
       params['bound_coherence_weight'] + 
       params['alpha_stability_weight'] + 
       params['spectral_score_weight']
   )
   if weight_sum <= 0:
       return False  # Combinaison invalide
   ```

2. **Limitation du nombre de combinaisons**
   ```python
   if len(valid_combinations) > max_combinations:
       combinations_to_test = valid_combinations[:max_combinations]
   ```

3. **Gestion des erreurs robuste**
   ```python
   try:
       # Opérations d'optimisation
   except Exception as e:
       logger.error(f"Erreur lors de l'optimisation: {str(e)}")
       # Retour des résultats partiels
   ```

### Défis et limitations actuelles

La version 1.0.0 présente encore certaines limitations en termes de gestion des ressources :

1. **Consommation mémoire**
   - L'optimisation par grid search nécessite une mémoire importante
   - Les backtests multiples peuvent saturer la RAM disponible

2. **Temps de calcul**
   - L'exploration exhaustive est computationnellement intense
   - Certaines optimisations peuvent prendre plusieurs heures

3. **Extensibilité**
   - L'approche actuelle ne se parallélise pas nativement
   - Limitations pour le traitement de très grandes quantités de données

## Configuration et utilisation

### Initialisation basique

```python
qaaf = QAAFCore(
    initial_capital=30000.0,
    trading_costs=0.001,  # 10 points de base
    start_date='2020-01-01',
    end_date='2024-12-31'
)

results = qaaf.run_full_analysis(
    optimize_metrics=True,
    optimize_threshold=True,
    run_validation=True,
    profile='balanced'
)
```

### Configuration avancée

```python
# Configuration personnalisée des métriques
qaaf.metrics_calculator.update_parameters(
    volatility_window=30,
    spectral_window=60,
    min_periods=20
)

# Configuration personnalisée de l'allocateur
qaaf.adaptive_allocator.update_parameters(
    min_btc_allocation=0.1,
    max_btc_allocation=0.9,
    sensitivity=0.8,
    observation_period=3
)

# Configuration personnalisée du backtester
qaaf.backtester.update_parameters(
    rebalance_threshold=0.07
)
```

### Visualisation des résultats

```python
# Affichage du résumé
qaaf.print_summary()

# Visualisation des résultats
qaaf.visualize_results()

# Sauvegarde des résultats
results_dir = save_and_visualize_results(qaaf, results)
```

## Perspectives d'évolution

### Priorités pour QAAF 1.x.x

La feuille de route pour les versions mineures futures se concentre sur la consolidation et l'optimisation :

1. **QAAF 1.1.0 (Priorité HAUTE)**
   - Refactorisation modulaire complète
   - Correction du bug de comparaison avec benchmarks
   - Mise en place de l'infrastructure de test

2. **QAAF 1.2.0 (Priorité HAUTE)**
   - Système de mise en cache local des données
   - Parallélisation des backtests
   - Optimisations mémoire

3. **QAAF 1.3.0 (Priorité MOYENNE)**
   - Tests de robustesse avancés
   - Rapports améliorés et visualisations interactives
   - API de configuration simplifiée

### Vision pour QAAF 2.0.0

Le développement de QAAF 2.0.0 représente l'objectif stratégique à moyen terme, avec plusieurs innovations majeures :

1. **Métriques optimisées**
   - Sous-ensemble optimal des métriques actuelles
   - Métriques composites à haute valeur signalétique
   - Élimination des redondances informationnelles

2. **Optimisation bayésienne**
   - Remplacement du grid search exhaustif
   - Exploration adaptative de l'espace des paramètres
   - Convergence plus rapide vers les solutions optimales

3. **Architecture distribuée**
   - Calcul parallèle natif
   - Stockage et traitement optimisés des données
   - API pour intégration avec d'autres systèmes

4. **Intégration du Machine Learning**
   - Détection des phases de marché par clustering
   - Apprentissage des relations non-linéaires entre métriques
   - Identification avancée des signaux de rebalancement

### Applications futures

Au-delà de l'alliage BTC/PAXG, QAAF pourrait être appliqué à d'autres paires d'actifs complémentaires :

- **Or/Actions (GLD/SPY)** - Protection contre l'inflation avec participation à la croissance
- **Bitcoin/Obligations (BTC/TLT)** - Combinaison extrêmes opposés du spectre risque/rendement
- **Stablecoins/Actifs de croissance (USDC/ETH)** - Exposition contrôlée avec base stable
- **Matières premières/Technologies (GSG/QQQ)** - Couverture inflation et exposition innovation
- **Multi-alliages** - Combinaisons de trois actifs ou plus pour des propriétés encore plus sophistiquées

La structure modulaire de QAAF 1.0.0 pose les fondations nécessaires pour ces évolutions futures, tout en offrant dès maintenant une performance impressionnante avec la paire BTC/PAXG.

## Installation et utilisation

### Installation depuis GitHub

#### Sur un environnement de développement personnel

```bash
# Clonage du dépôt
git clone https://github.com/dravitch/qaaf-project.git
cd qaaf-project

# Installation des dépendances
pip install -r requirements.txt

# Installation du package en mode développement (optionnel)
pip install -e .
```

#### Sur Google Colab

```python
# Installation depuis GitHub
!git clone https://github.com/dravitch/qaaf.git
%cd qaaf


# Installation des dépendances
!pip install -r requirements.txt

# Ajout du chemin au PYTHONPATH
import sys
sys.path.append('/content/qaaf')
print(sys.path)

# Vérification de l'installation
!python -c "from qaaf.core.qaaf_core import QAAFCore; print('Installation réussie!')"
```

#### Sur Kaggle

```python
# Installation depuis GitHub
!git clone https://github.com/dravitch/qaaf.git
%cd qaaf
# Installation des dépendances
!pip install -r requirements.txt

# Ajout du chemin au PYTHONPATH
import os
os.chdir('/kaggle/working')

# Vérification de l'installation
!python -c "from qaaf.core.qaaf_core import QAAFCore; print('Installation réussie!')"
```

### Configuration de l'environnement

QAAF détecte automatiquement la disponibilité du GPU et optimise ses calculs en conséquence :

```python
from qaaf.utils.gpu_utils import initialize_gpu_support, GPU_AVAILABLE

# Vérification de la disponibilité du GPU
status = "disponible" if GPU_AVAILABLE else "non disponible"
print(f"Support GPU: {status}")
```

### Exemple d'utilisation basique

```python
from core.qaaf_core import QAAFCore
from data.synthetic_data import generate_sample_data

# Génération de données de test (pour démonstration)
data = generate_sample_data(periods=500, with_scenarios=True)

# Initialisation de QAAF
qaaf = QAAFCore(
    initial_capital=30000.0,
    trading_costs=0.001,  # 10 points de base
    start_date='2020-01-01',
    end_date='2021-12-31'
)

# Utilisation des données générées
qaaf.data = data

# Analyse des phases de marché
qaaf.analyze_market_phases()

# Calcul des métriques
qaaf.calculate_metrics()

# Calcul du score composite avec des poids égaux
weights = {
    'vol_ratio': 0.25,
    'bound_coherence': 0.25, 
    'alpha_stability': 0.25,
    'spectral_score': 0.25
}
qaaf.calculate_composite_score(weights)

# Calcul des allocations adaptatives
qaaf.calculate_adaptive_allocations()

# Exécution du backtest
qaaf.run_backtest()

# Affichage des résultats
qaaf.print_summary()

# Visualisation des résultats
qaaf.visualize_results()
```

### Utilisation avancée avec optimisation

```python
# Exécution de l'analyse complète avec optimisation
results = qaaf.run_full_analysis(
    optimize_metrics=True,       # Activer l'optimisation des métriques
    optimize_threshold=True,     # Activer l'optimisation du seuil
    run_validation=True,         # Activer la validation out-of-sample
    profile='balanced'           # Profil d'optimisation
)

# Accès aux résultats
if 'metrics_results' in results:
    best_weights = results['metrics_results']['best_combinations']['balanced']['normalized_weights']
    print(f"Poids optimaux: {best_weights}")

if 'validation_results' in results:
    test_performance = results['validation_results']['test']['performance']
    print(f"Performance sur données de test: {test_performance['total_return']:.2f}%")
```

### Optimisation pour environnements à ressources limitées

Pour les environnements comme Google Colab ou Kaggle avec des limitations de ressources, QAAF peut être configuré pour minimiser la consommation de mémoire :

```python
# Exécution avec optimisation des ressources
results = qaaf.run_full_analysis(
    optimize_metrics=True,       
    optimize_threshold=False,    # Désactivé pour économiser la mémoire
    run_validation=False,        # Désactivé pour économiser la mémoire
    profile='balanced'
)
```

### Création de visualisations personnalisées

```python
from qaaf.utils.visualizer import plot_performance_comparison, plot_metrics

# Visualisation de la performance
fig1 = plot_performance_comparison(
    qaaf.performance,
    qaaf.data,
    qaaf.allocations,
    log_scale=False,
    title="Performance de la stratégie QAAF"
)

# Visualisation des métriques
fig2 = plot_metrics(
    qaaf.metrics,
    qaaf.market_phases,
    save_path="metriques_qaaf.png"
)
```
## Objectifs de la version 1.0.0

Cette première version stable de QAAF vise à :

1. **Établir une architecture modulaire** permettant des extensions futures
2. **Démontrer la validité de l'approche basée sur métriques** sans recours à la prédiction directe
3. **Optimiser l'allocation d'actifs complémentaires** pour créer des "alliages financiers" performants
4. **Offrir un framework adaptatif** capable de s'ajuster aux différentes phases de marché
5. **Exploiter les capacités GPU** pour accélérer significativement les calculs intensifs

La version 1.0.0 est à considérer comme un socle solide sur lequel pourront s'appuyer des améliorations futures, notamment l'intégration de techniques d'apprentissage automatique plus avancées et l'extension à des portefeuilles multi-actifs.

## Recommandations d'utilisation

1. **Exploration et recherche** : Utilisez cette version pour explorer les relations entre métriques et performances dans différentes conditions de marché
2. **Expérimentation** : Testez différents profils d'optimisation et contraintes pour comprendre leur impact
3. **Validation approfondie** : Appliquez systématiquement les tests de robustesse avant tout déploiement
4. **Extension** : Le framework est conçu pour être étendu avec vos propres métriques et stratégies d'allocation

## Feuille de route pour versions futures

Consultez la section [Perspectives d'évolution](#perspectives-dévolution) pour plus de détails sur les améliorations prévues pour les versions futures de QAAF.
La documentation pour l'utilisation du framework QAAF continue comme suit :

## Intégration avec d'autres outils

QAAF peut s'intégrer facilement avec d'autres outils et bibliothèques d'analyse financière :

### Intégration avec pandas-ta

```python
import pandas as pd
import pandas_ta as ta
from qaaf.core.qaaf_core import QAAFCore

# Préparation des données avec indicateurs techniques supplémentaires
def prepare_enhanced_data(data):
    enhanced_data = data.copy()
    
    for asset in ['BTC', 'PAXG']:
        df = enhanced_data[asset]
        # Ajout d'indicateurs techniques
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        enhanced_data[asset] = df
    
    return enhanced_data

# Utilisation avec QAAF
qaaf = QAAFCore()
qaaf.data = prepare_enhanced_data(qaaf.data)
```

### Export des résultats

```python
import json
from qaaf.utils.visualizer import save_performance_summary

# Export des résultats en format texte
summary = save_performance_summary(
    qaaf.results,
    qaaf.results.get('comparison'),
    file_path="qaaf_performance_summary.md"
)

# Export des résultats en JSON
with open('qaaf_results.json', 'w') as f:
    json.dump({
        'metrics': qaaf.results['metrics'],
        'total_transactions': len(qaaf.backtester.transaction_history),
        'total_fees': qaaf.fees_evaluator.get_total_fees()
    }, f, indent=2)
```

## Diagnostics et résolution de problèmes

### 1. Vérification de l'environnement

Cette section aide les utilisateurs à s'assurer que leur environnement est correctement configuré avec les versions requises des bibliothèques.

```python
import sys
import numpy as np
import pandas as pd
import logging

def check_environment():
    """
    Vérifie les versions de Python et des bibliothèques essentielles.
    Fournit également des informations sur la disponibilité du GPU.
    """
    print("=== Environnement ===")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    # Vérification GPU (CuPy)
    try:
        import cupy
        print(f"CuPy version: {cupy.__version__}")
        print(f"CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        print("CuPy non disponible. Le calcul GPU est désactivé.")
    except AttributeError:
        print("CuPy installé, mais CUDA non détecté.")

    # Vérification GPU (PyTorch)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch non disponible.")

    print("=======================")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Exécuter la vérification
check_environment()
```

**Améliorations :**

* Ajout de la vérification de PyTorch et de l'affichage du nom du GPU.
* Gestion plus précise des erreurs liées à CuPy (distinction entre `ImportError` et `AttributeError`).
* Utilisation du module `logging` pour une meilleure gestion des logs.
* Formatage de la sortie pour une meilleure lisibilité.

### 2. Gestion des erreurs courantes

Cette section guide les utilisateurs à travers les solutions aux problèmes les plus fréquents qu'ils peuvent rencontrer lors de l'utilisation du framework QAAF.

#### 2.1 Erreurs de mémoire (Colab/Kaggle)

Les environnements Colab et Kaggle peuvent avoir des ressources limitées. Voici comment gérer les erreurs de mémoire :

```python
# 1. Réduire la plage de données
qaaf = QAAFCore(start_date='2022-01-01', end_date='2022-06-30')  # Exemple: 6 mois au lieu de plusieurs années
logger.info("Plage de données réduite pour économiser la mémoire.")

# 2. Désactiver les optimisations coûteuses
results = qaaf.run_full_analysis(
    optimize_metrics=True,
    optimize_threshold=False,  # Désactiver l'optimisation du seuil
    run_validation=False,      # Désactiver la validation
    run_robustness=False,      # Désactiver les tests de robustesse
    profile='balanced'
)
logger.info("Optimisations et validations désactivées pour économiser la mémoire.")

# 3. Libérer manuellement la mémoire GPU (si applicable)
try:
    from qaaf.utils.gpu_utils import clear_memory  # Assurez-vous que cet utilitaire existe
    clear_memory()
    logger.info("Mémoire GPU libérée.")
except ImportError:
    logger.warning("Fonction clear_memory non disponible.")
except NameError:
    logger.warning("Fonction clear_memory non définie.")
```

**Améliorations :**

* Utilisation du module `logging` pour informer l'utilisateur des actions entreprises.
* Ajout de la désactivation des tests de robustesse comme option pour économiser la mémoire.
* Gestion des exceptions lors de l'appel à `clear_memory`.
* Documentation plus claire des stratégies.

#### 2.2 AttributeError: 'QAAFCore' object has no attribute '...'

Cette erreur se produit lorsque vous essayez d'appeler une méthode qui n'est pas définie dans la classe `QAAFCore`.

**Causes possibles et solutions :**

* **Nom de la méthode incorrect :** Vérifiez l'orthographe et la casse du nom de la méthode. Utilisez l'autocomplétion de votre IDE ou consultez la documentation.
* **Version obsolète du code :** Assurez-vous que vous utilisez la dernière version du code du dépôt. Redémarrez le runtime et réexécutez toutes les cellules.
* **Ordre d'appel des méthodes incorrect :** Certaines méthodes peuvent dépendre de l'exécution préalable d'autres méthodes (par exemple, `load_data()` doit être appelé avant `analyze_market_phases()`). Consultez la documentation ou le code source pour l'ordre correct.

    ```python
    qaaf = QAAFCore()
    qaaf.load_data()  # Appel explicite de load_data()
    qaaf.analyze_market_phases()  # Maintenant, cela devrait fonctionner
    ```

* **Objet `qaaf` incorrect :** Vérifiez que `qaaf` est bien une instance de `QAAFCore` :

    ```python
    print(type(qaaf))
    ```

#### 2.3 ModuleNotFoundError: No module named 'qaaf'

Cette erreur indique que Python ne trouve pas le package `qaaf`.

**Causes possibles et solutions :**

* **`qaaf` n'est pas installé :** Assurez-vous d'avoir correctement installé le package. Si vous utilisez Colab ou Kaggle, cela peut impliquer de cloner le dépôt et d'ajouter le chemin au `sys.path`.

    ```python
    import sys
    if '/path/to/qaaf' not in sys.path:  # Remplacez '/path/to/qaaf' par le chemin réel
        sys.path.append('/path/to/qaaf')
    ```

* **Erreur dans les imports relatifs :** Si l'erreur se produit à l'intérieur du code de `qaaf` lui-même, cela peut être dû à des imports relatifs incorrects. Assurez-vous que les imports utilisent la syntaxe correcte (`from .module import ...`).

J'espère que cette version mise à jour sera plus complète et utile pour les utilisateurs de votre code!