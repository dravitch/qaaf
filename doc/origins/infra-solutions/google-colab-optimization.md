# Optimisation pour Google Colab

## Gestion de la mémoire en environnement contraint

Ce document présente les techniques d'optimisation pour exécuter efficacement des scripts gourmands en ressources dans Google Colab, en particulier pour des modèles complexes comme QAAF.

### Table des matières

1. [Gestion de la mémoire CPU](#gestion-de-la-mémoire-cpu)
2. [Utilisation efficace du GPU](#utilisation-efficace-du-gpu)
3. [Optimisation de la structure de code](#optimisation-de-la-structure-de-code)
4. [Paramètres recommandés pour QAAF](#paramètres-recommandés-pour-qaaf)
5. [Supervision des ressources](#supervision-des-ressources)

## Gestion de la mémoire CPU

### Éviter le stockage de résultats intermédiaires volumineux

**Problème :** Le stockage de grandes structures de données temporaires consomme beaucoup de RAM.

**Solutions :**
- **Traiter par lots :** Diviser les données en petits lots plutôt que de tout charger d'un coup
- **Utiliser des générateurs :** Ils ne chargent les données en mémoire que lorsque c'est nécessaire
  ```python
  # Au lieu de
  results = [process(x) for x in large_data]
  
  # Utiliser
  def result_generator():
      for x in large_data:
          yield process(x)
  ```

### Utiliser efficacement les tableaux NumPy

**Problème :** Les tableaux NumPy peuvent consommer beaucoup de RAM s'ils ne sont pas utilisés avec précaution.

**Solutions :**
- **Spécifier des types de données :** Utiliser `np.float32` au lieu de `np.float64` par défaut
  ```python
  # Au lieu de
  array = np.zeros(1000000)  # Utilise float64 par défaut
  
  # Utiliser
  array = np.zeros(1000000, dtype=np.float32)  # Réduit la consommation de mémoire de 50%
  ```

- **Opérations en place :** Modifier les tableaux sans créer de copies
  ```python
  # Au lieu de
  B = A + 1  # Crée un nouveau tableau
  
  # Utiliser
  A += 1  # Modifie A en place
  ```

### Nettoyer les variables inutilisées

**Problème :** Les variables qui ne sont plus nécessaires restent en mémoire.

**Solutions :**
- **Mot-clé `del` :** Supprimer explicitement les grandes variables
  ```python
  del large_variable
  ```
- **Portée des fonctions :** Définir les variables dans des fonctions pour qu'elles soient automatiquement libérées
  ```python
  def process_data():
      # Variable locale, libérée après l'exécution
      large_array = np.random.random((10000, 10000))
      result = large_array.mean()
      return result
  ```

## Utilisation efficace du GPU

### Activer l'accélérateur GPU

1. Dans le menu de Colab : **Exécution** > **Modifier le type d'exécution**
2. Sélectionner **GPU** dans le menu déroulant
3. Cliquer sur **Enregistrer**

### Vérifier l'activation du GPU

```python
# Vérifier si le GPU est disponible
!nvidia-smi
```

### Optimiser l'utilisation du GPU

**Problème :** Les calculs peuvent être effectués sur CPU même si le GPU est disponible.

**Solutions :**
- **Utiliser des opérations vectorisées :** Privilégier les opérations matricielles NumPy
- **Éviter les boucles Python :** Elles s'exécutent sur CPU et sont souvent plus lentes

### Libérer la mémoire GPU

```python
# Libérer la mémoire GPU entre les exécutions
import gc
gc.collect()
```

## Optimisation de la structure de code

### Vectoriser les opérations

**Problème :** Les boucles sont moins efficaces que les opérations vectorisées.

**Solutions :**
- **Utiliser les opérations vectorisées de NumPy :**
  ```python
  # Au lieu de
  result = []
  for i in range(len(data)):
      result.append(data[i] * 2)
  
  # Utiliser
  result = data * 2  # Opération vectorisée
  ```

### Réduire les appels de fonctions

**Problème :** Les appels de fonctions peuvent entraîner des coûts de performance.

**Solutions :**
- **Éviter les appels inutiles dans les boucles :**
  ```python
  # Au lieu de
  for i in range(n):
      x = expensive_function()  # Appelé n fois avec le même résultat
  
  # Utiliser
  x = expensive_function()  # Appelé une seule fois
  for i in range(n):
      # Utiliser x
  ```

### Mise en cache des résultats

**Problème :** Recalcul de données identiques.

**Solutions :**
- **Stocker les résultats calculés :**
  ```python
  # Décorateur de mise en cache
  from functools import lru_cache
  
  @lru_cache(maxsize=None)
  def expensive_calculation(x):
      # Calcul coûteux
      return result
  ```

## Paramètres recommandés pour QAAF

Pour exécuter QAAF 1.0.0 dans Google Colab avec une utilisation optimale des ressources :

### Configuration minimale

```python
qaaf, results = run_qaaf(
    optimize_metrics=True,       # Activé car c'est l'essence de QAAF 1.0.0
    optimize_threshold=False,    # Désactivé pour économiser la mémoire
    run_validation=False,        # Désactivé pour économiser la mémoire
    profile='balanced',          # Profil standard
    verbose=False                # Réduit les logs
)
```

### Réduction de la grille de paramètres

Modifier la méthode `define_parameter_grid()` dans la classe `QAAFOptimizer` :

```python
def define_parameter_grid(self):
    """Définit une grille de paramètres réduite pour Google Colab"""
    return {
        # Fenêtres de calcul réduites
        'volatility_window': [30],                     # Valeur unique
        'spectral_window': [60],                       # Valeur unique
        'min_periods': [20],                           # Valeur unique
        
        # Poids des métriques avec moins d'options
        'vol_ratio_weight': [0.0, 0.3, 0.6],           # 3 options au lieu de 8
        'bound_coherence_weight': [0.0, 0.3, 0.6],     # 3 options au lieu de 8
        'alpha_stability_weight': [0.0, 0.3, 0.6],     # 3 options au lieu de 8
        'spectral_score_weight': [0.0, 0.3, 0.6],      # 3 options au lieu de 8
        
        # Paramètres d'allocation réduits
        'min_btc_allocation': [0.2, 0.4],              # 2 options au lieu de 4
        'max_btc_allocation': [0.6, 0.8],              # 2 options au lieu de 4
        'sensitivity': [1.0],                          # Valeur unique
        
        # Paramètres de rebalancement réduits
        'rebalance_threshold': [0.03, 0.05],           # 2 options au lieu de 5
        'observation_period': [7]                      # Valeur unique
    }
```

### Limitation du nombre de combinaisons

```python
# Dans l'appel de run_metrics_optimization
metrics_results = self.run_metrics_optimization(profile=profile, max_combinations=500)
```

## Supervision des ressources

### Surveiller l'utilisation de la RAM

```python
# Afficher l'utilisation de la RAM
!free -h
```

### Surveiller l'utilisation du GPU

```python
# Vérifier l'utilisation du GPU
!nvidia-smi
```

### Utiliser la barre de progression pour les longues opérations

```python
from tqdm.notebook import tqdm

for params in tqdm(combinations_to_test, desc="Optimisation en cours..."):
    # Traitement
```

## Conclusion

En appliquant ces techniques d'optimisation, vous pouvez significativement améliorer les performances et la stabilité de scripts gourmands en ressources comme QAAF dans Google Colab. Ces techniques permettent de maximiser l'utilisation des ressources limitées disponibles et d'éviter les crashs liés à une saturation de la mémoire.

Le compromis entre l'exhaustivité de l'analyse et les contraintes matérielles peut être géré efficacement en ajustant les paramètres et la structure du code en fonction des besoins spécifiques du projet.