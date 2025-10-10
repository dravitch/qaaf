# QAAF v1.1 - Quantitative Algorithmic Asset Framework

Framework d'analyse et de trading algorithmique pour portefeuilles BTC/PAXG avec allocation adaptative.

## 🚀 Démarrage Rapide

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/dravitch/qaaf.git
cd qaaf

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier l'installation
python check.py --fix
```

### Utilisation Basique

```python
from qaaf.core import QAAFCore

# Créer une instance
qaaf = QAAFCore(
    initial_capital=30000,
    trading_costs=0.001,
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Lancer l'analyse complète
results = qaaf.run_full_analysis()
```

## 🛠️ Scripts de Développement

### `check.py` - Vérification et Réparation

Vérifie que la structure du projet est correcte et répare automatiquement les problèmes courants.

```bash
# Vérifier seulement
python check.py

# Vérifier et réparer
python check.py --fix
```

### `test.py` - Tests Fonctionnels

Teste que QAAF fonctionne correctement.

```bash
# Test rapide (6 mois de données)
python test.py --quick

# Test complet (5 ans de données)
python test.py
```

### `save.py` - Sauvegarde sur GitHub

Sauvegarde automatiquement vos modifications sur GitHub.

```bash
# Avec message personnalisé
python save.py "Description des changements"

# Avec message par défaut
python save.py
```

## 📁 Structure du Projet

```
qaaf/
├── core/               # Moteur principal QAAF
│   └── qaaf_core.py   # Classe principale
├── metrics/            # Calcul des métriques
│   └── calculator.py  # 4 métriques primaires
├── market/             # Analyse de marché
│   └── phase_analyzer.py  # Détection des phases
├── allocation/         # Stratégies d'allocation
│   └── adaptive_allocator.py  # Allocation adaptative
├── transaction/        # Gestion des transactions
│   ├── fees_evaluator.py  # Calcul des frais
│   └── backtester.py      # Moteur de backtest
├── data/               # Gestion des données
│   └── data_manager.py    # Chargement des données
└── validation/         # Validation et tests
    ├── out_of_sample.py   # Validation OOS
    └── robustness.py      # Tests de robustesse
```

## 🔬 Méthodologie QAAF

QAAF utilise 4 métriques primaires pour évaluer les opportunités d'allocation:

1. **Ratio de Volatilité**: Mesure la volatilité relative du ratio PAXG/BTC
2. **Cohérence des Bornes**: Vérifie que le ratio reste dans les bornes naturelles
3. **Stabilité d'Alpha**: Évalue la stabilité des allocations
4. **Score Spectral**: Analyse tendancielle et oscillatoire

Ces métriques sont combinées en un score composite qui guide l'allocation adaptative.

## 📊 Exemples d'Utilisation

### Analyse Complète avec Optimisation

```python
from qaaf.core import QAAFCore

qaaf = QAAFCore()

# Analyse complète avec optimisation
results = qaaf.run_full_analysis(
    optimize_metrics=True,
    optimize_threshold=True,
    run_validation=True,
    profile='balanced'
)

# Afficher les résultats
qaaf.print_summary()
qaaf.visualize_results()
```

### Backtest Simple

```python
# Charger les données
qaaf.load_data()

# Analyser le marché
qaaf.analyze_market_phases()

# Calculer les métriques
qaaf.calculate_metrics()
qaaf.calculate_composite_score()

# Calculer les allocations
qaaf.calculate_adaptive_allocations()

# Exécuter le backtest
results = qaaf.run_backtest()

# Résultats
print(f"Rendement: {results['metrics']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

### Validation Out-of-Sample

```python
# Test de robustesse avec validation OOS
validation_results = qaaf.run_out_of_sample_validation(
    test_ratio=0.3,
    profile='balanced'
)

# Afficher le résumé
qaaf.validator.print_validation_summary()
```

## 🧪 Tests et Validation

### Exécuter les Tests

```bash
# Tests unitaires (à venir)
pytest tests/

# Test d'intégration
python test.py

# Validation complète
python test.py && python check.py
```

### Critères de Validation

Un algorithme QAAF est considéré comme valide si:

- ✅ Sharpe ratio out-of-sample > 0.5
- ✅ Dégradation de performance < 40% entre in-sample et out-of-sample
- ✅ Drawdown maximum < 30%
- ✅ Performance supérieure à un portfolio 60/40 statique

## 🔄 Workflow de Développement

### Branches

- `main`: Version stable
- `develop`: Développement continu
- `v1.x-fixes`: Corrections de version spécifique
- `feature/*`: Nouvelles fonctionnalités

### Processus de Contribution

1. Créer une branche depuis `develop`
2. Faire vos modifications
3. Vérifier: `python check.py --fix`
4. Tester: `python test.py`
5. Sauvegarder: `python save.py "Description"`
6. Créer une Pull Request vers `develop`

## 📈 Roadmap

### v1.1 (En cours)
- [x] Structure modulaire fonctionnelle
- [x] Scripts de diagnostic et test automatiques
- [ ] Documentation complète
- [ ] Tests unitaires

### v1.2 (Prévu)
- [ ] Optimisation GPU complète
- [ ] Support multi-actifs (au-delà de BTC/PAXG)
- [ ] Dashboard de visualisation interactif
- [ ] API REST pour intégration externe

### v2.0 (Vision)
- [ ] Trading en temps réel
- [ ] Machine learning pour optimisation dynamique
- [ ] Support de multiples exchanges
- [ ] Gestion multi-portefeuilles

## 📝 License

MIT License - Voir le fichier LICENSE pour plus de détails.

## 🤝 Support

- **Issues**: https://github.com/dravitch/qaaf/issues
- **Discussions**: https://github.com/dravitch/qaaf/discussions
- **Documentation**: https://github.com/dravitch/qaaf/wiki

## 📚 Ressources

- [Guide de Démarrage](docs/getting_started.md)
- [Architecture Détaillée](docs/architecture.md)
- [Méthodologie QAAF](docs/methodology.md)
- [FAQ](docs/faq.md)

---

**Note**: QAAF est un outil de recherche et d'analyse. Utilisez-le à vos propres risques en environnement de production.
