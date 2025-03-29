# scripts/manual_test.py
from qaaf.core.qaaf_core import QAAFCore
import matplotlib.pyplot as plt


def main ():
    """Test manuel interactif du flux complet"""
    # Initialisation de QAAF
    qaaf=QAAFCore (
        initial_capital=30000.0,
        trading_costs=0.001,
        start_date='2020-01-01',
        end_date='2022-12-31',
        use_gpu=False  # Changez à True pour tester l'accélération GPU
    )

    # Chargement des données (avec mesure du temps)
    import time
    start_time=time.time ()
    qaaf.load_data ()
    print (f"Chargement des données: {time.time () - start_time:.2f} secondes")

    # Affichage des informations sur les données
    for key,df in qaaf.data.items ():
        print (f"{key}: {len (df)} points de données du {df.index[0]} au {df.index[-1]}")

    # Analyse des phases de marché
    start_time=time.time ()
    qaaf.analyze_market_phases ()
    print (f"Analyse des phases: {time.time () - start_time:.2f} secondes")

    # Distribution des phases
    phase_counts=qaaf.market_phases.value_counts ()
    print ("Distribution des phases de marché:")
    for phase,count in phase_counts.items ():
        print (f"  {phase}: {count} jours ({count / len (qaaf.market_phases) * 100:.1f}%)")

    # Calcul des métriques
    start_time=time.time ()
    qaaf.calculate_metrics ()
    print (f"Calcul des métriques: {time.time () - start_time:.2f} secondes")

    # Affichage des statistiques des métriques
    for name,series in qaaf.metrics.items ():
        print (f"{name}: min={series.min ():.4f}, max={series.max ():.4f}, mean={series.mean ():.4f}")

    # Suite du flux...
    # [...]

    # Affichage des résultats finaux
    if qaaf.results:
        qaaf.print_summary ()
        qaaf.visualize_results ()


if __name__ == "__main__":
    main ()