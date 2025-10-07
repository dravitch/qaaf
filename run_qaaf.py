'''
# Fonction pour exécuter dans Google Colab
def run_qaaf (optimize_metrics: bool = True,optimize_threshold: bool = True):
    """Exécute le framework QAAF dans Google Colab"""

    print ("\n🔹 QAAF (Quantitative Algorithmic Asset Framework) - Version 1.0.8 🔹")
    print ("=" * 70)
    print ("Ajout d'un module d'optimisation avancé des métriques et des frais")
    print ("Identification de combinaisons optimales selon différents profils de risque/rendement")
        '''


"""
Point d'entrée principal pour exécuter QAAF
"""
import logging
from qaaf.core.qaaf_core import QAAFCore

def run_qaaf(
    optimize_metrics: bool = True,
    optimize_threshold: bool = True,
    run_validation: bool = True,
    profile: str = 'balanced',
    verbose: bool = True
):
    """
    Fonction principale d'exécution de QAAF 1.0.0
    """
    # Configuration du niveau de logging
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    print("\n🔹 QAAF (Quantitative Algorithmic Asset Framework) - Version 1.0.0 🔹")
    print("=" * 70)
    
    # Configuration
    initial_capital = 30000.0
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    trading_costs = 0.001
    
    print(f"\nConfiguration:")
    print(f"- Capital initial: ${initial_capital:,.2f}")
    print(f"- Période d'analyse: {start_date} à {end_date}")
    print(f"- Frais de transaction: {trading_costs:.2%}")
    print(f"- Profil d'optimisation: {profile}")
    
    # Initialisation
    qaaf = QAAFCore(
        initial_capital=initial_capital,
        trading_costs=trading_costs,
        start_date=start_date,
        end_date=end_date,
        allocation_min=0.1,
        allocation_max=0.9
    )
    
    # Exécution de l'analyse complète
    print("\n📊 Démarrage de l'analyse...\n")
    
    try:
        results = qaaf.run_full_analysis(
            optimize_metrics=optimize_metrics,
            optimize_threshold=optimize_threshold,
            run_validation=run_validation,
            profile=profile
        )
        
        print("\n✅ Analyse complétée avec succès!")
        return qaaf, results
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None, None


if __name__ == "__main__":
    qaaf, results = run_qaaf()
