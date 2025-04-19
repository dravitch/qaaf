def main ():
    """Test des composants QAAF avec des données de test"""
    # Configuration du logging
    logging.basicConfig (level=logging.INFO)
    logger.info ("Test des composants QAAF")

    # Création de données de test
    dates=pd.date_range (start='2020-01-01',end='2024-02-17',freq='W-MON')
    n_points=len (dates)

    # Simulation de séries de prix
    np.random.seed (42)
    btc_values=pd.Series (np.random.normal (1,0.02,n_points).cumprod (),index=dates)
    xaut_values=pd.Series (np.random.normal (1,0.01,n_points).cumprod (),index=dates)

    # Simulation de portfolio avec alpha = 0.6 BTC, 0.4 XAUT
    alpha=0.6
    portfolio_values=alpha * btc_values + (1 - alpha) * xaut_values

    # Test du MetricsCalculator
    logger.info ("\nTest du MetricsCalculator:")
    metrics_calc=MetricsCalculator ()

    # Calcul des métriques primaires
    vol_ratio=metrics_calc.volatility_ratio (portfolio_values,btc_values,xaut_values)
    bound_coherence=metrics_calc.bound_coherence (portfolio_values,btc_values,xaut_values)
    alpha_stability=metrics_calc.alpha_stability (pd.Series ([alpha] * n_points))
    spectral=metrics_calc.spectral_score (portfolio_values,btc_values,xaut_values)

    logger.info (f"Ratio de volatilité: {vol_ratio:.4f}")
    logger.info (f"Cohérence des bornes: {bound_coherence:.4f}")
    logger.info (f"Stabilité de l'alpha: {alpha_stability:.4f}")
    logger.info (f"Score spectral: {spectral:.4f}")

    # Test du QAAFCore
    logger.info ("\nTest du QAAFCore:")
    qaaf=QAAFCore ()

    metrics={
        'volatility_ratio':vol_ratio,
        'bound_coherence':bound_coherence,
        'alpha_stability':alpha_stability,
        'spectral_score':spectral
    }

    composite_score=metrics_calc.calculate_composite_score (metrics)
    logger.info (f"Score composite: {composite_score:.4f}")

    # Test de validation des poids
    weights={'BTC':0.6,'XAUT':0.4}
    valid=qaaf.validate_weights (weights)
    logger.info (f"\nValidation des poids: {'✓' if valid else '✗'}")

    return {
        'metrics':metrics,
        'composite_score':composite_score,
        'weights_valid':valid
    }


if __name__ == "__main__":
    results=main ()