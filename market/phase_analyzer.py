"""
Module d'analyse des phases de marché pour QAAF
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Tentative d'importation de CuPy pour l'accélération GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketPhaseAnalyzer:
    """
    Analyseur des phases de marché pour QAAF

    Identifie les phases de marché (haussier, baissier, consolidation)
    et leur niveau de volatilité
    """

    def __init__(self,
                 short_window: int = 20,
                 long_window: int = 50,
                 volatility_window: int = 30,
                 use_gpu: bool = None):
        """
        Initialise l'analyseur des phases de marché

        Args:
            short_window: Fenêtre courte pour les moyennes mobiles
            long_window: Fenêtre longue pour les moyennes mobiles
            volatility_window: Fenêtre pour le calcul de volatilité
            use_gpu: Utiliser le GPU si disponible (None pour auto-détection)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_window = volatility_window

        # Configuration GPU
        self.use_gpu = GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)
        self.xp = cp if self.use_gpu else np

        if self.use_gpu:
            logger.info("GPU acceleration enabled for market phase analysis")

    def identify_market_phases(self, btc_data: pd.DataFrame) -> pd.Series:
        """
        Identifie les phases de marché (haussier, baissier, consolidation)

        Args:
            btc_data: DataFrame des données BTC

        Returns:
            Série des phases de marché
        """
        # Extraction des prix de clôture
        close = btc_data['close']

        # Calcul des moyennes mobiles
        ma_short = close.rolling(window=self.short_window).mean()
        ma_long = close.rolling(window=self.long_window).mean()

        # Calcul du momentum (rendement sur la fenêtre courte)
        momentum = close.pct_change(periods=self.short_window)

        # Calcul de la volatilité
        volatility = close.pct_change().rolling(window=self.volatility_window).std() * np.sqrt(252)

        # Conditions pour chaque phase (optimisées vectoriellement)
        # Conversion en arrays NumPy/CuPy pour accélération
        ma_short_arr = self.xp.array(ma_short)
        ma_long_arr = self.xp.array(ma_long)
        momentum_arr = self.xp.array(momentum)

        # CORRECTION: Initialiser avec le bon dtype pour éviter le warning
        # Utiliser dtype='object' pour permettre les strings
        phases = pd.Series('consolidation', index=close.index, dtype='object')

        # Identification des phases haussières
        bullish_condition = ((ma_short > ma_long) & (momentum > 0.1)) | (momentum > 0.2)
        phases[bullish_condition] = 'bullish'

        # Identification des phases baissières
        bearish_condition = ((ma_short < ma_long) & (momentum < -0.1)) | (momentum < -0.2)
        phases[bearish_condition] = 'bearish'

        # Identification des phases de forte volatilité
        volatility_median = volatility.rolling(window=100).median()
        high_volatility = volatility > volatility_median * 1.5

        # CORRECTION: Créer combined_phases avec le bon dtype dès le départ
        combined_phases = pd.Series(index=phases.index, dtype='object')

        # Combinaison efficace des phases et niveaux de volatilité
        # Vectorisation pour améliorer les performances
        vol_suffix = high_volatility.map({True: '_high_vol', False: '_low_vol'})
        combined_phases = phases + vol_suffix

        return combined_phases

    def analyze_metrics_by_phase(self,
                                 metrics: Dict[str, pd.Series],
                                 market_phases: pd.Series) -> Dict:
        """
        Analyse les métriques par phase de marché

        Args:
            metrics: Dictionnaire des métriques
            market_phases: Série des phases de marché

        Returns:
            Dictionnaire d'analyse par phase
        """
        unique_phases = market_phases.unique()
        phase_analysis = {}

        for phase in unique_phases:
            phase_mask = market_phases == phase
            phase_data = {}

            for metric_name, metric_series in metrics.items():
                phase_values = metric_series[phase_mask]

                if len(phase_values) > 0:
                    phase_data[metric_name] = {
                        'mean': phase_values.mean(),
                        'std': phase_values.std(),
                        'min': phase_values.min(),
                        'max': phase_values.max(),
                        'median': phase_values.median()
                    }

            phase_analysis[phase] = phase_data

        return phase_analysis