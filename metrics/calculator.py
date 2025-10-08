"""
Module de calcul des métriques fondamentales QAAF
"""
# Dans qaaf/metrics/calculator.py

import pandas as pd
import numpy as np
from typing import Dict,Optional,Tuple
import logging

logger = logging.getLogger(__name__)

# Dans MetricsCalculator ou d'autres classes qui utilisent GPU
from qaaf.utils.gpu_utils import get_array_module, GPU_AVAILABLE

# Tentative d'importation de CuPy pour l'accélération GPU
try:
    import cupy as cp

    GPU_AVAILABLE=True
except ImportError:
    cp=np
    GPU_AVAILABLE=False


class MetricsCalculator:
    """
    Calculateur des métriques QAAF

    Cette classe implémente les quatre métriques primaires du framework QAAF:
    1. Ratio de Volatilité
    2. Cohérence des Bornes
    3. Stabilité d'Alpha
    4. Score Spectral
    """

    def __init__ (self,
                  volatility_window: int = 30,
                  spectral_window: int = 60,
                  min_periods: int = 20,
                  use_gpu: bool = None):
        """
        Initialise le calculateur de métriques

        Args:
            volatility_window: Fenêtre pour le calcul de la volatilité
            spectral_window: Fenêtre pour le calcul des composantes spectrales
            min_periods: Nombre minimum de périodes pour les calculs
            use_gpu: Si True, utilise le GPU si disponible. Si None, détecte automatiquement.
        """
        # Imports nécessaires
        import numpy as np
        from qaaf.utils.gpu_utils import GPU_AVAILABLE

        # Paramètres de base
        self.volatility_window=volatility_window
        self.spectral_window=spectral_window
        self.min_periods=min_periods

        # Configuration GPU
        self.use_gpu=GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)

        if self.use_gpu:
            import cupy as cp
            self.xp=cp
            print ("GPU acceleration enabled for metrics calculation")
        else:
            self.xp=np

    def update_parameters (self,
                           volatility_window: Optional[int] = None,
                           spectral_window: Optional[int] = None,
                           min_periods: Optional[int] = None):
        """
        Met à jour les paramètres du calculateur

        Args:
            volatility_window: Fenêtre pour le calcul de la volatilité
            spectral_window: Fenêtre pour le calcul des composantes spectrales
            min_periods: Nombre minimum de périodes pour les calculs
        """
        if volatility_window is not None:
            self.volatility_window=volatility_window

        if spectral_window is not None:
            self.spectral_window=spectral_window

        if min_periods is not None:
            self.min_periods=min_periods

    def calculate_metrics (self,data: Dict[str,pd.DataFrame],
                           alpha: Optional[pd.Series] = None) -> Dict[str,pd.Series]:
        """
        Calcule toutes les métriques primaires QAAF

        Args:
            data: Dictionnaire contenant les DataFrames ('BTC', 'PAXG', 'PAXG/BTC')
            alpha: Série des allocations (optionnel)

        Returns:
            Dictionnaire avec les séries temporelles des métriques
        """
        # Extraction des données
        btc_data=data['BTC']
        paxg_data=data['PAXG']
        paxg_btc_data=data['PAXG/BTC']

        # Calcul des rendements
        btc_returns=btc_data['close'].pct_change ().dropna ()
        paxg_returns=paxg_data['close'].pct_change ().dropna ()
        paxg_btc_returns=paxg_btc_data['close'].pct_change ().dropna ()

        # Alignement des indices
        common_index=btc_returns.index.intersection (paxg_returns.index).intersection (paxg_btc_returns.index)
        btc_returns=btc_returns.loc[common_index]
        paxg_returns=paxg_returns.loc[common_index]
        paxg_btc_returns=paxg_btc_returns.loc[common_index]

        # 1. Ratio de Volatilité
        vol_ratio=self._calculate_volatility_ratio (paxg_btc_returns,btc_returns,paxg_returns)

        # 2. Cohérence des Bornes
        bound_coherence=self._calculate_bound_coherence (paxg_btc_data,btc_data,paxg_data)

        # 3. Stabilité d'Alpha (si alpha est fourni)
        if alpha is not None:
            alpha_stability=self._calculate_alpha_stability (alpha)
        else:
            # Créer un alpha statique de 0.5 pour démonstration
            alpha_static=pd.Series (0.5,index=common_index)
            alpha_stability=self._calculate_alpha_stability (alpha_static)

        # 4. Score Spectral
        spectral_score=self._calculate_spectral_score (paxg_btc_data,btc_data,paxg_data)

        # Résultats
        metrics={
            'vol_ratio':vol_ratio,
            'bound_coherence':bound_coherence,
            'alpha_stability':alpha_stability,
            'spectral_score':spectral_score
        }

        # Validation des résultats (avec log de problèmes potentiels)
        self._validate_metrics (metrics)

        return metrics

    def _calculate_volatility_ratio (self,
                                     paxg_btc_returns: pd.Series,
                                     btc_returns: pd.Series,
                                     paxg_returns: pd.Series) -> pd.Series:
        """
        Calcule le ratio de volatilité σ_C(t) / max(σ_A(t), σ_B(t))

        Une valeur proche de 0 est meilleure (volatilité faible par rapport aux actifs)
        Une valeur > 1 indique une volatilité plus élevée que les actifs sous-jacents
        """
        # Conversion vers arrays NumPy/CuPy pour accélération
        paxg_btc_arr=self.xp.array (paxg_btc_returns)
        btc_arr=self.xp.array (btc_returns)
        paxg_arr=self.xp.array (paxg_returns)

        # Calcul des volatilités mobiles (optimisé GPU si disponible)
        vol_paxg_btc=self._rolling_std (paxg_btc_arr,self.volatility_window) * self.xp.sqrt (252)
        vol_btc=self._rolling_std (btc_arr,self.volatility_window) * self.xp.sqrt (252)
        vol_paxg=self._rolling_std (paxg_arr,self.volatility_window) * self.xp.sqrt (252)

        # Calcul du maximum des volatilités sous-jacentes
        max_vol=self.xp.maximum (vol_btc,vol_paxg)

        # Calcul du ratio (avec gestion des divisions par zéro)
        ratio=self.xp.divide (vol_paxg_btc,max_vol,
                              out=self.xp.ones_like (vol_paxg_btc),
                              where=max_vol != 0)

        # Limiter les valeurs pour éviter les extrêmes
        ratio=self.xp.clip (ratio,0.1,10.0)

        # Conversion vers pandas Series pour l'interface standard
        if self.use_gpu:
            ratio=cp.asnumpy (ratio)

        result=pd.Series (ratio,index=paxg_btc_returns.index).fillna (1.0)
        return result

    def _rolling_std (self,arr,window):
        """Calcul optimisé d'écart-type mobile"""
        # Implémentation GPU-compatible de rolling standard deviation
        # Cette méthode est une version simplifiée et doit être complétée
        # pour correspondre exactement au comportement de pandas
        result=self.xp.zeros_like (arr)
        for i in range (len (arr)):
            if i < window - 1:
                result[i]=self.xp.nan
            else:
                result[i]=self.xp.std (arr[i - window + 1:i + 1],ddof=1)
        return result

    def _calculate_bound_coherence (self,
                                    paxg_btc_data: pd.DataFrame,
                                    btc_data: pd.DataFrame,
                                    paxg_data: pd.DataFrame) -> pd.Series:
        """
        Calcule la cohérence des bornes P(min(A,B) ≤ C ≤ max(A,B))

        Une valeur proche de 1 est meilleure (prix entre les bornes naturelles)
        Une valeur proche de 0 indique un prix en dehors des bornes
        """
        # Extraction des séries de prix
        paxg_btc_prices=paxg_btc_data['close']
        btc_prices=btc_data['close']
        paxg_prices=paxg_data['close']

        # Normalisation pour comparaison (base 100)
        common_index=paxg_btc_prices.index.intersection (btc_prices.index).intersection (paxg_prices.index)
        start_date=common_index[0]

        norm_paxg_btc=paxg_btc_prices.loc[common_index] / paxg_btc_prices.loc[start_date] * 100
        norm_btc=btc_prices.loc[common_index] / btc_prices.loc[start_date] * 100
        norm_paxg=paxg_prices.loc[common_index] / paxg_prices.loc[start_date] * 100

        # Calcul des bornes
        min_bound=pd.concat ([norm_btc,norm_paxg],axis=1).min (axis=1)
        max_bound=pd.concat ([norm_btc,norm_paxg],axis=1).max (axis=1)

        # Vérification si le prix est dans les bornes
        in_bounds=(norm_paxg_btc >= min_bound) & (norm_paxg_btc <= max_bound)

        # Calcul de la cohérence sur une fenêtre mobile
        coherence=in_bounds.rolling (window=self.volatility_window,
                                     min_periods=self.min_periods).mean ()

        return coherence.fillna (0.5)

    def _calculate_alpha_stability (self,alpha: pd.Series) -> pd.Series:
        """
        Calcule la stabilité d'alpha -σ(α(t))

        Une valeur proche de 0 est meilleure (allocations stables)
        Une valeur très négative indique des changements fréquents d'allocation
        """
        # Calcul de la volatilité des allocations
        alpha_volatility=alpha.rolling (window=self.volatility_window,
                                        min_periods=self.min_periods).std ()

        # Inversion du signe (-σ) pour que les valeurs proches de 0 soient meilleures
        stability=-alpha_volatility

        # Normalisation entre 0 et 1
        normalized_stability=(stability - stability.min ()) / (stability.max () - stability.min () + 1e-10)

        return normalized_stability.fillna (1.0)

    def _calculate_spectral_score (self,
                                   paxg_btc_data: pd.DataFrame,
                                   btc_data: pd.DataFrame,
                                   paxg_data: pd.DataFrame) -> pd.Series:
        """
        Calcule le score spectral (combinaison de tendance et d'oscillation)

        Une valeur proche de 1 indique un bon équilibre tendance/oscillation
        """
        # 1. Composante tendancielle (70%)
        trend_score=self._calculate_trend_component (paxg_btc_data,btc_data,paxg_data)

        # 2. Composante oscillatoire (30%)
        oscillation_score=self._calculate_oscillation_component (paxg_btc_data,btc_data,paxg_data)

        # Score combiné
        spectral_score=0.7 * trend_score + 0.3 * oscillation_score

        return spectral_score.fillna (0.5)

    def _calculate_trend_component (self,
                                    paxg_btc_data: pd.DataFrame,
                                    btc_data: pd.DataFrame,
                                    paxg_data: pd.DataFrame) -> pd.Series:
        """Calcule la composante tendancielle du score spectral"""
        # Moyennes mobiles
        ma_paxg_btc=paxg_btc_data['close'].rolling (window=self.spectral_window,
                                                    min_periods=self.min_periods).mean ()
        ma_btc=btc_data['close'].rolling (window=self.spectral_window,
                                          min_periods=self.min_periods).mean ()
        ma_paxg=paxg_data['close'].rolling (window=self.spectral_window,
                                            min_periods=self.min_periods).mean ()

        # Synthèse des actifs sous-jacents
        ma_combined=(ma_btc + ma_paxg) / 2

        # Pour éviter des complications, nous retournons une mesure simplifiée
        # de la différence entre le ratio et la moyenne des actifs
        trend_diff=(ma_paxg_btc - ma_combined).abs ()
        max_diff=ma_combined.max () - ma_combined.min ()

        # Normalisation entre 0 et 1 (1 = bonne tendance, 0 = mauvaise)
        trend_score=1 - (trend_diff / (max_diff + 1e-10)).clip (0,1)

        return trend_score

    def _calculate_oscillation_component (self,
                                          paxg_btc_data: pd.DataFrame,
                                          btc_data: pd.DataFrame,
                                          paxg_data: pd.DataFrame) -> pd.Series:
        """Calcule la composante oscillatoire du score spectral"""
        # Calcul des rendements
        returns_paxg_btc=paxg_btc_data['close'].pct_change ()
        returns_btc=btc_data['close'].pct_change ()
        returns_paxg=paxg_data['close'].pct_change ()

        # Calcul des volatilités
        vol_paxg_btc=returns_paxg_btc.rolling (window=self.volatility_window,
                                               min_periods=self.min_periods).std ()
        vol_btc=returns_btc.rolling (window=self.volatility_window,
                                     min_periods=self.min_periods).std ()
        vol_paxg=returns_paxg.rolling (window=self.volatility_window,
                                       min_periods=self.min_periods).std ()

        # Score oscillatoire basé sur le rapport de volatilité
        max_vol=pd.concat ([vol_btc,vol_paxg],axis=1).max (axis=1)
        vol_ratio=vol_paxg_btc / max_vol.replace (0,np.nan)

        # Normalisation (1 = bonne oscillation, 0 = mauvaise)
        # Une bonne oscillation a un ratio de volatilité optimal (ni trop élevé, ni trop faible)
        osc_score=1 - (vol_ratio - 0.5).abs ().clip (0,0.5) * 2

        return osc_score.fillna (0.5)

    def _validate_metrics (self,metrics: Dict[str,pd.Series]) -> None:
        """Valide les métriques calculées et log les anomalies"""
        for name,series in metrics.items ():
            # Vérification des NaN
            nan_count=series.isna ().sum ()
            if nan_count > 0:
                logger.warning (f"La métrique {name} contient {nan_count} valeurs NaN")

            # Vérification des valeurs extrêmes
            if name == 'vol_ratio':
                if (series > 5).any ():
                    logger.warning (f"Valeurs de vol_ratio > 5 détectées")
            elif name == 'bound_coherence':
                if (series < 0.2).any ():
                    logger.warning (f"Faible cohérence des bornes détectée (< 0.2)")

            # Vérification des sauts brusques
            diff=series.diff ().abs ()
            mean_diff=diff.mean ()
            max_diff=diff.max ()
            if max_diff > 5 * mean_diff:
                logger.warning (f"Sauts importants détectés dans {name}: max={max_diff}, mean={mean_diff}")

    def normalize_metrics (self,metrics: Dict[str,pd.Series]) -> Dict[str,pd.Series]:
        """
        Normalise les métriques entre 0 et 1 pour une comparaison équitable

        Args:
            metrics: Dictionnaire des métriques calculées

        Returns:
            Dictionnaire des métriques normalisées
        """
        normalized_metrics={}

        for name,series in metrics.items ():
            min_val=series.min ()
            max_val=series.max ()

            if max_val > min_val:
                normalized_metrics[name]=(series - min_val) / (max_val - min_val)
            else:
                normalized_metrics[name]=series - min_val

        return normalized_metrics
