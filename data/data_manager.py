"""
Module de gestion des données pour QAAF
Intègre les meilleures pratiques Yahoo Finance
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List, Tuple
import logging
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)


def standardize_yahoo_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les données Yahoo Finance pour garantir la cohérence
    
    Gère:
    - MultiIndex (quand yfinance retourne plusieurs niveaux de colonnes)
    - Conversion des noms de colonnes en minuscules
    - Validation de la structure
    
    Args:
        data: DataFrame brut de yfinance
    
    Returns:
        DataFrame standardisé avec colonnes en minuscules
    
    Raises:
        ValueError: Si les données sont vides ou invalides
    """
    if data.empty:
        raise ValueError("DataFrame vide fourni à standardize_yahoo_data")
    
    # Copie pour ne pas modifier l'original
    df = data.copy()
    
    # CRITIQUE: Aplatir le MultiIndex si présent
    if isinstance(df.columns, pd.MultiIndex):
        # Prendre le premier niveau (les noms de colonnes: Open, High, Low, Close, Volume)
        df.columns = df.columns.droplevel(1)
        logger.debug("MultiIndex aplati en Index simple")
    
    # Convertir tous les noms de colonnes en minuscules
    df.columns = df.columns.str.lower()
    
    return df


def validate_yahoo_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Valide que les données Yahoo Finance contiennent toutes les colonnes requises
    
    Args:
        df: DataFrame à valider
        symbol: Symbole de l'actif (pour les messages d'erreur)
    
    Raises:
        ValueError: Si des colonnes requises sont manquantes
    """
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    actual_columns = set(df.columns)
    
    missing_columns = required_columns - actual_columns
    
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes pour {symbol}: {missing_columns}. "
            f"Colonnes présentes: {actual_columns}"
        )
    
    logger.debug(f"Validation réussie pour {symbol}: toutes les colonnes OHLCV présentes")


class DataManager:
    """
    Gestionnaire des données pour QAAF

    Cette classe s'occupe de:
    - Charger les données depuis des sources externes (Yahoo Finance)
    - Stocker et récupérer les données en cache
    - Standardiser et valider les données
    - Préparer les données pour l'analyse QAAF
    """

    def __init__(self,
                 data_dir: str = "./data/historical",
                 use_cache: bool = True):
        """
        Initialise le gestionnaire de données

        Args:
            data_dir: Répertoire de stockage des données
            use_cache: Utiliser le cache pour éviter les téléchargements répétés
        """
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.data_cache = {}

        # Création du répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"DataManager initialisé (cache: {use_cache}, dir: {data_dir})")

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """
        Génère le chemin pour le fichier de cache

        Args:
            symbol: Symbole de l'actif
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Chemin du fichier de cache
        """
        cache_filename = f"{symbol}_{start_date}_{end_date}.csv"
        return os.path.join(self.data_dir, cache_filename)

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Récupère les données pour un symbole avec standardisation Yahoo Finance

        Tente d'abord de charger depuis le cache, sinon télécharge les données

        Args:
            symbol: Symbole de l'actif (ex: 'BTC-USD', 'PAXG-USD')
            start_date: Date de début (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')

        Returns:
            DataFrame standardisé avec colonnes: open, high, low, close, volume
        
        Raises:
            ValueError: Si les données sont invalides ou vides
            RuntimeError: Si le téléchargement échoue
        """
        # Vérification du cache en mémoire
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"✅ {symbol}: Chargé depuis cache mémoire")
            return self.data_cache[cache_key]

        # Vérification du cache sur disque
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        if self.use_cache and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                
                # Validation du cache
                validate_yahoo_data(df, symbol)
                
                self.data_cache[cache_key] = df
                logger.info(f"✅ {symbol}: Chargé depuis cache disque ({len(df)} jours)")
                return df
            except Exception as e:
                logger.warning(f"⚠️  Cache corrompu pour {symbol}, re-téléchargement: {e}")
                # Supprimer le cache corrompu
                try:
                    os.remove(cache_path)
                except:
                    pass

        # Téléchargement des données depuis Yahoo Finance
        try:
            logger.info(f"📥 Téléchargement {symbol} ({start_date} → {end_date})")
            
            # Téléchargement avec paramètres explicites
            raw_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Évite le warning et simplifie les données
            )

            if raw_data.empty:
                raise ValueError(f"Aucune donnée retournée par Yahoo Finance pour {symbol}")

            # STANDARDISATION (gère le MultiIndex automatiquement)
            df = standardize_yahoo_data(raw_data)
            
            # VALIDATION
            validate_yahoo_data(df, symbol)
            
            # Informations de debug
            logger.info(
                f"✅ {symbol}: {len(df)} jours chargés "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )
            logger.debug(f"   Colonnes: {list(df.columns)}")
            logger.debug(f"   Prix moyen: ${df['close'].mean():.2f}")

            # Sauvegarde dans le cache disque
            if self.use_cache:
                df.to_csv(cache_path)
                logger.debug(f"💾 Cache disque créé: {os.path.basename(cache_path)}")

            # Mise en cache mémoire
            self.data_cache[cache_key] = df

            return df

        except ValueError as e:
            # Erreur de validation ou données vides
            logger.error(f"❌ Données invalides pour {symbol}: {e}")
            raise
        except Exception as e:
            # Erreur de téléchargement ou autre
            logger.error(f"❌ Erreur téléchargement {symbol}: {type(e).__name__}: {e}")
            raise RuntimeError(f"Impossible de charger {symbol}: {e}")

    def prepare_qaaf_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prépare toutes les données nécessaires pour l'analyse QAAF:
        - BTC-USD
        - PAXG-USD
        - PAXG/BTC (ratio calculé)

        Args:
            start_date: Date de début (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')

        Returns:
            Dictionnaire avec les clés:
            - 'BTC': DataFrame des données Bitcoin
            - 'PAXG': DataFrame des données PAXG (or tokenisé)
            - 'PAXG/BTC': DataFrame du ratio PAXG/BTC
        
        Raises:
            ValueError: Si les données sont invalides ou incompatibles
        """
        logger.info("=" * 60)
        logger.info("📊 Préparation des données QAAF")
        logger.info(f"   Période: {start_date} → {end_date}")
        logger.info("=" * 60)
        
        # Récupération des données individuelles
        btc_data = self.get_data('BTC-USD', start_date, end_date)
        paxg_data = self.get_data('PAXG-USD', start_date, end_date)

        # Alignement des indices temporels (strictement BTC ∩ PAXG)
        common_index = btc_data.index.intersection(paxg_data.index)

        if len(common_index) == 0:
            raise ValueError(
                f"Aucune date commune entre BTC et PAXG. "
                f"BTC: {len(btc_data)} jours, PAXG: {len(paxg_data)} jours"
            )

        # Filtrer sur les dates communes et remplir NaN si nécessaire
        btc_data = btc_data.reindex(common_index).ffill().bfill()
        paxg_data = paxg_data.reindex(common_index).ffill().bfill()

        logger.info(f"✅ Alignement temporel: {len(common_index)} jours communs")

        # Calcul du ratio PAXG/BTC
        paxg_btc_ratio = paxg_data['close'] / btc_data['close']

        # Création d'un DataFrame complet pour PAXG/BTC (indexé sur common_index)
        paxg_btc_data = pd.DataFrame({
            'open': paxg_data['open'] / btc_data['open'],
            'high': paxg_data['high'] / btc_data['high'],
            'low': paxg_data['low'] / btc_data['low'],
            'close': paxg_btc_ratio,
            'volume': paxg_data['volume']  # Approximation
        }, index=common_index)

        # Statistiques du ratio (debug)
        ratio_mean = paxg_btc_ratio.mean()
        ratio_std = paxg_btc_ratio.std()
        ratio_min = paxg_btc_ratio.min()
        ratio_max = paxg_btc_ratio.max()

        logger.info(f"📊 Ratio PAXG/BTC calculé:")
        logger.info(f"   Moyenne: {ratio_mean:.6f} (±{ratio_std:.6f})")
        logger.info(f"   Min: {ratio_min:.6f}, Max: {ratio_max:.6f}")
        logger.info(f"   Variation: {((ratio_max/ratio_min - 1) * 100):.1f}%")

        logger.info("✅ Préparation des données QAAF terminée")
        logger.info("=" * 60)

        return {
            'BTC': btc_data,
            'PAXG': paxg_data,
            'PAXG/BTC': paxg_btc_data
        }


    def clear_cache(self, memory_only: bool = False):
        """
        Vide le cache des données

        Args:
            memory_only: Si True, vide uniquement le cache mémoire
                        Si False, vide aussi le cache disque
        """
        # Vider le cache mémoire
        cache_size = len(self.data_cache)
        self.data_cache = {}
        logger.info(f"🧹 Cache mémoire vidé ({cache_size} entrées supprimées)")

        # Vider également le cache disque si demandé
        if not memory_only:
            count = 0
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    try:
                        os.remove(os.path.join(self.data_dir, file))
                        count += 1
                    except Exception as e:
                        logger.warning(f"⚠️  Impossible de supprimer {file}: {e}")
            
            logger.info(f"🧹 Cache disque vidé ({count} fichiers supprimés)")


# Fonction utilitaire pour test rapide
def test_data_manager():
    """Test rapide du DataManager avec gestion des erreurs"""
    print("\n🧪 Test du DataManager\n")
    
    manager = DataManager(use_cache=False)  # Pas de cache pour le test
    
    try:
        # Test BTC
        print("Test BTC-USD...")
        btc = manager.get_data('BTC-USD', '2024-01-01', '2024-01-10')
        print(f"✅ BTC: {len(btc)} jours, colonnes: {list(btc.columns)}")
        
        # Test PAXG
        print("\nTest PAXG-USD...")
        paxg = manager.get_data('PAXG-USD', '2024-01-01', '2024-01-10')
        print(f"✅ PAXG: {len(paxg)} jours, colonnes: {list(paxg.columns)}")
        
        # Test préparation complète
        print("\nTest préparation QAAF...")
        data = manager.prepare_qaaf_data('2024-01-01', '2024-01-10')
        print(f"✅ QAAF data préparé: {list(data.keys())}")
        print(f"   Ratio moyen PAXG/BTC: {data['PAXG/BTC']['close'].mean():.6f}")
        
        print("\n🎉 Tous les tests réussis!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Lancer le test si exécuté directement
    test_data_manager()