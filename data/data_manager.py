"""
Module de gestion des données pour QAAF
"""

import pandas as pd
import numpy as np
import os
from typing import Dict,Optional,List,Tuple
import logging
import yfinance as yf
from datetime import datetime

logger=logging.getLogger (__name__)


class DataManager:
    """
    Gestionnaire des données pour QAAF

    Cette classe s'occupe de:
    - Charger les données depuis des sources externes
    - Stocker et récupérer les données en cache
    - Préparer les données pour l'analyse
    """

    def __init__ (self,
                  data_dir: str = "./data/historical",
                  use_cache: bool = True):
        """
        Initialise le gestionnaire de données

        Args:
            data_dir: Répertoire de stockage des données
            use_cache: Utiliser le cache pour éviter les téléchargements répétés
        """
        self.data_dir=data_dir
        self.use_cache=use_cache
        self.data_cache={}

        # Création du répertoire de données s'il n'existe pas
        os.makedirs (data_dir,exist_ok=True)

    def _get_cache_path (self,symbol: str,start_date: str,end_date: str) -> str:
        """
        Génère le chemin pour le fichier de cache

        Args:
            symbol: Symbole de l'actif
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Chemin du fichier de cache
        """
        cache_filename=f"{symbol}_{start_date}_{end_date}.csv"
        return os.path.join (self.data_dir,cache_filename)

    def get_data (self,symbol: str,start_date: str,end_date: str) -> pd.DataFrame:
        """
        Récupère les données pour un symbole

        Tente d'abord de charger depuis le cache, sinon télécharge les données

        Args:
            symbol: Symbole de l'actif
            start_date: Date de début
            end_date: Date de fin

        Returns:
            DataFrame des données de l'actif
        """
        # Vérification du cache en mémoire
        cache_key=f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Vérification du cache sur disque
        cache_path=self._get_cache_path (symbol,start_date,end_date)
        if self.use_cache and os.path.exists (cache_path):
            try:
                df=pd.read_csv (cache_path,index_col=0,parse_dates=True)
                self.data_cache[cache_key]=df  # Mise en cache mémoire
                logger.info (f"Données chargées depuis le cache pour {symbol}")
                return df
            except Exception as e:
                logger.warning (f"Erreur lors du chargement depuis le cache pour {symbol}: {e}")

        # Téléchargement des données si pas en cache
        try:
            logger.info (f"Téléchargement des données pour {symbol}")
            df=yf.download (symbol,start=start_date,end=end_date,progress=False)

            if df.empty:
                raise ValueError (f"Aucune donnée disponible pour {symbol}")

            # Standardisation des colonnes
            df.columns=df.columns.str.lower ()

            # Sauvegarde dans le cache disque
            if self.use_cache:
                df.to_csv (cache_path)

            # Mise en cache mémoire
            self.data_cache[cache_key]=df

            return df

        except Exception as e:
            logger.error (f"Erreur lors du téléchargement pour {symbol}: {e}")
            raise

    def prepare_qaaf_data (self,start_date: str,end_date: str) -> Dict[str,pd.DataFrame]:
        """
        Prépare toutes les données nécessaires pour l'analyse QAAF:
        - BTC-USD
        - PAXG-USD
        - PAXG/BTC (calculé)

        Args:
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Dictionnaire des DataFrames préparés
        """
        # Récupération des données
        btc_data=self.get_data ('BTC-USD',start_date,end_date)
        paxg_data=self.get_data ('PAXG-USD',start_date,end_date)

        # Alignement des indices
        common_index=btc_data.index.intersection (paxg_data.index)
        btc_data=btc_data.loc[common_index]
        paxg_data=paxg_data.loc[common_index]

        # Calcul du ratio PAXG/BTC
        paxg_btc_ratio=paxg_data['close'] / btc_data['close']

        # Création d'un DataFrame complet pour PAXG/BTC
        paxg_btc_data=pd.DataFrame ({
            'open':paxg_data['open'] / btc_data['open'],
            'high':paxg_data['high'] / btc_data['high'],
            'low':paxg_data['low'] / btc_data['low'],
            'close':paxg_btc_ratio,
            'volume':paxg_data['volume']  # Approximation
        },index=common_index)

        return {
            'BTC':btc_data,
            'PAXG':paxg_data,
            'PAXG/BTC':paxg_btc_data
        }

    def clear_cache (self,memory_only: bool = False):
        """
        Vide le cache

        Args:
            memory_only: Si True, vide uniquement le cache mémoire
        """
        # Vider le cache mémoire
        self.data_cache={}

        # Vider également le cache disque si demandé
        if not memory_only:
            for file in os.listdir (self.data_dir):
                if file.endswith ('.csv'):
                    try:
                        os.remove (os.path.join (self.data_dir,file))
                    except Exception as e:
                        logger.warning (f"Impossible de supprimer {file}: {e}")