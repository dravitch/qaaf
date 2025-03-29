"""
Générateur de données synthétiques pour les tests du framework QAAF.
Permet de créer des séries temporelles représentatives du comportement des actifs
crypto sans nécessiter de données réelles ou de base de données.
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from typing import Dict,Tuple,Optional,List


class SyntheticDataGenerator:
    """
    Générateur de données synthétiques pour tester le framework QAAF.
    Crée des séries temporelles simulées pour BTC, PAXG et leur ratio.
    """

    def __init__ (self,
                  start_date: str = '2020-01-01',
                  periods: int = 365,
                  seed: Optional[int] = 42):
        """
        Initialise le générateur de données synthétiques.

        Args:
            start_date: Date de début des données (format 'YYYY-MM-DD')
            periods: Nombre de périodes à générer (jours)
            seed: Graine pour la génération de nombres aléatoires (pour reproductibilité)
        """
        self.start_date=start_date
        self.periods=periods
        self.dates=pd.date_range (start=start_date,periods=periods)

        # Initialisation du générateur de nombres aléatoires
        if seed is not None:
            np.random.seed (seed)

    def generate_btc_data (self,
                           start_price: float = 10000.0,
                           volatility: float = 0.03,
                           trend: float = 0.001) -> pd.DataFrame:
        """
        Génère des données synthétiques pour Bitcoin.

        Args:
            start_price: Prix initial
            volatility: Volatilité quotidienne
            trend: Tendance quotidienne moyenne

        Returns:
            DataFrame avec colonnes OHLCV pour Bitcoin
        """
        # Génération des rendements logarithmiques
        returns=np.random.normal (trend,volatility,self.periods)

        # Application d'une tendance cyclique (bullish/bearish)
        cycle=np.sin (np.linspace (0,4 * np.pi,self.periods)) * 0.002
        returns=returns + cycle

        # Calcul des prix
        log_prices=np.log (start_price) + np.cumsum (returns)
        close_prices=np.exp (log_prices)

        # Création du DataFrame
        btc_data=pd.DataFrame ({
            'open':close_prices * np.exp (np.random.normal (-0.005,0.01,self.periods)),
            'high':close_prices * np.exp (np.random.normal (0.01,0.01,self.periods)),
            'low':close_prices * np.exp (np.random.normal (-0.01,0.01,self.periods)),
            'close':close_prices,
            'volume':np.random.lognormal (10,1,self.periods)
        },index=self.dates)

        return btc_data

    def generate_paxg_data (self,
                            start_price: float = 1800.0,
                            volatility: float = 0.01,
                            trend: float = 0.0002,
                            correlation_with_btc: float = -0.2) -> pd.DataFrame:
        """
        Génère des données synthétiques pour PAXG, potentiellement corrélées avec BTC.

        Args:
            start_price: Prix initial
            volatility: Volatilité quotidienne
            trend: Tendance quotidienne moyenne
            correlation_with_btc: Corrélation avec les rendements BTC

        Returns:
            DataFrame avec colonnes OHLCV pour PAXG
        """
        # Génération des rendements de base
        independent_returns=np.random.normal (trend,volatility,self.periods)

        # Ajout d'une composante cyclique plus lente que BTC
        cycle=np.sin (np.linspace (0,2 * np.pi,self.periods)) * 0.001
        independent_returns=independent_returns + cycle

        # Si nous avons généré des données BTC, nous pouvons ajouter une corrélation
        btc_df=self.generate_btc_data () if correlation_with_btc != 0 else None

        if btc_df is not None and correlation_with_btc != 0:
            btc_returns=np.diff (np.log (btc_df['close'].values),prepend=np.log (btc_df['close'].iloc[0]))

            # Création de rendements corrélés
            correlated_component=correlation_with_btc * btc_returns
            combined_returns=independent_returns + correlated_component
        else:
            combined_returns=independent_returns

        # Calcul des prix
        log_prices=np.log (start_price) + np.cumsum (combined_returns)
        close_prices=np.exp (log_prices)

        # Création du DataFrame
        paxg_data=pd.DataFrame ({
            'open':close_prices * np.exp (np.random.normal (-0.002,0.005,self.periods)),
            'high':close_prices * np.exp (np.random.normal (0.004,0.005,self.periods)),
            'low':close_prices * np.exp (np.random.normal (-0.004,0.005,self.periods)),
            'close':close_prices,
            'volume':np.random.lognormal (8,1,self.periods)
        },index=self.dates)

        return paxg_data

    def generate_market_data (self,with_special_scenarios: bool = False) -> Dict[str,pd.DataFrame]:
        """
        Génère un ensemble complet de données de marché synthétiques pour QAAF.

        Args:
            with_special_scenarios: Si True, inclut des scénarios spéciaux comme des crashs

        Returns:
            Dictionnaire avec les DataFrames pour BTC, PAXG et PAXG/BTC
        """
        # Génération des données BTC
        btc_data=self.generate_btc_data ()

        # Génération des données PAXG avec légère corrélation négative avec BTC
        paxg_data=self.generate_paxg_data (correlation_with_btc=-0.2)

        # Ajout de scénarios spéciaux si demandé
        if with_special_scenarios:
            self._add_special_scenarios (btc_data,paxg_data)

        # Calcul du ratio PAXG/BTC
        paxg_btc_ratio=paxg_data['close'] / btc_data['close']

        # Création du DataFrame pour le ratio
        paxg_btc_data=pd.DataFrame ({
            'open':paxg_data['open'] / btc_data['open'],
            'high':paxg_data['high'] / btc_data['high'],
            'low':paxg_data['low'] / btc_data['low'],
            'close':paxg_btc_ratio,
            'volume':(paxg_data['volume'] + btc_data['volume']) / 2  # Approximation
        },index=self.dates)

        return {
            'BTC':btc_data,
            'PAXG':paxg_data,
            'PAXG/BTC':paxg_btc_data
        }

    def _add_special_scenarios (self,btc_data: pd.DataFrame,paxg_data: pd.DataFrame) -> None:
        """
        Ajoute des scénarios spéciaux aux données (crash, pump, etc.).
        Modifie les DataFrames en place.

        Args:
            btc_data: DataFrame BTC à modifier
            paxg_data: DataFrame PAXG à modifier
        """
        # Position pour un crash BTC (au quart de la série)
        crash_position=self.periods // 4
        crash_length=10

        # Appliquer un crash de 30% sur 10 jours pour BTC
        crash_factor=np.linspace (1.0,0.7,crash_length)
        btc_data.iloc[crash_position:crash_position + crash_length]*=crash_factor[:,np.newaxis]

        # PAXG est plus stable pendant le crash (baisse de seulement 5%)
        stable_factor=np.linspace (1.0,0.95,crash_length)
        paxg_data.iloc[crash_position:crash_position + crash_length]*=stable_factor[:,np.newaxis]

        # Position pour un pump BTC (aux trois quarts de la série)
        pump_position=3 * self.periods // 4
        pump_length=15

        # Appliquer un pump de 50% sur 15 jours pour BTC
        pump_factor=np.linspace (1.0,1.5,pump_length)
        btc_data.iloc[pump_position:pump_position + pump_length]*=pump_factor[:,np.newaxis]

        # PAXG est presque stable pendant le pump (hausse de seulement 3%)
        paxg_pump_factor=np.linspace (1.0,1.03,pump_length)
        paxg_data.iloc[pump_position:pump_position + pump_length]*=paxg_pump_factor[:,np.newaxis]

    def generate_market_phases (self) -> pd.Series:
        """
        Génère des phases de marché synthétiques correspondant aux données générées.

        Returns:
            Série pandas avec les phases de marché
        """
        # Création de phases basées sur des périodes
        phases=[]

        # Division approximative pour l'exemple
        first_quarter=self.periods // 4
        middle=self.periods // 2
        third_quarter=3 * self.periods // 4

        # Générer les phases
        for i in range (self.periods):
            if i < first_quarter:
                phase='bullish_low_vol'
            elif i < middle:
                phase='bearish_high_vol'
            elif i < third_quarter:
                phase='consolidation_low_vol'
            else:
                phase='bullish_high_vol'

            phases.append (phase)

        return pd.Series (phases,index=self.dates)

    def generate_testing_datasets (self,
                                   train_ratio: float = 0.7,
                                   validation_ratio: float = 0.15,
                                   test_ratio: float = 0.15) -> Dict[str,Dict[str,pd.DataFrame]]:
        """
        Génère des ensembles de données d'entraînement, validation et test.

        Args:
            train_ratio: Proportion des données pour l'entraînement
            validation_ratio: Proportion des données pour la validation
            test_ratio: Proportion des données pour le test

        Returns:
            Dictionnaire avec les ensembles train, validation et test
        """
        # Vérification que les ratios somment à 1
        if abs (train_ratio + validation_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError ("Les ratios doivent sommer à 1.0")

        # Génération des données de marché
        market_data=self.generate_market_data ()

        # Calcul des indices pour la division
        train_end=int (self.periods * train_ratio)
        validation_end=train_end + int (self.periods * validation_ratio)

        # Division des données
        train_data={
            key:df.iloc[:train_end].copy () for key,df in market_data.items ()
        }

        validation_data={
            key:df.iloc[train_end:validation_end].copy () for key,df in market_data.items ()
        }

        test_data={
            key:df.iloc[validation_end:].copy () for key,df in market_data.items ()
        }

        return {
            'train':train_data,
            'validation':validation_data,
            'test':test_data
        }


def generate_sample_data (periods: int = 100,with_scenarios: bool = False) -> Dict[str,pd.DataFrame]:
    """
    Fonction utilitaire pour générer rapidement des données d'exemple.

    Args:
        periods: Nombre de périodes à générer
        with_scenarios: Si True, inclut des scénarios spéciaux

    Returns:
        Données de marché synthétiques
    """
    generator=SyntheticDataGenerator (periods=periods)
    return generator.generate_market_data (with_special_scenarios=with_scenarios)


if __name__ == "__main__":
    # Exemple d'utilisation et génération de quelques visualisations
    import matplotlib.pyplot as plt

    generator=SyntheticDataGenerator (periods=500)
    market_data=generator.generate_market_data (with_special_scenarios=True)

    # Visualisation des prix
    plt.figure (figsize=(15,10))

    # Prix BTC normalisés
    plt.subplot (2,1,1)
    btc_normalized=market_data['BTC']['close'] / market_data['BTC']['close'].iloc[0]
    paxg_normalized=market_data['PAXG']['close'] / market_data['PAXG']['close'].iloc[0]

    plt.plot (btc_normalized,label='BTC')
    plt.plot (paxg_normalized,label='PAXG')
    plt.title ('Prix normalisés (base 1)')
    plt.legend ()
    plt.grid (True)

    # Ratio PAXG/BTC
    plt.subplot (2,1,2)
    plt.plot (market_data['PAXG/BTC']['close'],color='green')
    plt.title ('Ratio PAXG/BTC')
    plt.grid (True)

    plt.tight_layout ()
    plt.show ()