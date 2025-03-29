import unittest
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from unittest.mock import patch,MagicMock

from qaaf.data.data_manager import DataManager,YFinanceSource,DataSource


class TestYFinanceSource (unittest.TestCase):
    def setUp (self):
        self.source=YFinanceSource ()

    @patch ('qaaf.data.data_manager.yf.download')
    def test_fetch_data_success (self,mock_download):
        # Créer un DataFrame simulé comme retour de yf.download
        df=pd.DataFrame ({
            'Open':[100,101,102],
            'High':[105,106,107],
            'Low':[95,96,97],
            'Close':[103,104,105],
            'Volume':[1000,1100,1200]
        },index=pd.date_range (start='2023-01-01',periods=3))

        mock_download.return_value=df

        # Appeler fetch_data
        result=self.source.fetch_data ('BTC-USD','2023-01-01','2023-01-03')

        # Vérifier que yf.download a été appelé avec les bons paramètres
        mock_download.assert_called_once_with ('BTC-USD',start='2023-01-01',end='2023-01-03',progress=False)

        # Vérifier que les colonnes sont bien standardisées
        self.assertIn ('open',result.columns)
        self.assertIn ('high',result.columns)
        self.assertIn ('low',result.columns)
        self.assertIn ('close',result.columns)
        self.assertIn ('volume',result.columns)

    @patch ('qaaf.data.data_manager.yf.download')
    def test_fetch_data_empty (self,mock_download):
        # Simuler un DataFrame vide
        mock_download.return_value=pd.DataFrame ()

        # Vérifier que l'exception ValueError est levée
        with self.assertRaises (ValueError):
            self.source.fetch_data ('NONEXISTENT','2023-01-01','2023-01-03')

    def test_standardize_data (self):
        # Créer un DataFrame avec un MultiIndex comme yfinance peut retourner
        cols=pd.MultiIndex.from_product ([['Open','High','Low','Close','Volume'],['BTC-USD']])
        df=pd.DataFrame (
            np.random.randn (3,5),
            index=pd.date_range (start='2023-01-01',periods=3),
            columns=cols
        )

        # Standardiser les données
        result=self.source._standardize_data (df)

        # Vérifier les colonnes
        self.assertEqual (set (result.columns),{'open','high','low','close','volume'})

    def test_validate_data (self):
        # Créer un DataFrame valide
        df_valid=pd.DataFrame ({
            'open':[100,101],
            'high':[105,106],
            'low':[95,96],
            'close':[103,104],
            'volume':[1000,1100]
        })

        # Créer un DataFrame invalide (manque la colonne 'volume')
        df_invalid=pd.DataFrame ({
            'open':[100,101],
            'high':[105,106],
            'low':[95,96],
            'close':[103,104]
        })

        # Créer un DataFrame vide
        df_empty=pd.DataFrame ()

        # Tester la validation
        self.assertTrue (self.source.validate_data (df_valid))
        self.assertFalse (self.source.validate_data (df_invalid))
        self.assertFalse (self.source.validate_data (df_empty))


class TestDataManager (unittest.TestCase):
    def setUp (self):
        # Créer un mock de DataSource pour éviter les appels réseau
        self.mock_source=MagicMock (spec=DataSource)
        self.data_manager=DataManager (data_source=self.mock_source)

    def test_get_data_from_cache (self):
        # Configurer le mock pour retourner un DataFrame
        df=pd.DataFrame ({'close':[100,101]})
        self.mock_source.fetch_data.return_value=df

        # Premier appel, devrait utiliser fetch_data
        result1=self.data_manager.get_data ('BTC-USD','2023-01-01','2023-01-03')
        self.mock_source.fetch_data.assert_called_once ()

        # Réinitialiser le mock
        self.mock_source.fetch_data.reset_mock ()

        # Deuxième appel avec les mêmes paramètres, devrait utiliser le cache
        result2=self.data_manager.get_data ('BTC-USD','2023-01-01','2023-01-03')
        self.mock_source.fetch_data.assert_not_called ()

        # Vérifier que les deux résultats sont identiques
        pd.testing.assert_frame_equal (result1,result2)

    def test_prepare_qaaf_data (self):
        # Configurer le mock pour retourner des DataFrames
        btc_data=pd.DataFrame ({
            'open':[100,101],
            'high':[105,106],
            'low':[95,96],
            'close':[103,104],
            'volume':[1000,1100]
        },index=pd.date_range (start='2023-01-01',periods=2))

        paxg_data=pd.DataFrame ({
            'open':[1800,1810],
            'high':[1850,1860],
            'low':[1780,1790],
            'close':[1820,1830],
            'volume':[500,550]
        },index=pd.date_range (start='2023-01-01',periods=2))

        # Configurer le mock pour retourner ces DataFrames
        self.mock_source.fetch_data.side_effect=[btc_data,paxg_data]

        # Appeler prepare_qaaf_data
        result=self.data_manager.prepare_qaaf_data ('2023-01-01','2023-01-03')

        # Vérifier que le résultat contient les clés attendues
        self.assertIn ('BTC',result)
        self.assertIn ('PAXG',result)
        self.assertIn ('PAXG/BTC',result)

        # Vérifier que le ratio PAXG/BTC est correctement calculé
        expected_ratio=paxg_data['close'] / btc_data['close']
        pd.testing.assert_series_equal (result['PAXG/BTC']['close'],expected_ratio)


if __name__ == '__main__':
    unittest.main ()