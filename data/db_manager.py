# qaaf/data/db_manager.py
import os
import json
import pandas as pd
from typing import Dict,Optional,Union
import logging

logger=logging.getLogger (__name__)


class DatabaseManager:
    """Gestionnaire de base de données pour QAAF avec support pour SQLite et PostgreSQL"""

    def __init__ (self,db_type: str = "sqlite",
                  sqlite_path: str = './data/results/qaaf_results.db',
                  pg_params: Optional[Dict] = None):
        """
        Initialise le gestionnaire de base de données

        Args:
            db_type: Type de base de données ('sqlite' ou 'postgres')
            sqlite_path: Chemin du fichier SQLite (si db_type='sqlite')
            pg_params: Paramètres de connexion PostgreSQL (si db_type='postgres')
        """
        self.db_type=db_type
        self.connection=None

        if db_type == "sqlite":
            self._init_sqlite_db (sqlite_path)
        elif db_type == "postgres":
            if pg_params is None:
                raise ValueError ("Les paramètres PostgreSQL sont requis")
            self._init_postgres_db (pg_params)
        else:
            raise ValueError (f"Type de base de données non supporté: {db_type}")

    def _init_sqlite_db (self,db_path: str):
        """Initialise une base de données SQLite"""
        import sqlite3

        os.makedirs (os.path.dirname (db_path),exist_ok=True)
        self.connection=sqlite3.connect (db_path)
        self._create_tables_sqlite ()

    def _init_postgres_db (self,pg_params: Dict):
        """Initialise une connexion PostgreSQL"""
        import psycopg2

        self.connection=psycopg2.connect (**pg_params)
        self._create_tables_postgres ()

    def _create_tables_sqlite (self):
        """Crée les tables pour SQLite"""
        cursor=self.connection.cursor ()

        # Création des tables (même structure que pour PostgreSQL)
        cursor.execute ('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            profile TEXT,
            start_date TEXT,
            end_date TEXT,
            initial_capital REAL,
            final_value REAL,
            total_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            total_fees REAL,
            params TEXT,
            description TEXT
        )
        ''')

        # Autres tables...

        self.connection.commit ()

    def _create_tables_postgres (self):
        """Crée les tables pour PostgreSQL"""
        cursor=self.connection.cursor ()

        # Création des tables (adaptée pour PostgreSQL)
        cursor.execute ('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id SERIAL PRIMARY KEY,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            profile TEXT,
            start_date TEXT,
            end_date TEXT,
            initial_capital REAL,
            final_value REAL,
            total_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            total_fees REAL,
            params JSONB,
            description TEXT
        )
        ''')

        # Autres tables...

        self.connection.commit ()

    # Méthodes communes d'interaction avec la base
    def save_backtest_results (self,results,profile,params,description=''):
        """Sauvegarde les résultats d'un backtest"""
        # Logique adaptative selon le type de base
        # ...