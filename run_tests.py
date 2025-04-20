#!/usr/bin/env python
# run_tests.py
import os
import sys
import pytest


def main ():
    """Fonction principale qui exécute les tests pytest."""
    # Ajout du répertoire parent au path
    sys.path.insert (0,os.path.abspath (os.path.dirname (__file__)))

    # Arguments par défaut
    pytest_args=[
        '--cov=qaaf',
        '--cov-report=term',
        '--cov-report=html',
        '-v'
    ]

    # Ajout des arguments de ligne de commande
    pytest_args.extend (sys.argv[1:])

    # Exécution des tests
    return pytest.main (pytest_args)


if __name__ == '__main__':
    sys.exit (main ())