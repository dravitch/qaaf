# qaaf/utils/setup_environment.py
import subprocess
import sys
import logging

logger=logging.getLogger (__name__)


def check_cupy_installation ():
    """Vérifie si l'installation de CuPy est correcte et propose des corrections."""
    try:
        import cupy
        return True
    except ImportError:
        logger.info ("CuPy n'est pas installé. L'accélération GPU n'est pas disponible.")
        return False
    except Exception as e:
        logger.warning (f"Problème avec l'installation de CuPy: {str (e)}")
        return False


def fix_cupy_installation ():
    """Tente de corriger l'installation de CuPy."""
    try:
        # Désinstaller toutes les versions de CuPy
        packages=["cupy","cupy-cuda11x","cupy-cuda12x","cupy-cuda110","cupy-cuda111","cupy-cuda112"]
        for pkg in packages:
            try:
                subprocess.check_call ([sys.executable,"-m","pip","uninstall","-y",pkg])
                logger.info (f"Désinstallation de {pkg} réussie.")
            except:
                pass

        # Réinstaller CuPy pour la bonne version CUDA
        # Vous pouvez ajuster selon votre environnement
        subprocess.check_call ([sys.executable,"-m","pip","install","cupy-cuda12x"])
        logger.info ("Réinstallation de CuPy réussie.")

        return True
    except Exception as e:
        logger.error (f"Échec de la correction de l'installation de CuPy: {str (e)}")
        return False


def setup_environment ():
    """Configure l'environnement pour QAAF."""
    if not check_cupy_installation ():
        user_input=input ("Voulez-vous tenter de corriger l'installation de CuPy? (o/n): ")
        if user_input.lower () in ['o','oui','y','yes']:
            if fix_cupy_installation ():
                print ("Installation de CuPy corrigée avec succès!")
            else:
                print ("Impossible de corriger l'installation de CuPy. L'accélération GPU sera désactivée.")


if __name__ == "__main__":
    setup_environment ()