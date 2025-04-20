import setuptools
import subprocess
import sys
import logging

def resolve_cupy_conflicts():
    """
    Tente de résoudre les conflits d'installation de CuPy.
    (Code de la fonction resolve_cupy_conflicts, légèrement adapté)
    """
    logger = logging.getLogger('setup')  # Use a setup-specific logger
    logger.info("Tentative de résolution des conflits CuPy...")

    conflicting_packages = ['cupy-cuda11x', 'cupy-cuda12x']
    try:
        installed_packages = [pkg.split('==')[0] for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split()]
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de la récupération des packages installés: {e}")
        return False

    for pkg in conflicting_packages:
        if pkg in installed_packages:
            try:
                logger.warning(f"Désinstallation de {pkg}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg])
            except subprocess.CalledProcessError as e:
                logger.error(f"Erreur lors de la désinstallation de {pkg}: {e}")
                return False

    try:
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode()
        cuda_major_version = int(cuda_version.split('release ')[1].split(',')[0].split('.')[0])
        cupy_version_to_install = f'cupy-cuda{cuda_major_version}x'
        logger.info(f"CUDA détecté, installation de {cupy_version_to_install}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', cupy_version_to_install])

    except FileNotFoundError:
        logger.info("CUDA non détecté, installation de cupy")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cupy'])
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'installation de CuPy: {e}")
        return False
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la gestion de CuPy: {e}")
        return False

    logger.info("Résolution des conflits CuPy terminée.")
    return True

# Configurer le logging pour setup.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Exécuter la résolution CuPy (avant setuptools.setup)
resolve_cupy_conflicts()

setuptools.setup(
    name='qaaf',
    version='0.1.0',  # Change this to your desired version
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'yfinance>=0.2.55',
        'tqdm>=4.62.0',
        'pytest>=6.2.0',
        # 'cupy-cuda11x>=11.0.0'  #  Do not hardcode CuPy version here; handled by resolve_cupy_conflicts
    ],
    # ... other setup parameters ...
)