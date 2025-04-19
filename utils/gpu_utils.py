"""
Module d'utilitaires pour l'utilisation du GPU dans QAAF.
Fournit des fonctions pour détecter et utiliser le GPU lorsque disponible.
"""

import logging
import numpy as np
from typing import Any,Union,Tuple

# Configuration du logging
logger=logging.getLogger (__name__)

# Variables globales pour le statut GPU
GPU_AVAILABLE=False
GPU_MODULE=None


def initialize_gpu_support () -> bool:
    """
    Initialise le support GPU en vérifiant la disponibilité de CuPy ou PyTorch.

    Returns:
        bool: True si le GPU est disponible et utilisable, False sinon
    """
    global GPU_AVAILABLE,GPU_MODULE

    # Vérification de CuPy en premier
    try:
        import cupy as cp
        GPU_MODULE=cp
        # Test d'allocation pour vérifier le fonctionnement
        try:
            test_array=cp.array ([1,2,3])
            test_result=cp.sum (test_array)
            GPU_AVAILABLE=True
            logger.info (f"Support GPU activé via CuPy {cp.__version__}")
            return True
        except Exception as e:
            logger.warning (f"CuPy est installé mais l'initialisation a échoué: {str (e)}")
    except ImportError:
        logger.debug ("CuPy n'est pas installé, tentative avec PyTorch...")

    # Tentative avec PyTorch si CuPy n'est pas disponible
    try:
        import torch
        if torch.cuda.is_available ():
            # Test pour vérifier le fonctionnement
            try:
                test_tensor=torch.tensor ([1,2,3]).cuda ()
                test_result=torch.sum (test_tensor)
                GPU_AVAILABLE=True
                GPU_MODULE=torch
                logger.info (f"Support GPU activé via PyTorch {torch.__version__}")
                return True
            except Exception as e:
                logger.warning (f"PyTorch CUDA est disponible mais l'initialisation a échoué: {str (e)}")
        else:
            logger.debug ("PyTorch est installé mais CUDA n'est pas disponible")
    except ImportError:
        logger.debug ("PyTorch n'est pas installé")

    logger.info ("Support GPU non disponible, utilisation de NumPy sur CPU")
    return False


def get_array_module ():
    """
    Retourne le module à utiliser pour les calculs (NumPy ou module GPU).

    Returns:
        module: NumPy ou le module GPU (CuPy ou PyTorch) si disponible
    """
    global GPU_AVAILABLE,GPU_MODULE

    if not GPU_AVAILABLE:
        return np

    return GPU_MODULE


def to_device (data: Any,use_gpu: bool = None) -> Any:
    """
    Transfère les données vers le GPU ou le CPU selon la disponibilité et les préférences.

    Args:
        data: Données à transférer (ndarray, DataFrame, Series, etc.)
        use_gpu: Forcer l'utilisation du GPU (True) ou du CPU (False), ou None pour auto

    Returns:
        Les données transférées vers le dispositif approprié
    """
    global GPU_AVAILABLE,GPU_MODULE

    # Détermination de l'utilisation du GPU
    should_use_gpu=GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)

    if not should_use_gpu:
        # Conversion vers CPU si nécessaire
        if GPU_MODULE == 'cupy' and hasattr (data,'get'):
            return data.get ()
        elif GPU_MODULE == 'torch' and hasattr (data,'cpu'):
            return data.cpu ().numpy ()
        return data

    # Conversion vers GPU
    if GPU_MODULE is None:
        return data

    if isinstance (data,np.ndarray):
        if GPU_MODULE.__name__ == 'cupy':
            return GPU_MODULE.array (data)
        elif GPU_MODULE.__name__ == 'torch':
            return GPU_MODULE.tensor (data).cuda ()

    # Pour les types pandas, conversion manuelle
    elif hasattr (data,'values'):
        # Pour pandas DataFrame et Series
        try:
            if GPU_MODULE.__name__ == 'cupy':
                return GPU_MODULE.array (data.values)
            elif GPU_MODULE.__name__ == 'torch':
                return GPU_MODULE.tensor (data.values).cuda ()
        except:
            logger.warning (f"Échec de la conversion vers GPU pour {type (data)}")
            return data

    return data


def mem_info () -> Tuple[float,float]:
    """
    Obtient des informations sur l'utilisation de la mémoire GPU.

    Returns:
        Tuple[float, float]: (mémoire utilisée en MB, mémoire totale en MB)
    """
    global GPU_AVAILABLE,GPU_MODULE

    if not GPU_AVAILABLE:
        return (0,0)

    if GPU_MODULE.__name__ == 'cupy':
        try:
            mempool=GPU_MODULE.get_default_memory_pool ()
            used=mempool.used_bytes () / (1024 * 1024)  # MB
            total=mempool.total_bytes () / (1024 * 1024)  # MB
            return (used,total)
        except:
            return (0,0)
    elif GPU_MODULE.__name__ == 'torch':
        try:
            used=GPU_MODULE.cuda.memory_allocated () / (1024 * 1024)  # MB
            total=GPU_MODULE.cuda.get_device_properties (0).total_memory / (1024 * 1024)  # MB
            return (used,total)
        except:
            return (0,0)

    return (0,0)


def clear_memory () -> None:
    """
    Libère la mémoire GPU si possible.
    """
    global GPU_AVAILABLE,GPU_MODULE

    if not GPU_AVAILABLE:
        return

    if GPU_MODULE.__name__ == 'cupy':
        try:
            mempool=GPU_MODULE.get_default_memory_pool ()
            mempool.free_all_blocks ()
            logger.debug ("Mémoire GPU (CuPy) libérée")
        except Exception as e:
            logger.warning (f"Échec de la libération de mémoire GPU (CuPy): {str (e)}")
    elif GPU_MODULE.__name__ == 'torch':
        try:
            GPU_MODULE.cuda.empty_cache ()
            logger.debug ("Mémoire GPU (PyTorch) libérée")
        except Exception as e:
            logger.warning (f"Échec de la libération de mémoire GPU (PyTorch): {str (e)}")


# Initialisation automatique au chargement du module
initialize_gpu_support ()

# Interface simplifiée pour l'utilisation
xp=get_array_module ()