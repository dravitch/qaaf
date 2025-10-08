"""
Module de gestion des erreurs et débogage pour QAAF
Permet un traçage complet des erreurs dans la pipeline
"""

import logging
import traceback
import sys
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class QAAFErrorHandler:
    """
    Gestionnaire centralisé des erreurs pour QAAF
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialise le gestionnaire d'erreurs
        
        Args:
            verbose: Si True, affiche les erreurs détaillées sur stderr
        """
        self.verbose = verbose
        self.error_history = []
        
    def log_error(self, error: Exception, context: str = "", additional_info: dict = None):
        """
        Enregistre une erreur avec son contexte complet
        
        Args:
            error: L'exception capturée
            context: Description du contexte où l'erreur s'est produite
            additional_info: Informations supplémentaires (dict)
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "additional_info": additional_info or {}
        }
        
        self.error_history.append(error_info)
        
        # Log vers le logger
        logger.error("=" * 80)
        logger.error(f"❌ ERREUR DANS: {context}")
        logger.error("=" * 80)
        logger.error(f"Type: {error_info['error_type']}")
        logger.error(f"Message: {error_info['error_message']}")
        if additional_info:
            logger.error(f"Info supplémentaire: {additional_info}")
        logger.error("\nTraceback complet:")
        logger.error(error_info['traceback'])
        logger.error("=" * 80)
        
        # Affichage sur stderr si verbose
        if self.verbose:
            print("\n" + "=" * 80, file=sys.stderr)
            print(f"❌ ERREUR CRITIQUE: {context}", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Type: {error_info['error_type']}", file=sys.stderr)
            print(f"Message: {error_info['error_message']}", file=sys.stderr)
            if additional_info:
                print(f"\nInformations supplémentaires:", file=sys.stderr)
                for key, value in additional_info.items():
                    print(f"  - {key}: {value}", file=sys.stderr)
            print("\nTraceback complet:", file=sys.stderr)
            print(error_info['traceback'], file=sys.stderr)
            print("=" * 80, file=sys.stderr)
    
    def get_last_error(self) -> Optional[dict]:
        """
        Retourne la dernière erreur enregistrée
        """
        return self.error_history[-1] if self.error_history else None
    
    def get_error_summary(self) -> str:
        """
        Génère un résumé des erreurs enregistrées
        """
        if not self.error_history:
            return "Aucune erreur enregistrée"
        
        summary = f"\n{'=' * 80}\n"
        summary += f"📋 RÉSUMÉ DES ERREURS ({len(self.error_history)} erreur(s))\n"
        summary += f"{'=' * 80}\n\n"
        
        for i, error in enumerate(self.error_history, 1):
            summary += f"{i}. [{error['timestamp']}] {error['context']}\n"
            summary += f"   Type: {error['error_type']}\n"
            summary += f"   Message: {error['error_message']}\n"
            if error['additional_info']:
                summary += f"   Info: {error['additional_info']}\n"
            summary += "\n"
        
        summary += f"{'=' * 80}\n"
        return summary
    
    def clear_history(self):
        """
        Efface l'historique des erreurs
        """
        self.error_history = []


def safe_execute(error_handler: QAAFErrorHandler, context: str, **kwargs):
    """
    Décorateur pour exécuter une fonction avec gestion d'erreurs
    
    Args:
        error_handler: Instance de QAAFErrorHandler
        context: Description du contexte
        **kwargs: Informations supplémentaires à logger
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **func_kwargs) -> Any:
            try:
                return func(*args, **func_kwargs)
            except Exception as e:
                error_handler.log_error(
                    error=e,
                    context=f"{context} - {func.__name__}",
                    additional_info=kwargs
                )
                raise
        return wrapper
    return decorator


class PipelineTracker:
    """
    Suit l'exécution d'une pipeline étape par étape
    """
    
    def __init__(self, pipeline_name: str = "QAAF Pipeline"):
        """
        Initialise le tracker de pipeline
        
        Args:
            pipeline_name: Nom de la pipeline
        """
        self.pipeline_name = pipeline_name
        self.steps = []
        self.current_step = None
        self.start_time = None
        
    def start_step(self, step_name: str, description: str = ""):
        """
        Démarre une nouvelle étape
        """
        self.current_step = {
            "name": step_name,
            "description": description,
            "start_time": datetime.now(),
            "end_time": None,
            "status": "running",
            "error": None,
            "duration": None
        }
        
        print(f"\n▶️  {step_name}...", flush=True)
        if description:
            print(f"   {description}", flush=True)
    
    def end_step(self, success: bool = True, error: Optional[Exception] = None):
        """
        Termine l'étape courante
        """
        if self.current_step is None:
            return
        
        self.current_step["end_time"] = datetime.now()
        self.current_step["duration"] = (
            self.current_step["end_time"] - self.current_step["start_time"]
        ).total_seconds()
        self.current_step["status"] = "success" if success else "failed"
        self.current_step["error"] = str(error) if error else None
        
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {self.current_step['name']} - {'OK' if success else 'ÉCHEC'}", flush=True)
        
        if error:
            print(f"   Erreur: {error}", flush=True)
        
        self.steps.append(self.current_step)
        self.current_step = None
    
    def get_summary(self) -> str:
        """
        Génère un résumé de l'exécution
        """
        if not self.steps:
            return "Aucune étape exécutée"
        
        total_duration = sum(step["duration"] for step in self.steps if step["duration"])
        success_count = sum(1 for step in self.steps if step["status"] == "success")
        failed_count = len(self.steps) - success_count
        
        summary = f"\n{'=' * 80}\n"
        summary += f"📊 RÉSUMÉ D'EXÉCUTION: {self.pipeline_name}\n"
        summary += f"{'=' * 80}\n"
        summary += f"Total d'étapes: {len(self.steps)}\n"
        summary += f"Réussies: {success_count} ✅\n"
        summary += f"Échouées: {failed_count} ❌\n"
        summary += f"Durée totale: {total_duration:.2f}s\n"
        summary += f"\nDétail des étapes:\n"
        summary += f"{'-' * 80}\n"
        
        for i, step in enumerate(self.steps, 1):
            status = "✅" if step["status"] == "success" else "❌"
            summary += f"{i}. {status} {step['name']}"
            if step['duration']:
                summary += f" ({step['duration']:.2f}s)"
            summary += "\n"
            if step['description']:
                summary += f"   {step['description']}\n"
            if step['error']:
                summary += f"   Erreur: {step['error']}\n"
        
        summary += f"{'=' * 80}\n"
        return summary


# Instance globale pour faciliter l'utilisation
global_error_handler = QAAFErrorHandler(verbose=True)


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module error_handler\n")
    
    handler = QAAFErrorHandler(verbose=True)
    tracker = PipelineTracker("Test Pipeline")
    
    # Test 1: Étape réussie
    tracker.start_step("Test 1", "Opération simple")
    try:
        result = 1 + 1
        tracker.end_step(success=True)
    except Exception as e:
        handler.log_error(e, "Test 1")
        tracker.end_step(success=False, error=e)
    
    # Test 2: Étape échouée
    tracker.start_step("Test 2", "Opération qui échoue")
    try:
        result = 1 / 0
        tracker.end_step(success=True)
    except Exception as e:
        handler.log_error(e, "Test 2", additional_info={"operation": "division by zero"})
        tracker.end_step(success=False, error=e)
    
    # Affichage des résumés
    print(tracker.get_summary())
    print(handler.get_error_summary())