# check_integrity.py  
# version 0.3
import os
import ast
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_EXT = ".py"

def find_python_files(root, include_tests=False):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(TARGET_EXT):
                if not include_tests and "test" in filename:
                    continue
                yield os.path.join(dirpath, filename)

def check_file_integrity(filepath, verbose=False):
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        errors.append(f"❌ SyntaxError in {filepath} at line {e.lineno}: {e.msg}")
        return errors
    except Exception as e:
        errors.append(f"❌ Failed to parse {filepath}: {str(e)}")
        return errors

    orphan_lines = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("return") or "analysis[" in stripped:
            if not any(
                isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= getattr(node, "end_lineno", lineno)
                for node in ast.walk(tree)
            ):
                orphan_lines.append(lineno)

    if orphan_lines:
        for lineno in orphan_lines:
            errors.append(f"⚠️ Orphan line in {filepath} at line {lineno}: likely outside any function")

    if verbose and not errors:
        print(f"✅ {filepath} passed.")

    return errors

def check_optimizer_consistency():
    """Vérifie la cohérence de l'utilisation de l'optimiseur"""
    
    core_path = 'qaaf/core/qaaf_core.py'
    
    if not os.path.exists(core_path):
        return False, "❌ qaaf_core.py introuvable"
    
    with open(core_path, 'r') as f:
        core_content = f.read()
    
    # Vérifier les différentes configurations possibles
    uses_grid_search = 'from qaaf.optimization.grid_search import GridSearchOptimizer' in core_content
    uses_qaaf_optimizer = 'from qaaf.optimization.grid_search import QAAFOptimizer' in core_content
    optimizer_disabled = 'self.optimizer = None' in core_content
    optimizer_commented = '# from qaaf.optimization' in core_content
    
    if optimizer_disabled or optimizer_commented:
        return True, "✅ Optimiseur désactivé (OK pour v1.1)"
    elif uses_grid_search and not uses_qaaf_optimizer:
        return True, "✅ Utilise GridSearchOptimizer (recommandé pour v1.1)"
    elif uses_qaaf_optimizer and not uses_grid_search:
        return False, "❌ Utilise QAAFOptimizer qui n'existe pas"
    elif uses_grid_search and uses_qaaf_optimizer:
        return False, "❌ Import incohérent (GridSearchOptimizer ET QAAFOptimizer)"
    else:
        return True, "⚠️  Aucun optimiseur détecté (peut être normal)"

def main():
    parser = argparse.ArgumentParser(description="🔍 QAAF Integrity Checker")
    parser.add_argument("--include-tests", action="store_true", help="Inclure les fichiers de test")
    parser.add_argument("--verbose", action="store_true", help="Afficher les fichiers valides")
    parser.add_argument("--only-errors", action="store_true", help="Afficher uniquement les erreurs")
    args = parser.parse_args()

    print("🔍 Scanning Python files for integrity issues...\n")
    
    all_errors = []
    
    # Vérification des fichiers Python
    for filepath in find_python_files(PROJECT_ROOT, include_tests=args.include_tests):
        errors = check_file_integrity(filepath, verbose=args.verbose)
        if errors:
            all_errors.extend(errors)
            if not args.only_errors:
                print("\n".join(errors))
    
    # Vérification spécifique de l'optimiseur
    print("\n" + "=" * 60)
    print("🔧 Vérification de l'optimiseur")
    print("=" * 60)
    
    optimizer_ok, optimizer_msg = check_optimizer_consistency()
    print(optimizer_msg)
    
    if not optimizer_ok:
        all_errors.append(optimizer_msg)
        print("\n💡 Solution: python disable_optimizer.py")
    
    # Résumé final
    print("\n" + "=" * 60)
    if all_errors:
        print(f"❌ Found {len(all_errors)} issue(s) across the codebase.")
        sys.exit(1)
    else:
        print("✅ All Python files passed integrity checks.")
        sys.exit(0)

if __name__ == "__main__":
    main()