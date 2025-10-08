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

# Dans check_integrity.py, section "Vérifications du code"

def check_optimizer_consistency():
    """Vérifie que l'optimiseur est cohérent"""
    
    # Vérifier quel optimiseur est utilisé
    core_path = 'qaaf/core/qaaf_core.py'
    with open(core_path, 'r') as f:
        core_content = f.read()
    
    uses_grid_search = 'from qaaf.optimization.grid_search import' in core_content
    uses_qaaf_optimizer = 'from qaaf.optimization.qaaf_optimizer import' in core_content
    
    if uses_grid_search and not uses_qaaf_optimizer:
        print("✅ Utilise GridSearchOptimizer (simple, recommandé pour v1.1)")
        return True
    elif uses_qaaf_optimizer and not uses_grid_search:
        print("⚠️  Utilise QAAFOptimizer (complexe, risqué pour v1.1)")
        return True
    elif uses_grid_search and uses_qaaf_optimizer:
        print("❌ Utilise LES DEUX optimiseurs (incohérent!)")
        return False
    else:
        print("❌ N'utilise AUCUN optimiseur")
        return False
        
def main():
    parser = argparse.ArgumentParser(description="🔍 QAAF Integrity Checker")
    parser.add_argument("--include-tests", action="store_true", help="Inclure les fichiers de test")
    parser.add_argument("--verbose", action="store_true", help="Afficher les fichiers valides")
    parser.add_argument("--only-errors", action="store_true", help="Afficher uniquement les erreurs")
    args = parser.parse_args()

    print("🔍 Scanning Python files for integrity issues...\n")
    all_errors = []
    for filepath in find_python_files(PROJECT_ROOT, include_tests=args.include_tests):
        errors = check_file_integrity(filepath, verbose=args.verbose)
        if errors:
            all_errors.extend(errors)
            if not args.only_errors:
                print("\n".join(errors))

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} issue(s) across the codebase.")
        sys.exit(1)
    else:
        print("✅ All Python files passed integrity checks.")
        sys.exit(0)

if __name__ == "__main__":
    main()
