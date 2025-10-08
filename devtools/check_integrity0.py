import os
import ast
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_EXT = ".py"

def find_python_files(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(TARGET_EXT):
                yield os.path.join(dirpath, filename)

def check_file_integrity(filepath):
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
            # Check if line is inside a function
            if not any(
                isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= node.end_lineno
                for node in ast.walk(tree)
                if hasattr(node, "end_lineno")
            ):
                orphan_lines.append(lineno)

    if orphan_lines:
        for lineno in orphan_lines:
            errors.append(f"⚠️ Orphan line in {filepath} at line {lineno}: likely outside any function")

    return errors

def main():
    print("🔍 Scanning Python files for integrity issues...\n")
    all_errors = []
    for filepath in find_python_files(PROJECT_ROOT):
        errors = check_file_integrity(filepath)
        if errors:
            all_errors.extend(errors)

    if all_errors:
        print("\n".join(all_errors))
        print(f"\n❌ Found {len(all_errors)} issue(s) across the codebase.")
        sys.exit(1)
    else:
        print("✅ All Python files passed integrity checks.")

if __name__ == "__main__":
    main()
