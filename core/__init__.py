# Commandes terminal pour éditer qaaf_core.py
cd qaaf/core
# Utilisez votre éditeur préféré pour modifier qaaf_core.py
# Par exemple: nano qaaf_core.py, vim qaaf_core.py ou ouvrez-le dans votre IDE
# Exemple pour le module core
cat > qaaf/core/__init__.py << EOF
from qaaf.core.qaaf_core import QAAFCore

__all__ = ['QAAFCore']
EOF