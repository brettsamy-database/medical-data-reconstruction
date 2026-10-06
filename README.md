# Environnement virtuel
.venv/
env/
venv/

# Cache Python
__pycache__/
*.pyc
.pytest_cache/

# Jupyter Notebooks
.ipynb_checkpoints/

# Fichiers de données lourds (ne pas envoyer sur GitHub)
data/raw/*
data/processed/*
data/gold_standard/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/gold_standard/.gitkeep

# Fichiers de configuration sensibles (clés API, secrets)
.env
config/secrets.json

# Fichiers système / IDE
.DS_Store
.vscode/
.idea/