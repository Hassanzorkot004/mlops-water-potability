


# =============================
#  MAKEFILE – Projet MLOps
# =============================

# --- VARIABLES ---
PYTHON=python3
MAIN=main.py
REQ=requirements.txt


# =============================
#  I. INSTALLATION
# =============================
install:
	@echo "📦 Installation des dépendances..."
	pip install -r $(REQ)



# =============================
#  II. CI – CODE QUALITY
# =============================

# Formatage du code (Black)
format:
	@echo "🎨 Formatage du code avec Black..."
	black .

# Analyse qualité du code (pylint)
lint:
	@echo "🔍 Vérification de la qualité du code..."
	pylint *.py

# Sécurité du code (bandit)
security:
	@echo "🛡️ Analyse de sécurité..."
	bandit -r .

# Tout CI
ci: format lint security
	@echo "✅ CI COMPLET : format + lint + sécurité"



# =============================
#  III. PIPELINE ML
# =============================

# 1) Préparer les données
prepare:
	@echo "🧹 Étape : Préparation des données..."
	$(PYTHON) $(MAIN) --prepare

# 2) Entraîner le modèle
train:
	@echo "🤖 Étape : Entraînement du modèle..."
	$(PYTHON) $(MAIN) --train

# 3) Validation / Évaluation
validate:
	@echo "📊 Étape : Validation du modèle..."
	$(PYTHON) $(MAIN) --validate



# =============================
#  IV. TESTS UNITAIRES
# =============================
test:
	@echo "🧪 Exécution des tests..."
	pytest -q



# =============================
#  V. PIPELINE COMPLET
# =============================
all: install ci prepare train validate test
	@echo "🎉 Pipeline complet exécuté avec succès !"



# =============================
# VI. CLEAN (OPTIONNEL)
# =============================
clean:
	@echo "🧽 Nettoyage des fichiers temporaires..."
	rm -rf __pycache__
	rm -f *.pkl
