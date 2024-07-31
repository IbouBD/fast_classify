# Utiliser une image officielle de Python comme image de base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par l'application Flask
EXPOSE 5000

# Commande par défaut (à remplacer par le service dans docker-compose)
CMD ["flask", "run", "--host=0.0.0.0"]