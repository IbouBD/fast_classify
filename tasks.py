from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
from utils import*
from database import create_app
from flask import jsonify
from flask_login import current_user
import time

app, celery = create_app()

#app = Celery('tasks', backend='rpc://', broker='redis://localhost:6379/0')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Charger le modèle VGG pré-entraîné
vgg = models.vgg19(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(vgg.features.children()), vgg.avgpool)
feature_extractor.eval()

    # Préparer la transformation pour les images
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(image.unsqueeze(0))
        return features.flatten().numpy()
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de {image_path}: {e}")
        return None


def organize_files(df, image_folder, sorted_folder):
    # Vérifier et créer le dossier de destination si nécessaire
    if not os.path.exists(sorted_folder):
        os.makedirs(sorted_folder)

    # Obtenir les clusters uniques et les fichiers associés
    clusters = df['cluster'].unique()
    clustered_files = {cluster: df[df['cluster'] == cluster]['image'].tolist() for cluster in clusters}

    # Parcourir chaque cluster et organiser les fichiers
    for cluster in clusters:
        current_cluster = clustered_files[cluster]
        folder_path = os.path.join(sorted_folder, f'group_{cluster + 1}')

        # Vérifier et créer le sous-dossier pour le cluster si nécessaire
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Déplacer chaque image dans le sous-dossier correspondant
        for img in current_cluster:
            src_path = os.path.join(image_folder, img)
            dest_path = os.path.join(folder_path, img)

            # Vérifier si le fichier source existe avant de le déplacer
            if not os.path.exists(src_path):
                print(f"Le fichier source {src_path} n'existe pas.")
                continue
            try:
                shutil.move(src_path, dest_path)
            except Exception as e:
                print(f"Erreur lors du déplacement de {src_path} vers {dest_path}: {e}")

@celery.task(bind=True)
def process_images(self, image_data, nb_cluster, upload_folder, sorted_folder):
    pca = PCA(n_components=10)

    features = []
    image_names = []

    for filename, file_path in image_data:
        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            image_names.append(filename)

    features_array = np.array(features)
    print(f"Features array shape: {features_array.shape}")

    # tsne = TSNE(n_components=3, random_state=0, perplexity=30)
    # projections = tsne.fit_transform(features_array)
    features_pca = pca.fit_transform(features_array)
    # Conversion des features en DataFrame
    df = pd.DataFrame(features_pca)
    df['image'] = image_names
    
    # Réduction de dimensionnalité avec UMAP
    umap_model = umap.UMAP(n_neighbors=20, min_dist=0.5, metric='euclidean', n_components=3)
    projections = umap_model.fit_transform(df.iloc[:, :-2])  # Enlever les colonnes 'image' et 'class' avant l'ajustement

    # Ajouter les nouvelles coordonnées réduites au DataFrame
    df['x'] = projections[:, 0]
    df['y'] = projections[:, 1]
    df['z'] = projections[:, 2]

    km = KMeans(random_state = 0, n_init = 10, max_iter=200,n_clusters=nb_cluster)
    df['cluster'] = km.fit_predict(df[['x', 'y', 'z']])

    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        title='3D space with your data',
        hover_data=['image'],
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=5))

    output_path = os.path.join('static', 'cluster_plot.html')
    fig.write_html(output_path)


    
    print("Calling organize_files function")
    organize_files(df, upload_folder, sorted_folder)
    print("organize_files function executed successfully")

    
    return output_path

@celery.task
def del_file():
    one_minute_ago = time.time() - 180

    if current_user.is_authenticated:
        user_zip_folder = os.path.join(app.config['ZIP_FOLDER'], str(current_user.id))
        user_sorted_folder = os.path.join(app.config['SORTED_FOLDER'], str(current_user.id))
    else:
        user_zip_folder = os.path.join(app.config['ZIP_FOLDER'], current_user.id)
        user_sorted_folder = os.path.join(app.config['SORTED_FOLDER'], current_user.id)

    deleted_files = []

    try:
        if os.path.exists(user_zip_folder):
            for filename in os.listdir(user_zip_folder):
                file_path = os.path.join(user_zip_folder, filename)
                if os.path.isfile(file_path):
                    mtime = os.path.getmtime(file_path)
                    if mtime < one_minute_ago:
                        os.remove(file_path)
                        deleted_files.append(filename)

        if os.path.exists(user_sorted_folder):
            for folder in os.listdir(user_sorted_folder):
                folder_path = os.path.join(user_sorted_folder, folder)
                
                # Check if it's a directory before proceeding
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, filename)
                        
                        # Check if it's a file before proceeding
                        if os.path.isfile(file_path):
                            mtime = os.path.getmtime(file_path)
                            if mtime < one_minute_ago:
                                os.remove(file_path)
                                deleted_files.append(file_path)
                    
                    # After all files are checked, remove the folder if it's empty
                    if not os.listdir(folder_path):
                        shutil.rmtree(folder_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print({"deleted_files": deleted_files, "status": "deleted" if deleted_files else "no files to delete"})
    return jsonify({"deleted_files": deleted_files, "status": "deleted" if deleted_files else "no files to delete"})