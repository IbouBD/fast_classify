import os
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import umap.umap_ as umap


UPLOAD_FOLDER = 'uploads'
SORTED_FOLDER = 'data'
ARCHIVE_NAME = 'zips'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if os.path.exists(SORTED_FOLDER):
    shutil.rmtree(SORTED_FOLDER)

if not os.path.exists(SORTED_FOLDER):
    os.makedirs(SORTED_FOLDER)

if os.path.exists(ARCHIVE_NAME):
    shutil.rmtree(ARCHIVE_NAME)

if not os.path.exists(ARCHIVE_NAME):
    os.makedirs(ARCHIVE_NAME)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_files(zip_path, user_sorted_folder, user_zip_folder):
    try:
        shutil.rmtree(user_sorted_folder)
        os.remove(zip_path)
        shutil.rmtree(user_zip_folder)
    except Exception as e:
        print(f"Error during cleanup: {e}")