import os
import shutil
import tempfile
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import umap.umap_ as umap

UPLOAD_FOLDER = 'uploads'
SORTED_FOLDER = 'static/data'
ARCHIVE_NAME = 'zips'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


if os.path.exists(ARCHIVE_NAME):
        one_minute_ago = time.time() - 180
        for folder in os.listdir(ARCHIVE_NAME):
            folder_path = os.path.join(ARCHIVE_NAME, folder)
            if os.path.isdir(folder_path):
                mtime = os.path.getmtime(folder_path)
                print("******",mtime)
                if mtime < one_minute_ago:
                    print(folder_path, "*******")


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


def del_file(zip_folder, sorted_folder):
    if os.path.exists(zip_folder):
        for filename in os.listdir(zip_folder):
            file_path = os.path.join(zip_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    if os.path.exists(sorted_folder):
        for filename in os.listdir(sorted_folder):
            file_path = os.path.join(sorted_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)



def check_time(user_zip_folder, user_sorted_folder):
    
        del_file(user_zip_folder, user_sorted_folder)
