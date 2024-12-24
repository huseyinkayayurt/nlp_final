import os


def create_dir(folder_name):
    dir_path = os.path.join(folder_name, "")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
