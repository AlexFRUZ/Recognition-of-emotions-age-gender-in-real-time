import os
import shutil

source_dir = r"D:\Train\Cyber_Bulling\Train"
target_dir = r"D:\Train\Cyber_Bulling\Test"

os.makedirs(target_dir, exist_ok=True)

for folder_name in os.listdir(source_dir):
    source_folder_path = os.path.join(source_dir, folder_name)
    target_folder_path = os.path.join(target_dir, folder_name)

    if os.path.isdir(source_folder_path):
        os.makedirs(target_folder_path, exist_ok=True)

        files = os.listdir(source_folder_path)

        for file_name in files[:1600]:
            source_file_path = os.path.join(source_folder_path, file_name)
            target_file_path = os.path.join(target_folder_path, file_name)
            shutil.move(source_file_path, target_file_path)

print("Переміщення завершено.")
