import os
import shutil
from sklearn.model_selection import train_test_split

# 원본 데이터 폴더 경로
source_folder = 'Mydata'

# 대상 폴더 경로
dataset_folder = 'dataset'
val_dataset_folder = 'val_dataset'

# 각 하위 폴더 이름
subfolders = ['1', '2', '3', '4', '5', '6', '7']

# 각 폴더를 순회하면서 파일을 분할하여 복사합니다.
for subfolder in subfolders:
    source_path = os.path.join(source_folder, subfolder)
    dataset_path = os.path.join(dataset_folder, subfolder)
    val_dataset_path = os.path.join(val_dataset_folder, subfolder)

    # 대상 폴더가 존재하지 않으면 생성합니다.
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(val_dataset_path, exist_ok=True)

    # 원본 폴더의 모든 파일 목록을 가져옵니다.
    files = os.listdir(source_path)
    
    # 파일을 train과 val로 나눕니다.
    train_files, val_files = train_test_split(files, test_size=4, random_state=42)

    # train 파일들을 dataset 폴더로 복사합니다.
    for file_name in train_files:
        src_file = os.path.join(source_path, file_name)
        dst_file = os.path.join(dataset_path, file_name)
        shutil.copy(src_file, dst_file)

    # val 파일들을 val_dataset 폴더로 복사합니다.
    for file_name in val_files:
        src_file = os.path.join(source_path, file_name)
        dst_file = os.path.join(val_dataset_path, file_name)
        shutil.copy(src_file, dst_file)

print("데이터셋 분할 및 복사가 완료되었습니다.")
