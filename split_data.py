from pathlib import Path
import cv2
import numpy as np


def load_data_to_files(data_dir: Path, name: str) -> None:
    source_data_folder = data_dir / name
    target_data_folder = Path("VGG-Face2-Data/")
    files = source_data_folder.glob("**/*")

    train, val, test = np.split(files, [int(0.8 * len(files)), int(0.9 * len(files))])
    
    train_target_folder = target_data_folder / train / name
    test_target_folder = target_data_folder / test / name
    val_target_folder = target_data_folder / val / name

    train_target_folder.mkdir(parents=True, exist_ok=True)
    for jpg in train:
        img = cv2.imread(source_data_folder / jpg)
        cv2.imwrite(train_target_folder / jpg, img)

    test_target_folder.mkdir(parents=True, exist_ok=True)
    for jpg in test:
        img = cv2.imread(source_data_folder / jpg)
        cv2.imwrite(test_target_folder / jpg, img)
    
    val_target_folder.mkdir(parents=True, exist_ok=True)
    for jpg in val:
        img = cv2.imread(source_data_folder / jpg)
        cv2.imwrite(val_target_folder / jpg, img)


if __name__ == "__main__":    
    source_data_path = Path("VGG-Face2")

    for pers in source_data_path.glob("**/*"):
        load_data_to_files(source_data_path, pers)
