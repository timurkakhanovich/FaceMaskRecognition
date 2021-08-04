import os
import cv2
import numpy as np

def load_data_to_files(data_dir, name):
    folder = os.path.join(data_dir, name)
    files = os.listdir(folder)

    train, val, test = np.split(files, [int(0.8 * len(files)), int(0.9 * len(files))])

    os.mkdir('VGG-Face2-Data/train/' + name)
    for jpg in train:
        img = cv2.imread(folder + '/' + jpg)
        cv2.imwrite('VGG-Face2-Data/train/' + name + '/' + jpg, img)

    os.mkdir('VGG-Face2-Data/test/' + name)
    for jpg in test:
        img = cv2.imread(folder + '/' + jpg)
        cv2.imwrite('VGG-Face2-Data/test/' + name + '/' + jpg, img)
    
    os.mkdir('VGG-Face2-Data/val/' + name)
    for jpg in val:
        img = cv2.imread(folder + '/' + jpg)
        cv2.imwrite('VGG-Face2-Data/val/' + name + '/' + jpg, img)

if __name__ == "__main__":
    #os.chdir(r'E:\\Projects\\CourseWork\\face_mask_recognition')
    
    for pers in os.listdir('VGG-Face2'):
        load_data_to_files('VGG-Face2/', pers)
