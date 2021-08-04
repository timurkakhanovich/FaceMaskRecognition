import dlib
import cv2
import os
import numpy as np
import imutils
from imutils.face_utils import rect_to_bb, shape_to_np
from tqdm import tqdm

def fill_padding(image):
    ww = 220
    hh = 220
    color = (0, 0, 0)
    
    ht, wt, cc = image.shape
    result = np.full((ww, hh, cc), color)

    if wt > ht:
        newImg = cv2.resize(image, (int(220/wt*ht), 220))
    elif wt < ht:
        newImg = cv2.resize(image, (220, int(220/ht*wt)))

    ht, wt, _ = newImg.shape

    xx = (ww - wt) // 2
    yy = (hh - ht) // 2

    result[yy : yy+ht, xx : xx+wt] = newImg

    return result

def mesh_image(img_path, detector, predictor, save_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    
    if len(rects) == 1:
        shape = predictor(gray, rects[0])
        shape = shape_to_np(shape)

        colors = [(255, 255, 255), (0, 0, 0), (255, 0, 0)]
        curr_color = int(np.random.choice(3, 1, p=[0.6, 0.3, 0.1]))

        cv2.fillPoly(image, [np.take(shape, list(range(6, 15)) + [29, 1], axis=0)], 
                    color=colors[curr_color])

        x_min, y_min = min(shape[:, 0]), min(shape[:, 1])
        x_max, y_max = max(shape[:, 0]), max(shape[:, 1])

        image = image[y_min-1000 : y_max+1000, x_min-1000 : x_max+1000]
        image = fill_padding(image)
        cv2.imwrite(save_path, image)
    else:
        return

def main():
    os.chdir(r'E:\\Projects\\CourseWork\\face_mask_recognition')
    imagePath = r'VGG-Face2-Data\\'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for name in tqdm(os.listdir(imagePath)[100:106]):
        if not os.path.exists(r'data2\\masked\\' + name):
            os.mkdir(r'data2\\masked\\' + name)

        currNamePath = imagePath + name
        for img in os.listdir(currNamePath):
            try:
                mesh_image(img_path=currNamePath + '\\' + img, detector=detector, 
                            predictor=predictor, save_path=r'data2\\masked\\' + name + '\\' + img)
            except:
                continue

if __name__ == "__main__":
    main()
