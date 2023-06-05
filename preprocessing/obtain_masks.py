import h5py, csv, re, os, cv2, math, numpy as np, argparse
from random import seed, random

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='images', help="Path to the .hdf5 file with the data")
parser.add_argument('--save_path', type=str, default='warsaw_data.hdf5', help="Path to the file where the masks will be saved")
args = parser.parse_args()

seed(123)
save_path = args.save_path  # path to save the hdf5 file
saved_data = args.data_path  # path where data is saved

def get_image_mask(img):
    width = img.shape[0]
    height = img.shape[1]
    
    centre = (img.shape[1]/2, img.shape[0]/2)
    img_circle = cv2.GaussianBlur(img, (9, 9), 0)
    img_circle = cv2.threshold(img_circle, 130, 255, cv2.THRESH_TRUNC)[1]
    normalizedImg = np.zeros((img.shape[0], img.shape[1]))
    normalizedImg = cv2.normalize(img_circle,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_circle = cv2.threshold(normalizedImg, 130, 255, cv2.THRESH_BINARY)[1]
    img_circle = cv2.GaussianBlur(img_circle, (9, 9), 0)

    circles = cv2.HoughCircles(img_circle, cv2.HOUGH_GRADIENT, 1.5, img.shape[0], param1=200, param2=25, minRadius=int(img.shape[0]/6), maxRadius=int(img.shape[0]/3))
    mask = np.zeros((img.shape[0], img.shape[1]))

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            (xb, yb, rb) = (x, y, r)
            if r > 17:
                r = 17
            if x < 28 or x > 37:
                if x >= 45:
                    x = 32
                else:
                    x = width // 2 + (x - width // 2) // 2
            if y < 28 or y > 38:
                if y > 40:
                    y = 32
                else:
                    y = height // 2 + (y - height // 2) // 2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(img, (x, y), r, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
            
            break
    return mask

def create_dataset():
    f = h5py.File(saved_data)
    label1 = f['id']                 # labels for the identity recognition network
    label2 = f['dis']                # labels for the task-related classification network (glaucoma)
    label3 = f['set']                # labels for the dataset to which the sample belongs (train - 0, test - 1 or validation - 2)
    x     =  f['images']             # image data
    masks = []
    for img in x:
        mask = get_image_mask(img)
        masks.append(mask)

    masks = np.asarray(masks)
    print(masks.shape)

    hf = h5py.File(save_path, 'w')  # open the file in append mode

    hf.create_dataset('id', data=label1)
    hf.create_dataset('dis', data=label2)
    hf.create_dataset('set', data=label3)
    hf.create_dataset('masks', data=masks)
    hf.close()

create_dataset()
