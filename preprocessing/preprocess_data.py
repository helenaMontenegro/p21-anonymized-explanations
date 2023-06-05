import h5py, csv, re, os, cv2, math, numpy as np, argparse
from random import seed, random

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='images', help="Path to the folder with the images")
parser.add_argument('--annotations_path', type=str, default='medical-description-v2-1.csv', help="Path to the csv with annotations")
parser.add_argument('--save_path', type=str, default='warsaw_data.hdf5', help="Path to the file where the data will be saved")
args = parser.parse_args()

seed(123)
img_size = (64, 64)
train_data_prob = 0.8
base_path = args.image_path
annotations_path = args.annotations_path
save_path = args.save_path

def normalize_image(img, eye):
    img = img[0:img.shape[0]-11, 0:img.shape[1]]
    if eye == 'right':
        img = cv2.flip(img, 1)

    centre = (img.shape[1]/2, img.shape[0]/2)

    img_circle = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_circle = cv2.GaussianBlur(img_circle, (9, 9), 0)
    img_circle = cv2.threshold(img_circle, 130, 255, cv2.THRESH_TRUNC)[1]

    normalizedImg = np.zeros((img.shape[0], img.shape[1]))
    normalizedImg = cv2.normalize(img_circle,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img_circle = cv2.threshold(normalizedImg, 130, 255, cv2.THRESH_BINARY)[1]
    img_circle = cv2.GaussianBlur(img_circle, (9, 9), 0)

    circles = cv2.HoughCircles(img_circle, cv2.HOUGH_GRADIENT, 1.5, img.shape[0], param1=200, param2=25, minRadius=int(img.shape[0]/6), maxRadius=int(img.shape[0]/3))
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if abs(centre[0]-x) > 100 or abs(centre[1]-y) > 100:
                break
            
            M = np.float32([[1, 0, centre[0] - x],[0, 1, centre[1] - y]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode = cv2.BORDER_REPLICATE)
            break

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_annotations():
    glaucoma_id = []
    with open(annotations_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        i = 0
        for row in csv_reader:
            if i == 0:
                i += 1
                continue
            j = 1
            while j <= 2:
                diseases = row[j]
                diseases = diseases.split(',')
                diseases = list(filter(None, diseases))
                for dis in diseases:
                    if dis[0] == ' ':
                        dis = dis[1:]
                    if dis == 'glaucoma' or dis == 'secondary glaucoma':
                        eye = 'right'
                        if j == 1:
                            eye = 'left'
                        glaucoma_id.append(row[0] + "-" + eye)
                j+=1
    return glaucoma_id

def create_dataset():
    label_ids = [] # ids
    label_dis = [] # disease
    label_set = [] # training or testing set
    images = []
    name = []
    names = []
    glaucoma_id = read_annotations()

    for i in os.listdir(base_path):
        name_path = os.path.join(base_path, i)
        p_id = int(i[:4])
        eye = i[4:]
        for k in os.listdir(name_path):
            if(re.search('IG', k)):
                names.append(k)
                img = cv2.imread(os.path.join(name_path, k))
                img = normalize_image(img, eye)
                images.append(img)
                rand = random()
                if (rand <= 0.7):
                    label_set.append(0)
                elif rand <= 0.9:
                    label_set.append(1)
                else:
                    label_set.append(2)

                if p_id not in name:
                    print("Patient ID = " + str(p_id))
                    name.append(p_id)
                label_ids.append(name.index(p_id))
                
                if str(p_id) + "-" + eye in glaucoma_id:
                    label_dis.append(1)
                else:
                    label_dis.append(0)

    images = np.asarray(images)
    names = np.asarray(names)
    print(images.shape)
    print("Number of images with glaucoma: " + str(label_dis.count(1)))
    print("Total number of images: " + str(len(label_dis)))

    hf = h5py.File(save_path, 'w')  # open the file in append mode

    hf.create_dataset('id', data=label_ids)
    hf.create_dataset('dis', data=label_dis)
    hf.create_dataset('set', data=label_set)
    hf.create_dataset('images', data=images)
    hf.close()

create_dataset()
