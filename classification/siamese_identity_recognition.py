import tensorflow, h5py, numpy as np, matplotlib.pyplot as plt, argparse, math
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=80, help="Number of epochs to train the network")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
parser.add_argument('--data_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the data")
parser.add_argument('--save_folder', type=str, default='id_net', help="Folder where the weights of the model will be saved")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
filename = args.data_file
save_folder = args.save_folder
adam = Adam(lr=1e-4)

###################
#### Load Data ####
###################

f = h5py.File(filename)

label1 = f['id']
label1 = np.asarray(label1)
label2 = f['dis']
label2 = np.asarray(label2)
label3 = f['set']
label3 = np.asarray(label3)
x = f['images']
x = np.asarray(x)
if len(x.shape) == 3:
    x = np.reshape(x, (-1, 64, 64, 1))

num_pp = len(np.unique(label1)) # number of identities

# split data into training, validation and test
idx_train = np.asarray(np.where(label3 == 0))
idx_test = np.asarray(np.where(label3 == 1))
idx_valid = np.asarray(np.where(label3 == 2))

x_train = x[idx_train[0,:],:,:,:]
x_test = x[idx_test[0,:],:,:,:]
x_valid = x[idx_valid[0,:],:,:,:]
y_train1 = label1[idx_train[0,:]]
y_test1 = label1[idx_test[0,:]]
y_valid1 = label1[idx_valid[0,:]]
y_train2 = label2[idx_train[0,:]]
y_test2 = label2[idx_test[0,:]]
y_valid2 = label2[idx_valid[0,:]]

# normalize images
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
x_valid = (x_valid - 127.5) / 127.5
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_valid = x_valid.astype('float16')

#########################
#### Get paired data ####
#########################

def get_training_pairs(x_set, labels):
    pairs = []
    pair_labels = []
    for index in range(len(x_set)):
        same_class = np.where(labels == labels[index])[0]
        # obtain pairs of the same class, each pair is unique
        for i in same_class:
            if i > index:
                pairs.append([x_set[index], x_set[i]])
                pair_labels.append(1)
        # obtain pairs of different classes
        # to ensure balanced data, we calculate the number of pairs to generate based on the
        # total number of pairs generated for the same class
        num_diff_class = math.ceil((len(same_class)-1) / 2)
        existing_ids = list(range(0, num_pp))
        existing_ids.remove(labels[index])
        ids = np.random.choice(existing_ids, size=num_diff_class, replace=False)
        for i in range(len(ids)):
            id_class = np.where(labels == ids[i])[0]
            if (len(id_class) != 0):
                chosen_index = np.random.choice(id_class, size=1, replace=False)[0]
                pairs.append([x_set[index], x_set[chosen_index]])
                pair_labels.append(0)
    return np.asarray(pairs), np.asarray(pair_labels)

def get_test_pairs(x_set, labels):
    pairs = []
    pair_labels = []
    for index in range(len(x_set)):
        for i in range(index + 1, len(x_set)):
            pairs.append([x_set[index], x_set[i]])
            pair_labels.append(int(labels[index] == labels[i]))
    return np.asarray(pairs), np.asarray(pair_labels)

def gaussian_noise(image):
    row, col, ch= image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, -1, 1)
    return noisy_image

def add_noise_to_data(pair_set, label_set):
    noisy_set = []
    noisy_labels = []
    for i in range(len(pair_set)):
        noisy_set.append([pair_set[i][0], gaussian_noise(pair_set[i][1])])
        noisy_labels.append(label_set[i])
    noisy_set = np.asarray(noisy_set)
    noisy_labels = np.asarray(noisy_labels)
    new_set = np.concatenate((pair_set, noisy_set))
    new_labels = np.concatenate((label_set, noisy_labels))
    return new_set, new_labels

pair_train, label_train = add_noise_to_data(get_training_pairs(x_train, y_train1))
pair_valid, label_valid = add_noise_to_data(get_training_pairs(x_valid, y_valid1))
pair_test, label_test = add_noise_to_data(get_training_pairs(x_test, y_test1))

def contrastive_loss(y_true, y_pred, margin=10):
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def cnn_model(embedding_dim=256):
    input_imgs = Input((x_train.shape[1:]))
    x = Conv2D(embedding_dim // 8, (5, 5), strides = (2,2), padding = 'same', name = 'id_conv1')(input_imgs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(embedding_dim // 4, (5, 5), strides = (2,2), padding = 'same', name = 'id_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(embedding_dim // 2, (3, 3), strides = (2,2), padding = 'same', name = 'id_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(embedding_dim, (3, 3), strides = (2,2),  padding = 'same', name = 'id_conv4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_dim)(x)

    return Model(input_imgs, x)

img_a = Input(x_train.shape[1:])
img_b = Input(x_train.shape[1:])
cnn = cnn_model()
cnn.summary()
latent_a = cnn(img_a)
latent_b = cnn(img_b)

distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([latent_a, latent_b])
distance_model = Model(inputs=[img_a, img_b], outputs=distance)

distance_model.compile(loss=contrastive_loss, optimizer=adam)
distance_model.summary()

callbacks = [
    EarlyStopping(monitor='loss',
                  patience=20,
                  verbose=1,
                  mode='auto'),
    ModelCheckpoint(filepath=save_folder + '/siamese_weights_{epoch:03d}.h5',
                    monitor='val_loss',
                    save_weights_only=True,
                    save_best_only=True),
    History(),
]

if os.path.isdir(save_folder) == False:
    os.mkdir(save_folder)

history = distance_model.fit([pair_train[:,0], pair_train[:,1]], label_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([pair_valid[:,0], pair_valid[:,1]], label_valid),
                    shuffle=True, callbacks=callbacks)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.tif')
plt.close()

######################
#### Test Results ####
######################

true_pairs = pair_test[np.where(label_test == 1)[0]]
fake_pairs = pair_test[np.where(label_test == 0)[0]]
predictions_true = distance_model.predict([true_pairs[:,0], true_pairs[:,1]], batch_size=1)
predictions_fake = distance_model.predict([fake_pairs[:,0], fake_pairs[:,1]], batch_size=1)
avg_true = np.average(predictions_true)
avg_fake = np.average(predictions_fake)
p_avg_true = len(np.where(predictions_true < (avg_true+avg_fake)/2)[0]) / len(predictions_true)
p_avg_fake = len(np.where(predictions_fake > (avg_true+avg_fake)/2)[0]) / len(predictions_fake)
print('Whole Average: ' + str((avg_true+avg_fake)/2))
print('\n--- True pairs ----')
print('Average: ' + str(avg_true))
print('Maximum: ' + str(np.max(predictions_true)))
print('Percentage under whole average: ' + str(p_avg_true))
p_avg_true = len(np.where(predictions_true < 0.5)[0]) / len(predictions_true)
print('Percentage over 0.5: ' + str(p_avg_fake))
print('\n--- Fake pairs ----')
print('Average: ' + str(avg_fake))
print('Minimum: ' + str(np.min(predictions_fake)))
print('Percentage over whole average: ' + str(p_avg_fake))
p_avg_fake = len(np.where(predictions_fake > 0.5)[0]) / len(predictions_fake)
print('Percentage over 0.5: ' + str(p_avg_fake))
