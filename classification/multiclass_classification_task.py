import tensorflow, h5py, numpy as np, matplotlib.pyplot as plt, argparse
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help="Number of epochs to train the network")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
parser.add_argument('--data_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the data")
parser.add_argument('--save_folder', type=str, default='id_net', help="Folder where model will be saved")
parser.add_argument('--task', type=str, default='identity', help="Classification task to perform: disease or identity")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
filename = args.data_file
save_folder = args.save_folder
task = args.task
if task != 'identity' and task != 'disease':
    print('The selected task does not exist. Try: identity or disease')
    exit(-1)

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

if task == 'identity':
    num_classes = len(np.unique(label1))
    y_train = to_categorical(y_train1, num_classes)
    y_test = to_categorical(y_test1, num_classes)
    y_valid = to_categorical(y_valid1, num_classes)
else:
    num_classes = len(np.unique(label2))
    y_train = to_categorical(y_train2, num_classes)
    y_test = to_categorical(y_test2, num_classes)
    y_valid = to_categorical(y_valid2, num_classes)

######################
#### Define Model ####
######################

def conv_block(d_input, k, name):
    h1 = Conv2D(32, (k, k), activation = 'relu', padding = 'same', name = name)(d_input)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)
    return h1

d0 = Input((x_train.shape[1:]))
h = conv_block(d0, 5, 'id_conv1')
h = conv_block(h, 5, 'id_conv2')
h = conv_block(h, 3, 'id_conv3')
h = conv_block(h, 3, 'id_conv4')
h = Flatten()(h)
h = Dense(32, activation = 'relu', name = 'id_dense1')(h)
h = Dropout(0.25)(h)
output = Dense(num_classes, activation = 'softmax', name = 'id_dense2')(h)

model = Model(d0, output)

opt = Adam(lr = 2e-4, decay = 1e-6)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto'),
    ModelCheckpoint(filepath=save_folder + 'id_net_{epoch:03d}.h5', monitor='val_acc', save_best_only=True),
    History(),
]

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), shuffle=True, callbacks = callbacks)

results = model.evaluate(x_test, y_test)
print('Results: ' + str(results))

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
