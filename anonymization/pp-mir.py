import os, random, h5py, numpy as np, tensorflow, matplotlib.pyplot as plt, argparse
from tensorflow.keras.layers import Conv2D, Input, Lambda, Reshape, Dense, Dropout, Activation, Flatten, LeakyReLU, Conv2DTranspose
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs to train the network")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
parser.add_argument('--data_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the data")
parser.add_argument('--masks_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the segmentation masks")
parser.add_argument('--save_folder', type=str, default='id_net', help="Folder where the model will be saved")
parser.add_argument('--dis_file', type=str, default='disease_model.h5', help="Name of file with the disease model")
parser.add_argument('--id_file', type=str, default='identity_model.h5', help="Name of file with the identity model")
parser.add_argument('--infer', action='store_true', help="Loads VAE and generates anonymized data.")
parser.add_argument('--encoder_weights', type=str, default='encoder_weights.h5', help="Name of file with the weights of the encoder")
parser.add_argument('--decoder_weights', type=str, default='decoder_weights.h5', help="Name of file with the weights of the decoder")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
data_file = args.data_file
masks_file = args.masks_file
save_folder = args.save_folder
disease_model = args.dis_file
identity_model = args.id_file
encoder_weights = args.encoder_weights
decoder_weights = args.decoder_weights
infer = args.infer
grayscale = False
epsilon_std = 1
opt = Adam(lr=2e-5, decay=1e-6)
dopt = Adam(lr=2e-5, decay=1e-6)
z_dim = 128
units = 256
img_rows, img_cols = 64, 64

###################
#### Load data ####
###################

f = h5py.File(data_file)
f_masks = h5py.File(masks_file)

label1 = f['id']                 # labels for the identity recognition network
label1 = np.asarray(label1)
label2 = f['dis']                # labels for the task-related classification network (glaucoma)
label2 = np.asarray(label2)
label3 = f['set']                # labels for the dataset to which the sample belongs (train - 0, test - 1 or validation - 2)
label3 = np.asarray(label3)
x = f['images']                  # image data
x = np.asarray(x)
masks = f_masks['masks']         # masks data
masks = np.asarray(masks)

if len(x.shape) == 3:
    grayscale = True
    x = np.reshape(x, (-1, 64, 64, 1))
masks = np.reshape(masks, (-1, 64, 64, 1))

num_pp = len(np.unique(label1))
num_ep = len(np.unique(label2))

# split data into training, validation
idx_train = np.asarray(np.where(label3 == 0))
idx_test  = np.asarray(np.where(label3 == 1))
idx_valid  = np.asarray(np.where(label3 == 2))

x_train = x[idx_train[0,:]]
x_test = x[idx_test[0,:]]
x_valid = x[idx_valid[0,:]]
y_train1 = label1[idx_train[0,:]]
y_test1 = label1[idx_test[0,:]]
y_valid1 = label1[idx_valid[0,:]]
y_train2 = label2[idx_train[0,:]]
y_test2 = label2[idx_test[0,:]]
y_valid2 = label2[idx_valid[0,:]]
mask_train = masks[idx_train[0,:]]
mask_test = masks[idx_test[0,:]]
mask_valid = masks[idx_valid[0,:]]

# normalize data
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
x_valid = (x_valid - 127.5) / 127.5
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_valid = x_valid.astype('float16')

mask_train = mask_train / np.amax(mask_train)
mask_test = mask_test / np.amax(mask_train)
mask_valid = mask_valid / np.amax(mask_train)
mask_train = mask_train.astype('float16')
mask_test = mask_test.astype('float16')
mask_valid = mask_valid.astype('float16')

y_train1 = keras.utils.to_categorical(y_train1, num_pp)
y_valid1  = keras.utils.to_categorical(y_valid1, num_pp)
y_train2 = keras.utils.to_categorical(y_train2, num_ep)
y_valid2  = keras.utils.to_categorical(y_valid2, num_ep)

######################
#### Define Model ####
######################

# load pre-trained models
dis_model = load_model(disease_model)
id_model = load_model(identity_model)
dis_model_results_valid = dis_model.predict(x_valid)

# define loss functions
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(averaged_samples, y_pred):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients = K.square(gradients)
    gradients = K.sum(gradients, axis=np.arange(1, len(gradients.shape)).tolist())
    gradients = K.sqrt(gradients)
    gradient_penalty = K.square(1 - gradients)
    return K.mean(gradient_penalty)

def KL_loss(y_true, y_pred):
    z_mean = y_pred[:, 0:z_dim]
    z_log_var = y_pred[:, z_dim:2 * z_dim]
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)

def iris_reconstruction_loss(real_image, generated_image, mask):
    iris_real_image = real_image * mask
    iris_generated_image = generated_image * mask
    return K.sum(K.square(iris_real_image - iris_generated_image))

def get_averaged_samples(real_images, fake_images):
    weights = np.random.uniform(size=(batch_size, 1, 1, 1))
    return (weights * real_images) + ((1 - weights) * fake_images)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(K.square(z_log_sigma) / 2) * epsilon

# define variational autoencoder
def encoder_feature_extractor(h, dropout=0.3):
    k = 8
    h = Conv2D(int(units / 8), (k, k), strides=(2, 2), border_mode='same', kernel_initializer=init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(int(units / 4), (k, k), strides=(2, 2), border_mode='same', kernel_initializer=init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(int(units / 2), (k, k), strides=(2, 2), border_mode='same', kernel_initializer=init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(units, (k, k), strides=(2, 2), border_mode='same', kernel_initializer=init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dropout(dropout)(h)
    h = LeakyReLU(0.2)(h)
    return h

def model_encoder(input_shape, units=512, dropout=0.3):
    init = RandomNormal(stddev=0.02)
    k = 8
    x = Input(input_shape)
    mask = Input(input_shape)
    h1 = encoder_feature_extractor(x, dropout)
    h2 = encoder_feature_extractor(mask, dropout)
    h = keras.layers.concatenate([h1, h2])    
    h = Flatten()(h)
    mean = Dense(z_dim, name="encoder_mean")(h)
    logvar = Dense(z_dim, name="encoder_sigma", activation='sigmoid')(h)

    z = Lambda(sampling, output_shape=(z_dim,))([mean, logvar])
    h2 = keras.layers.concatenate([mean, logvar])
    return Model([x, mask], [z, h2], name='Encoder')


def model_decoder():
    init = RandomNormal(stddev=0.02)
    k = 8
    x = Input(shape=(z_dim,))
    h = Dense(4 * 4 * 128, activation='relu')(x)
    h = Reshape((4, 4, 128))(h)
    h = Conv2DTranspose(units, (k, k), strides=(2, 2), padding='same', activation='relu', kernel_initializer=init)(h)  # 32*32*64
    h = BatchNormalization(momentum=0.8)(h)
    h = Conv2DTranspose(int(units / 2), (k, k), strides=(2, 2), padding='same', activation='relu', kernel_initializer=init)(h)  # 64*64*64
    h = BatchNormalization(momentum=0.8)(h)
    h = Conv2DTranspose(int(units / 2), (k, k), strides=(2, 2), padding='same', activation='relu', kernel_initializer=init)(h)  # 8*6*64
    h = BatchNormalization(momentum=0.8)(h)

    depth = 3
    if grayscale:
        depth = 1

    h = Conv2DTranspose(depth, (k, k), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=init)(h)  # 8*6*64
    return Model([x], h, name="Decoder")

# build discriminator
input_shape = x_train.shape[1:]
loss_weights_1 = Input(shape=(1,), name='disc_1')
loss_weights_2 = Input(shape=(1,),name='disc_2')
loss_weights_3 = Input(shape=(1,),name='disc_3')
loss_weights_4 = Input(shape=(1,),name='disc_8')
targets1 = Input(shape = (1,),name='disc_4')
targets2 = Input(shape = (num_pp,),name='disc_5')
targets3 = Input(shape = (num_ep,),name='disc_6')
d_input = Input(shape = input_shape,name='disc_7')
mask = Input(shape = input_shape,name='disc_mask')
loss_weights_w = Input(shape=(1,))
loss_weights_p = Input(shape=(1,))
loss_weights_0 = Input(shape=(1,))
init = RandomNormal(stddev=0.02)
k = 8
x = Conv2D(32, (k, k), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer=init, name='id_conv1')(d_input)
x = LeakyReLU(0.2)(x)

x = Conv2D(64, (k, k), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv2')(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(128, (k, k), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv3')(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(256, (k, k), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv4')(x)
x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(256, name='ds')(x)
x = LeakyReLU(0.2)(x)
x = Dropout(0.5)(x)
output_binary = Dense(1, name='bin_real')(x)

discriminator = Model([d_input, loss_weights_w, loss_weights_p, targets1], output_binary)

disc_loss = loss_weights_w * wasserstein_loss(targets1, output_binary) + loss_weights_p * gradient_penalty_loss(d_input, output_binary)
discriminator.add_loss(disc_loss)
discriminator.compile(optimizer=dopt, loss=None)
print('################### DISCRIMINATOR ###################')
discriminator.summary()

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)
make_trainable(dis_model, False)
make_trainable(id_model, False)

# build generative adversarial network

gan_loss_weights_vae = Input(shape = (1,))
gan_targets_vae  = Input(shape = (z_dim*2,))

encoder = model_encoder(z_dim=z_dim, input_shape=input_shape, units=units, dropout=0.2)
if infer:
    encoder.load_weights(encoder_weights)
encoder.compile(loss='binary_crossentropy', optimizer=opt)
print('################### ENCODER ###################')
encoder.summary()

decoder = model_decoder(z_dim=z_dim)
if infer:
    encoder.load_weights(decoder_weights)
decoder.compile(loss='binary_crossentropy', optimizer=opt)
print('################### DECODER ###################')
decoder.summary()


# define gan
[z, mean_var] = encoder([d_input, mask])
xpred = decoder([z])
output_binary = discriminator([xpred, loss_weights_w, loss_weights_p, targets1])
output_identity = id_model(xpred)
output_glaucoma = dis_model(xpred)
output_glaucoma_ori = dis_model(d_input)

gan = Model([d_input, gan_loss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, loss_weights_4, loss_weights_w, loss_weights_p, \
             gan_targets_vae, targets1, targets2, targets3, mask],\
            [mean_var, xpred, output_binary, output_identity, output_glaucoma, output_glaucoma_ori])

gan_loss = gan_loss_weights_vae * KL_loss(gan_targets_vae, mean_var) + \
          loss_weights_1 * wasserstein_loss(targets1, output_binary) + \
          loss_weights_2 * categorical_crossentropy(targets2, output_identity) + \
          loss_weights_3 * categorical_crossentropy(output_glaucoma_ori, output_glaucoma) + \
          loss_weights_4 * iris_reconstruction_loss(d_input, xpred, mask)

gan.add_loss(gan_loss)
gan.compile(optimizer = opt, loss = None)
print('################### GAN ###################')
gan.summary()

def plot_generated_images(epoch, idx=0, examples=10, dim=(10, 10), figsize=(10, 10)):
    n = 6 
    plt.figure(figsize=(12, 4))

    sample = x_test[idx:idx+n]
    mask_sample = mask_test[idx:idx+n]

    [z, mean_var] = encoder.predict([sample, mask_sample])
    generated_images = decoder.predict([z])

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        ori = sample[i]
        ori = np.uint8(ori * 127.5 + 127.5)
        if grayscale:
            ori = ori.reshape(img_rows, img_cols)
            plt.imshow(ori, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(ori)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        rec = generated_images[i]
        rec = np.uint8(rec * 127.5 + 127.5)
        if grayscale:
            rec = rec.reshape(img_rows, img_cols)
            plt.imshow(rec, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(rec)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Path to be created
    plt.savefig(path + '/results_epoch_' + str(epoch) +'.jpg')
    plt.close()

def train():
    batch_count = x_train.shape[0] / batch_size
    best_loss = []
    for ee in range(1, epochs + 1):
        print('-' * 20, 'Epoch %d' % ee, '-' * 20)
        train_loss = []
        train_dfake_loss = []
        train_dreal_loss = []
        train_davg_loss = []
        for e in tqdm(range(int(batch_count))):

            y0_dist_real = np.ones((batch_size, 1))*-1
            y0_dist_fake = np.ones((batch_size, 1))

            n_discriminator = 5
            make_trainable(discriminator, True)
            discriminator.trainable = True
            for _ in range(n_discriminator):
                idx = random.sample(range(0, x_train.shape[0]), batch_size)  # train discriminator twice more than the generator
                image_batch = x_train[idx]  # real data
                mask_batch = mask_train[idx]
                [z, mean_var] = encoder.predict([image_batch, mask_batch])
                generated_images = decoder.predict([z])
                loss_weights_0 = np.zeros(shape = (batch_size,))
                loss_weights_1 = np.ones(shape = (batch_size,))

                # train discriminator with real images
                d_loss_real = discriminator.train_on_batch([image_batch, loss_weights_1, loss_weights_0, y0_dist_real], y=None)
                train_dreal_loss.append(np.average(d_loss_real[0]))
                
                # train discriminator with fake images
                d_loss_fake = discriminator.train_on_batch([generated_images, loss_weights_1, loss_weights_0, y0_dist_fake], y=None)
                train_dfake_loss.append(np.average(d_loss_fake[0]))
                
                # train discriminator with weighted averaged images
                loss_weights_p = loss_weights_1 * 10
                averaged_samples = get_averaged_samples(image_batch, generated_images)
                d_loss_fake = discriminator.train_on_batch([averaged_samples, loss_weights_0, loss_weights_p, loss_weights_0], y=None)
                train_davg_loss.append(np.average(d_loss_fake[0]))

            make_trainable(discriminator, False)
            discriminator.trainable = False
            for ii in range(0, 2):
                idx = random.sample(range(0, x_train.shape[0]), batch_size)
                image_batch = x_train[idx]
                mask_batch = mask_train[idx]

                mean_var_ref = np.ones((batch_size, z_dim * 2))
                y1_batch = y_train1[idx]
                y2_batch = y_train2[idx]

                x = np.random.uniform(1/(num_pp*2),2/num_pp,(batch_size, num_pp))
                total = np.sum(x, axis=1)
                needed = 1.0 - total
                uniform_dist = [x[i] + needed[i]*1/num_pp for i in range(len(x))]
                uniform_dist = np.asarray(uniform_dist)

                y0_batch = np.ones((batch_size, 1))*-1
                loss_weights_0 = np.zeros(shape = (batch_size,))
                gan_loss_weights_vae = np.ones(shape = (batch_size,))*0.002
                loss_weights_1 = np.ones(shape = (batch_size,))*0.4
                loss_weights_2 = np.ones(shape = (batch_size,))*1
                loss_weights_3 = np.ones(shape = (batch_size,))*0.001
                loss_weights_4 = np.ones(shape = (batch_size,))*0.002
                g_loss = gan.train_on_batch([image_batch, gan_loss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, loss_weights_4, \
                                             loss_weights_0, loss_weights_0, mean_var_ref, y0_batch, uniform_dist, y2_batch, mask_batch], y = None)
                train_loss.append(np.average(g_loss[0]))
        print('Generator train loss: ' + str(np.average(train_loss)))
        print('Discriminator real loss: ' + str(np.average(train_dreal_loss)))
        print('Discriminator fake loss: ' + str(np.average(train_dfake_loss)))
        print('Discriminator penalty loss: ' + str(np.average(train_davg_loss)))
        
        [z, mean_var] = encoder.predict([x_valid, mask_valid])
        generated_images = decoder.predict([z])
        
        dis_results = dis_model.evaluate(generated_images, dis_model_results_valid)
        print('Disease Recognition:' + str(dis_results))
        predictions = id_model.predict(generated_images)
        uniform_dist = np.ones((y_valid1.shape[0],num_pp)) * 1 / num_pp
        id_results = np.average(K.eval(categorical_crossentropy(K.constant(uniform_dist), K.constant(predictions))))
        print('Identity Recognition:' + str(id_results))
        mask_similarity = np.average(K.eval(iris_reconstruction_loss(K.constant(x_valid), K.constant(generated_images), K.constant(mask_valid))))
        print('Iris Reconstruction:' + str(mask_similarity))
        valid_loss = dis_results[0] + id_results + mask_similarity * 0.001
        print('Validation Loss: ' + str(valid_loss))
        
        plot_generated_images(epoch=ee, idx=0)
        if ee > 900 and (len(best_loss) == 0 or (len(best_loss) > 0 and valid_loss <= best_loss[len(best_loss)-1])):
            best_loss.append(valid_loss)
            discriminator.save_weights(save_folder + '/discriminator_weights_'+ str(ee) +'.h5')
            encoder.save_weights(save_folder + '/encoder_weights_'+ str(ee) +'.h5')
            decoder.save_weights(save_folder + '/decoder_weights_'+ str(ee) +'.h5')


if not infer:
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)

    path = save_folder + "/images"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    train()

#####################################
#### Generate Anonymized Dataset ####
#####################################

def generate_dataset():
    save_path = './generated_data.hdf5'  # path to save the hdf5 file
    [z, mean_var] = encoder.predict([x_test, mask_test])
    generated_images = decoder.predict([z])
    images = np.asarray(generated_images)
    hf = h5py.File(save_path, 'w')
    hf.create_dataset('id', data=y_test1)
    hf.create_dataset('dis', data=y_test2)
    hf.create_dataset('images', data=images)

generate_dataset()
