import os, random, h5py, numpy as np, tensorflow, matplotlib.pyplot as plt, argparse
from tensorflow.keras.layers import Conv2D, Input, Lambda, Reshape, Dense, Dropout, Activation, Flatten, LeakyReLU, Conv2DTranspose
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train the network")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size during training")
parser.add_argument('--data_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the data")
parser.add_argument('--masks_file', type=str, default='warsaw_data.hdf5', help="Name of the file with the segmentation masks")
parser.add_argument('--save_folder', type=str, default='id_net', help="Folder where the model will be saved")
parser.add_argument('--dis_file', type=str, default='disease_model.h5', help="Name of file with the disease model")
parser.add_argument('--id_weights', type=str, default='identity_weights.h5', help="Name of file with the weights of the identity model")
parser.add_argument('--infer', action='store_true', help="Loads VAE and generates anonymized data.")
parser.add_argument('--training_factuals', action='store_true', help="Trains the model to anonymize images.")
parser.add_argument('--encoder_weights', type=str, default='encoder_weights.h5', help="Name of file with the weights of the encoder")
parser.add_argument('--decoder_weights', type=str, default='decoder_weights.h5', help="Name of file with the weights of the decoder")
parser.add_argument('--cf_weights', type=str, default='decoder_weights.h5', help="Name of file with the weights of the counterfactuals decoder")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
data_file = args.data_file
masks_file = args.masks_file
save_folder = args.save_folder
disease_model = args.dis_file
identity_weights = args.id_weights
encoder_weights = args.encoder_weights
decoder_weights = args.decoder_weights
counterfactual_weights = args.cf_weights
training_factuals = args.training_factuals
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

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

####################################
####### Load Siamese Network #######
####################################

def contrastive_loss(y_true, y_pred, margin=1):
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

def load_siamese_net(file):
    adam = Adam(lr=1e-3)
    img_a = Input(x_train.shape[1:])
    img_b = Input(x_train.shape[1:])
    cnn = cnn_model()
    cnn.summary()
    latent_a = cnn(img_a)
    latent_b = cnn(img_b)

    distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([latent_a, latent_b])
    distance_model = Model(inputs=[img_a, img_b], outputs=distance)

    distance_model.load_weights(file)
    distance_model.compile(loss=contrastive_loss, optimizer=adam)
    return distance_model

##########################
# Load pretrained models #
##########################

dis_model = load_model(disease_model)
id_distance_model = load_siamese_net(identity_weights)
dis_model_results_valid = dis_model.predict(x_valid)
dis_model_results_valid = np.where(dis_model_results_valid < 0.5, 0, 1)

n_samples_a = Input(shape = (num_pp, x_train.shape[1:][0], x_train.shape[1:][1], x_train.shape[1:][2]))
n_samples_b = Input(shape = x_train.shape[1:])


na = Lambda(lambda x: tensorflow.slice(x, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1]))(n_samples_a)
na = Lambda(lambda x: K.reshape(x, [-1, 64, 64, 1]))(na)

n_distances = id_distance_model([na, n_samples_b])
for n in range(1, num_pp):
    na = Lambda(lambda x: tensorflow.slice(x, [0, n, 0, 0, 0], [-1, 1, -1, -1, -1]))(n_samples_a)
    na = Lambda(lambda x: K.reshape(x, [-1, 64, 64, 1]))(na)
    n_distances = keras.layers.concatenate([n_distances, id_distance_model([na, n_samples_b])])

id_distance_model_n = Model(inputs=[n_samples_a, n_samples_b], outputs=n_distances)
id_distance_model_n.summary()

make_trainable(dis_model, False)
make_trainable(id_distance_model, False)

opt  = Adam(lr=2e-5, decay=1e-6)    #TODO: place optimisers closer to where they'll be used
dopt = Adam(lr=2e-5, decay=1e-6)

########################
#### Loss functions ####
########################

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

def l2_norm(factual, counterfactual):
    return K.sum(K.square(factual - counterfactual))

def get_averaged_samples(real_images, fake_images):
    weights = np.random.uniform(size=(batch_size, 1, 1, 1))
    return (weights * real_images) + ((1 - weights) * fake_images)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], z_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(K.square(z_log_sigma) / 2) * epsilon



###########################
# Variational Autoencoder #
###########################

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

def model_encoder(z_dim, input_shape, units=512, dropout=0.3):
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

def model_decoder(z_dim):
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
    return Model([x], h)

###########################
# Build the discriminator #
###########################

input_shape = x_train.shape[1:] #(img_rows, img_cols, 1)
loss_weights_1= Input(shape=(1,), name='loss_1')
loss_weights_2= Input(shape=(1,), name='loss_2')
loss_weights_3= Input(shape=(1,), name='loss_3')
loss_weights_4= Input(shape=(1,), name='loss_4')
loss_weights_5= Input(shape=(1,), name='loss_5')
loss_weights_6= Input(shape=(1,), name='loss_6')
loss_weights_7= Input(shape=(1,), name='loss_7')
targets1  = Input(shape = (1,), name='targ_1')
targets2  = Input(shape = (num_pp,), name='targ_2')
targets3  = Input(shape = (1,), name='targ_3')
d_input   = Input(shape = input_shape, name='input')
mask   = Input(shape = input_shape, name='mask')
all_imgs_id = Input(shape = (num_pp, input_shape[0], input_shape[1], input_shape[2]), name='all_img')
loss_weights_w= Input(shape=(1,), name='loss_w')
loss_weights_p= Input(shape=(1,), name='loss_p')
loss_weights_0= Input(shape=(1,), name='loss_0')
init = RandomNormal(stddev=0.02)
rep_field = 8
x = Conv2D(32, (rep_field, rep_field), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer=init, name='id_conv1')(d_input)
x = LeakyReLU(0.2)(x)

x = Conv2D(64, (rep_field, rep_field), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv2')(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(128, (rep_field, rep_field), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv3')(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(256, (rep_field, rep_field), strides=(2, 2), padding='same', kernel_initializer=init, name='id_conv4')(x)
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

make_trainable(discriminator, False)
discriminator.trainable = False

#######################
# Build the GAN model #
#######################

gan_loss_weights_vae = Input(shape = (1,))
gan_targets_vae  = Input(shape = (z_dim*2,))

encoder = model_encoder(z_dim=z_dim, input_shape=input_shape, units=units, dropout=0.3)
if not training_factuals:
    encoder.load_weights(encoder_weights)
    make_trainable(encoder, False)
    encoder.trainable = False
encoder.compile(loss='binary_crossentropy', optimizer=opt)
print('################### ENCODER ###################')
encoder.summary()

factuals_decoder = model_decoder(z_dim=z_dim)
if training_factuals:
    make_trainable(factuals_decoder, True)
    factuals_decoder.trainable = True
else:
    factuals_decoder.load_weights(decoder_weights)
    make_trainable(factuals_decoder, False)
    factuals_decoder.trainable = False
factuals_decoder.compile(loss='binary_crossentropy', optimizer=opt)
print('################### DECODER ###################')
factuals_decoder.summary()

counterfactuals_decoder = model_decoder(z_dim=z_dim)

if training_factuals:
    make_trainable(counterfactuals_decoder, False)
    counterfactuals_decoder.trainable = False
else:
    if infer:
        counterfactuals_decoder.load_weights(counterfactual_weights)
    make_trainable(counterfactuals_decoder, True)
    counterfactuals_decoder.trainable = True
counterfactuals_decoder.compile(loss='binary_crossentropy', optimizer=opt)
print('################### DECODER ###################')
counterfactuals_decoder.summary()

### GAN formulation ###
[z, mean_var] = encoder([d_input, mask])
xpred = factuals_decoder([z])
cf_xpred = counterfactuals_decoder([z])
output_binary = discriminator([xpred, loss_weights_w, loss_weights_p, targets1])
output_binary_cf = discriminator([cf_xpred, loss_weights_w, loss_weights_p, targets1])
output_ori_identity = id_distance_model([d_input, xpred])
output_identity = id_distance_model_n([all_imgs_id, xpred])
output_glaucoma = dis_model(xpred)
output_glaucoma_cf = dis_model(cf_xpred)
output_glaucoma_ori = dis_model(d_input)

gan = Model([d_input, gan_loss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, loss_weights_4, loss_weights_5,  loss_weights_6, \
            loss_weights_7, loss_weights_w, loss_weights_p, gan_targets_vae, targets1, targets2, targets3, mask, all_imgs_id],\
            [mean_var, output_binary, output_identity, output_glaucoma, output_glaucoma_cf, output_binary_cf, output_glaucoma_ori, output_ori_identity])

gan_loss = gan_loss_weights_vae*KL_loss(gan_targets_vae, mean_var) + \
          loss_weights_1*wasserstein_loss(targets1, output_binary) + \
          loss_weights_2*losses.categorical_crossentropy(targets2, output_identity) + \
          loss_weights_3*losses.categorical_crossentropy(output_glaucoma_ori, output_glaucoma) + \
          loss_weights_4*iris_reconstruction_loss(d_input, xpred, mask) + \
          loss_weights_5*contrastive_loss(targets3, output_ori_identity) + \
          loss_weights_6*losses.categorical_crossentropy(1-tensorflow.math.round(output_glaucoma_ori), output_glaucoma_cf) + \
          loss_weights_7*iris_reconstruction_loss(xpred, cf_xpred, 1-mask)

gan.add_loss(gan_loss)
gan.compile(optimizer = opt, loss = None)
print('################### GAN ###################')
gan.summary()

def plot_generated_images(epoch, idx=0):
    n = 6
    plt.figure(figsize=(12, 4))

    sample = x_test[idx:idx+n]
    mask_sample = mask_test[idx:idx+n]

    [z, mean_var] = encoder.predict([sample, mask_sample])
    generated_images = factuals_decoder.predict([z])
    generated_counterfactuals = counterfactuals_decoder.predict([z])

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        ori = sample[i]
        ori = np.uint8(ori * 127.5 + 127.5)
        if grayscale:
            ori = ori.reshape(img_rows, img_cols)
            plt.imshow(ori, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(ori)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display generated
        ax = plt.subplot(3, n, i + 1 + n)
        rec = generated_images[i]
        rec = np.uint8(rec * 127.5 + 127.5)
        if grayscale:
            rec = rec.reshape(img_rows, img_cols)
            plt.imshow(rec, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(rec)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2*n)
        rec = generated_counterfactuals[i]
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

def get_image_per_identity(dset = 'train'):
    anchor = []
    for i in range(num_pp):
        y1 = label1[idx_train[0,:]]
        if dset == 'valid':
            y1 = label1[idx_valid[0,:]]
        ids = np.where(y1 == i)[0]
        if len(ids) != 0:
            anchor.append(np.random.choice(ids))
    return anchor

def train():
    global training_factuals
    batchCount = x_train.shape[0] / batch_size
    best_epoch = 0
    best_loss = []
    for ee in range(1, epochs + 1):
        print('-' * 20, 'Epoch %d' % ee, '-' * 20)
        train_loss=[]
        train_dfake_loss=[]
        train_dreal_loss=[]
        for e in tqdm(range(int(batchCount))):
            if training_factuals:
                y0_dist_real = np.ones((batch_size, 1)) * -1
                y0_dist_fake = np.ones((batch_size, 1))

                n_discriminator = 5
                make_trainable(discriminator, True)
                discriminator.trainable = True
                for _ in range(n_discriminator):
                    idx = random.sample(range(0, x_train.shape[0]), batch_size)  # train discriminator twice more than the generator
                    image_batch = x_train[idx]  # real data
                    mask_batch = mask_train[idx]
                    [z, mean_var] = encoder.predict([image_batch, mask_batch])
                    generated_images = factuals_decoder.predict([z])
                    loss_weights_0 = np.zeros(shape = (batch_size,))
                    loss_weights_1 = np.ones(shape = (batch_size,))

                    # Train discriminator with real images
                    d_loss_real = discriminator.train_on_batch([image_batch, loss_weights_1, loss_weights_0, y0_dist_real], y=None)
                    train_dreal_loss.append(np.average(d_loss_real[0]))
                    
                    # Train discriminator with fake images
                    d_loss_fake = discriminator.train_on_batch([generated_images, loss_weights_1, loss_weights_0, y0_dist_fake], y=None)
                    train_dfake_loss.append(np.average(d_loss_fake[0]))
                    
                    # Train discriminator with weighted averaged images
                    loss_weights_p = loss_weights_1*10
                    averaged_samples = get_averaged_samples(image_batch, generated_images)
                    d_loss_fake = discriminator.train_on_batch([averaged_samples, loss_weights_0, loss_weights_p, loss_weights_0], y=None)
                    train_davg_loss.append(np.average(d_loss_fake[0]))

                make_trainable(discriminator, False)
                discriminator.trainable = False
            idx = random.sample(range(0, x_train.shape[0]), batch_size)
            image_batch = x_train[idx]
            mask_batch = mask_train[idx]

            mean_var_ref = np.ones((batch_size, z_dim * 2))
            y0_batch = np.ones((batch_size, 1))*-1
            y1_batch = np.zeros((batch_size, num_pp))
            y1_batch_ori = np.zeros((batch_size, 1))

            idx2 = [get_image_per_identity()]
            idx2 = np.repeat(idx2, batch_size, axis=0)
            image_batch2 = x_train[idx2]

            loss_weights_0 = np.zeros(shape = (batch_size,))
            gan_loss_weights_vae = np.ones(shape = (batch_size,))*0.002
                
            loss_weights_1 = np.ones(shape = (batch_size,))*0.4     # realism weight in factuals
            loss_weights_2 = np.ones(shape = (batch_size,))*10      # overall privacy weight in factuals
            loss_weights_3 = np.ones(shape = (batch_size,))*2       # glaucoma weight in factuals
            loss_weights_4 = np.ones(shape = (batch_size,))*0.0005  # glaucoma reconstruction weight in factuals
            loss_weights_5 = np.ones(shape = (batch_size,))*5       # privacy weight in factuals
            #For counterfactuals
            loss_weights_6 = np.ones(shape = (batch_size,))         # glaucoma weight in counterfactuals
            loss_weights_7 = np.ones(shape = (batch_size,))*0.001   # reconstruction weight in counterfactuals

            if training_factuals:
                g_loss = gan.train_on_batch([image_batch, gan_loss_weights_vae, loss_weights_1, loss_weights_2, loss_weights_3, \
                    loss_weights_4, loss_weights_5, loss_weights_0, loss_weights_0, loss_weights_0, loss_weights_0, mean_var_ref, y0_batch, \
                    y1_batch, y1_batch_ori, mask_batch, image_batch2], y = None)
            else:
                g_loss = gan.train_on_batch([image_batch, loss_weights_0, loss_weights_0, loss_weights_0, loss_weights_0, \
                loss_weights_0, loss_weights_0, loss_weights_6, loss_weights_7, loss_weights_0, loss_weights_0, mean_var_ref, y0_batch, \
                y1_batch, y1_batch_ori, mask_batch, image_batch2], y = None)
                
            train_loss.append(g_loss[0])
        print('Generator train loss: ' + str(np.average(train_loss)))
        print('Discriminator real loss: ' + str(np.average(train_dreal_loss)))
        print('Discriminator fake loss: ' + str(np.average(train_dfake_loss)))
        gen_loss.append(np.average(train_loss))
        disc_loss.append(0.5 * (np.average(train_dreal_loss) + np.average(train_dfake_loss)))
        plot_generated_images(epoch=ee, idx=0)

        if training_factuals:
            [z, mean_var] = encoder.predict([x_valid, mask_valid])
            generated_images = factuals_decoder.predict([z])
            
            dis_results = dis_model.evaluate(generated_images, dis_model_results_valid)
            print('Disease Recognition:' + str(dis_results))
            distance = np.average(id_distance_model.predict([x_valid, generated_images]))
            print('Identity Distance: ' + str(distance))
            if distance > 2:
                distance = 0
            valid_loss = dis_results[0] + id_results
            mask_similarity = np.average(K.eval(iris_reconstruction_loss(K.constant(x_valid), K.constant(generated_images), K.constant(mask_valid))))
            print('Iris Reconstruction:' + str(mask_similarity))
            valid_loss = dis_results[0] * 2 - distance * 1 + mask_similarity * 0.001
            print('Validation Loss: ' + str(valid_loss))
            if ee > 800 and (len(best_loss) == 0 or (len(best_loss) > 0 and valid_loss <= best_loss[len(best_loss)-1])):
                best_loss.append(valid_loss)
                best_epoch = ee
                encoder.save_weights(save_folder + '/encoder_weights_'+ str(ee) +'.h5')
                factuals_decoder.save_weights(save_folder + '/decoder_weights_'+ str(ee) +'.h5')
            elif ee - best_epoch >= patience or ee >= epochs:
                counterfactuals_decoder.load_weights(save_folder + '/decoder_weights_' + str(best_epoch) + '.h5')
                make_trainable(counterfactuals_decoder, True)
                counterfactuals_decoder.trainable = True
                counterfactuals_decoder.compile(loss='binary_crossentropy', optimizer=opt)
                factuals_decoder.load_weights(save_folder + '/decoder_weights_' + str(best_epoch) + '.h5')
                make_trainable(factuals_decoder, False)
                factuals_decoder.trainable = False
                factuals_decoder.compile(loss='binary_crossentropy', optimizer=opt)
                encoder.load_weights(save_folder + '/encoder_weights_' + str(best_epoch) + '.h5')
                make_trainable(encoder, False)
                encoder.trainable = False
                encoder.compile(loss='binary_crossentropy', optimizer=opt)
                best_loss = []
                training_factuals = False
        else:
            [z, mean_var] = encoder.predict([x_valid, mask_valid])
            generated_factuals = factuals_decoder.predict([z])
            generated_counterfactuals = counterfactuals_decoder.predict([z])
            dis_results = dis_model.evaluate(generated_counterfactuals, 1-dis_model_results_valid)
            print('Disease Recognition:' + str(dis_results))
            distance = 0
            for i in range(len(generated_factuals)):
                distance += np.sum(np.square(generated_factuals[i]*(1-mask_valid[i]) - generated_counterfactuals[i]*(1-mask_valid[i])))
            distance /= len(generated_factuals)
            print('Distance: ' + str(distance))
            valid_loss = dis_results[0] + distance*0.0005
            if len(best_loss) == 0 or (len(best_loss) > 0 and valid_loss <= best_loss[len(best_loss)-1]):
                best_loss.append(valid_loss)
                counterfactuals_decoder.save_weights(save_folder + '/cf_decoder_weights_'+ str(ee) +'.h5')
            if ee - best_epoch >= epochs_cf:
                break
        
    np.savetxt('gen_loss.csv', np.asarray(gen_loss), delimiter=',')
    np.savetxt('disc_loss.csv', np.asarray(disc_loss), delimiter=',')

if not infer:
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)

    path = save_folder + "/images"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    train()

# Generate dataset using trained model
def generate_dataset():
    save_path = './generated_data.hdf5'  # path to save the hdf5 file
    [z, mean_var] = encoder.predict([x_test, mask_test])
    generated_images = factuals_decoder.predict([z])
    generated_counterfactuals = counterfactuals_decoder.predict([z])
    images = np.asarray(generated_images)
    images2 = np.asarray(generated_counterfactuals)
    hf = h5py.File(save_path, 'w')
    hf.create_dataset('id', data=y_test1)
    hf.create_dataset('dis', data=y_test2)
    hf.create_dataset('images', data=images)
    hf.create_dataset('counterfactuals', data=images2)
generate_dataset()
