import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDNN_LOGINFO_DBG'] = '1'
# os.environ['CUDNN_LOGDEST_DBG'] = 'stdout'


class DCGAN:
    input_shape = None
    seed_dimension = None
    output_shape = None
    optimizer = None

    # work around for CUDNN
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 6})
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    #

    def __init__(self, input_shape, seed_dimension, output_shape, model_save_path):

        self.input_shape = input_shape
        self.seed_dimension = seed_dimension
        self.output_shape = output_shape
        self.model_save_path = model_save_path

        self.starting_epoch = None

        if len(input_shape) == 3:
            self.channels = input_shape[2]
        else:
            self.channels = 1

        self.optimizer = tf.compat.v1.train.AdamOptimizer(0.00005, 0.5)

        print("Building the discriminator")
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        print("Building the generator")
        self.generator = self.build_generator()

        print("Building combined model")
        # self.discriminator.trainable = False
        for l in self.discriminator.layers:
            l.trainable = False

        random_input = keras.layers.Input(shape=(self.seed_dimension,))
        generated_image = self.generator(random_input)

        validity = self.discriminator(generated_image)

        # combined model is a custom model where the input is the seed, processed by the generator and the output
        # is the validity computed by the discriminator.

        self.combined = keras.models.Model(random_input, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        self.combined.summary()
        self.load_model(self.model_save_path)

    def build_generator(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(units=4096, activation='relu', input_dim=self.seed_dimension))
        model.add(keras.layers.Reshape((4, 4, 256)))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("tanh"))

        model.summary()

        inputs = keras.layers.Input(shape=(self.seed_dimension,))
        generated_image = model(inputs)

        return keras.models.Model(inputs, generated_image)

    def build_discriminator(self):

        model = keras.models.Sequential()

        model.add(
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=self.input_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        model.summary()

        inputs = keras.layers.Input(shape=self.input_shape)

        validity = model(inputs)

        return keras.models.Model(inputs, validity)

    def train(self, ds, batch_size, epochs):

        normalization = np.vectorize(lambda s: s / 127.5 - 1)

        dataset = len(ds) * [None]

        for n, data in enumerate(ds):
            dataset[n] = normalization(data.reshape(self.input_shape))

        real_img_labels = np.ones((batch_size, 1))
        fake_img_labels = np.zeros((batch_size, 1))

        t_epoch = t = time.time()
        for epoch in range(1, epochs + 1):
            # pick a random half of images from dataset (the other half will be filled with fake images)
            indices = np.random.randint(0, len(dataset), batch_size)
            real_images = [dataset[i] for i in indices]

            # Generate random noise for a whole batch.
            seed = np.random.normal(0, 1, (batch_size, self.seed_dimension))
            # Generate a batch of new images.
            fake_images = self.generator.predict(seed)

            # train discriminator on real images
            discr_real_loss = self.discriminator.train_on_batch([real_images], real_img_labels)

            # train discriminator on fake images
            discr_fake_loss = self.discriminator.train_on_batch(fake_images, fake_img_labels)

            discr_loss = (np.add(discr_real_loss, discr_fake_loss)) * 0.5

            # train the combined model ( basically only the generator because discriminator is not trainable in this
            # model)

            # We put the seeds as inputs and the labels are going to be all ones
            # This because our goal is to trick the discriminator to evaluate all fake images as real images
            # So every batch, firstly we train the discriminator on the images produced by the generator model of the
            # previous batch and then we improve the generator with the new discriminator
            # The process will end when the loss of the discriminator hits 50%, cause this means that it's just guessing
            # and is no more capable to discriminate properly a fake image from a real one

            generator_loss = self.combined.train_on_batch(seed, real_img_labels)

            if not epoch % 100:
                print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" %
                      (epoch, discr_loss[0], 100 * discr_loss[1], generator_loss))

            if not epoch % 1000:
                self.combined.save_weights(self.model_save_path + "/combined")
                elapsed = time.time() - t_epoch
                t_epoch = time.time()
                log_file = open(self.model_save_path + '/log.txt', "a+")
                date = time.strftime("%d/%m/%y %H:%M:%S")
                log_file.write(
                    "\n" + str(epoch + self.starting_epoch) + '\t' + date + '\t' + '{:.4f}'.format(elapsed) + "s")
                log_file.close()
                collage = self.generate_collage(5, 5, show=False)
                collage.save("./predictions/" + str(epoch + self.starting_epoch) + ".png", 'PNG')

        elapsed = time.time() - t
        print("Elapsed " + str(elapsed))

    def generate_img(self, seed):
        img = self.generator.predict(seed)

        denormalize = np.vectorize(lambda s: int((s + 1) * 127.5))

        image = len(img[0]) * [None]

        for n, pix in enumerate(img[0]):
            image[n] = denormalize(pix)

        image = (np.array(image, dtype=np.uint8)).reshape(self.output_shape)

        image = Image.fromarray(image, "RGB")
        image = image.resize((128, 128))
        return image

    def generate_collage(self, X, Y, show=True, dim=128):
        collage = Image.new('RGB', (128 * X, 128 * Y))
        for i in range(X):
            for j in range(Y):
                seed = np.random.normal(0, 1, (1, 100))
                image = self.generate_img(seed)
                collage.paste(image, (128 * i, 128 * j))
        if show:
            collage.show()

        return collage

    def load_model(self, path):
        try:
            print("Loading weights from " + path)
            self.combined.load_weights(path + "/combined")
        except:
            print("Model weights not found, ignoring...")

        log_file_w = open(path + "/log.txt", 'a+')
        log_file_r = open(path + "/log.txt", 'r+')

        logs = log_file_r.readlines()
        if len(logs) != 0:
            last_log = logs[-1]
            fields = last_log.split('\t')
            try:
                last_epoch = int(fields[0])
                self.starting_epoch = last_epoch
                print("Starting epoch: " + str(self.starting_epoch))
            except:
                print("Error reading log file")
                self.starting_epoch = 0
        else:
            log_file_w.write("Epoch Number\tDate\tElapsed Time\n")
            self.starting_epoch = 0

        log_file_w.close()
        log_file_r.close()


class DCGAN_small(DCGAN):

    def build_generator(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(units=16384, activation='relu', input_dim=self.seed_dimension))
        model.add(keras.layers.Reshape((16, 16, 64)))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("tanh"))

        model.summary()

        inputs = keras.layers.Input(shape=(self.seed_dimension,))
        generated_image = model(inputs)

        return keras.models.Model(inputs, generated_image)

    def build_discriminator(self):
        model = keras.models.Sequential()

        model.add(
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=self.input_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        model.summary()

        inputs = keras.layers.Input(shape=self.input_shape)

        validity = model(inputs)

        return keras.models.Model(inputs, validity)


class DCGAN_large_img(DCGAN):

    def build_generator(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(units=16384, activation='relu', input_dim=self.seed_dimension))
        model.add(keras.layers.Reshape((16, 16, 64)))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(keras.layers.Activation("tanh"))

        model.summary()

        inputs = keras.layers.Input(shape=(self.seed_dimension,))
        generated_image = model(inputs)

        return keras.models.Model(inputs, generated_image)

    def build_discriminator(self):
        model = keras.models.Sequential()

        model.add(
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=self.input_shape))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.8))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        model.summary()

        inputs = keras.layers.Input(shape=self.input_shape)

        validity = model(inputs)

        return keras.models.Model(inputs, validity)

    def train(self, ds, batch_size, epochs):

        normalization = np.vectorize(lambda s: s / 127.5 - 1)

        dataset = len(ds) * [None]

        for n, data in enumerate(ds):
            dataset[n] = normalization(data.reshape(self.input_shape))

        real_img_labels = np.ones((batch_size, 1))
        fake_img_labels = np.zeros((batch_size, 1))

        t_epoch = t = time.time()
        for epoch in range(1, epochs + 1):
            # pick a random half of images from dataset (the other half will be filled with fake images)
            indices = np.random.randint(0, len(dataset), batch_size)
            real_images = [dataset[i] for i in indices]

            # Generate random noise for a whole batch.
            seed = np.random.normal(0, 1, (batch_size, self.seed_dimension))
            # Generate a batch of new images.
            fake_images = self.generator.predict(seed)

            # train discriminator on real images
            discr_real_loss = self.discriminator.train_on_batch([real_images], real_img_labels)

            # train discriminator on fake images
            discr_fake_loss = self.discriminator.train_on_batch(fake_images, fake_img_labels)

            discr_loss = (np.add(discr_real_loss, discr_fake_loss)) * 0.5

            # train the combined model ( basically only the generator because discriminator is not trainable in this
            # model)

            # We put the seeds as inputs and the labels are going to be all ones
            # This because our goal is to trick the discriminator to evaluate all fake images as real images
            # So every batch, firstly we train the discriminator on the images produced by the generator model of the
            # previous batch and then we improve the generator with the new discriminator
            # The process will end when the loss of the discriminator hits 50%, cause this means that it's just guessing
            # and is no more capable to discriminate properly a fake image from a real one

            generator_loss = self.combined.train_on_batch(seed, real_img_labels)

            if not epoch % 1:
                print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" %
                      (epoch, discr_loss[0], 100 * discr_loss[1], generator_loss))

            if not epoch % 100:
                self.combined.save_weights(self.model_save_path + "/combined")
                elapsed = time.time() - t_epoch
                t_epoch = time.time()
                log_file = open(self.model_save_path + '/log.txt', "a+")
                date = time.strftime("%d/%m/%y %H:%M:%S")
                log_file.write(
                    "\n" + str(epoch + self.starting_epoch) + '\t' + date + '\t' + '{:.4f}'.format(elapsed) + "s")
                log_file.close()
                collage = self.generate_collage(5, 5, show=False, dim=256)
                collage.save("./predictions/" + str(epoch + self.starting_epoch) + ".png", 'PNG')

        elapsed = time.time() - t
        print("Elapsed " + str(elapsed))
