import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow_addons as tfa
import SimpleITK as sitk

from warnings import filterwarnings
filterwarnings('ignore')

class DiamondGAN():
    def __init__(self, image_shape=(200, 200, 2), input_dir='', output_dir=''):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = tfa.layers.InstanceNormalization

        # Generator
        self.G_A2B = self.modelGenerator(name='G_A2B_model')

        # ======= Initialize training ==========
        self.load_model_and_generate_synthetic_images(input_dir, output_dir)

#===============================================================================
# Architecture functions
    def c7Ak(self, x, k):
        x = KL.Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = KL.Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = KL.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        # second layer
        x = KL.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = KL.add([x, x0])
        return x

    def uk(self, x, k):
        x = KL.Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

#===============================================================================
# Models
    def modelGenerator(self, name=None):
        # Specify input
        input_img = KL.Input(shape=self.img_shape)
        mask = KL.Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 48)
        # Layer 2
        x = self.dk(x, 72)
        # Layer 3
        x = self.dk(x, 128)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        # Layer 13
        x = self.uk(x, 72)
        # Layer 14
        x = self.uk(x, 48)
        x = ReflectionPadding2D((3, 3))(x)
        x = KL.Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = KL.Activation('tanh')(x)  # They say they use Relu but really they do not
        x_atten = KL.Multiply()([x, mask])
        return KM.Model(inputs=[input_img, mask], outputs=[x, x_atten], name=name)


    def load_model_and_weights(self, model):
        try:
            path_to_weights = 'G_A2B_model.hdf5'
            if not os.path.isfile(path_to_weights):
                import urllib.request
                URL = 'https://syncandshare.lrz.de/getlink/fiJrCQiDY4rP4M2cv2mSXmZf/G_A2B_model.hdf5'
                urllib.request.urlretrieve(URL, path_to_weights)
            model.load_weights(path_to_weights)
        except:
            print('Automatically download pre-trained model failed')
            print('Please download model under the link https://drive.google.com/file/d/1BkBc-_yTabEOf1_HJxNjccV9kdg5Dgu5/view manually')
            exit()
    
    def load_model_and_generate_synthetic_images(self, input_dir, output_dir):

        def crop_or_pad(input_array, std_size):
            dim_2 = np.shape(input_array)[1]
            dim_3 = np.shape(input_array)[2]
            rows = std_size[0]
            cols = std_size[1]
            array_1 = np.zeros([np.shape(input_array)[0], rows, cols], dtype = 'float32')
            
            if dim_2 <=rows and dim_3<=cols: 
                array_1[:, int((rows - dim_2)/2):(int((rows - dim_2)/2)+ dim_2), int((cols - dim_3)/2):(int((cols - dim_3)/2)+dim_3)] = input_array[:, :, :]
            elif dim_2>=rows and dim_3>=cols: 
                array_1[:, :, :] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), int((dim_3-cols)/2):(int((dim_3-cols)/2)+cols)]
            elif dim_2>=rows and dim_3<=cols: 
                array_1[:, :, int((cols-dim_3)/2):(int((cols-dim_3)/2)+dim_3)] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), :]
            elif dim_2<=rows and dim_3>=cols: 
                array_1[:, int((rows-dim_2)/2):(int((rows-dim_2)/2)+ dim_2), :] = input_array[:, :, int((dim_3 -cols)/2):(int((dim_3 -cols)/2)+cols)]
            return array_1
        
        # load pre-trained generator
        self.load_model_and_weights(self.G_A2B)

        dirs = os.listdir(input_dir)
        dirs.sort()
        for dir_name in dirs:
            print(f'Parsing data from dir: {dir_name}')
            if not os.path.exists(output_dir+'/'+dir_name):
                os.mkdir(output_dir+'/'+dir_name)
            brain_img = sitk.ReadImage(os.path.join(input_dir, dir_name, 't1_bet_mask.nii.gz'))
            brain_array = sitk.GetArrayFromImage(brain_img)
            brain_array = np.nan_to_num(brain_array, copy=True)

            # test_S1_test
            t1_img = sitk.ReadImage(os.path.join(input_dir, dir_name, 't1.nii.gz'))
            t1_array = sitk.GetArrayFromImage(t1_img)
            t1_array = np.nan_to_num(t1_array, copy=True)
            t1_array = t1_array*brain_array
            t1_array_m = crop_or_pad(t1_array, [200, 200])
            # test_S2_test
            t2_img = sitk.ReadImage(os.path.join(input_dir, dir_name, 't2.nii.gz'))
            t2_array = sitk.GetArrayFromImage(t2_img)
            t2_array = np.nan_to_num(t2_array, copy=True)
            t2_array = t2_array*brain_array
            t2_array_m = crop_or_pad(t2_array, [200, 200])

            S1_test = t1_array_m[..., np.newaxis]
            S2_test = t2_array_m[..., np.newaxis]
            S_test = np.concatenate((S1_test, S2_test), axis = -1)

            syn_images, atten_syn_images = self.G_A2B.predict([S_test, np.zeros(np.shape(S_test))])
            syn_images = (syn_images+1)*127.5

            flair_syn_orig = crop_or_pad(syn_images[:,:,:,0], [np.shape(t1_array)[1], np.shape(t1_array)[2]])
            dir_syn_orig = crop_or_pad(syn_images[:,:,:,1], [np.shape(t1_array)[1], np.shape(t1_array)[2]])

            flair_image = sitk.GetImageFromArray(flair_syn_orig, isVector=False)
            flair_image.CopyInformation(t1_img)
            dir_image = sitk.GetImageFromArray(dir_syn_orig, isVector=False)
            dir_image.CopyInformation(t1_img)

            sitk.WriteImage(flair_image, os.path.join(output_dir+'/'+dir_name, 'syn_flair.nii.gz'))
            sitk.WriteImage(dir_image, os.path.join(output_dir + '/' + dir_name, 'syn_dir.nii.gz'))
        print('synthetic images have been generated and placed in {}'.format(output_dir))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(KL.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [KL.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
             'padding': self.padding
        })
        return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="input path to the test set", default="../test_samples")
    parser.add_argument("-o", "--output_dir", help="output path of the predicted results", default="../test_samples")
    args = parser.parse_args()
    GAN = DiamondGAN(input_dir=args.input_dir, output_dir=args.input_dir)
