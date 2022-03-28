
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
import warnings
import os
import logging
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel(logging.FATAL)
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

######################################################################################################################################################################
#PROJECTOR CLASS FOR SINGLECOIL INFERENCE
class Projector:
    def __init__(self):
        self.num_steps                  = 1200          #number of inference iterations
        self.dlatent_avg_samples        = 10000         #number of averaged randomly generated vectors 
        self.initial_learning_rate      = 0.01          #fixed inference optimization learning rate
        self.initial_noise_factor       = 0.05          #noise factor's initial value 
        self.noise_ramp_length          = 0.75          #ratio of inference iterations to run noşse optimization
        self.regularize_noise_weight    = 0             #regularization control parameter for noise optimization
        self.verbose                    = False         #option to print informative expressions about network initialization
        self.clone_net                  = True          #clone generator network to initialize tensorflow graph for inference optimization
        self._D                         = None
        self._G                         = None
        self._minibatch_size            = None
        self._dlatent_avg               = None
        self._dlatent_std               = None
        self._noise_vars                = None
        self._noise_init_op             = None
        self._noise_normalize_op        = None
        self._dlatents_var              = None
        self._noise_in                  = None
        self._dlatents_expr             = None
        self._images_expr               = None
        self._target_images_var         = None
        self._lpips                     = None
        self._dist                      = None
        self._loss                      = None
        self._reg_sizes                 = None
        self._lrate_in                  = None
        self._opt                       = None
        self._opt_step                  = None
        self._cur_step                  = None
        self.mask                       = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)
            
######################################################################################################################################################################
#FOURIER TRANSFORMATION FUNCTIONS FOR NUMPY ARRAYS AND TENSORS
    def fft2c(self, im):
        return tf.compat.v1.signal.fftshift(tf.compat.v1.signal.fft2d(tf.compat.v1.signal.ifftshift(im))) 
    def ifft2c(self, d):
        return tf.compat.v1.signal.fftshift(tf.compat.v1.signal.ifft2d(tf.compat.v1.signal.ifftshift(d)))

    def fft2c_np(self,im):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 
    def ifft2c_np(self,d):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

######################################################################################################################################################################
#DEFINITION FOR LATENT VECTORS, NOISE PARAMETERS, AND TRAINED GENERATOR WEIGHTS 
    def set_network(self, G, dset, minibatch_size=1):
        assert minibatch_size == 1
        self._G = G
        self.initial_G = G.clone()
        self._minibatch_size = minibatch_size
        if self._G is None:
            return
        if self.clone_net:
            self._G = self._G.clone()
        
        # Site-specific inital latents, random latent vector is generated in an unknown test set introduced
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        if dset == "IXI":
            self._dlatent_avg = np.load('datasets/latents/IXI_latent.npy')
        elif dset == "fastMRI":
            self._dlatent_avg = np.load('datasets/latents/fastMRI_latent.npy')
        elif dset == "brats":
            self._dlatent_avg = np.load('datasets/latents/brats_latent.npy')
        else:
            print("Unknown Client Detected")
            dlatent_avg0 = np.load('datasets/latents/IXI_latent.npy')
            dlatent_avg1 = np.load('datasets/latents/fastMRI_latent.npy')
            dlatent_avg2 = np.load('datasets/latents/brats_latent.npy')
            self._dlatent_avg = (dlatent_avg0 + dlatent_avg1 + dlatent_avg2)/3
            
        
        # Noise and weight optimization declarations
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        self.weights_ops = []
        weight_init_ops = []
        self.initial_weights = []
        
        for w in self._G.vars:
            m = self._G.vars[w]
            m_copy = self.initial_G.vars[w]
            self.initial_weights.append(m_copy)
            self.weights_ops.append(m)
            weight_init_ops.append(tf.compat.v1.assign(m, m_copy))
        self._weight_init_op = tf.group(*weight_init_ops)  
        
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._G.vars:
                break
            v = self._G.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.compat.v1.assign(v, tf.random.normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.compat.v1.reduce_mean(v)
            noise_std = tf.compat.v1.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.compat.v1.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

    
        # Network output expressions
        self._info('Building image output graph...')
        # Input latent placeholder
        self._dlatents_var = tf.compat.v1.Variable(tf.ones([1,512]),name = 'dlatents_var')
        # Undersampling mask placeholder - represents site specific imaging operator dependent on acc rate and undersampling pattern
        self.mask = tf.compat.v1.Variable(tf.zeros([256,256], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        # One-hot coded vector placeholder to carry out site information
        self.labels = tf.Variable(tf.zeros([1,3]),name = 'labels', trainable=False)
        # Input noise placeholder
        self._noise_in = tf.compat.v1.placeholder(tf.float32, [], name='noise_in')
        
        # Generate fake images from Generator network based on input latent and label
        self._images_expr = self._G.get_output_for(self._dlatents_var, self.labels)
        # Network output image data range scaling option
        proc_images_expr = (self._images_expr + 1.0) *(255.0/2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.compat.v1.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
            
        # K-space loss and gradient loss definitions
        self._info('Building loss graph...')
        # Generate real&imag channel concatenated fully sampled(target) image
        self._target_images_var = tf.compat.v1.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        # Convert target image to complex variable
        self.target_images_var_complex = tf.cast(self._target_images_var, dtype=tf.complex64)
        # Target fully sampled k-space
        self.full_kspace_org_image = self.fft2c(self.target_images_var_complex[0,0,:,:])
        # Multiply by undersampling mask to generate undersampled k-space based on imaging operator
        self.undersampled_kspace_org_image = tf.compat.v1.math.multiply(self.full_kspace_org_image, self.mask)
        # Generate network output fake image in desired data range
        self.proc_images_expr_complex = tf.cast(proc_images_expr, dtype=tf.complex64)
        # Generate network output fake image's k-space
        self.full_kspace_gen_image = self.fft2c(self.proc_images_expr_complex[0,0,:,:])
        # Multiply generated image's k-space to obtain estimated undersampled points to define k-space L2 loss
        self.undersampled_kspace_gen_image = tf.compat.v1.math.multiply(self.full_kspace_gen_image,self.mask)
        diff = self.undersampled_kspace_org_image - self.undersampled_kspace_gen_image  
        self.Kloss = tf.math.sqrt(tf.compat.v1.reduce_mean( tf.math.square(tf.math.real(diff)) + tf.math.square(tf.math.imag(diff)) ))
        # Define gradient loss to prevent noise amplification
        self.TVloss = tf.compat.v1.reduce_sum(tf.compat.v1.image.total_variation(tf.transpose(self.proc_images_expr_complex, perm=[0, 2, 3, 1])))
        # Combine both losses 
        self._loss = self.Kloss  + 0.0001*self.TVloss 
        
        # Noise regularization graph, closed by default
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.compat.v1.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.compat.v1.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.compat.v1.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.compat.v1.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var]+ self.weights_ops)
        self._opt_step = self._opt.apply_updates()

    def start(self, target_images, mask, labels):
        assert self._G is not None

        # self._G.reset_vars()

        self.target_images_initial = target_images
        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images.copy(), dtype='float32')
        print("BEFORE TARGET RANGE: ",np.min(target_images)," ",np.max(target_images))
        target_images = (target_images + 1.0) *(255.0/2)
        target_images = np.tile(target_images, [1,1,1,1])
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
        
        self.target_images = target_images
        print("AFTER TARGET RANGE: ",np.min(target_images)," ",np.max(target_images))
        # Initialize optimization state.
        self._info('Initializing optimization state...')
        
        print("Labels:", labels,labels.shape, labels.dtype)
        tflib.set_vars({self._target_images_var: target_images,self._dlatents_var: self._dlatent_avg, self.mask:mask, self.labels:labels})
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / 1500
        noise_strength = self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        learning_rate = self.initial_learning_rate
        

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, loss = tflib.run([self._opt_step, self._loss], feed_dict)
        
        tflib.run(self._noise_normalize_op)

        self._cur_step += 1


    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_var)

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def untouched_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})

    def get_mask(self):
        return tflib.run(self.mask)
    
    def apply_dc(self, output, reference, us_mask, coil_map = None, pad_x = None, pad_y = None):
        output = output[0,0,:,:]
        output[output<-1]=-1
        
        output_min = np.min(output)
        output_max = np.max(output)
        output = (output - output_min)/(output_max - output_min)
        
        reference = reference[0,0,:,:]
        reference_min = np.min(reference)
        reference_max = np.max(reference)
        reference = (reference - reference_min)/(reference_max - reference_min)
        
        kspace_out = self.fft2c_np(output) 
        full_kspace_org_image = self.fft2c_np(np.complex64(reference))
        
        kspace_out[us_mask>0] = full_kspace_org_image[us_mask>0]
        output_dc = np.float32(np.abs(self.ifft2c_np(kspace_out)))
        psnr = compute_psnr(reference,output_dc)
        ssim = compute_ssim(reference,output_dc)
        
        return output_dc, reference, psnr, ssim



#----------------------------------------------------------------------------
