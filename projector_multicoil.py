
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import dnnlib
import dnnlib.tflib as tflib
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

#----------------------------------------------------------------------------


class Projector:
    def __init__(self):
        self.num_steps                  = 1200
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.01
        self.initial_noise_factor       = 0.05
        self.noise_ramp_length          = 0.75
        self.verbose                    = False
        self.clone_net                  = True
        self._D                         = None
        self._G                         = None
        self._minibatch_size            = None
        self._dlatent_avg               = None
        self._noise_vars                = None
        self._noise_init_op             = None
        self._noise_normalize_op        = None
        self._dlatents_var              = None
        self._noise_in                  = None
        self._images_expr               = None
        self._target_images_var         = None
        self._loss                      = None
        self._lrate_in                  = None
        self._opt                       = None
        self._opt_step                  = None
        self._cur_step                  = None
        self.contrast                   = None
        self.pad_x                      = None
        self.pad_y                      = None
        self.initial_weights            = None
        self.us_image                   = None
        self.Kloss                      = None
        self.TVloss                     = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def fft2c_multi_np(self,im):
        array = []
        for i in range(im.shape[2]):
            image = im[:,:,i]
            array.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))))
        return np.stack(array[:],axis=2)

    def ifft2c_multi_np(self,d):
        array = []
        for i in range(d.shape[2]):
            data = d[:,:,i]
            array.append(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data))))
        return np.stack(array[:],axis=2)


    def fft2c_multi(self,im):
        array = []
        for i in range(im.shape[2]):
            image = im[:,:,i]
            array.append(tf.compat.v1.signal.fftshift(tf.compat.v1.signal.fft2d(tf.compat.v1.signal.ifftshift(image))))
        return tf.stack(array[:],axis=2)

    def ifft2c_multi(self,d):
        array = []
        for i in range(d.shape[2]):
            data = d[:,:,i]
            array.append(tf.compat.v1.signal.fftshift(tf.compat.v1.signal.ifft2d(tf.compat.v1.signal.ifftshift(data))))
        return tf.stack(array[:],axis=2)


    def fft2c(self, im):
        return tf.compat.v1.signal.fftshift(tf.compat.v1.signal.fft2d(tf.compat.v1.signal.ifftshift(im))) 
    def ifft2c(self, d):
        return tf.compat.v1.signal.fftshift(tf.compat.v1.signal.ifft2d(tf.compat.v1.signal.ifftshift(d)))

    def fft2c_np(self,im):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 
    def ifft2c_np(self,d):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

    def set_network(self, G, dataset_name, minibatch_size=1):
        assert minibatch_size == 1
        self._G = G
        self.initial_G = G.clone()
        self._minibatch_size = minibatch_size
        if self._G is None:
            return
        if self.clone_net:
            self._G = self._G.clone()

        # Find dlatent stats.
        if dataset_name == "fast_mri_brain":
            self._dlatent_avg = np.load('datasets/latents/fastMRI_latent.npy')
        elif dataset_name == "umram_mri_brain":
            self._dlatent_avg = np.load('datasets/latents/umram_latent.npy')
        elif dataset_name == "fast_mri_knee":
            self._dlatent_avg = np.load('datasets/latents/fastMRI_knee_latent.npy')
        else:
            print("Unknown Client Detected")
            dlatent_avg0 = np.load('datasets/latents/fastMRI_latent.npy')
            dlatent_avg1 = np.load('datasets/latents/umram_latent.npy')
            dlatent_avg2 = np.load('datasets/latents/fastMRI_knee_latent.npy')
            self._dlatent_avg = (dlatent_avg0 + dlatent_avg1 + dlatent_avg2)/3

        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        self.weights_ops = []
        self.initial_weights = []
        weight_init_ops = []
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

        # Image output graph.
        self._info('Building image output graph...')
        self._dlatents_var = tf.compat.v1.Variable(tf.ones([1,512]),name = 'dlatents_var')
        if self.contrast == 'T1f' or self.contrast=='FLAIRf':
            self.mask = tf.compat.v1.Variable(tf.zeros([320,256], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        elif self.contrast == 'T2f':
            self.mask = tf.compat.v1.Variable(tf.zeros([384,288], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        elif self.contrast == 'T1u' or self.contrast == 'T2u' or self.contrast == 'PDu' :
            self.mask = tf.compat.v1.Variable(tf.zeros([248,192], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        else:
            self.mask = tf.compat.v1.Variable(tf.zeros([320,320], dtype=tf.complex64), trainable=False, dtype=tf.complex64)
        
        self.pad_x = int((512 - self.mask.shape[0]) // 2)
        self.pad_y = int((512 - self.mask.shape[1]) // 2)
        self.coil_map = tf.compat.v1.Variable(tf.zeros([(512- 2 * self.pad_x) ,(512- 2 * self.pad_y) ,5], dtype=tf.complex64), trainable=False, dtype=tf.complex64)

        self._noise_in = tf.compat.v1.placeholder(tf.float32, [], name='noise_in')
        self.labels = tf.compat.v1.Variable(tf.zeros([1,3]),name = 'labels', trainable=False)
        
        self._images_expr = self._G.get_output_for(self._dlatents_var, self.labels, randomize_noise=False)
        # Loss graph.
        self._info('Building loss graph...')
        self._target_images_var = tf.compat.v1.Variable(tf.zeros(self._images_expr.shape), name='target_images_var')
        self.us_image = tf.compat.v1.Variable(tf.zeros(self.coil_map.shape, dtype=tf.complex64),name='us_image_var', dtype=tf.complex64)

        self.target_images_var_complex = tf.squeeze(tf.complex(self._target_images_var[:,0,:,:], self._target_images_var[:,1,:,:]))
        self.target_images_var_complex = tf.stack([self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex,self.target_images_var_complex],axis=2)
        self.target_images_var_complex = self.target_images_var_complex[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y), :]
        self.full_org_image_coil_separate = tf.compat.v1.math.multiply(self.target_images_var_complex, self.coil_map)
        self.coil_seperate_mask = tf.stack([self.mask, self.mask, self.mask, self.mask, self.mask], axis=2)
        self.full_kspace_org_image_coil_separate = self.fft2c_multi(self.full_org_image_coil_separate)
        self.undersampled_kspace_org_image_coil_separate = tf.compat.v1.math.multiply(self.full_kspace_org_image_coil_separate ,self.coil_seperate_mask) 

        self.proc_images_expr_complex = tf.squeeze(tf.complex(self._images_expr[:,0,:,:],self._images_expr[:,1,:,:]))
        self.proc_images_expr_complex = self.proc_images_expr_complex[self.pad_x:(512-self.pad_x), self.pad_y:(512-self.pad_y)]
        self.proc_images_expr_complex = tf.stack([self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex,self.proc_images_expr_complex],axis=2)
        self.proc_images_expr_complex_coil_separate = tf.compat.v1.math.multiply(self.proc_images_expr_complex, self.coil_map)
        self.full_kspace_gen_image = self.fft2c_multi(self.proc_images_expr_complex_coil_separate)
        self.undersampled_kspace_gen_image_coil_separate = tf.math.multiply(self.full_kspace_gen_image,self.coil_seperate_mask)
        diff = self.undersampled_kspace_org_image_coil_separate - self.undersampled_kspace_gen_image_coil_separate
        self.Kloss = tf.math.sqrt(tf.compat.v1.reduce_mean( tf.math.square(tf.math.real(diff)) + tf.math.square(tf.math.imag(diff)) ))
        self.TVloss = tf.compat.v1.reduce_sum(tf.compat.v1.image.total_variation(self.proc_images_expr_complex_coil_separate))
        self._loss = self.Kloss  + 0.0001*self.TVloss

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.compat.v1.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self.weights_ops)
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images, self.mask, self.coil_map)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        return pres

    def start(self, target_images, mask, coil_map, labels):
        assert self._G is not None
        # self._G.reset_vars()
        self.target_images_initial = target_images

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images.copy(), dtype='float32')
        target_images = (target_images)
        target_images = np.tile(target_images, [1,1,1,1])
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
        self.target_images = target_images
        # Initialize optimization state.
        self._info('Initializing optimization state...')
        print("Labels:", labels,labels.shape, labels.dtype)
        tflib.set_vars({self._target_images_var: target_images,self._dlatents_var: self._dlatent_avg, self.mask:mask, self.coil_map :coil_map, self.labels:labels})
        tflib.run(self._noise_init_op)
        tflib.run(self._weight_init_op)
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
       a = tflib.run(self._images_expr, {self._noise_in: 0})
       my_images = np.zeros([512,512,2])
       my_images[:,:,0] =  a[0,0,:,:]
       my_images[:,:,1] =  a[0,1,:,:]
       return my_images

    def get_mask(self):
        return tflib.run(self.mask)
    
    def get_coil_maps(self):
        return tflib.run(self.coil_map)
    
    def apply_dc(self, output, reference, us_mask, coil_map, pad_x, pad_y):
    
        output_complex = output[pad_x:512-pad_x,pad_y:512-pad_y,0] + 1j*output[pad_x:512-pad_x,pad_y:512-pad_y,1]
        output_complex = output_complex/np.max(np.abs(output_complex))
        reference_complex = reference[0,0,pad_x:512-pad_x,pad_y:512-pad_y] + 1j*reference[0,1,pad_x:512-pad_x,pad_y:512-pad_y]
        reference_complex = reference_complex/np.max(np.abs(reference_complex))
        
        output_dc = np.zeros(output_complex.shape,dtype='complex64')
        for coil_ind in range(coil_map.shape[2]):
            out_mapped = output_complex*coil_map[:,:,coil_ind]
            ref_mapped = reference_complex*coil_map[:,:,coil_ind]
            out_fft = self.fft2c_np(out_mapped)
            ref_fft = self.fft2c_np(ref_mapped)
            out_fft[us_mask>0] = ref_fft[us_mask>0]
            output_dc += self.ifft2c_np(out_fft)*np.conjugate(coil_map[:,:,coil_ind])
        
        output_dc = np.abs(output_dc)
        reference_abs = np.abs(reference_complex)
        reference_abs[reference_abs>1] = 1
        psnr = compute_psnr(reference_abs,output_dc)
        ssim = compute_ssim(reference_abs,output_dc)
        
        return output_dc, reference, psnr, ssim

#----------------------------------------------------------------------------
