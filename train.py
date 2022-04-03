# Training initialization file for FedGIMP for both singlecoil and multicoil cases
# Stylegan1 code base and style-based generator library was used as prior work
# The code assumes local sites have the same image dimensions and same number of MRI slices
import copy
import dnnlib
from dnnlib import EasyDict
import os

######################################################################################################################################################################
#DESCRIPTION LABEL TO RESULT FOLDER AND DICTIONARIES TO STORE TRAINING AND SCHEDULING OPTIONS
desc          = 'FedGIMP'                                                             
train         = EasyDict(run_func_name='training.training_loop.training_loop')         
sched         = EasyDict()                                                             
submit_config = dnnlib.SubmitConfig()                                                  

######################################################################################################################################################################
#DETERMINE HOW MANY CLIENTS WILL BE AGGREGATED AND FOR HOW MANY COMMUNICATION ROUNDS THE TRAINING WILL CONTINUE
train.client_num = 3
train.total_comm_rounds = 100

######################################################################################################################################################################
#ENTER THE LATEST NETWORK PICKLE AND HOW MANY IMAGES WERE SEEN UP TO THAT PICKLE TO CONTINUE TRAINING
#DO NOT EDIT BELOW FOR FROM SCRATCH RUNS
train.resume_run_id = None
train.resume_kimg = 0.0 # thousands of images

######################################################################################################################################################################
#NUMBER OF SLICES PER LOCAL SITE
slice_per_client = 0.864 # thousands of images: 0.84 for singlecoil, 0.864 for multicoil in the example training set

######################################################################################################################################################################
#NUMBER OF GPUS TO USE AND DEFAULT SCHEDULING PARAMETERS FOR STYLE-BASED GENERATOR
desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
os.environ["CUDA_VISIBLE_DEVICES"]="0"

######################################################################################################################################################################
#CHOOSE WHETHER TO ALLOW PROGRESSIVE TRAINING SCHEDULING OR NON-PROGRESSIVE FIXED SCHEDULE TRAINING
# IMPORTANT NOTE: 
# PROGRESSIVE SCHEDULING IS DIRECTLY INHERITED FROM STYLE-BASED GENERATOR IMPLEMENTATION AND HEAVLIY INCREASES COMPUTATIONAL LOAD PER SITE
# THIS IS OFFERED AS AN OPTION TO SHOW THAT INCREASED PER SITE TRAINIG GENERATES BETTER FAKE MR IMAGES
if 0:
    desc += '-withgrowing'; sched.lod_initial_resolution = 8; sched.lod_training_kimg = 600;sched.lod_transition_kimg = 600;train.total_kimg = 25000
    sched.tick_kimg_base          = 160     
    sched.tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20}
else:
    desc += '-withoutgrowing'; sched.lod_training_kimg = 10000;sched.lod_transition_kimg = 0;train.total_kimg = 10000
    train.epoch_per_round = 1
    sched.tick_kimg_base = train.epoch_per_round * slice_per_client
    sched.tick_kimg_dict = {4: sched.tick_kimg_base, 8:sched.tick_kimg_base, 16:sched.tick_kimg_base, 32:sched.tick_kimg_base, 64:sched.tick_kimg_base, 128:sched.tick_kimg_base, 256:sched.tick_kimg_base, 
                            512:sched.tick_kimg_base, 1024:sched.tick_kimg_base}
                                      
######################################################################################################################################################################
#DATA CASE AND DATASET PATHS
dataset       = EasyDict()                                                             
if 0:
    desc += '-singlecoil'; dataset.coil_case = "singlecoil"
    dataset.client1_path = "datasets/TFRecords/singlecoil/IXI/train/"    
    dataset.client2_path = "datasets/TFRecords/singlecoil/fastMRI/train/"      
    dataset.client3_path = "datasets/TFRecords/singlecoil/brats/train/"    
    sched.lod_initial_resolution = 256; 
else:
    desc += '-multicoil'; dataset.coil_case = "multicoil"
    dataset.client1_path = "datasets/TFRecords/multicoil/fastMRI_brain/train/"    
    dataset.client2_path = "datasets/TFRecords/multicoil/fastMRI_knee/train/"      
    dataset.client3_path = "datasets/TFRecords/multicoil/umram_brain/train/" 
    sched.lod_initial_resolution = 512;  


######################################################################################################################################################################
#DETERMINE WHETHER THE SITE INFORMATION WILL BE INHERITED BY THE NETWORK
if 1:
    desc += '-cond'; dataset.max_label_size = 'full' # conditioned on full label
else:
    desc += '-uncond'; dataset.max_label_size = 0 # conditioned on full label
    
######################################################################################################################################################################
#GENERATE RESULT DIRECTORY TO SAVE NETWORK SNAPSHOTS AND GENERATED PRIORS
result_dir = 'results/training/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print("Result Path Created: " + result_dir)
else:
    print("Path Already Exists")
 
######################################################################################################################################################################
#DEFAULT OPTIONS FOR STYLE-BASED GENERATOR
G             = EasyDict(func_name='training.networks_stylegan.G_style')               
D             = EasyDict(func_name='training.networks_stylegan.D_basic')               
G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          
D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          
sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           
D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) 
grid          = EasyDict(size='1080p', layout='random')                                                       
tf_config     = {'rnd.np_random_seed': 1000}  
train.mirror_augment = False

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(result_dir)
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
