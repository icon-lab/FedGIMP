# FedGIMP
Official TensorFlow implementation of Federated Learning of Generative Image Priors for MRI Reconstruction (FedGIMP)
# Demo
You can use the following links to download pretrained network examples of singlecoil and multicoil cases as well as corresponding training and test datasets. 

Example Training & Test Sets: https://drive.google.com/drive/folders/1_n8JynaPRQcPmu4TwYF6x6zOFPqLYlx8?usp=sharing

Pretrained Network: https://drive.google.com/drive/folders/15FiUVr7w3NmW92PFc-tQPJA_sfUta2aF?usp=sharing

If you are using Anaconda, you can export the required virtual environment from "environment.yaml" file with `$ conda env create -f environment.yaml` command.

Edit lines 29 and 55 in `train.py` to switch between singlecoil and multicoil trainings, multicoil is set by default. Configure & Run `train.py` directly.

Use below for singlecoil inference:
```python run_projector_kspace.py project-images --network_pkl pretrained_networks/FedGIMP-840-1gpu-withoutgrowing-singlecoil-cond/G-snapshot-100.pkl --tfr-dir datasets/TFRecords/singlecoil/IXI/test/ --h5-dir datasets/h5files/singlecoil/ --result_dir results/inference --case singlecoil --dataset IXI --us_case poisson --acc_rate 3 --num-images 210 --gpu 0```

Use below for multicoil inference:
```python run_projector_kspace.py project-images --network_pkl pretrained_networks/FedGIMP-864-1gpu-withoutgrowing-multicoil-cond/G-snapshot-100.pkl --tfr-dir datasets/TFRecords/multicoil/fastMRI_brain/test/ --h5-dir datasets/h5files/multicoil/ --result_dir results/inference --case multicoil --dataset fastMRI_brain --us_case poisson --acc_rate 3 --num-images 216 --gpu 0```


# Dataset
- IXI dataset: https://brain-development.org/ixi-dataset/
- fastMRI dataset: https://fastmri.med.nyu.edu/
- brats dataset: https://www.med.upenn.edu/sbia/brats2018/registration.html

FedGIMP uses TFRecords files in run time, so images (optionally site labels) should be converted to this format with `data_prep.py`. Either single channel magnitude images or three channel complex images [real, imag, dummy] could be fed to the network. In the example training and test sets, singlecoil images are fed as magnitude images whereas multicoil images are fed as complex images to the network. 

Only image prior is learnt throughout the training phase so that either magnitude or coil combined images are used during training. Imaging operator is required in test phase with complementary coil sensitivity maps (multicoil only) and undersampling masks in hdf5 data format (.mat extension is used). 

- Use the following hdf5keys with channels for multicoil complex datasets: 
  - "map" -> undersampling mask -> [x, y, #slices]
  - "coil_maps" -> coil sensitivity maps -> [x, y, #slices, #coils]
  - "images_fs" -> coil combined fully sampled images -> [x, y, #slices] 
- Use the following hdf5keys for singlecoil magnitude datasets: 
  - "us_map" -> undersampling mask -> [x, y, #slices]
  - "data_fs" -> fully sampled images -> [x, y, #slices]

Coil-sensitivity-maps are estimated using ESPIRIT (http://people.eecs.berkeley.edu/~mlustig/Software.html). Network implementations use libraries from Stylegan (https://github.com/NVlabs/stylegan) and Stylegan-2 (https://github.com/NVlabs/stylegan2) repositories.

# Prerequisites

Note: Anaconda environment import from .yaml file is recommended

- Python 3.6
- CuDNN 8.2.1
- Tensorflow 2.5.0

# Acknowledgements

This code uses libraries from StyleGAN (https://github.com/NVlabs/stylegan) and StyleGAN-2 (https://github.com/NVlabs/stylegan2) repositories.

For questions/comments please send me an email: gokberk@ee.bilkent.edu.tr




