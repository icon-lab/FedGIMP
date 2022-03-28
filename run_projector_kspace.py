import argparse
import dnnlib
import sys
import h5py
import os
import numpy as np
from training import misc
import warnings
warnings.filterwarnings("ignore")

def fft2c(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im))) 
def ifft2c(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

def project_image(proj, targets, png_prefix, num_snapshots, mask, contrast, labels, case, coil_map = None, pad_x = None, pad_y = None):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    if case == "multicoil":
        targets_255_real = targets[0,0,pad_x:512-pad_x,pad_y:512-pad_y]
        targets_255_imag = targets[0,1,pad_x:512-pad_x,pad_y:512-pad_y]
        targets_255 = targets_255_real + 1j * targets_255_imag
        proj.start(targets, mask, coil_map,labels)
    elif case == "singlecoil":
        targets_255 = (targets[0, 0, :, :] + 1) * 255 / 2
        proj.start(targets, mask, labels)
        
    targets_abs = np.abs(targets_255)
    noisy_targets = fft2c((targets_255))
    noisy_targets = np.multiply(mask,noisy_targets)
    noisy_targets_images = np.float32(np.abs(ifft2c(noisy_targets)))
    np.save('zerofilled.npy',noisy_targets_images)

    misc.save_image_grid(noisy_targets_images[np.newaxis], png_prefix + 'zero-fill_target.png', drange=[np.min(noisy_targets_images),np.max(noisy_targets_images)])
    misc.save_image_grid(targets_abs[np.newaxis][np.newaxis], png_prefix + 'target.png', drange=[np.min(targets_abs),np.max(targets_abs)])
    
    psnr = []
    ssim = []
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        untouched_images = proj.untouched_images()
        output_dc, reference, temp_psnr, temp_ssim = proj.apply_dc(untouched_images, targets, mask, coil_map, pad_x, pad_y)

        psnr.append(temp_psnr)
        ssim.append(temp_ssim)
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(output_dc[np.newaxis][np.newaxis], png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[np.min(output_dc), np.max(output_dc)])
            np.save(png_prefix + '-untouched_images.npy', untouched_images)
    return psnr,ssim



def project_multicoil_images(args):
    from training import dataset_float as dataset
    import projector_multicoil as projector
    
    contrast_list = []
    if args['dataset_name'] == 'fast_mri_brain':
        contrast_list = ['T1f','T2f','FLAIRf']
    if args['dataset_name'] == 'fast_mri_knee':
        contrast_list = ['PDf','PDFSf']
    elif args['dataset_name'] == 'umram_mri_brain':
        contrast_list = ['T1u', 'T2u', 'PDu']
    else:
        print("Please update the datasets")
        
    dataset_obj = dataset.load_dataset(data_dir=args['tfr_dir'], max_label_size=3, repeat=False, shuffle_mb=0)
    
    psnr = []
    ssim = []
    avg_psnr = 0
    avg_ssim = 0
    for contrast_ind in range(len(contrast_list)):
        contrast = contrast_list[contrast_ind]
        filename = args['h5_dir']+ args['dataset_name'] + "/" + contrast[:-1] + "/" + contrast[:-1] + "_under_sampled_" + args['acc_rate'] + "x_multicoil_test.mat"
        
        proj = projector.Projector()
        proj.contrast = contrast
        
        f = h5py.File(filename, 'r')
        us_masks = np.transpose(f['map'])
        coil_maps = np.transpose(f['coil_maps'])
        images_fs = np.transpose(f['images_fs'])
        maps = coil_maps['real'] + 1j * coil_maps['imag']

        G = misc.load_pkl(args['network_pkl'])
        
        G.print_layers()
        proj.set_network(G, args['dataset_name'])
        
        for image_idx in range(us_masks.shape[2]):
            im, _labels = dataset_obj.get_minibatch_np(1)
            image_fs = np.zeros([1, 3, 512, 512], dtype='float32')
            abs_image_fs = np.abs(images_fs['real'][:,:,image_idx] + 1j*images_fs['imag'][:,:,image_idx])
            pad_x = int((512 - images_fs.shape[0]) // 2)
            pad_y = int((512 - images_fs.shape[1]) // 2)
            image_fs[:,0,pad_x:512-pad_x,pad_y:512-pad_y] = images_fs['real'][:,:,image_idx]/np.max(abs_image_fs) 
            image_fs[:,1,pad_x:512-pad_x,pad_y:512-pad_y] = images_fs['imag'][:,:,image_idx]/np.max(abs_image_fs) 
            image_fs[:,2,:,:] = np.ones([512,512])
            mask = us_masks[:,:,image_idx]
            coil_map = maps[:,:,image_idx,:] 
            proj.image_idx = image_idx
            print('Projecting image %d/%d ...' % (image_idx + us_masks.shape[2]*contrast_ind, args['num_images']))
            temp_psnr, temp_ssim = project_image(proj, targets=image_fs, png_prefix=args['result_dir']+'/0_image%04d-' % (image_idx+us_masks.shape[2]*contrast_ind), num_snapshots=args['num_snapshots'],mask=mask, coil_map = coil_map, contrast=contrast, labels=_labels,case=args['case'], pad_x=pad_x,pad_y=pad_y)
            psnr.append(temp_psnr)
            ssim.append(temp_ssim)
            avg_psnr = (avg_psnr*(image_idx) +temp_psnr[len(temp_psnr)-1])/(image_idx + 1)
            avg_ssim = (avg_ssim*(image_idx) +temp_ssim[len(temp_ssim)-1])/(image_idx + 1)
    
            print("Current Average PSNR: ", avg_psnr," Average SSIM: ", avg_ssim)
            
    
    np.save(args["result_dir"]+'/psnr.npy', psnr)
    np.save(args["result_dir"]+'/ssim.npy', ssim)
    
    
def project_singlecoil_images(args):
    from training import dataset as dataset
    import projector_singlecoil as projector
    
    dataset_obj = dataset.load_dataset(data_dir=args['tfr_dir'], max_label_size=3, repeat=False, shuffle_mb=0)
    
    proj = projector.Projector()
    
    filename = args['h5_dir'] + args['us_case'] + "/" + args['dataset_name'] + "_usx" + args['acc_rate'] + "/test/data.mat"
    f = h5py.File(filename, 'r')
    us_masks = np.transpose(f['us_map'])
    images_fs = np.transpose(f['data_fs'])

    G= misc.load_pkl(args['network_pkl'])
    
    proj.set_network(G, args['dataset_name'])
      
    psnr = []
    ssim = []
    avg_psnr = 0
    avg_ssim = 0
    for image_idx in range(us_masks.shape[2]):  
        mask = us_masks[:, :, image_idx]
        image_fs = images_fs[:, :, image_idx]
        image_fs_min = np.min(image_fs)
        image_fs_max = np.max(image_fs)
        image_fs = (image_fs - image_fs_min)/(image_fs_max - image_fs_min)
        image_fs = (image_fs - 0.5)*2
        image_fs = image_fs[np.newaxis][np.newaxis]
        
        print('Projecting image %d/%d ...' % (image_idx, args['num_images']))
        _, _labels = dataset_obj.get_minibatch_np(1)
        temp_psnr, temp_ssim = project_image(proj, targets=image_fs, labels=_labels, png_prefix=args['result_dir'] + '/0_image%04d-' % image_idx, num_snapshots=args['num_snapshots'], mask=mask, coil_map = None, contrast = None, case=args['case'], pad_x=None, pad_y=None)
        
        psnr.append(temp_psnr)
        ssim.append(temp_ssim)
        avg_psnr = (avg_psnr*(image_idx) +temp_psnr[len(temp_psnr)-1])/(image_idx + 1)
        avg_ssim = (avg_ssim*(image_idx) +temp_ssim[len(temp_ssim)-1])/(image_idx + 1)

        print("Current Average PSNR: ", avg_psnr," Average SSIM: ", avg_ssim)
        
    np.save(args["result_dir"]+'/psnr.npy', psnr)
    np.save(args["result_dir"]+'/ssim.npy',ssim)

# #----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''k-space projector for both singlecoil and multicoil.

        Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_images_parser = subparsers.add_parser('project-images', help='Project multicoil images')
    project_images_parser.add_argument('--network_pkl', help='Network pickle filename', dest='network_pkl', required=True)
    project_images_parser.add_argument('--result_dir', help='Result folder path', dest='result_dir', required=True)
    project_images_parser.add_argument('--tfr-dir', help='TFRecords root directory', dest='tfr_dir', required=True)
    project_images_parser.add_argument('--h5-dir', help='H5 root directory', dest='h5_dir', required=True)
    project_images_parser.add_argument('--dataset', help='Test dataset', dest='dataset_name', required=True)
    project_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=1)
    project_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=216)
    project_images_parser.add_argument('--contrast', dest='contrast', default='T1f')
    project_images_parser.add_argument('--acc_rate', dest='acc_rate', help='3, 6 or 9')
    project_images_parser.add_argument('--case', dest='case', help='singlecoil or multicoil', default='multicoil')
    project_images_parser.add_argument('--us_case', dest='us_case', help='uniform or poisson', default='poisson')
    project_images_parser.add_argument('--network', dest='network', default='FedGIMP')
    project_images_parser.add_argument('--gpu', dest='gpu_id', default='0')
    
    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)
    kwargs = vars(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['gpu_id']
    
    dnnlib.tflib.init_tf()
    
    result_folder_name = kwargs['dataset_name']+'-'+kwargs['case']+'-set-'+kwargs['network']+'-net-'+kwargs['us_case']+'-sampling-x'+kwargs['acc_rate']
    kwargs["result_dir"] = kwargs["result_dir"]+"/"+result_folder_name
    if not os.path.exists(kwargs["result_dir"]):
        os.makedirs(kwargs["result_dir"])
        print("Result Path Created: " + kwargs["result_dir"])
    else:
        print("Path Already Exists")
        
    del kwargs['command']
    del kwargs['network']
    del kwargs['gpu_id']
        
    if args.case == 'singlecoil':
        project_singlecoil_images(kwargs)
    elif args.case == 'multicoil':
        project_multicoil_images(kwargs)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
