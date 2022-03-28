# Main training loop for FedGIMP

import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import train
from training import misc

######################################################################################################################################################################
#DYNAMIC RANGE NORMALIZATIONS BEFORE FEEDING DATA TO NETWORKS 
#ONLY SINGLE AND THREE CHANNEL DATA ARE ACCEPTED

def process_reals(x, lod, mirror_augment, drange_data, drange_net, coil_case):
    with tf.compat.v1.keras.backend.name_scope('ProcessReals'):
        with tf.compat.v1.keras.backend.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            if coil_case == "singlecoil":
                x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.compat.v1.keras.backend.name_scope('MirrorAugment'):
                s = tf.compat.v1.shape(x)
                mask = tf.random.uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.compat.v1.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.compat.v1.keras.backend.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.compat.v1.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.compat.v1.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.compat.v1.keras.backend.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.compat.v1.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

######################################################################################################################################################################
#TRAINING SCHEDULING FUNCTION TO CORRELATE CURRENT NETWORK RESOLUTION, MINIBATCH SIZE, AND LEARNING RATE WITH NETWORK PROGRESS
#NOT IMPORTANT IF NO-GROWING OPTION IS SELECTED

def training_schedule(
    cur_nimg,
    training_set,
    num_gpus,
    lod_initial_resolution,        # Image resolution used at the beginning.
    lod_training_kimg,      # Thousands of real images to show before doubling the resolution. #original 600
    lod_transition_kimg,      # Thousands of real images to show when fading in new layers.       #original 600
    tick_kimg_base,      # Default interval of progress snapshots.
    tick_kimg_dict ,          # Resolution-specific overrides.
    minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
    minibatch_dict          = {},       # Resolution-specific overrides.
    max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
    G_lrate_base            = 0.001,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.001,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0):     # Duration of learning rate ramp-up.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = training_set.resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

######################################################################################################################################################################
#MAIN TRAINING SCRIPT

def training_loop(
    submit_config,
    total_comm_rounds,
    epoch_per_round,
    client_num,
    G_args      = {}, D_args = {},      # arguments for generator and discriminator networks
    G_opt_args  = {}, D_opt_args = {},  # arguments for generator and discriminator networks optimizers
    G_loss_args = {}, D_loss_args = {}, # arguments for generator and discriminator losses
    dataset_args            = {},       # arguments for how to load datasets
    sched_args              = {},       # arguments to control learning schedule
    grid_args               = {},       # arguments for how to construct image prior grid
    # metric_arg_list         = [],       
    tf_config               = {},       # arguments TensorFlow graph initialization
    D_repeats               = 1,        # number of times the discriminator is repeated before generator loss calculation
    minibatch_repeats       = 4,        # number of minibatches to run before loss calculation
    reset_opt_for_new_lod   = True,     # optimizer reset flag once the network progresses to higher resolution layers (only important for withgrowing option)
    total_kimg              = 15000,    # thousands of training images
    mirror_augment          = False,    # allow mirror augmetation for slices?
    drange_net              = [-1,1],   # input dynamic range to feed data to network
    image_snapshot_ticks    = 1,        # communication round period to save generated image priors
    network_snapshot_ticks  = 10,       # communication round period to save network snapshots
    resume_run_id           = None,     # network pickle path to continue training from 
    resume_kimg             = 0.0,      # number of images seen by the continues network pickle 
    resume_time             = 0.0):     # Assumed wallclock time at the beginning. Affects reporting.

    # Initialize dnnlib library and TensorFlow graph 
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)
    print('Building TensorFlow graph...')
    with tf.compat.v1.keras.backend.name_scope('Inputs'), tf.compat.v1.device('/cpu:0'):
        lod_in          = tf.compat.v1.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.compat.v1.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.compat.v1.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        
    # Dataset loading options for singlecoil and multicoil data
    print(dataset_args)
    if dataset_args.coil_case == "singlecoil":
        from training import dataset as dataset
    elif dataset_args.coil_case == "multicoil":
        from training import dataset_float as dataset
    else:
        print("Invalid Coil Case Option")
        return 0
    
    # Client dictionary to store everything about each local client : networks, optimizers, loss functions, datasets, training schedule
    client_dict = {}
    for client in range(1,client_num+1):
        client_dict["client%d_data" % client] = dataset.load_dataset(data_dir=dataset_args["client%d_path" % client], verbose=True, max_label_size=dataset_args.max_label_size)
        if client == 1:
            G = tflib.Network('G', num_channels=client_dict["client%d_data" % client].shape[0], resolution=client_dict["client%d_data" % client].shape[1], label_size=client_dict["client%d_data" % client].label_size, **G_args)
        client_dict["client%d_G" % client] = G.clone()
        client_dict["client%d_D" % client] = tflib.Network('D', num_channels=client_dict["client%d_data" % client].shape[0], resolution=client_dict["client%d_data" % client].shape[1], label_size=client_dict["client%d_data" % client].label_size, **D_args)
        client_dict["client%d_G_opt" % client] = tflib.Optimizer(name='TrainG', learning_rate=lrate_in, **G_opt_args)
        client_dict["client%d_D_opt" % client] = tflib.Optimizer(name='TrainD', learning_rate=lrate_in, **D_opt_args)
        with tf.compat.v1.keras.backend.name_scope('GPU0'), tf.compat.v1.device('/gpu:0'):
            client_dict["client%d_lod_assign_ops" % client] = [tf.compat.v1.assign(client_dict["client%d_G" % client].find_var('lod'), lod_in), tf.compat.v1.assign(client_dict["client%d_D" % client].find_var('lod'), lod_in)]
            client_dict["client%d_reals" % client], client_dict["client%d_labels" % client] = client_dict["client%d_data" % client].get_minibatch_tf()
            client_dict["client%d_reals" % client] = process_reals(client_dict["client%d_reals" % client], lod_in, mirror_augment, client_dict["client%d_data" % client].dynamic_range, drange_net,dataset_args.coil_case)
            with tf.compat.v1.keras.backend.name_scope('G_loss_client%d'%client), tf.control_dependencies(client_dict["client%d_lod_assign_ops" % client]):
                client_dict["client%d_G_loss" % client] = dnnlib.util.call_func_by_name(G=client_dict["client%d_G" % client], D=client_dict["client%d_D" % client], 
                                                                                        opt=client_dict["client%d_G_opt" % client], training_set=client_dict["client%d_data" % client], 
                                                                                        minibatch_size=minibatch_split, **G_loss_args)
            with tf.compat.v1.keras.backend.name_scope('D_loss_client%d'%client), tf.control_dependencies(client_dict["client%d_lod_assign_ops" % client]):
                client_dict["client%d_D_loss" % client] = dnnlib.util.call_func_by_name(G=client_dict["client%d_G" % client], D=client_dict["client%d_D" % client],
                                                                                        opt=client_dict["client%d_D_opt" % client], training_set=client_dict["client%d_data" % client],
                                                                                        minibatch_size=minibatch_split, reals=client_dict["client%d_reals" % client], labels=client_dict["client%d_labels" % client], **D_loss_args)
        client_dict["client%d_G_opt" % client].register_gradients(tf.compat.v1.reduce_mean(client_dict["client%d_G_loss" % client]), client_dict["client%d_G" % client].trainables)
        client_dict["client%d_D_opt" % client].register_gradients(tf.compat.v1.reduce_mean(client_dict["client%d_D_loss" % client]), client_dict["client%d_D" % client].trainables)
        client_dict["client%d_G_train_op" % client] = client_dict["client%d_G_opt" % client].apply_updates()
        client_dict["client%d_D_train_op" % client] = client_dict["client%d_D_opt" % client].apply_updates()
        client_dict["client%d_grid_size" % client], client_dict["client%d_grid_reals" % client], client_dict["client%d_grid_labels" % client], client_dict["client%d_grid_latents" % client] = misc.setup_snapshot_image_grid(G, client_dict["client%d_data" % client], **grid_args)
        client_dict["client%d_sched" % client] = training_schedule(cur_nimg=total_kimg*1000, training_set=client_dict["client%d_data" % client], num_gpus=submit_config.num_gpus, **sched_args)
        client_dict["client%d_grid_fakes" % client] = G.run(client_dict["client%d_grid_latents" % client], client_dict["client%d_grid_labels" % client], is_validation=True, minibatch_size=client_dict["client%d_sched" % client].minibatch//submit_config.num_gpus)
        misc.save_image_grid(client_dict["client%d_grid_reals" % client], os.path.join(submit_config.run_dir, 'client%d_reals.png'%client), drange=client_dict["client%d_data" % client].dynamic_range, grid_size=client_dict["client%d_grid_size" % client])
        misc.save_image_grid(client_dict["client%d_grid_fakes" % client], os.path.join(submit_config.run_dir, 'client%d_fakes%06d.png' % (client,resume_kimg)), drange=drange_net, grid_size=client_dict["client%d_grid_size" % client])
    
    # Training loop through communication rounds
    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_round = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    while cur_round < total_comm_rounds:
        if ctx.should_stop(): break
    
        for client in range(1,client_num+1):
            client_dict["client%d_sched" % client] = training_schedule(cur_nimg=cur_nimg, training_set=client_dict["client%d_data" % client], num_gpus=submit_config.num_gpus, **sched_args)
            client_dict["client%d_data" % client].configure(client_dict["client%d_sched" % client].minibatch // submit_config.num_gpus, client_dict["client%d_sched" % client].lod)
            if reset_opt_for_new_lod:
                if np.floor(client_dict["client%d_sched" % client].lod) != np.floor(prev_lod) or np.ceil(client_dict["client%d_sched" % client].lod) != np.ceil(prev_lod):
                    client_dict["client%d_G_opt" % client].reset_optimizer_state(); client_dict["client%d_D_opt" % client].reset_optimizer_state()
            if client == client_num:
                prev_lod = client_dict["client%d_sched" % client_num].lod
            # Run training ops.
            for _mb_repeat in range(minibatch_repeats):
                for _D_repeat in range(D_repeats):
                    tflib.run([client_dict["client%d_D_train_op" % client]], {lod_in: client_dict["client%d_sched" % client].lod, lrate_in: client_dict["client%d_sched" % client].D_lrate, 
                                                                              minibatch_in: client_dict["client%d_sched" % client].minibatch})
                    if client == client_num:
                        cur_nimg += client_dict["client%d_sched" % client].minibatch
                tflib.run([client_dict["client%d_G_train_op" % client]], {lod_in: client_dict["client%d_sched" % client].lod, lrate_in: client_dict["client%d_sched" % client].G_lrate, 
                                                                          minibatch_in: client_dict["client%d_sched" % client].minibatch})
        
        # End of 1 epoch for all clients
        if cur_nimg >= tick_start_nimg + client_dict["client%d_sched" % client_num].tick_kimg * 1000:
            # Federated Averaging for generator networks accross all the clients
            G.aggregate_networks_parametric(client_dict, client_num)
            for client in range(1,client_num+1):
                client_dict["client%d_G"%client].copy_vars_from(G) 
            
            cur_round += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # Report progress.
            print('round %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f ' % (
                autosummary('Progress/Round', cur_round),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', client_dict["client%d_sched" % client_num].lod),
                autosummary('Progress/minibatch', client_dict["client%d_sched" % client_num].minibatch),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save Generated Image Priors Based on Each Local Client's Label and Some Random Latent Vector 
            if cur_round % image_snapshot_ticks == 0:
                for client in range(1,client_num+1):
                    grid_fakes = G.run(client_dict["client%d_grid_latents" % client], client_dict["client%d_grid_labels" % client], is_validation=True, minibatch_size=client_dict["client%d_sched" % client].minibatch//submit_config.num_gpus)
                    misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'client%d_fakes%06d.png' % (client,(cur_nimg // 1000))), drange=drange_net, grid_size=client_dict["client%d_grid_size" % client])
            
            # Save Generator and Discriminator network 
            if cur_round % network_snapshot_ticks == 0 or cur_round == 1 or cur_round == total_comm_rounds:
                misc.save_pkl((G), os.path.join(submit_config.run_dir, 'G-snapshot-%d.pkl' % cur_round))
                for client in range(1,client_num+1):
                    misc.save_pkl((client_dict["client%d_D" % client]), os.path.join(submit_config.run_dir, 'client%dD-snapshot-%d.pkl' % (client,cur_round)))
            
            ctx.update('%.2f' % client_dict["client%d_sched" % client_num].lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    ctx.close()