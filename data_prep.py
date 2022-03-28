#COMPLEX AND MAGNITUDE TFRECORDS DATA PREPARATION TOOL, CONFIGURE ACCORDINGLY
import warnings
warnings.filterwarnings("ignore")

if 1:
    from dataset_tool import create_from_hdf5_complex

    h5_path1 = "H5PATH FOR THE FIRST CONTRAST"
    h5_path2 = "H5PATH FOR THE SECOND CONTRAST"
    h5_path3 = "H5PATH FOR THE THIRD CONTRAST"

    tf_path = "FINAL TFRECORDS SAVE PATH"
    #3 different contrasts are assumed to be used, h5_key is the key for fully sampled images in h5file
    #label index is to determine site information of the dataset to be converted to TFRecords, choose None for no site information
    #choose whether to shuffle
    create_from_hdf5_complex(tf_path, h5_path1, h5_path2, h5_path3, h5_key = "images_fs", label_index = 0, shuffle = 1)
else:
    from dataset_tool import create_from_hdf5_magnitude

    h5_path = "H5PATH FOR ALL MAGNITUDE IMAGES OF ONE DATASET"

    tf_path = "FINAL TFRECORDS SAVE PATH"
    #3 different clients are assumed to be used, h5_key is the key for fully sampled images in h5file
    #label index is to determine site information of the dataset to be converted to TFRecords, choose None for no site information
    #choose whether to shuffle
    create_from_hdf5_magnitude(tf_path, h5_path, h5_key = "data_fs", label_index = 0, shuffle = 1)

