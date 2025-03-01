from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/media/jiawen/Dataset_JiaWenZ1/DATASET_TEST/got-10k'
    # settings.got10k_path = '/media/titan3/Dataset_JiaWenZ2/SOT/GOT10K'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/media/titan3/Dataset_JiaWenZ2/SOT/LaSOTExtsion_zip'
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/media/jiawen/Dataset_JiaWenZ1/DATASET/LaSOTBenchmark'
    # settings.lasot_path = '/media/titan3/Dataset_JiaWenZ2/SOT/LaSOTBenchmark'
    settings.network_path = ''  # Where tracking networks are stored.
    settings.nfs_path = '/media/titan3/Dataset_JiaWenZ/SOT/nfs'
    settings.otb_path = '/media/titan3/Dataset_JiaWenZ2/SOT/OTB100'
    settings.prj_dir = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github'
    settings.result_plot_path = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github/test/result_plots'
    settings.results_path = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github'
    settings.segmentation_path = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github/test/segmentation_results'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/titan3/Dataset_JiaWenZ/SOT/TNL2K_test_subset'
    settings.tpl_path = ''
    # settings.trackingnet_path = '/media/titan3/Dataset_JiaWenZ/SOT/TrackingNet'
    settings.trackingnet_path = '/media/jiawen/Dataset_JiaWenZ1/DATASET_TEST/TrackingNet'
    settings.uav_path = '/media/titan3/Dataset_JiaWenZ2/SOT/UAV123'
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
