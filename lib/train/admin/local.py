class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/jiawen/File_JiaWenZ1/AsymTrack/codes/AsymTrack-github/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = ''
        self.lasot_dir = '/media/jiawen/Dataset_JiaWenZ1/DATASET/LaSOTBenchmark'
        self.got10k_dir = '/media/jiawen/Dataset_JiaWenZ1/DATASET/GOT10K/train'
        self.trackingnet_dir = '/media/jiawen/Dataset_JiaWenZ1/DATASET/TrackingNet'
        self.coco_dir = '/media/jiawen/Dataset_JiaWenZ1/DATASET/COCO-2017'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.imagenet1k_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
