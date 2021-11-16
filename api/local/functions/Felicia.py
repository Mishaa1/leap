import torch # Assume torch is loaded in cloud/sites
import torchvision
import io
import pandas as pd
from PIL import Image
from felicia import privacy_disc


server = FeliciaServer()

# ----------------------
BATCH_SIZE = {2: 512, 3: 256, 4: 128, 5: 32, 6: 32, 7: 8, 8: 8, 9: 4, 10:4} # key is log2 resolution
TRAIN_STEP_RATIO = {k: BATCH_SIZE[2]/v for k, v in BATCH_SIZE.items()}
STEPS_PER_PHASE = 2000 # Number of steps before growing GAN - show around one million images per phase (2000 * 512)
TICK_INTERVAL = 400    # Number of steps before running saving a checkpoints

PDISC_DELAY = 100 # Number of iterations before starting privacy discriminator training
PDISC_LAMBDA = 0  # Weight of privacy penalty

# ---------------------
LEARNING_RATE = 1e-3
BETA_1 = 0.0
BETA_2 = 0.99
EPSILON = 1e-8

# ----------------------

def get_model(hyperparams):
    model =  setup_training_params(pdisc_res_dir, pdisc_delay, pdisc_lambda, batch_size, target_res, train_step_ratio, learning_rate, beta_1, beta_2, epsilon)
    return model


def get_criterion(hyperparams):
    return torch.nn.CrossEntropyLoss()

def get_dataloader(hyperparams, data):
    
    def custom_collate(batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch[1] = batch[1].reshape(-1, 1)
        return batch

    class TFRecordExporter:
    """
    Class for preprocessing and creating TFRecrods from images
    """
    def __init__(self, tfrecord_dir, expected_images):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.start_log2_res     = 2
        self.resolution_log2    = 6
        self.tfr_writers        = []
        self.resolutions        = [x for x in range(self.start_log2_res, self.resolution_log2+1)]

        print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

        tfr_opt = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
        for res in self.resolutions:
            tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (res)
            self.tfr_writers.append(tf.io.TFRecordWriter(tfr_file, tfr_opt))

        def add_image(self, img_path):
        """
        Add a single image to all tf records
        resizes image to different target sizes and adds to appropriate records
        Args:
            img_path (string): full image path
        """
        for i, tfr_writer in enumerate(self.tfr_writers):
            image = dataset.load_image(2**self.resolutions[i], img_path).numpy()
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())

        self.cur_images += 1
     
      
    transforms_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize((224,224)), 
                                                       torchvision.transforms.RandomHorizontalFlip(),
                                                       torchvision.transforms.RandomVerticalFlip(),
                                                       torchvision.transforms.RandomRotation(20),
                                                       torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                                       torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transforms_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Resize((224,224)),  
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    site_id = hyperparams.get("site_id")
    train_ids = hyperparams["train_ids"]
    if site_id is not None:
        #first_id = int(site_id * len(train_ids) / hyperparams["num_sites"])
        #last_id = int((site_id * len(train_ids) / hyperparams["num_sites"]) + (len(train_ids) / hyperparams["num_sites"]))
        first_id = int(site_id * len(train_ids) / 15)
        last_id = int((site_id * len(train_ids) / 15) + (len(train_ids) / 15))
        train_ids = train_ids[first_id:last_id]
    
    dataset_train = HAMDataset(ids=train_ids,
                               csv_file="/home/stolet/ham10000/HAM10000_metadata.csv",
                               root_dir="/home/stolet/ham10000/HAM10000_images_part_1",
                               transform=transforms_train) 
    dataset_val = HAMDataset(ids=hyperparams["val_ids"],
                             csv_file="/home/stolet/ham10000/HAM10000_metadata.csv",
                             root_dir="/home/stolet/ham10000/HAM10000_images_part_1",
                             transform=transforms_val)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=hyperparams["batch_size"],
                                                   shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=hyperparams["batch_size"],
                                                 shuffle=True)
     
    return dataloader_train, dataloader_val
