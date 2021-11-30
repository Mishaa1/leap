"""
ProGAN model and runtime parameters
"""
#----------------------------------------------------------------------------
# Dataset parameters
PATH = "/datadrive/histopathologic-cancer-detection/train/*.tif"
RESULTS_DIR = "results/"
IMAGE_RESOLUTION = 64
BUFFER_SIZE = 200 # for loading images

#----------------------------------------------------------------------------
# Training parameters
BATCH_SIZE = {2: 64, 3: 32, 4: 16, 5: 16, 6: 4, 7: 4, 8: 4, 9: 4, 10:4} # key is log2 resolution
TRAIN_STEP_RATIO = {k: BATCH_SIZE[2]/v for k, v in BATCH_SIZE.items()}
STEPS_PER_PHASE = 20000 # Number of steps before growing GAN
TICK_INTERVAL = 4000

#----------------------------------------------------------------------------
# Optimization parameters
LEARNING_RATE = 1e-3
BETA_1 = 0.0
BETA_2 = 0.99
EPSILON = 1e-8
