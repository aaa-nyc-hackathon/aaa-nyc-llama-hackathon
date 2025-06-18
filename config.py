import os

TMP_DIRNAME_IMAGES = os.path.join(os.getcwd(), "images/cache/segmentation")
TMP_DIRNAME_VIDEOS = os.path.join(os.getcwd(), "videos/cache/segmentation")
MAX_FRAMES = 500
HF_MODEL = "facebook/sam2-hiera-large"
# HF_MODEL = "facebook/sam2-hiera-tiny"