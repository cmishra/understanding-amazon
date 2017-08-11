import pandas
import torch
from PIL import Image
import os
import json
import numpy as np


class SatelliteData(torch.utils.data.Dataset):

    def __init__(self, filepath, target_json, compose, logger, debug_mode):
        logger.info("Initializing dataset.")
        images_names = os.listdir(filepath)
        if debug_mode:
            images_names = images_names[:(64*30)]
        images = {}
        for f in images_names:
            f_key = f[:-4]
            with Image.open(os.path.join(filepath, f)) as img_pil:
                images[f_key] = img_pil.convert("RGB")

        with open(target_json, 'r') as f:
            self.targets = json.load(f)
                
        self.images = images
        self.compose = compose
        self.keys = list(self.images.keys())
        logger.info("Initialized dataset. %d rows" % len(self))


    def __getitem__(self, index):
        img = self.keys[index]
        return self.compose(self.images[img]), torch.FloatTensor(self.targets[img])


    def __len__(self):
        return len(self.images)


