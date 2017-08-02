import pandas
import torch
from PIL import Image
import os
import logging

class SatelliteData(torch.utils.data.Dataset):

    def __init__(self, filepath, target_csv, compose):
        logging.log(20, "Initializing dataset")
        targets = pandas.read_csv(target_csv)
        tag_list = set()
        for tag_description in targets['tags']:
            tags = tag_description.split(' ')
            for t in tags:
                tag_list.add(t)
        for t in tag_list:
            targets[t] = targets['tags'].apply(lambda x: 1 if t in x.split(' ') else 0)
        self.targets = targets

        images_names = os.listdir(filepath)
        images = {}
        for f in images_names:
            with Image.open(os.path.join(filepath, f)) as f_bytes:
                images[f] = f_bytes
        self.images = images
        self.compose = compose
        logging.log(20, "Initialized dataset. %d rows" % len(self))


    def __getitem__(self, index):
        target_row = self.targets.iloc[index]
        img = target_row['image_name']
        y = target_row[1:]
        return self.compose(self.images[img]), y


    def __len__(self):
        return len(self.images)
