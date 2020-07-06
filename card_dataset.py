
import os
import json
import cv2
from torch.utils.data import Dataset
import torch


class CardDataset(Dataset):

    def __init__(self, image_folder, gt_folder, image_size):

        self.images = []
        self.groundtruths = []

        image_subfolders = os.listdir(image_folder)

        for image_subfolder in image_subfolders:

            print('processing {}'.format(image_subfolder))

            with open(os.path.join(gt_folder, image_subfolder + '.json')) as json_file:

                data = json.load(json_file)

                for key, value in data.items():

                    image_name = value['filename']
                    print(image_name)
                    image = cv2.imread(os.path.join(image_folder, image_subfolder, image_name))
                    image_height, image_width = image.shape[:2]

                    image = cv2.resize(image, (image_size, image_size))

                    regions = value['regions']

                    temp = {}
                    for region in regions:

                        label = region['region_attributes']['class']
                        x = region['shape_attributes']['cx']/image_width
                        y = region['shape_attributes']['cy']/image_height

                        temp[label] = [x, y]

                    # torch.from_numpy: convert to tensor
                    self.groundtruths.append(torch.FloatTensor([temp['ul'] + temp['bl'] + temp['br'] + temp['ur']]))
                    self.images.append(torch.from_numpy(image))


    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        return self.images[index], self.groundtruths[index]
