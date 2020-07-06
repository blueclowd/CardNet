import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CardNet
from card_dataset import CardDataset

state = torch.load('weights/epoch40.pth', map_location=torch.device('cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.DataParallel(CardNet())
model = model.to(device)
model.load_state_dict(state['model'])
model.eval()

test_data = CardDataset(image_folder='data_test/images', gt_folder='data_test/groundtruths', image_size=512)
test_dataloader = DataLoader(test_data, batch_size=5)

with torch.no_grad():
    for i, (images, targets) in enumerate(test_dataloader):
        images = images.to(device)
        targets = images.to(device)

        images = images.permute(0, 3, 1, 2).float()
        targets = targets.float()

        tic = time.time()
        predictions = model(images)
        print('Inference time: {}'.format(time.time() - tic))



        for img_idx in range(len(images)):

            ul_x, ul_y = predictions[img_idx][0].item(), predictions[img_idx][1].item()
            bl_x, bl_y = predictions[img_idx][2].item(), predictions[img_idx][3].item()
            br_x, br_y = predictions[img_idx][4].item(), predictions[img_idx][5].item()
            ur_x, ur_y = predictions[img_idx][6].item(), predictions[img_idx][7].item()

            image_np = images[img_idx].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)

            cv2.circle(image_np, (int(ul_x * 512), int(ul_y * 512)), 3, (255, 255, 255), -1)
            cv2.putText(image_np, text='ul', org=(int(ul_x * 512), int(ul_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))

            cv2.circle(image_np, (int(bl_x * 512), int(bl_y * 512)), 3, (255, 255, 255), -1)
            cv2.putText(image_np, text='bl', org=(int(bl_x * 512), int(bl_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))

            cv2.circle(image_np, (int(br_x * 512), int(br_y * 512)), 3, (255, 255, 255), -1)
            cv2.putText(image_np, text='br', org=(int(br_x * 512), int(br_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))

            cv2.circle(image_np, (int(ur_x * 512), int(ur_y * 512)), 3, (255, 255, 255), -1)
            cv2.putText(image_np, text='ur', org=(int(ur_x * 512), int(ur_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))

            cv2.imwrite(str(img_idx) + '.jpg', image_np)
