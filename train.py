
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from model import CardNet
from torchsummary import summary
from card_dataset import CardDataset
from torch.utils.data import DataLoader

image_size = 512

train_data = CardDataset(image_folder='data/images', gt_folder='data/groundtruths', image_size=image_size)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)

train_size = len(train_dataloader.dataset)

# for i_batch, (images, groundtruths) in enumerate(train_dataloader):
#
#     image_np = images[0].numpy().astype(np.uint8)
#
#     gt = groundtruths[0][0]
#
#     ul_x, ul_y = gt[0].item(), gt[1].item()
#     bl_x, bl_y = gt[2].item(), gt[3].item()
#     br_x, br_y = gt[4].item(), gt[5].item()
#     ur_x, ur_y = gt[6].item(), gt[7].item()
#
#     cv2.circle(image_np, (int(ul_x * image_size), int(ul_y * image_size)), 3, (255, 255, 255), -1)
#     cv2.putText(image_np, text='ul', org=(int(ul_x * 512), int(ul_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 255, 255))
#
#     cv2.circle(image_np, (int(bl_x * image_size), int(bl_y * image_size)), 3, (255, 255, 255), -1)
#     cv2.putText(image_np, text='bl', org=(int(bl_x * 512), int(bl_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 255, 255))
#
#     cv2.circle(image_np, (int(br_x * image_size), int(br_y * image_size)), 3, (255, 255, 255), -1)
#     cv2.putText(image_np, text='br', org=(int(br_x * 512), int(br_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 255, 255))
#
#     cv2.circle(image_np, (int(ur_x * image_size), int(ur_y * image_size)), 3, (255, 255, 255), -1)
#     cv2.putText(image_np, text='ur', org=(int(ur_x * 512), int(ur_y * 512)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 255, 255))
#
#     cv2.imwrite('train{}.jpg'.format(i_batch), image_np)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(CardNet())
model = model.to(device)

summary(model, (3, 512, 512))

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

n_epoch = 50

for epoch in range(n_epoch):

    print('Epoch {}'.format(epoch))

    accumulate_loss = 0

    for batch_index, (images, targets) in enumerate(train_dataloader):

        # Clear old gradients from the last step
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        # <batch size, image channels image height, image width>
        images = images.permute(0, 3, 1, 2).float()
        targets = targets.float()

        predictions = model(images)
        loss = criterion(predictions, targets.view(-1, 8))

        # Computer derivative of the loss w.r.t. the paraemters using backpropagation
        loss.backward()

        # Update the parameters based on the gradients of the parameters
        optimizer.step()

        accumulate_loss += loss
        # print('batch:{} loss: {}'.format(batch_index, loss))

    if epoch % 20 == 0:

        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, 'weights/epoch{}.pth'.format(epoch))

    print('loss: {:.5f}'.format(accumulate_loss/train_size))





