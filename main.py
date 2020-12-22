import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
import os
import argparse
import time
import tqdm
from utils import *
from model import *
import random
import numpy as np

seed= 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description = 'Imagenette2 Classification Task')
    parser.add_argument('--lr', default = 5e-4, type = float, help = 'learning rate')
    parser.add_argument('--epoch', default = 300, type = int, help = 'epochs')
    parser.add_argument('--batch_size', default = 24, type = int, help = 'batch size')
    parser.add_argument('--cuda', default = torch.cuda.is_available(), type = bool)
    parser.add_argument('--log_dir', default = './runs')
    parser.add_argument('--ckpt_dir', default = './model')
    parser.add_argument('--model', default = 'vgg11')
    args = parser.parse_args()

    epochs = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    device = torch.device("cuda" if args.cuda else "cpu")
    log_dir = args.log_dir
    model_name = args.model
    ckpt_dir = args.ckpt_dir
    data_path = './datasets/imagenette2'

    writer = SummaryWriter(log_dir)
    train_data_path = os.path.join(data_path, 'train')
    val_data_path = os.path.join(data_path, 'val')

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = ImageFolder(root = train_data_path, transform=train_transform)
    val_set = ImageFolder(root = val_data_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = batch_size, shuffle = False)

    if model_name=='alexnet':
        model = AlexNet(n_classes=10).to(device)
    elif model_name in ['vgg11','vgg16','vgg19']:
        model = VGG(n_classes=10, mode =model_name, in_channels=3).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    criterion = nn.CrossEntropyLoss()

    train_meter = AccuracyMeter((1,5))
    val_meter = AccuracyMeter((1,5))

    best_val_loss = float('inf')

    for epoch in range(epochs):

        s = time.time()

        train_loss_total = 0
        val_loss_total = 0
        train_meter.reset()
        val_meter.reset()

        model.train()
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            train_loss = criterion(predictions, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_total += train_loss.item()
            train_meter.update(predictions, labels)
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                predictions = model(inputs)
                val_loss = criterion(predictions, labels)

                val_loss_total += val_loss.item()
                val_meter.update(predictions, labels)

        train_loss_mean = train_loss_total/len(train_set)
        val_loss_mean = val_loss_total/len(val_set)

        train_acc_dict = train_meter.as_dict()
        val_acc_dict = val_meter.as_dict()

        t = time.time()
        epoch_min, epoch_sec = epoch_time(s, t)

        print("\n Epoch : %03d / %03d | Elapsed Time %02d min %02d sec"%(epoch + 1, epochs, epoch_min, epoch_sec))
        print("\t Train Loss : %.4f | Val Loss : %.4f"%(train_loss_mean, val_loss_mean))
        print("\t Train Acc@1 : %.4f | Train Acc@5 : %.4f"%(train_acc_dict['Acc@1'], train_acc_dict['Acc@5']))
        print("\t Val   Acc@1 : %.4f | Val   Acc@5 : %.4f"%(val_acc_dict['Acc@1'], val_acc_dict['Acc@5']))
        writer.add_scalars(''.join([str(model_name),'/Train and Val Loss']), {'Train Loss' : train_loss_mean, 'Val Loss' : val_loss_mean}, epoch + 1)
        writer.add_scalars(''.join([str(model_name), '/Train and Val Top 1 Acc']), {'Train Acc@1' : train_acc_dict['Acc@1'], 'Val Acc@1' : val_acc_dict['Acc@1']}, epoch + 1)
        writer.add_scalars(''.join([str(model_name), '/Train and Val Top 5 Acc']), {'Train Acc@5' : train_acc_dict['Acc@5'], 'Val Acc@5' : val_acc_dict['Acc@5']}, epoch + 1)

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save(model.state_dict(), os.path.join(ckpt_dir, '.'.join([model_name, 'pt'])))

        if (epoch % 10 == 0) & (epoch <= 200):
            scheduler.step()
        #
        # for i in [0, 4, 8, 11, 15, 18]:
        #     print(torch.sum(model.conv_layers[i].weight.data))


    writer.close()
if __name__ == '__main__':
    main()