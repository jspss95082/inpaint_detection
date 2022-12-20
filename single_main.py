
from dataset import InpaintDataset
from models import mymodel
import os
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from train import train_one_iter
import argparse
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import wandb


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 100)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = mymodel()
    model = model.to(device)
    path = "../Inpaint_dataset/"
    train_data = InpaintDataset(data_dir_list = [path], use_custom_transform = True)
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    test_data = InpaintDataset(data_dir_list = [path], use_custom_transform = True)
    test_loader = DataLoader(
        test_data,
        1,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)
    wandb.init(project="first")
    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (origin, inpaint, mask) in enumerate(train_loader):
            origin, inpaint, mask = origin.to(device), inpaint.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = train_one_iter(model, optimizer, origin, inpaint, mask)
        wandb.log({"loss": loss})
        
    
        print(epoch, loss)
        if epoch % 50 == 0:
            torch.save(model, 'save_models/' + str(epoch) + '_model.pt')
    

    # toimage = transforms.ToPILImage()
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, (origin, inpaint, mask) in enumerate(test_loader):
    #         origin, inpaint, mask = origin.to(device), inpaint.to(device), mask.to(device)
    #         pred = model.forward(inpaint)
    #         pred = pred[0].to('cpu')
    #         mask = mask[0].to('cpu')
    #         img1 = toimage(pred)
    #         img2 = toimage(mask)
    #         img1.save("img1.jpg")
    #         img2.save("img2.jpg")
    #         break


    
