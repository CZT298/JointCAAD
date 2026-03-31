

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:27:30 2025

@author: zyserver
"""

# from torchvision.transforms import transforms as T
import torch
from torch import nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
# from model.unetr import UNETR
from utils.dice import dice_score
from model.transunet_3d import TransUNet
from model.swinUnetr import SwinUNETR
from model.UXNet.network_backbone import UXNET
from model.Unet3d import UNet
#from dataload import train_dataload
import torch.optim as optim 
from dataload import train_dataload , val_dataload
# from diceloss import MultiClassSoftDiceLoss
from model.Unet_plus_plus_3d import UNet as UNetPlus
from monai.networks import one_hot
from monai.losses import DiceCELoss,DiceFocalLoss
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
from model.mednextv1.MedNextV1 import  MedNeXt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
import random
random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def init_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


def train_model(model,  optimizer, train_loader, num_epochs=40):
    # 将历史最小的loss（取值范围是[0,1]）初始化为最大值1
    # min_loss = 3
    max_val_dice = 0
    for epoch in range(num_epochs):
        # model.train()
        # 3个epoch不优化则降低学习率
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='min', 
                                                    factor=0.5, 
                                                    patience=3, 
                                                    threshold=0.0001,
                                                    threshold_mode='rel',
                                                    min_lr=1e-6)
        epoch_iterator = tqdm( #进度条
           train_loader, desc="Training (X / X Steps) (loss=X.X,dice=X.X)", dynamic_ncols=True
       )
        epoch_loss = 0
        epoch_dice_1 = 0
        epoch_dice_2 = 0
        epoch_dice_3 = 0

        for step,batch in enumerate(epoch_iterator):
            x, y = batch['image'].float(), batch['label'].float()#(1,1,512,512,?)
            # print(y.shape)
            y = one_hot(y, num_classes=4, dim=1)
            # print(y.shape)
            x, y = x.to(device), y.to(device)
            # print(x.shape,y.shape)

            optimizer.zero_grad()        
            out = model(x) 
            #print(out.shape)
            #criterion = DiceCELoss(ce_weight = torch.tensor([1.15, 13.68, 19.84, 159.12]).to(device),
                                 #   lambda_dice=0.5, lambda_ce=0.5)
            criterion = DiceCELoss(lambda_dice=0.5, lambda_ce=0.5)
            # criterion = DiceFocalLoss(lambda_focal=0.5,lambda_dice=0.5)
            # criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(y,out)
            loss.backward()
            optimizer.step()
            
            # print(dice)
            dice_1 = dice_score(y[:,1],out[:,1])
            dice_2 = dice_score(y[:,2],out[:,2])
            dice_3 = dice_score(y[:,3],out[:,3])
            epoch_loss += loss.item()
                    
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) (loss=%5f dice_TL=%2.5f dice_FL=%2.5f dice_CA=%2.5f)" % (
                    epoch+1, step+1, len(train_loader), loss.item(),dice_1,dice_2,dice_3)
            )
            epoch_dice_1 += dice_1
            epoch_dice_2 += dice_2
            epoch_dice_3 += dice_3
            
        ave_loss = epoch_loss / len(train_loader) 
        ave_dice_1 = epoch_dice_1 / len(train_loader)                                                                                                       
        ave_dice_2 = epoch_dice_2 / len(train_loader)
        ave_dice_3 = epoch_dice_3 / len(train_loader)
        log(f'Epoch {epoch+1}/{num_epochs}, loss:{ave_loss:.5f},dice_TL:{ave_dice_1:.5f},dice_FL:{ave_dice_2:.5f},dice_CA:{ave_dice_3:.5f}')
        print(f'Epoch {epoch+1}/{num_epochs}, loss:{ave_loss:.5f},dice_TL:{ave_dice_1:.5f},dice_FL:{ave_dice_2:.5f},dice_CA:{ave_dice_3:.5f}') 
        # if min_loss > ave_loss: 
        torch.save(model.state_dict(), pth)            
            
        
        if (epoch+1) % 3 == 0:
            val_dice = val()
            if val_dice > max_val_dice:
                torch.save(model.state_dict(), val_pth )
                max_val_dice = val_dice
    return model 
            
            
            
    
# 训练模型
def train():
    #log时间
    # 导入历史保存的权重作为训练初始权重
    model.load_state_dict(torch.load(pth, map_location='cpu'))  # JY11.21,加载之前的训练结果，到model中
    
    # 梯度下降的优化器，使用默认学习率
    optimizer = optim.AdamW(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    train_loader = train_dataload(1)
    # 开始训练
    train_model(model,  optimizer, train_loader)
    
def val():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    model.load_state_dict(torch.load(pth, map_location='cpu'))
    # model.eval()
    val_loader = val_dataload()
    epoch_iterator = tqdm( #进度条
                          val_loader, desc="Testing (X / X Steps) (val_dice=X.X)", dynamic_ncols=True
                          )
    epoch_dice_1 = 0
    epoch_dice_2 = 0
    epoch_dice_3 = 0
    for step,batch in enumerate(epoch_iterator):
        x, y = batch['image'].float(), batch['label'].float()#(1,1,512,512,?)
        # print(x.shape)
        y = one_hot(y, num_classes=4, dim=1)
        x, y = x.to(device), y.to(device)
        # print(x.shape)
        with torch.no_grad():
            out = model(x)
            #out = sliding_window_inference(x, (128,128,128), sw_batch_size=1, predictor= model, overlap=0.2)
            out = torch.where(out <= 0.5, torch.tensor(0, dtype=torch.float).to(device), out)
            out = torch.where(out > 0.5, torch.tensor(1, dtype=torch.float).to(device), out)
            dice_1 = dice_score(y[:,1],out[:,1])
            dice_2 = dice_score(y[:,2],out[:,2])
            dice_3 = dice_score(y[:,3],out[:,3])     
            epoch_iterator.set_description(
            "Testing (%d / %d Steps) (val_dice_1=%2.5f,val_dice_2=%2.5f,val_dice_3=%2.5f)" % (
            step, len(val_loader), dice_1, dice_2, dice_3 )
            )
     
        epoch_dice_1 += dice_1 
        epoch_dice_2 += dice_2 
        epoch_dice_3 += dice_3 
   
    ave_dice_1 = epoch_dice_1 / len(val_loader)
    ave_dice_2 = epoch_dice_2 / len(val_loader)
    ave_dice_3 = epoch_dice_3 / len(val_loader)
    ave_dice = (ave_dice_1 +ave_dice_2 + ave_dice_3) / 3
    log(f'validation ave_val_dice: {ave_dice_1:.5f}, {ave_dice_2:.5f}, {ave_dice_3:.5f}, {ave_dice:.5f}')
    print(f'validation ave_val_dice: {ave_dice_1:.5f}, {ave_dice_2:.5f}, {ave_dice_3:.5f}, {ave_dice:.5f}') 
    
    return  ave_dice    
log_path = './trainning_log_both/mednext.txt'
def log(str):
    f = open(log_path, 'a')
    f.write(str + '\n')
    f.close()
    
if __name__ == '__main__':
    log(f'\nNOW TIME:{datetime.now()}')
    print(f'NOW TIME:{datetime.now()}')
    pth = './pth_both/train/mednext.pth'
    val_pth = './pth_both/val/mednext_val.pth'

    #model = UNet(1,4)
    #model = UNetPlus(1,4)
    # model = Vnet(1,4)
    #model = SwinUNETR(img_size =(96,96,192),
     #                      in_channels=1,
     #                     out_channels=4)
    #model = TransUNet(img_dim=(128, 128, 128),
     #                  in_channels=1,
     #                 out_channels=128,
     #                 head_num=4,
      #                mlp_dim=512,
       #                block_num=8,
       #                patch_dim=16,
        #              class_num=4)
    # model = nn.DataParallel(model)
    # model.apply(init_weights)
    model = MedNeXt(
            in_channels = 1, 
            n_channels = 32,
            n_classes = 4,
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            #exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            # block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = None,
            dim = '3d',
            grn=True
            )
    model = model.to(device)
    
    # 参数解析
    # with torch.cuda.device(0):
    # 训练
    # with torch.no_grad():
    # val()
    train() 

        
            
            
        
        
        
        
        