##### 학습 스크립트 구현 (train.py)
### 실제 학습 루프(Epoch, Batch, Optimizer, Log)를 실행.
### 논문의 Self-Reconstruction(자기 복원) 전략을 구현
### 원본 이미지의 조건지도를 추출하고, 이를 그대로 조건으로 넣어 원본을 다시 만들어내도록 학습
### 주요 Loss 가중치:
### GAN: 1.0, Feature Matching: 10.0, VGG: 10.0
### Identity: 5.0, Cycle: 10.0 (논문 3.5.4 기준)

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Custom Modules
from networks import SPADEGenerator, MultiscaleDiscriminator
from losses import VGGLoss, IdentityLoss, CycleConsistencyLoss
from dataset import FaceDataset

def train(opt):
    # --- [1] Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join("data", "checkpoints", opt.name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # --- [2] DataLoader ---
    dataset = FaceDataset(opt.data_root, image_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset Size: {len(dataset)}")

    # --- [3] Models ---
    # label_nc = 3 (RGB Maps) + 1 (Mask) = 4 (실제론 파싱 클래스 수에 따라 다름)
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    netD = MultiscaleDiscriminator(input_nc=3+4).to(device) # Img(3) + Cond(4)

    netG.train()
    netD.train()

    # --- [4] Optimizers ---
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # --- [5] Losses ---
    criterionGAN = nn.MSELoss() # LSGAN
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss().to(device)
    # weights 폴더에 model_ir_se50.pth가 있어야 함
    id_weights = "./weights/model_ir_se50.pth"
    criterionID = IdentityLoss(id_weights, device) if os.path.exists(id_weights) else None
    if criterionID is None: print("Warning: ID Loss Disabled (Weights not found)")

    criterionCycle = CycleConsistencyLoss().to(device)

    # --- [6] Training Loop ---
    print("Starting Training...")
    global_step = 0

    for epoch in range(opt.epochs):
        for i, data in enumerate(dataloader):
            real_img = data['image'].to(device)         # [-1, 1]
            spade_input = data['spade_input'].to(device) # Condition
            target_maps = data['target_maps'].to(device)
            mask = data['mask'].to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()

            # Fake Generation
            fake_img, _ = netG(real_img, spade_input) # Self-Reconstruction

            # Real Loss
            # D input: Concatenate Image + Condition
            real_pair = torch.cat([real_img, spade_input], dim=1)
            pred_real = netD(real_pair)
            loss_D_real = 0
            for pred in pred_real:
                loss_D_real += criterionGAN(pred, torch.ones_like(pred))

            # Fake Loss
            fake_pair = torch.cat([fake_img.detach(), spade_input], dim=1)
            pred_fake = netD(fake_pair)
            loss_D_fake = 0
            for pred in pred_fake:
                loss_D_fake += criterionGAN(pred, torch.zeros_like(pred))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizerG.zero_grad()

            # 1. GAN Loss (Fool Discriminator)
            fake_pair_G = torch.cat([fake_img, spade_input], dim=1)
            pred_fake_G = netD(fake_pair_G)
            loss_G_GAN = 0
            for pred in pred_fake_G:
                loss_G_GAN += criterionGAN(pred, torch.ones_like(pred))

            # 2. Feature Matching Loss (Stabilization)
            loss_G_Feat = 0
            # (D의 중간 feature를 가져오는 로직은 생략되었으나, networks.py 수정 시 추가 가능)
            # 여기서는 편의상 L1 Reconstruction으로 대체
            loss_G_Feat = criterionFeat(fake_img, real_img) * 10.0

            # 3. VGG Loss
            loss_G_VGG = criterionVGG(fake_img, real_img) * 10.0

            # 4. Identity Loss
            loss_G_ID = 0
            if criterionID:
                loss_G_ID = criterionID(real_img, fake_img) * 5.0

            # 5. Cycle Consistency Loss (본 연구 핵심)
            # 생성된 이미지에서 맵 추출 vs 원본 맵
            loss_G_Cycle = criterionCycle(fake_img, target_maps, mask) * 10.0

            # Total Loss
            loss_G = loss_G_GAN + loss_G_Feat + loss_G_VGG + loss_G_ID + loss_G_Cycle

            loss_G.backward()
            optimizerG.step()

            global_step += 1

            # --- [Logging] ---
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{opt.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

                writer.add_scalar('Loss/D', loss_D.item(), global_step)
                writer.add_scalar('Loss/G', loss_G.item(), global_step)
                writer.add_scalar('Loss/Cycle', loss_G_Cycle.item(), global_step)

            if i % 500 == 0:
                # 시각화 저장 (Real | Fake | Map)
                vis = torch.cat([real_img, fake_img], dim=3)
                vutils.save_image(vis, os.path.join(save_dir, f"epoch_{epoch}_step_{i}.png"), normalize=True)

        # Save Model Checkpoint
        if epoch % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(save_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_dir, f"netD_epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment_1', help='Experiment name')
    parser.add_argument('--data_root', type=str, default='./data/processed', help='Path to processed data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (Reduce if OOM)')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--img_size', type=int, default=512, help='Image resolution for training')

    opt = parser.parse_args()
    train(opt)
