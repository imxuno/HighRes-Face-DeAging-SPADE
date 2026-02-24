import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils
from tqdm import tqdm

from networks import SPADEGenerator, MultiscaleDiscriminator
from losses import VGGLoss, IdentityLoss, CycleConsistencyLoss
from dataset import FaceDataset


# -------------------------
# utils
# -------------------------
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp01(x):
    return torch.clamp(x, 0.0, 1.0)

def masked_l1(fake, real, mask, w_in=1.0, w_out=10.0):
    """
    피부 안(in)은 약하게, 피부 밖(out)은 강하게 원본 유지.
    """
    m = (mask > 0.5).float()
    loss_in = F.l1_loss(fake * m, real * m)
    loss_out = F.l1_loss(fake * (1 - m), real * (1 - m))
    return w_in * loss_in + w_out * loss_out

def edit_cond_contrast(cond, mask, factors):
    """
    cond: (B,3,H,W) [0,1]
    mask: (B,1,H,W) 0/1
    factors: (B,3,1,1)  0.6이면 대비(강도) 약화, 1.2면 강화
    """
    eps = 1e-6
    m = (mask > 0.5).float()
    m_sum = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)  # (B,1,1,1)

    # mask 내부 평균 유지
    mu = (cond * m).sum(dim=(2, 3), keepdim=True) / m_sum   # (B,3,1,1)
    out = mu + factors * (cond - mu)
    out = clamp01(out)
    out = out * m + cond * (1 - m)
    return out

def sample_factors(B, device, red_min, red_max, wr_min, wr_max, pore_min, pore_max):
    fr = torch.empty(B, 1, 1, 1, device=device).uniform_(red_min, red_max)
    fw = torch.empty(B, 1, 1, 1, device=device).uniform_(wr_min, wr_max)
    fp = torch.empty(B, 1, 1, 1, device=device).uniform_(pore_min, pore_max)
    return torch.cat([fr, fw, fp], dim=1)  # (B,3,1,1)


# -------------------------
# train
# -------------------------
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

    if opt.seed is not None:
        set_seed(opt.seed)

    save_dir = os.path.join("..", "data", "checkpoints", opt.name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, "logs"))

    print("--- Training Setup ---")
    print(f"Device: {device}")
    print(f"Save Directory: {save_dir}")
    print(f"Image Size: {opt.img_size}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"LR: {opt.lr}")
    print(f"Use VGG Loss:   {opt.use_vgg}")
    print(f"Use ID Loss:    {opt.use_id}")
    print(f"Use Cycle Loss: {opt.use_cycle}")
    print(f"cond_jitter_prob: {opt.cond_jitter_prob}")

    # Data
    dataset = FaceDataset(
        opt.data_root,
        is_train=True,
        image_size=opt.img_size,
        allow_dummy_maps=opt.allow_dummy_maps,
        flip_prob=opt.flip_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,   # Windows는 0 권장, GPU/환경 되면 2~4
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
    )
    print(f"Dataset Size: {len(dataset)}")
    print(f"Steps per epoch (approx): {len(dataloader)}")

    # Models
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    netD = MultiscaleDiscriminator(input_nc=7, num_D=opt.num_D).to(device)
    netG.train()
    netD.train()

    # Optim
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Losses
    criterionGAN = nn.MSELoss()  # LSGAN
    criterionVGG = VGGLoss().to(device) if opt.use_vgg else None

    criterionID = None
    if opt.use_id:
        id_weights_path = os.path.join("..", "weights", "model_ir_se50.pth")
        if os.path.exists(id_weights_path):
            criterionID = IdentityLoss(id_weights_path, device)
        else:
            print(f"[Warn] ID weights not found: {id_weights_path}")
            criterionID = None

    criterionCycle = CycleConsistencyLoss().to(device) if opt.use_cycle else None

    # Train Loop
    global_step = 0
    print("Starting Training Loop...")

    for epoch in range(opt.epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{opt.epochs}")

        for i, data in progress_bar:
            if data is None:
                continue

            real_img = data["image"].to(device)          # (B,3,H,W) [-1,1]
            spade_input = data["spade_input"].to(device) # (B,4,H,W) [0,1]+mask
            target_maps = data["target_maps"].to(device) # (B,3,H,W)
            mask = data["mask"].to(device)               # (B,1,H,W)

            # -------------------------
            # 1) Train Discriminator
            # -------------------------
            optimizerD.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_for_D, _ = netG(real_img, spade_input)

            real_pair = torch.cat([real_img, spade_input], dim=1)  # (B,7,H,W)
            fake_pair = torch.cat([fake_for_D, spade_input], dim=1)

            pred_real = netD(real_pair)
            pred_fake = netD(fake_pair)

            # (선택) label smoothing
            real_label = 0.9 if opt.label_smooth else 1.0

            loss_D_real = real_img.new_tensor(0.0)
            for pr in pred_real:
                loss_D_real = loss_D_real + criterionGAN(pr, torch.full_like(pr, real_label))

            loss_D_fake = real_img.new_tensor(0.0)
            for pf in pred_fake:
                loss_D_fake = loss_D_fake + criterionGAN(pf, torch.zeros_like(pf))

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizerD.step()

            # -------------------------
            # 2) Train Generator (base step)
            # -------------------------
            optimizerG.zero_grad(set_to_none=True)

            fake_img, _ = netG(real_img, spade_input)
            fake_pair_G = torch.cat([fake_img, spade_input], dim=1)
            pred_fake_G = netD(fake_pair_G)

            loss_G_GAN = real_img.new_tensor(0.0)
            for pf in pred_fake_G:
                loss_G_GAN = loss_G_GAN + criterionGAN(pf, torch.ones_like(pf))

            # ✅ mask 기반 재구성 (피부 안은 약하게, 밖은 강하게)
            loss_G_L1 = masked_l1(fake_img, real_img, mask, w_in=opt.l1_in, w_out=opt.l1_out) * opt.lambda_l1

            loss_G_VGG = real_img.new_tensor(0.0)
            if criterionVGG is not None:
                # 피부는 real로 채워서 VGG가 외곽/구조 위주로 보게
                m = (mask > 0.5).float()
                fake_bg = fake_img * (1 - m) + real_img * m
                loss_G_VGG = criterionVGG(fake_bg, real_img) * opt.lambda_vgg

            loss_G_ID = real_img.new_tensor(0.0)
            if criterionID is not None:
                loss_G_ID = criterionID(real_img, fake_img) * opt.lambda_id

            loss_G_Cycle = real_img.new_tensor(0.0)
            if criterionCycle is not None:
                loss_G_Cycle = criterionCycle(fake_img, target_maps, mask) * opt.lambda_cycle

            loss_G = (opt.lambda_gan * loss_G_GAN) + loss_G_L1 + loss_G_VGG + loss_G_ID + loss_G_Cycle
            loss_G.backward()
            optimizerG.step()

            # -------------------------
            # 3) Control Step (핵심)
            #    inference처럼 cond를 흔들어보고, 그 cond_edit를 만족하도록 학습
            # -------------------------
            if (criterionCycle is not None) and (opt.cond_jitter_prob > 0.0):
                if torch.rand(1).item() < opt.cond_jitter_prob:
                    optimizerG.zero_grad(set_to_none=True)

                    cond = spade_input[:, :3].float().clamp(0, 1)      # (B,3,H,W)
                    m = (spade_input[:, 3:4] > 0.5).float()            # (B,1,H,W)

                    factors = sample_factors(
                        B=cond.size(0),
                        device=device,
                        red_min=opt.red_min, red_max=opt.red_max,
                        wr_min=opt.wr_min, wr_max=opt.wr_max,
                        pore_min=opt.pore_min, pore_max=opt.pore_max,
                    )

                    cond_edit = edit_cond_contrast(cond, m, factors)
                    spade_edit = torch.cat([cond_edit, m], dim=1)

                    fake_edit, _ = netG(real_img, spade_edit)

                    # ✅ cycle target을 cond_edit로
                    loss_ctl_cycle = criterionCycle(fake_edit, cond_edit, m) * opt.lambda_cycle

                    # ✅ 피부 밖은 원본 유지(안정), 피부 안은 약하게만
                    loss_ctl_l1 = masked_l1(fake_edit, real_img, m, w_in=opt.l1_in, w_out=opt.l1_out) * opt.lambda_l1_control

                    loss_ctl_vgg = real_img.new_tensor(0.0)
                    if criterionVGG is not None and opt.lambda_vgg_control > 0:
                        fake_bg = fake_edit * (1 - m) + real_img * m
                        loss_ctl_vgg = criterionVGG(fake_bg, real_img) * opt.lambda_vgg_control

                    loss_ctl_id = real_img.new_tensor(0.0)
                    if criterionID is not None and opt.lambda_id_control > 0:
                        loss_ctl_id = criterionID(real_img, fake_edit) * opt.lambda_id_control

                    loss_control = loss_ctl_cycle + loss_ctl_l1 + loss_ctl_vgg + loss_ctl_id
                    loss_control.backward()
                    optimizerG.step()

                    if (global_step % 50) == 0:
                        writer.add_scalar("Loss/Control", loss_control.item(), global_step)
                        writer.add_scalar("Loss/Control_cycle", loss_ctl_cycle.item(), global_step)

            global_step += 1

            # progress
            if i % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "D": f"{loss_D.item():.3f}",
                        "G": f"{loss_G.item():.3f}",
                        "GAN": f"{loss_G_GAN.item():.3f}",
                    }
                )

            # logging
            if i % opt.log_every == 0:
                writer.add_scalar("Loss/D", loss_D.item(), global_step)
                writer.add_scalar("Loss/G", loss_G.item(), global_step)
                writer.add_scalar("Loss/G_GAN", loss_G_GAN.item(), global_step)
                writer.add_scalar("Loss/G_L1", loss_G_L1.item(), global_step)
                if criterionVGG is not None:
                    writer.add_scalar("Loss/G_VGG", loss_G_VGG.item(), global_step)
                if criterionID is not None:
                    writer.add_scalar("Loss/G_ID", loss_G_ID.item(), global_step)
                if criterionCycle is not None:
                    writer.add_scalar("Loss/G_Cycle", loss_G_Cycle.item(), global_step)

            # image save
            if i % opt.save_img_every == 0:
                vis = torch.cat([real_img, fake_img], dim=3)  # real | fake
                vutils.save_image(
                    vis,
                    os.path.join(save_dir, f"epoch_{epoch}_step_{i}.png"),
                    normalize=True,
                )

        # checkpoint
        if (epoch % opt.save_ckpt_every == 0) or (epoch == opt.epochs - 1):
            torch.save(netG.state_dict(), os.path.join(save_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_dir, f"netD_epoch_{epoch}.pth"))

    print("Training Finished.")
    writer.close()


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="experiment_1")
    parser.add_argument("--data_root", type=str, default="../data/processed")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--img_size", type=int, default=256)

    parser.add_argument("--num_workers", type=int, default=0)  # Windows는 0 권장
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_vgg", action="store_true")
    parser.add_argument("--use_id", action="store_true")
    parser.add_argument("--use_cycle", action="store_true")

    # dataset options
    parser.add_argument("--allow_dummy_maps", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)

    # D options
    parser.add_argument("--num_D", type=int, default=2)
    parser.add_argument("--label_smooth", action="store_true")

    # base loss weights
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=10.0)
    parser.add_argument("--lambda_vgg", type=float, default=10.0)
    parser.add_argument("--lambda_id", type=float, default=3.0)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)

    # recon balance
    parser.add_argument("--l1_in", type=float, default=1.0)
    parser.add_argument("--l1_out", type=float, default=10.0)

    # control step
    parser.add_argument("--cond_jitter_prob", type=float, default=0.7)
    parser.add_argument("--red_min", type=float, default=0.6)
    parser.add_argument("--red_max", type=float, default=1.2)
    parser.add_argument("--wr_min", type=float, default=0.25)
    parser.add_argument("--wr_max", type=float, default=1.4)
    parser.add_argument("--pore_min", type=float, default=0.35)
    parser.add_argument("--pore_max", type=float, default=1.3)

    parser.add_argument("--lambda_l1_control", type=float, default=6.0)
    parser.add_argument("--lambda_vgg_control", type=float, default=8.0)
    parser.add_argument("--lambda_id_control", type=float, default=0.0)

    # io/log
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_img_every", type=int, default=800)
    parser.add_argument("--save_ckpt_every", type=int, default=5)

    opt = parser.parse_args()

    print(f"Running Experiment: {opt.name}")
    print(f"Data Root: {opt.data_root}")

    try:
        train(opt)
    except Exception as e:
        print(f"Error occurred: {e}")
        input("Press Enter to exit...")
