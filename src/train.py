# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data._utils.collate import default_collate
# import torchvision.utils as vutils
# from tqdm import tqdm  # 진행률 표시

# # Custom Modules
# from networks import SPADEGenerator, MultiscaleDiscriminator
# from losses import VGGLoss, IdentityLoss, CycleConsistencyLoss
# from dataset import FaceDataset


# def safe_collate(batch):
#     """
#     Dataset에서 문제가 생기면 __getitem__이 None을 반환할 수 있으므로,
#     그 샘플을 배치에서 제거한 뒤 collate 하는 함수.
#     유효한 샘플이 하나도 없으면 None을 반환한다.
#     """
#     batch = [b for b in batch if b is not None]
#     if len(batch) == 0:
#         return None
#     return default_collate(batch)


# def train(opt):
#     # --- [1] Setup ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 체크포인트 저장 경로
#     save_dir = os.path.join("..", "data", "checkpoints", opt.name)
#     os.makedirs(save_dir, exist_ok=True)

#     # 로그 저장소
#     writer = SummaryWriter(os.path.join(save_dir, "logs"))

#     print(f"--- Training Setup ---")
#     print(f"Device: {device}")
#     print(f"Save Directory: {save_dir}")
#     print(f"Image Size: {opt.img_size}")
#     print(f"Batch Size: {opt.batch_size}")

#     # --- [2] DataLoader ---
#     dataset = FaceDataset(opt.data_root, image_size=opt.img_size)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=0,          # Windows 안정성을 위해 0
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=safe_collate # None 샘플 안전 처리
#     )
#     print(f"Dataset Size: {len(dataset)}")

#     # --- [3] Models ---
#     netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
#     netD = MultiscaleDiscriminator(input_nc=7).to(device)

#     netG.train()
#     netD.train()

#     # --- [4] Optimizers ---
#     optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

#     # --- [5] Losses ---
#     criterionGAN = nn.MSELoss()
#     criterionFeat = nn.L1Loss()
#     criterionVGG = VGGLoss().to(device)

#     id_weights_path = os.path.join("..", "weights", "model_ir_se50.pth")
#     if os.path.exists(id_weights_path):
#         print(f"Loading Identity Loss weights from: {id_weights_path}")
#         criterionID = IdentityLoss(id_weights_path, device)
#     else:
#         print(f"Warning: Identity Loss weights not found at {id_weights_path}")
#         criterionID = None

#     criterionCycle = CycleConsistencyLoss().to(device)

#     # --- [6] Training Loop ---
#     print("Starting Training Loop...")
#     global_step = 0

#     for epoch in range(opt.epochs):
#         progress_bar = tqdm(
#             enumerate(dataloader),
#             total=len(dataloader),
#             desc=f"Epoch {epoch}/{opt.epochs}"
#         )

#         for i, data in progress_bar:
#             # safe_collate가 None을 반환할 수 있으므로 체크
#             if data is None:
#                 continue

#             real_img = data["image"].to(device)          # (B,3,H,W)
#             spade_input = data["spade_input"].to(device) # (B,4,H,W)
#             target_maps = data["target_maps"].to(device) # (B,3,H,W)
#             mask = data["mask"].to(device)               # (B,1,H,W)

#             # --- Train Discriminator ---
#             optimizerD.zero_grad()

#             fake_img, _ = netG(real_img, spade_input)

#             # Real Loss
#             real_pair = torch.cat([real_img, spade_input], dim=1)
#             pred_real = netD(real_pair)
#             loss_D_real = 0.0
#             for pred in pred_real:
#                 loss_D_real = loss_D_real + criterionGAN(pred, torch.ones_like(pred))

#             # Fake Loss
#             fake_pair = torch.cat([fake_img.detach(), spade_input], dim=1)
#             pred_fake = netD(fake_pair)
#             loss_D_fake = 0.0
#             for pred in pred_fake:
#                 loss_D_fake = loss_D_fake + criterionGAN(pred, torch.zeros_like(pred))

#             loss_D = (loss_D_real + loss_D_fake) * 0.5
#             loss_D.backward()
#             optimizerD.step()

#             # --- Train Generator ---
#             optimizerG.zero_grad()

#             fake_pair_G = torch.cat([fake_img, spade_input], dim=1)
#             pred_fake_G = netD(fake_pair_G)
#             loss_G_GAN = 0.0
#             for pred in pred_fake_G:
#                 loss_G_GAN = loss_G_GAN + criterionGAN(pred, torch.ones_like(pred))

#             # 이미지 자체 L1, VGG, ID, Cycle
#             loss_G_Feat = criterionFeat(fake_img, real_img) * 10.0
#             loss_G_VGG = criterionVGG(fake_img, real_img) * 10.0

#             loss_G_ID = 0.0
#             if criterionID is not None:
#                 loss_G_ID = criterionID(real_img, fake_img) * 5.0

#             loss_G_Cycle = criterionCycle(fake_img, target_maps, mask) * 10.0

#             loss_G = loss_G_GAN + loss_G_Feat + loss_G_VGG + loss_G_ID + loss_G_Cycle
#             loss_G.backward()
#             optimizerG.step()

#             global_step += 1

#             # 진행률 바에 Loss 정보 표시
#             if i % 10 == 0:
#                 progress_bar.set_postfix(
#                     {
#                         "D_Loss": f"{loss_D.item():.3f}",
#                         "G_Loss": f"{loss_G.item():.3f}",
#                     }
#                 )

#             # TensorBoard Logging
#             if i % 100 == 0:
#                 writer.add_scalar("Loss/D", loss_D.item(), global_step)
#                 writer.add_scalar("Loss/G", loss_G.item(), global_step)
#                 if criterionID is not None:
#                     writer.add_scalar("Loss/ID", loss_G_ID.item(), global_step)

#             # Image Saving
#             if i % 500 == 0:
#                 # real | fake 를 가로로 붙여서 저장
#                 vis = torch.cat([real_img, fake_img], dim=3)
#                 vutils.save_image(
#                     vis,
#                     os.path.join(save_dir, f"epoch_{epoch}_step_{i}.png"),
#                     normalize=True,
#                 )

#         # Save Checkpoint
#         if epoch % 5 == 0 or epoch == opt.epochs - 1:
#             torch.save(netG.state_dict(), os.path.join(save_dir, f"netG_epoch_{epoch}.pth"))
#             torch.save(netD.state_dict(), os.path.join(save_dir, f"netD_epoch_{epoch}.pth"))

#     print("Training Finished.")
#     writer.close()


# if __name__ == "__main__":
#     import multiprocessing

#     multiprocessing.freeze_support()

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--name", type=str, default="experiment_1", help="Experiment name")
#     parser.add_argument(
#         "--data_root", type=str, default="../data/processed", help="Path to processed data"
#     )
#     parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
#     parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
#     parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
#     parser.add_argument("--img_size", type=int, default=512, help="Image resolution")

#     opt = parser.parse_args()

#     print(f"Running Experiment: {opt.name}")
#     print(f"Data Root: {opt.data_root}")

#     try:
#         train(opt)
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         input("Press Enter to exit...")


import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data._utils.collate import default_collate
import torchvision.utils as vutils
from tqdm import tqdm  # 진행률 표시

# Custom Modules
from networks import SPADEGenerator, MultiscaleDiscriminator
from losses import VGGLoss, IdentityLoss, CycleConsistencyLoss
from dataset import FaceDataset


def safe_collate(batch):
    """
    Dataset에서 문제가 생기면 __getitem__이 None을 반환할 수 있으므로,
    그 샘플을 배치에서 제거한 뒤 collate 하는 함수.
    유효한 샘플이 하나도 없으면 None을 반환한다.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def train(opt):
    # --- [1] Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 체크포인트 저장 경로
    save_dir = os.path.join("..", "data", "checkpoints", opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # 로그 저장소
    writer = SummaryWriter(os.path.join(save_dir, "logs"))

    print(f"--- Training Setup ---")
    print(f"Device: {device}")
    print(f"Save Directory: {save_dir}")
    print(f"Image Size: {opt.img_size}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"Use VGG Loss:   {opt.use_vgg}")
    print(f"Use ID Loss:    {opt.use_id}")
    print(f"Use Cycle Loss: {opt.use_cycle}")

    # --- [2] DataLoader ---
    dataset = FaceDataset(opt.data_root, image_size=opt.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,          # Windows 안정성을 위해 0
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate # None 샘플 안전 처리
    )
    print(f"Dataset Size: {len(dataset)}")

    # --- [3] Models ---
    netG = SPADEGenerator(input_nc=3, label_nc=4, z_dim=256).to(device)
    netD = MultiscaleDiscriminator(input_nc=7).to(device)

    netG.train()
    netD.train()

    # --- [4] Optimizers ---
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # --- [5] Losses ---
    criterionGAN = nn.MSELoss()
    criterionFeat = nn.L1Loss()

    # Light 모드에서는 아래 세 개를 옵션으로 켜고 끌 수 있게 처리
    criterionVGG = VGGLoss().to(device) if opt.use_vgg else None

    criterionID = None
    if opt.use_id:
        id_weights_path = os.path.join("..", "weights", "model_ir_se50.pth")
        if os.path.exists(id_weights_path):
            print(f"Loading Identity Loss weights from: {id_weights_path}")
            criterionID = IdentityLoss(id_weights_path, device)
        else:
            print(f"Warning: Identity Loss weights not found at {id_weights_path}")
            criterionID = None

    criterionCycle = CycleConsistencyLoss().to(device) if opt.use_cycle else None

    # --- [6] Training Loop ---
    print("Starting Training Loop...")
    global_step = 0

    for epoch in range(opt.epochs):
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch}/{opt.epochs}"
        )

        for i, data in progress_bar:
            # safe_collate가 None을 반환할 수 있으므로 체크
            if data is None:
                continue

            real_img = data["image"].to(device)          # (B,3,H,W)
            spade_input = data["spade_input"].to(device) # (B,4,H,W)
            target_maps = data["target_maps"].to(device) # (B,3,H,W)
            mask = data["mask"].to(device)               # (B,1,H,W)

            # --- Train Discriminator ---
            optimizerD.zero_grad()

            fake_img, _ = netG(real_img, spade_input)

            # Real Loss
            real_pair = torch.cat([real_img, spade_input], dim=1)
            pred_real = netD(real_pair)
            loss_D_real = real_img.new_tensor(0.0)
            for pred in pred_real:
                loss_D_real = loss_D_real + criterionGAN(pred, torch.ones_like(pred))

            # Fake Loss
            fake_pair = torch.cat([fake_img.detach(), spade_input], dim=1)
            pred_fake = netD(fake_pair)
            loss_D_fake = real_img.new_tensor(0.0)
            for pred in pred_fake:
                loss_D_fake = loss_D_fake + criterionGAN(pred, torch.zeros_like(pred))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()

            # --- Train Generator ---
            optimizerG.zero_grad()

            fake_pair_G = torch.cat([fake_img, spade_input], dim=1)
            pred_fake_G = netD(fake_pair_G)
            loss_G_GAN = real_img.new_tensor(0.0)
            for pred in pred_fake_G:
                loss_G_GAN = loss_G_GAN + criterionGAN(pred, torch.ones_like(pred))

            # 기본 재구성 L1 (Light 모드에서도 유지)
            loss_G_Feat = criterionFeat(fake_img, real_img) * 10.0

            # 선택적 Loss들 (옵션에 따라 계산/스킵)
            loss_G_VGG = real_img.new_tensor(0.0)
            if criterionVGG is not None:
                loss_G_VGG = criterionVGG(fake_img, real_img) * 10.0

            loss_G_ID = real_img.new_tensor(0.0)
            if criterionID is not None:
                loss_G_ID = criterionID(real_img, fake_img) * 5.0

            loss_G_Cycle = real_img.new_tensor(0.0)
            if criterionCycle is not None:
                loss_G_Cycle = criterionCycle(fake_img, target_maps, mask) * 10.0

            # 최종 Generator Loss
            loss_G = (
                loss_G_GAN
                + loss_G_Feat
                + loss_G_VGG
                + loss_G_ID
                + loss_G_Cycle
            )
            loss_G.backward()
            optimizerG.step()

            global_step += 1

            # 진행률 바에 Loss 정보 표시
            if i % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "D_Loss": f"{loss_D.item():.3f}",
                        "G_Loss": f"{loss_G.item():.3f}",
                    }
                )

            # TensorBoard Logging
            if i % 100 == 0:
                writer.add_scalar("Loss/D", loss_D.item(), global_step)
                writer.add_scalar("Loss/G", loss_G.item(), global_step)
                if criterionID is not None:
                    writer.add_scalar("Loss/ID", loss_G_ID.item(), global_step)
                if criterionVGG is not None:
                    writer.add_scalar("Loss/VGG", loss_G_VGG.item(), global_step)
                if criterionCycle is not None:
                    writer.add_scalar("Loss/Cycle", loss_G_Cycle.item(), global_step)

            # Image Saving
            if i % 500 == 0:
                # real | fake 를 가로로 붙여서 저장
                vis = torch.cat([real_img, fake_img], dim=3)
                vutils.save_image(
                    vis,
                    os.path.join(save_dir, f"epoch_{epoch}_step_{i}.png"),
                    normalize=True,
                )

        # Save Checkpoint
        if epoch % 5 == 0 or epoch == opt.epochs - 1:
            torch.save(netG.state_dict(), os.path.join(save_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(save_dir, f"netD_epoch_{epoch}.pth"))

    print("Training Finished.")
    writer.close()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="experiment_1", help="Experiment name")
    parser.add_argument(
        "--data_root", type=str, default="../data/processed", help="Path to processed data"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--img_size", type=int, default=512, help="Image resolution")

    # ---- Light / Heavy Loss 옵션 ----
    parser.add_argument(
        "--use_vgg",
        action="store_true",
        help="Use VGG perceptual loss (slow, default: off)",
    )
    parser.add_argument(
        "--use_id",
        action="store_true",
        help="Use Identity (ArcFace) loss (slow, default: off)",
    )
    parser.add_argument(
        "--use_cycle",
        action="store_true",
        help="Use Cycle Consistency loss on skin condition maps (slow, default: off)",
    )

    opt = parser.parse_args()

    print(f"Running Experiment: {opt.name}")
    print(f"Data Root: {opt.data_root}")

    try:
        train(opt)
    except Exception as e:
        print(f"Error occurred: {e}")
        input("Press Enter to exit...")
