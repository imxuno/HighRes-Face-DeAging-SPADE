import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        # 입력 채널을 출력 채널로 변환하는 Conv 추가
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat + feat_atten
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        
        # --- Context Path (ResNet18) ---
        self.resnet = models.resnet18(weights=None)
        
        # --- Spatial Path ---
        self.sp_conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.sp_conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.sp_conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)

        # --- ARM (Channel Reduction: 256/512 -> 128) ---
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        
        # --- Heads (Input 128 -> Output 128) ---
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        
        # Spatial Path Projection (64 -> 128)
        self.conv_s8 = ConvBNReLU(64, 128, ks=3, stride=1, padding=1)
        
        # --- FFM (128 + 128 -> 256) ---
        self.ffm = FeatureFusionModule(256, 256)
        
        # --- Output Layer ---
        self.conv_out = nn.Sequential(
            ConvBNReLU(256, 64, ks=3, stride=1, padding=1),
            nn.Conv2d(64, n_classes, kernel_size=1, bias=False)
        )
        
        # Aux Heads (For weight compatibility)
        self.conv_out16 = nn.Sequential(
            ConvBNReLU(128, 64, ks=3, stride=1, padding=1),
            nn.Conv2d(64, n_classes, kernel_size=1, bias=False)
        )
        self.conv_out32 = nn.Sequential(
            ConvBNReLU(128, 64, ks=3, stride=1, padding=1),
            nn.Conv2d(64, n_classes, kernel_size=1, bias=False)
        )

        self.init_weights()

    def forward(self, x):
        # Original Size
        H, W = x.size()[2:]
        
        # --- Spatial Path ---
        sp8 = self.sp_conv1(x)
        sp8 = self.sp_conv2(sp8)
        sp8 = self.sp_conv3(sp8)
        
        # --- Context Path ---
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        feat8 = self.resnet.layer1(x)
        feat16 = self.resnet.layer2(feat8)  # 128
        feat32 = self.resnet.layer3(feat16) # 256
        feat_final = self.resnet.layer4(feat32) # 512
        
        # --- ARM ---
        feat_arm32 = self.arm32(feat_final) # 512 -> 128
        feat_arm16 = self.arm16(feat32)     # 256 -> 128
        
        # --- Feature Fusion ---
        global_context = F.adaptive_avg_pool2d(feat_arm32, 1)
        feat_arm32 = feat_arm32 * global_context # This step is implied in some implementations
        
        # Upsampling
        feat32_up = F.interpolate(self.conv_head32(feat_arm32), size=feat_arm16.shape[2:], mode='nearest')
        feat16_up = F.interpolate(self.conv_head16(feat_arm16 + feat32_up), size=sp8.shape[2:], mode='nearest')
        
        feat_s8 = self.conv_s8(sp8)
        
        # FFM
        feat_fusion = self.ffm(feat_s8, feat16_up)
        
        # Output
        out = self.conv_out(feat_fusion)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        return [out]

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def load_state_dict(self, state_dict, strict=True):
        # Key Mapping for 79999_iter.pth
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            if k.startswith('cp.'): k = k[3:]
            new_state_dict[k] = v
        super().load_state_dict(new_state_dict, strict=False)