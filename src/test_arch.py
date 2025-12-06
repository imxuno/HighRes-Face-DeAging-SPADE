##### test_arch.py
### networks.py와 losses.py 테스트 파일.
### 검증: 아래 코드로 네트워크가 정상적으로 생성되는지 테스트
import torch
from networks import SPADEGenerator, MultiscaleDiscriminator

def test():
    # 배치 1, 3채널 이미지, 1024x1024
    img = torch.randn(1, 3, 1024, 1024)
    # 조건지도: 3채널 텍스처 + 19채널 파싱 마스크 = 22채널
    seg = torch.randn(1, 22, 1024, 1024)

    netG = SPADEGenerator(input_nc=3, label_nc=22)
    netD = MultiscaleDiscriminator(input_nc=3+22)

    # Generator Test
    fake_img, z = netG(img, seg)
    print(f"Generator Output: {fake_img.shape}") # Should be (1, 3, 1024, 1024)

    # Discriminator Test
    d_input = torch.cat([fake_img, seg], dim=1)
    pred = netD(d_input)
    print("Discriminator Check Passed")

if __name__ == "__main__":
    test()
