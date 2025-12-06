# HighRes-Face-DeAging-SPADE

---

### 개발 환경 구축

```bash
conda create -n face_deaging python=3.10 -y
conda activate face_deaging
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
(python -c "import torch; print(torch.cuda.is_available())")
pip install numpy pandas matplotlib scipy tqdm pyyaml
pip install opencv-python scikit-image pillow
pip install insightface onnxruntime-gpu
pip install tensorboard kornia einops
pip install -r requirements.txt
```

### 필수 모델

1. Face Parsing (BiSeNet)
   다운로드: [zllrunning/face-parsing.PyTorch GitHub](https://github.com/zllrunning/face-parsing.PyTorch)
2. Race Classifier (FairFace)
   다운로드: [dchen236/FairFace GitHub](https://github.com/dchen236/FairFace)

HighRes-Face-DeAging-SPADE/
│
├── data/ # 데이터셋 저장소 (용량 문제로 .gitignore 처리 필수)
│ ├── raw/ # AI-Hub/FFHQ 원본
│ ├── processed/ # 전처리 완료 (이미지, 마스크, 조건지도)
│ └── test_samples/ # 논문용 테스트 이미지
│
├── baselines/ # 비교 모델 (Submodule 또는 코드 복사)
│ ├── cyclegan/
│ ├── stargan_v2/
│ └── stylegan2/
│
├── core/ # 본 연구 핵심 코드
│ ├── networks.py # SPADE Generator, Discriminator
│ ├── losses.py # Cycle, ID, VGG Loss
│ └── feature_extract.py # Gabor, DoG, Guided Filter (전처리용)
│
├── utils/ # 유틸리티
│ ├── face_parsing/ # BiSeNet 관련
│ ├── race_classifier/ # FairFace 관련
│ └── metrics.py # FID, LPIPS, MAE 계산 코드
│
├── train.py # 메인 학습 코드
├── test.py # 추론 및 이미지 생성 코드
├── preprocess_data.py # 데이터 전처리 실행 스크립트
└── requirements.txt
