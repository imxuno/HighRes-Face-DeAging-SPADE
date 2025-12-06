# HighRes-Face-DeAging-SPADE

---

### 개발 환경 구축

```bash
conda create -n face_deaging python=3.10 -y
conda activate face_deaging
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
(설치 후 확인: python -c "import torch; print(torch.cuda.is_available())")
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
3. ArcFace
   다운로드: [Trezor/InsightFace_Pytorch GitHub](https://github.com/TreB1eN/InsightFace_Pytorch)

### 실행 가이드

1. 파일 저장: `dataset.py`, `train.py`를 프로젝트 루트 폴더에 저장
2. 데이터 확인: `data/processed/images`와 `data/processed/maps`에 `prprocess.py`로 생성한 데이터가 있는지 확인
3. 폴더 이동

```bash
cd HighRes-Face-DeAging-SPADE/src
```

4. 학습 시작

```bash
# 기본 학습 (배치 사이즈 4, 512x512 해상도) → 1024px 학습은 VRAM을 많이 먹음
python train.py --name my_first_run --batch_size 4 --img_size 512
```

---

- 시각적 결과 확인

```bash
# results 폴더에 원본, 주름제거, 모공제거, 홍조제거 이미지 4장이 생성됨
python test.py --checkpoint ./data/checkpoints/my_first_run/netG_epoch_100.pth
```

- 수치적 성능 확인

```bash
# MAE 점수
python evaluate.py --checkpoint ./data/checkpoints/my_first_run/netG_epoch_100.pth
```

- FID / LPIPS 계산
  설치: pip install pytorch-fid lpips
  사용법:
  test.py를 수정하여 테스트셋 전체의 **복원 이미지(Reconstruction)**를 한 폴더(results/fakes)에 저장
  원본 이미지 폴더(data/processed/images)와 비교
  FID: python -m pytorch_fid ./data/processed/images ./results/fakes
  LPIPS: lpips 라이브러리 예제 코드를 사용하여 두 폴더 간 거리를 계산

---

### 디렉토리 구조

```bash
HighRes-Face-DeAging-SPADE/
│
├── requirements.txt       # [Phase 1] 필수 라이브러리 목록
│
├── weights/               # [사전 준비] 다운로드 받은 Pretrained 모델들
│   ├── 79999_iter.pth                    # BiSeNet (Face Parsing)
│   ├── res34_fair_align_multi_7_20190809.pt # FairFace (Race Filter)
│   └── model_ir_se50.pth                 # ArcFace (Identity Loss)
│
├── data/                  # [Phase 2] 데이터 저장소
│   ├── raw/               # (Input) 원본 이미지 (AI-Hub, FFHQ 등)
│   ├── processed/         # (Output) 전처리 완료된 데이터
│   │   ├── images/        # 정렬된 얼굴 이미지
│   │   └── maps/          # 3채널 조건지도 (.npy)
│   └── checkpoints/       # (Train Output) 학습된 모델 가중치 (.pth) 및 로그
│
└── src/                   # [Main Code] 작성한 모든 소스 코드
    ├── preprocess.py      # [Phase 2] 전처리 및 조건지도 생성 스크립트
    ├── networks.py        # [Phase 3] SPADE 생성기 및 판별기 아키텍처
    ├── losses.py          # [Phase 3] Cycle, Identity, VGG 손실 함수
    ├── dataset.py         # [Phase 4] 데이터 로더 (Dataset Class)
    ├── train.py           # [Phase 4] 모델 학습 실행 스크립트
    ├── test.py            # [Phase 5] 추론 및 결과 이미지 생성 (시각화)
    ├── evaluate.py        # [Phase 5] 정량적 평가 (Attr-MAE 계산)
    └── test_arch.py       # [Phase 5] 네트워크 구조 테스트
```
