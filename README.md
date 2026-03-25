# HighRes-Face-DeAging-SPADE

---

## 개발 환경 구축

```bash
conda create -n face_deaging python=3.10 -y
conda activate face_deaging

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
(설치 후 확인: python -c "import torch; print(torch.cuda.is_available())")

pip install -r requirements.txt

또는

pip install numpy pandas matplotlib scipy tqdm pyyaml
pip install opencv-python scikit-image pillow
pip install insightface onnxruntime-gpu dlib
pip install tensorboard kornia einops albumentations
```

---

## 필수 모델

1. Face Parsing (BiSeNet): 피부 영역 추출용
   다운로드: [zllrunning/face-parsing.PyTorch GitHub](https://github.com/zllrunning/face-parsing.PyTorch)
   파일명: 79999_iter.pth

2. Race Classifier (FairFace)
   다운로드: [dchen236/FairFace GitHub](https://github.com/dchen236/FairFace)
   파일명: res34_fair_align_multi_7_20190809.pt

4. ArcFace: 정체성 보존 손실(Identity Loss) 계산용
   다운로드: [Trezor/InsightFace_Pytorch GitHub](https://github.com/TreB1eN/InsightFace_Pytorch)
   파일명: model_ir_se50.pth
   
**다운로드 받은 모델들을 루트 폴더 아래 weights 폴더 생성 후 넣어두기**

---

## 디렉토리 구조

```plain text
HighRes-Face-DeAging-SPADE/
│
└── src/
│   # [Main Code] 작성한 모든 소스 코드
│   ├── preprocess.py      # [Phase 2] 전처리 및 조건지도 생성 스크립트
│   ├── networks.py        # [Phase 3] SPADE 생성기 및 판별기 아키텍처
│   ├── losses.py          # [Phase 3] Cycle, Identity, VGG 손실 함수
│   ├── dataset.py         # [Phase 4] 데이터 로더 (Dataset Class)
│   ├── train.py           # [Phase 4] 모델 학습 실행 스크립트
│   ├── test.py            # [Phase 5] 추론 및 결과 이미지 생성 (시각화)
│   ├── evaluate.py        # [Phase 5] 정량적 평가 (Attr-MAE 계산)
│   └── test_arch.py       # [Phase 5] 네트워크 구조 테스트
├── requirements.txt       # [Phase 1] 필수 라이브러리 목록
├── data/                  # [Phase 2] 데이터 저장소
│   ├── raw/               # (Input) 원본 이미지 (AI-Hub, FFHQ 등)
│   ├── processed/         # (Output) 전처리 완료된 데이터
│   │   ├── images/        # 정렬된 얼굴 이미지
│   │   └── maps/          # 4채널 조건지도 (.npy)
│   └── checkpoints/       # (Train Output) 학습된 모델 가중치 (.pth) 및 로그
└── weights/               # [사전 준비] 다운로드 받은 Pretrained 모델들
    ├── 79999_iter.pth                    # BiSeNet (Face Parsing)
    ├── res34_fair_align_multi_7_20190809.pt # FairFace (Race Filter)
    └── model_ir_se50.pth                 # ArcFace (Identity Loss)
```
---

## 실행 가이드

- 모든 스크립트는 프로젝의 `src` 폴더 내부로 이동한 뒤 실행하는 것을 권장
```bash
cd HighRes-Face-DeAging-SPADE/src
```

1. 데이터 전처리 (Preprocessing)
- 원본 이미지를 입력받아 얼굴을 크롭하고, 주름/모공/홍조/피부마스크를 추출하여 학습용 `.npy` 맵과 이미지 생성
```bash
python preprocess.py --raw_dir ../data/raw --out_dir ../data/processed --save_debug
```

2. 데이터 품질 검증 (Quality Control)
- 전처리가 완료된 데이터 중, 마스크 영역이 너무 작거나 조건지도가 잘못 추출된 불량 데이터 필터링
```bash
python qc_dataset.py --data_root ../data/processed --report_csv ../data/qc_report.csv --action move
```

3. 모델 학습 (Training)
```bash
# 기본 학습 (배치 8, 256x256 해상도)
python train.py --name face_deaging_run_1 --batch_size 8 --img_size 256 --use_vgg --use_id --use_cycle
```

---

## 시각화 및 디에이징 추론 (Inference / Testing)

### 속성 제어 (`inference.py`) - 권장
- 특정 이미지 1장에 대해 홍조, 주름, 모공의 강도를 개별적인 수치로 제어하고, 피부 합성 지원
```bash
python inference.py \
    --checkpoint ../data/checkpoints/face_deaging_run_1/netG_epoch_100.pth \
    --data_root ../data/processed \
    --image_name test_target.png \
    --wrinkle_factor 0.2 \
    --pore_factor 0.3 \
    --redness_factor 0.5 \
    --blend_skin
```

### 전체 속성 제거 및 기복 복원 테스트 (`test.py`)
- 특정 채널 값을 0으로 일괄 설정하여 제어 결과 시각화
```bash
python test.py \
    --checkpoint ../data/checkpoints/face_deaging_run_1/netG_epoch_100.pth \
    --img_dir ../data/processed/images \
    --map_dir ../data/processed/maps \
    --output_dir ../results
```

### 정량적 평가 (Evaluation)
- 복원된 이미지에서 추출한 특징(홍조/주름/모공)과 목표 지도(Target Map) 간의 MAE(Mean Absolute Error)를 계산
```bash
python evaluate.py \
    --checkpoint ../data/checkpoints/face_deaging_run_1/netG_epoch_100.pth \
    --data_root ../data/processed \
    --img_size 256
```
---
