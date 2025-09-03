# AWS x SKKU Final Project - Team 6
## X-ray 이미지 기반 폐렴 진단 AI 모델

이 프로젝트는 AWS SageMaker를 활용하여 X-ray 이미지로부터 폐렴을 진단하는 딥러닝 모델을 개발하고 배포하는 프로젝트입니다.

## 🎯 프로젝트 개요

- **목적**: X-ray 이미지를 분석하여 정상/폐렴을 이진 분류하는 AI 모델 개발
- **모델**: EfficientNet-B0 기반 전이학습 모델
- **플랫폼**: AWS SageMaker를 통한 클라우드 기반 학습 및 배포
- **데이터**: X-ray 이미지 데이터셋 (정상/폐렴 이진 분류)

## 🏗️ 모델 아키텍처

- **백본**: EfficientNet-B0 (Hugging Face Hub의 X-ray 전학습 가중치 사용)
- **분류기**: 1-로짓 이진 분류 헤드 (Dropout 0.3 + Linear)
- **손실함수**: BCEWithLogitsLoss with Label Smoothing
- **옵티마이저**: AdamW with Weight Decay
- **스케줄러**: ReduceLROnPlateau (검증 정확도 기준)

## 📁 프로젝트 구조

```
├── config.py                 # AWS 설정 및 하이퍼파라미터
├── requirements.txt          # Python 패키지 의존성(key 값 제외된 상태)
├── sm_jobs/
│   └── sagemaker_train.py   # SageMaker 학습 작업 설정
└── train/
    ├── dataloader.py        # 데이터 로더 및 전처리
    ├── effb0_model.py       # EfficientNet-B0 모델 정의
    ├── train_main.py        # 메인 학습 스크립트
    ├── train_test.py        # 테스트 및 평가 스크립트
    ├── train_utils.py       # 학습 유틸리티 (EarlyStopping, 검증 등)
    └── transforms.py        # 이미지 전처리 및 증강
```

## 🔧 주요 기능

### 이미지 전처리 (`transforms.py`)
- **그레이스케일 변환**: 입력 이미지를 L 모드로 통일
- **CLAHE**: 지역 대비 향상 (50% 확률 적용)
- **리사이즈**: 짧은 변을 224px로 조정 (비율 유지)
- **데이터 증강**: 회전, 크롭, 플립, 색상 조정, 가우시안 블러
- **정규화**: ImageNet 통계 기반 정규화

### 모델 (`effb0_model.py`)
- Hugging Face Hub에서 X-ray 전학습 가중치 부분 로드
- 1-로짓 이진 분류 헤드로 교체
- TorchScript 모델 내보내기 지원

### 학습 (`train_main.py`)
- Mixed Precision Training (AMP) 지원
- Gradient Clipping
- Early Stopping
- 학습률 스케줄링
- 재현 가능한 결과를 위한 시드 설정

### 평가 (`train_test.py`)
- 다양한 메트릭 계산 (Accuracy, AUROC, AUPRC, F1, Precision, Recall)
- 혼동행렬 시각화
- 최적 임계값 탐색 및 저장

## 🚀 사용법

### 1. 환경 설정

```bash
# Python 3.9+ 환경에서
pip install -r requirements.txt
```

### 2. 로컬 학습

```bash
python train/train_main.py \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --img_size 224 \
    --use_amp \
    --scheduler plateau
```

### 3. SageMaker 학습

```bash
python sm_jobs/sagemaker_train.py
```

### 4. 테스트 실행

```bash
python train/train_test.py
```

## ⚙️ 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `epochs` | 100 | 학습 에폭 수 |
| `batch_size` | 32 | 배치 크기 |
| `learning_rate` | 1e-4 | 학습률 |
| `img_size` | 224 | 이미지 크기 |
| `weight_decay` | 1e-4 | 가중치 감쇠 |
| `label_smoothing` | 0.05 | 라벨 스무딩 |
| `max_grad_norm` | 1.0 | 그래디언트 클리핑 |

## 📊 모니터링 및 로깅

- **SageMaker 체크포인트**: 학습 중 모델 상태 자동 저장
- **메트릭 시각화**: 학습/검증 손실 및 정확도 곡선
- **성능 지표**: CSV 형태로 상세 메트릭 저장
- **혼동행렬**: 시각화된 분류 결과

## 🔧 AWS 설정

### 필수 환경 변수
- `SM_CHANNEL_TRAIN`: 학습 데이터 S3 경로
- `SM_CHANNEL_VAL`: 검증 데이터 S3 경로  
- `SM_CHANNEL_TEST`: 테스트 데이터 S3 경로
- `SM_MODEL_DIR`: 모델 저장 경로
- `SM_CHECKPOINT_DIR`: 체크포인트 저장 경로

### S3 버킷 구조
```
s3://your-bucket/
├── x-ray-v2/
│   ├── train/
│   ├── val/
│   └── test/
├── sagemaker-output/
├── sagemaker-code/
└── sagemaker-checkpoints/
```

## 📋 의존성

### 핵심 패키지
- `torch==1.13.1` - PyTorch 프레임워크
- `torchvision==0.14.1` - 컴퓨터 비전 유틸리티
- `huggingface_hub>=0.22.0` - HF 모델 다운로드
- `safetensors>=0.4.0` - 안전한 모델 저장

### 데이터 처리
- `numpy>=1.21.0` - 수치 계산
- `pandas>=1.3.0` - 데이터 조작
- `opencv-python>=4.5.0` - 이미지 처리
- `pillow>=10.0.0` - 이미지 I/O

### 시각화 및 평가
- `matplotlib>=3.4.0` - 플롯 생성
- `scikit-learn>=1.0.0` - 메트릭 계산
- `seaborn>=0.11.0` - 통계 시각화

### AWS 통합
- `boto3>=1.20.0` - AWS SDK
- `sagemaker` - SageMaker SDK

## 🎯 성능 목표

- **정확도**: 90% 이상
- **AUROC**: 0.95 이상
- **F1-Score**: 0.90 이상

## 📝 라이선스

이 프로젝트는 AWS x SKKU Final Project의 일부입니다.

## 👥 팀 정보

- **팀**: Team 6
- **기관**: AWS x SKKU
- **프로젝트**: MIMIC 데이터를 활용한 폐렴 환자의 급성 폐렴 조기 감지 모델

