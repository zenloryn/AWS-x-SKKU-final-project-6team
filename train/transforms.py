from typing import Tuple                 
import random                           
import torch                            
import cv2
from PIL import Image                    
from torchvision import transforms      
from torchvision.transforms import functional as F  
import numpy as np

# Pillow 10+ 에서는 Image.Resampling 사용, 예전(9.x 이하)에는 기존 alias로 fallback
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

# -------------------------
# Custom small utilities
# -------------------------

class ToGrayIfNot:
    # 입력 이미지를 항상 그레이스케일(L 모드)로 통일
    def __call__(self, img: Image.Image) -> Image.Image:
        # L-mode(그레이스케일)로 변환하여 채널 불일치(JPG 3ch/PNG 1ch 등) 문제를 제거
        return img.convert("L")

class RepeatTo3ch:
    """Tensor [1,H,W] -> [3,H,W] 복제"""
    # 1채널 텐서를 3채널로 복제하여 사전학습(Imagenet) 가중치의 입력 형식과 정합
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # 이미 3채널이면 그대로 반환 (중복 연산 방지)
        if t.ndim == 3 and t.size(0) == 3:
            return t
        # [H,W] 형태(채널 차원 없음)라면 채널 차원 추가
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [H,W] -> [1,H,W]
        # [1,H,W]를 [3,H,W]로 복제
        return t.repeat(3, 1, 1)

class RandomCLAHE:
    """
    p=1.0이면 항상 적용.
    clip_limit ~2.0, tile_grid_size=(8,8) 권장.
    """
    def __init__(self, p: float = 1.0, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.p = p                              # 적용 확률 (항상 ON이면 1.0)
        self.clip_limit = clip_limit            # CLAHE 대비 제한(과도한 대비 상승 방지)
        self.tile_grid_size = tile_grid_size    # 타일 크기 (지역 대비를 계산할 블록 크기)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        x = np.array(img, dtype=np.uint8)       # PIL 이미지를 uint8 배열로 변환 (L-mode 0~255)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size)
        )
        x = clahe.apply(x)                      # CLAHE 적용 (지역 대비 향상)
        return Image.fromarray(x, mode="L")     # 다시 PIL L 모드 이미지로 반환

class ResizeShortSide:
    """짧은 변을 target으로 비율 유지 리사이즈"""
    # 예: short_target=288이면 짧은 변이 288이 되도록 스케일 조정
    def __init__(self, short_target: int):
        self.short_target = short_target

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size                                     # 원본 너비/높이
        scale = self.short_target / min(w, h)               # 짧은 변 기준 스케일 팩터
        new_w, new_h = int(round(w * scale)), int(round(h * scale))  # 스케일링된 크기
        return img.resize((new_w, new_h), RESAMPLE_BILINEAR)   # 비율 유지 리사이즈 (bilinear 보간)

class CenterCropSquare:
    """가운데에서 정사각 CenterCrop"""
    # 리사이즈 후 중앙에서 원하는 크기(예: 240x240)로 크롭
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size                      # 현재 이미지 크기
        th, tw = self.size, self.size        # 목표 크기(정사각)
        i = max(0, int(round((h - th) / 2.0)))  # 세로 기준 중앙 시작점
        j = max(0, int(round((w - tw) / 2.0)))  # 가로 기준 중앙 시작점
        # 중앙에서 목표 크기만큼 잘라냄 (경계 안전 보정을 위해 min 사용)
        img = F.crop(img, i, j, min(th, h), min(tw, w))
        # 혹시 크롭 후 크기가 약간 다를 수 있어 최종 보장 리사이즈
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), RESAMPLE_BILINEAR)
        return img

MEAN = [0.533958, 0.533958, 0.533958]   
STD  = [0.228482, 0.228482, 0.228482]  

# -------------------------
# Builders
# -------------------------

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    # 학습(Train) 단계용 변환 파이프라인 정의
    return transforms.Compose([
        ToGrayIfNot(),                                            # 입력을 항상 그레이스케일로 통일
        RandomCLAHE(p=0.5, clip_limit=1.5, tile_grid_size=8),     
        ResizeShortSide(short_target=224),                         # 짧은 변을 224로 리사이즈(비율 유지)

        transforms.RandomRotation(degrees=3, fill=0),              # ±3° 회전, 가장자리는 0(검정)으로 채움

        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),

        transforms.RandomHorizontalFlip(p=0.5),                    # 좌우 반전(폐렴 이진 분류에서 허용)
        transforms.RandomApply(                                    # 밝기/대비만 약하게 조정(과도한 왜곡 방지)
            [transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.2
        ),
        transforms.RandomApply(                                    # 가우시안 블러를 낮은 확률로 적용(노이즈 완화)
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.05
        ),

        transforms.ToTensor(),                                     # PIL → Tensor (0~1 범위, [C,H,W])
        RepeatTo3ch(),                                             # [1,H,W] → [3,H,W] 복제(프리트레인 정합)
        transforms.Normalize(MEAN,STD),       
    ])

def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    # 검증/테스트 단계용 변환 파이프라인 (증강 없음)
    return transforms.Compose([
        ToGrayIfNot(),                                            # 입력을 그레이스케일로 통일
        # RandomCLAHE(p=1.0, clip_limit=1.5, tile_grid_size=8),    
        ResizeShortSide(short_target=224),                         # 짧은 변을 224로 리사이즈
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),  

        transforms.ToTensor(),                                     # PIL → Tensor
        RepeatTo3ch(),                                             # 1채널을 3채널로 복제
        transforms.Normalize(MEAN,STD),       
    ])