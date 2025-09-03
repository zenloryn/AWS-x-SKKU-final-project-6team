import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load

HF_REPO = "Dragonscypher/rayz_EfficientNet_B0"
HF_FILE = "model.safetensors"  # 없으면 "pytorch_model.bin"로 교체 가능

def _partial_load_from_hf(model: nn.Module) -> int:
    """
    HF 체크포인트에서 우리 모델과 shape가 일치하는 텐서만 골라 로드.
    (멀티라벨 헤드 등 shape 다른 키는 스킵)
    return: 로드된 파라미터 개수
    """
    # 1) 체크포인트 다운로드
    weight_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    # 2) state_dict 읽기
    try:
        ckpt = safe_load(weight_path)  # safetensors
    except Exception:
        ckpt = torch.load(weight_path, map_location="cpu")  # bin일 경우

    own = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if (k in own) and (own[k].shape == v.shape)}

    # 3) 부분 로드(strict=False)
    own.update(filtered)
    model.load_state_dict(own, strict=False)
    return len(filtered)

def get_model(img_size: int):
    """
    EfficientNet-B0 + (HF X-ray 전학습 가중치 '부분 로드') + 1-로짓 헤드
    - img_size는 summary 출력용으로만 사용 (전처리에서 맞춰줌)
    - 반환: (model, device)
    """
    # torchvision B0 생성 (ImageNet weights=None: HF 가중치로 덮어쓸 예정)
    try:
        model = models.efficientnet_b0(weights=None)
    except Exception:
        # 구버전 호환
        model = models.efficientnet_b0(pretrained=False)

    # 1-로짓 분류 헤드로 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(in_features, 1))

    # HF 전학습 가중치 부분 로드 (헤드 등 shape 불일치 파트는 자동 스킵)
    loaded = _partial_load_from_hf(model)
    print(f"[HF] Loaded {loaded} tensors from {HF_REPO}/{HF_FILE} (partial)")

    # 디바이스/요약 저장 (기존 B1 모듈과 인터페이스 동일)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device