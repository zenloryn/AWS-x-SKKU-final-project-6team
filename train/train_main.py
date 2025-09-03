from train.dataloader import get_dataloaders
from train.effb0_model import get_model
from train.train_utils import train_model, EarlyStopping
from train.train_test import run_test
import argparse
import torch
import os
import torch.nn as nn
import random
import numpy as np

def parse_args():
    # 하이퍼파라미터 및 이미지 크기 설정을 위한 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--run_test_after', action='store_true', help='학습 직후 test 실행')
    parser.add_argument('--use_amp', action='store_true', help='Enable mixed precision (AMP)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Grad clipping max-norm; <=0 to disable')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau'], help='LR scheduler type')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    
    return parser.parse_args()

def main():
    args = parse_args()

    args.run_test_after = args.run_test_after or os.getenv('RUN_TEST_AFTER', '0') == '1'

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True   # 재현성↑
    torch.backends.cudnn.benchmark = False      # 성능 튜너 off → 재현성↑

    # 학습/검증용 DataLoader 로드
    train_loader, val_loader, train_dataset = get_dataloaders(
        args.img_size, args.batch_size
    )

    # 모델 및 device(GPU or CPU) 준비
    model, device = get_model(args.img_size)

    # 손실함수 및 옵티마이저 정의
     # ----- 손실함수: Binary label smoothing 적용 -----
    class BCEWithLogitsLossSmooth(nn.Module):
        def __init__(self, eps: float = 0.0, pos_weight=None):
            super().__init__()
            self.eps = float(max(0.0, min(0.499, eps)))
            self.base = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def forward(self, logits, targets):
            # targets in {0,1} -> smooth toward 0.5
            if self.eps > 0:
                targets = targets * (1.0 - self.eps) + 0.5 * self.eps
            return self.base(logits, targets)
    criterion = BCEWithLogitsLossSmooth(eps=args.label_smoothing)

    # ----- 옵티마이저: AdamW + weight_decay -----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == 'plateau':
        # val_acc 최대화를 기준으로 학습률 감소
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=True
            )

    # EarlyStopping 구성: val_acc 최대화, patience=4, 최소 개선폭 0.001
    early_stopper = EarlyStopping(mode="max", patience=4, min_delta=0.001)

    use_amp = bool(args.use_amp)
    max_gn  = args.max_grad_norm if (args.max_grad_norm is not None and args.max_grad_norm > 0) else None

    # 모델 학습 시작
    train_model(
        model, device, train_loader, val_loader, train_dataset.classes,
        criterion, optimizer, args.epochs,
        early_stopping=early_stopper,
        use_amp=use_amp, max_grad_norm=max_gn, scheduler=scheduler, img_size=args.img_size
    )

    if args.run_test_after:
        print("[INFO] Running test right after training...")
        run_test(img_size=args.img_size, batch_size=args.batch_size)

if __name__ == '__main__':
    main()