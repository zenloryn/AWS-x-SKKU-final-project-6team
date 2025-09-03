import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import json

# SageMaker 표준 모델 디렉토리
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CKPT_DIR  = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints") 
REPORTS_DIR = os.path.join(CKPT_DIR, "reports-train")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

BEST_STATE_NAME = "best_state_dict.pth"  # 재학습/검증용
BEST_TS_NAME    = "best_model.pt"        # 배포/추론용 TorchScript

def get_best_ts_path() -> str:
    """TorchScript 저장/로딩 경로 (/opt/ml/model/...)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, BEST_TS_NAME)

def get_best_state_path() -> str:
    """베스트 state_dict 저장/로딩 경로 (/opt/ml/checkpoints/...)"""
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, BEST_STATE_NAME)

def validate(model, val_loader, criterion, device):
    # 검증 루프 실행
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            
            # ⬇️ 방어: list → Tensor, dtype/shape 고정
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float32)
            else:
                labels = labels.to(torch.float32)

            # BCEWithLogitsLoss에 맞춰 [B] 또는 [B,1] → [B]로 사용
            if labels.ndim > 1:
                labels = labels.view(-1)

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            logits = outputs.squeeze(1)                # [N,1] -> [N]
            loss = criterion(logits, labels.float())   # 타깃 float으로

            val_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)              # 동일 logits 사용
            predicted = (probs >= 0.5).long()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    val_loss /= total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_model(model, device, train_loader, val_loader, classes, criterion, optimizer, num_epochs, early_stopping=None, use_amp=False, max_grad_norm=None, scheduler=None, img_size: int = 224):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 각 에폭별 손실/정확도 기록용 리스트
    best_val_acc = -1.0                               # 지금까지의 최고 검증 정확도(초기에는 아주 작은 값으로 설정)

    # 단일 소스 경로 사용
    ts_path    = get_best_ts_path()
    state_path = get_best_state_path()
    os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (device.type == 'cuda')))

    for epoch in range(num_epochs):                   # 전체 에폭 반복
        model.train()                                 # 모델을 학습 모드로 전환(드롭아웃/BN 등 학습 동작)
        running_loss, correct, total = 0.0, 0, 0      # 에폭 동안 누적 손실/정답 수/샘플 수 초기화

        for images, labels in train_loader:           # 배치 단위로 학습 데이터 반복
            
            # ⬇️ 방어: list → Tensor, dtype/shape 고정
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.float32)
            else:
                labels = labels.to(torch.float32)
            if labels.ndim > 1:
                labels = labels.view(-1)

            images, labels = images.to(device), labels.to(device)  # GPU/CPU 등 디바이스로 텐서 이동
            optimizer.zero_grad()                     # 이전 스텝의 기울기(gradient) 초기화

            with torch.cuda.amp.autocast(enabled=(use_amp and (device.type == 'cuda'))):
                outputs = model(images)
                logits = outputs.squeeze(1)
                loss = criterion(logits, labels.float())

            if use_amp and (device.type == 'cuda'):
                scaler.scale(loss).backward()
                # 클리핑은 unscale 뒤에
                scaler.unscale_(optimizer)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            running_loss += loss.item() * images.size(0)  # 배치 손실 × 배치 크기 → 누적(평균 내기 위함)

            probs = torch.sigmoid(logits)              # 동일 logits 사용
            predicted = (probs >= 0.5).long()

            correct += (predicted == labels).sum().item()  # 정답 개수 누적
            total += labels.size(0)                   # 처리한 샘플 수 누적

        train_loss = running_loss / total             # 에폭 평균 학습 손실
        train_acc = correct / total                   # 에폭 평균 학습 정확도

        # 현재(마지막 업데이트된) 모델로 검증 수행 → 손실/정확도만 사용(라벨/예측은 베스트 평가에서 다시 얻을 것)
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss); val_losses.append(val_loss)  # 로그용 리스트에 추가
        train_accs.append(train_acc);   val_accs.append(val_acc)

        cur_lr = optimizer.param_groups[0].get("lr", None)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val_Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}"
              + (f", LR: {cur_lr:.2e}" if cur_lr is not None else ""))   # 진행 상황 출력

        # 베스트 검증 정확도 갱신 시: 단일 경로로 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), state_path)
            print(f"[Checkpoint] Best updated: Val_Acc={best_val_acc:.4f} -> {state_path}")

        # 스케줄러 step (Plateau는 val_acc 기준)
        if scheduler is not None:
            try:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            except Exception as e:
                print(f"[Scheduler][WARN] step failed: {e}")

        # EarlyStopping 훅: 개선 없으면 patience 후 중단
        if early_stopping is not None:                               # 조기 종료 사용 시
            early_stopping.step(val_acc)                             # 이번 에폭의 val_acc로 업데이트
            if early_stopping.should_stop:                           # 중단 신호가 켜졌다면
                print(f"[EarlyStopping] Stop triggered. Best={early_stopping.best:.4f}")  # 알림 출력
                break                                                # 학습 루프 종료

    # 모든 에폭 종료 후: 베스트 체크포인트를 불러와 '베스트 모델' 기준으로 최종 리포트 산출
    model.load_state_dict(torch.load(state_path, map_location=device))  # 단일 경로에서 로드
    model.to(device)                                    # 디바이스로 이동(안전 차원에서 재확인)
    _, _, all_labels, all_preds, all_probs = validate(model, val_loader, criterion, device)  # 베스트 모델로 다시 검증해 라벨/예측 수집

    # === PR 곡선 기반 F1 최대 임계값 탐색 & 저장 ===
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    # thresholds 길이는 precisions/recalls보다 1 짧음 주의
    best_idx = int(np.nanargmax(f1s)) if f1s.size else 0
    best_thresh = float(thresholds[max(best_idx - 1, 0)]) if thresholds.size else 0.5
    with open(os.path.join(MODEL_DIR, "val_selected_threshold.json"), "w") as f:
        json.dump(
            {
                "best_threshold": best_thresh,
                "val_best_f1": float(np.nanmax(f1s)) if f1s.size else None,
                "val_precision_at_best": float(precisions[best_idx]) if f1s.size else None,
                "val_recall_at_best": float(recalls[best_idx]) if f1s.size else None,
            },
            f,
            indent=2
        )

    # 혼동행렬/지표/곡선 등 리포트 생성 함수 호출(베스트 모델 기준의 결과를 전달)
    save_and_upload_results(
        train_accs, val_accs, train_losses, val_losses,  # 에폭별 로그(학습/검증 손실·정확도)
        all_labels, all_preds, classes                   # 베스트 모델에서 얻은 검증 셋 라벨/예측/클래스명
    )
    
    # === Export: Best state_dict -> TorchScript (best_model.pth) ===
    try:
        model_cpu = model.to('cpu').eval()
        example = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            scripted = torch.jit.trace(model_cpu, example)
        scripted.save(ts_path)   # /opt/ml/model/best_model.pt

        print(f"[Export] TorchScript saved to {ts_path} (will be packaged into model.tar.gz)")
        print(f"[Export] Kept best state_dict at {state_path} (synced via checkpoint_s3_uri).")
    except Exception as e:
        print(f"[Export][WARN] TorchScript export failed: {e}")

def save_and_upload_results(train_accs, val_accs, train_losses, val_losses, all_labels, all_preds, classes):
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1) 정확도/손실 곡선
    for name, values in zip(['acc', 'loss'], [[train_accs, val_accs], [train_losses, val_losses]]):
        plt.figure()
        plt.plot(values[0], label='Train'); plt.plot(values[1], label='Val')
        plt.xlabel('Epoch'); plt.ylabel(name.capitalize()); plt.legend()
        out_png = os.path.join(PLOTS_DIR, f"{name}_curve.png")
        plt.savefig(out_png, format='png', bbox_inches='tight'); plt.close()

    # 2) 혼동행렬
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues); plt.title('Confusion Matrix')
    out_cm = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out_cm, format='png', bbox_inches='tight'); plt.close()

    # 3) 성능지표 CSV
    metrics_df = pd.DataFrame({
        'accuracy':  [accuracy_score(all_labels, all_preds)],
        'precision': [precision_score(all_labels, all_preds, zero_division=0)],
        'recall':    [recall_score(all_labels, all_preds, zero_division=0)],
        'f1_score':  [f1_score(all_labels, all_preds, zero_division=0)],
    })
    out_csv = os.path.join(METRICS_DIR, "metrics.csv")
    metrics_df.to_csv(out_csv, index=False)

    print(f"[CHECKPOINTS] Reports saved under: {REPORTS_DIR}")
    print("[CHECKPOINTS] With estimator.checkpoint_s3_uri set, these will sync to your S3 checkpoints prefix.")

class EarlyStopping:
    """
    Early stopping on a monitored metric (default: maximize val_acc).
    - mode: "max" or "min"              # 최대화/최소화 모드
    - patience: epochs to wait           # 개선 없을 때 허용 에폭 수
    - min_delta: minimum improvement     # 개선으로 간주할 최소 변화량
    """
    def __init__(self, mode: str = "max", patience: int = 7, min_delta: float = 0.0):
        assert mode in ("max", "min")        # 모드 유효성 체크
        self.mode = mode                     # 모드 저장
        self.patience = patience             # 인내심(개선 없을 때 몇 에폭 기다릴지)
        self.min_delta = min_delta           # 최소 개선 폭
        self.best = None                     # 최고(또는 최저) 값 저장
        self.num_bad = 0                     # 연속 개선 실패 횟수
        self.should_stop = False             # 중단 플래그

    def step(self, value: float):
        # 모니터링 값(예: val_acc)을 한 에폭마다 입력받아 상태 업데이트
        if self.best is None:                # 첫 호출이면 현재 값을 베스트로
            self.best = value
            self.num_bad = 0
            return

        # 개선 판단: mode=max면 더 커져야, mode=min이면 더 작아져야 개선
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)

        if improved:                         # 개선되면
            self.best = value                # 베스트 갱신
            self.num_bad = 0                 # 실패 카운터 리셋
        else:                                # 개선 없으면
            self.num_bad += 1                # 실패 카운터 증가
            if self.num_bad >= self.patience:# 실패가 patience 이상 누적되면
                self.should_stop = True      # 조기 종료 신호 설정