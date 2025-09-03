import os, json
import torch
import pandas as pd
from train.dataloader import get_dataloaders_test
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경(SageMaker/서버)에서 그림 저장용
import matplotlib.pyplot as plt
# 단일 소스 경로 헬퍼 / 베스트 가중치 경로
from train.train_utils import get_best_state_path
# 현재 사용하는 모델 팩토리로 교체
from train.effb0_model import get_model

CKPT_DIR  = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints") 
REPORTS_DIR = os.path.join(CKPT_DIR, "reports-test")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

@torch.no_grad()
def run_test(
    img_size,
    batch_size,
):
    """
    최고 성능 가중치로 Test 실행.
    - img_size: 전처리(크롭 사이즈) 및 summary와 일치
    - threshold: None이면 저장된 파일을 우선 로드, 없으면 0.5
    - job_name: (선택) 실험명/잡명으로 결과 저장 폴더를 분리
    """
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1) 모델 준비
    model, device = get_model(img_size)

    # 2) 가중치 로드(학습과 동일 경로 헬퍼)
    weights_path = get_best_state_path()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Best weights not found: {weights_path}\n"
            f"- 학습 후 저장된 가중치가 맞는지 확인하세요."
        )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[TEST] Loaded weights: {weights_path}")

    # 3) 데이터 준비 (경로 고정)
    # 테스트용 DataLoader 로드
    test_loader = get_dataloaders_test(
        img_size, batch_size
    )

    # 4) 추론/수집
    all_labels, all_probs = [], []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logits = model(x)                         # [B,1]
        probs = torch.sigmoid(logits).squeeze(1)  # [B]
        all_labels.append(y.long())
        all_probs.append(probs)

    y_true = torch.cat(all_labels).cpu().numpy()
    p_hat = torch.cat(all_probs).cpu().numpy()
    # 저장된 임계값(있으면) 우선 적용, 없으면 0.5
    best_thresh = 0.5
    thr_path = os.path.join("/opt/ml/model", "val_selected_threshold.json")
    if os.path.exists(thr_path):
        try:
            best_thresh = float(json.load(open(thr_path))["best_threshold"])
        except Exception:
            pass
    y_pred = (p_hat >= best_thresh).astype("int64")

    # 6) 지표 계산
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # 안전 처리: 양/음성 모두 존재할 때만 ROC/PR 기반 지표 계산
    auroc = None
    auprc = None
    try:
        if len(set(y_true)) > 1:
            auroc = roc_auc_score(y_true, p_hat)
            auprc = average_precision_score(y_true, p_hat)
    except Exception:
        pass

    # Precision/Recall/F1 (threshold 기반) — 불안정 시 0으로 대체
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"[TEST] Accuracy: {acc:.4f}")
    if auroc is not None:
        print(f"[TEST] AUROC:   {auroc:.4f}")
    if auprc is not None:
        print(f"[TEST] AUPRC:   {auprc:.4f}")
    print(f"[TEST] Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(f"[TEST] Confusion Matrix:\n{cm}")

    # === S3 업로드 (테스트 산출물) ===
    # 7) 성능지표 CSV 저장
    metrics_df_test = pd.DataFrame({
        "weights_path": [weights_path],
        "img_size":     [img_size],
        "batch_size":   [batch_size],
        "best_threshold": [float(best_thresh)],
        "accuracy":     [float(acc)],
        "auroc":        [float(auroc) if auroc is not None else None],
        "auprc":        [float(auprc) if auprc is not None else None],
        "precision":    [float(prec)],
        "recall":       [float(rec)],
        "f1":           [float(f1)],
        "num_samples":  [int(len(test_loader.dataset))],
        "classes":      ["|".join(map(str, getattr(test_loader.dataset, "classes", ["0", "1"])))]
    })
    out_csv_test = os.path.join(METRICS_DIR, "metrics.csv")
    metrics_df_test.to_csv(out_csv_test, index=False)

    # 8) 혼동행렬 그림 저장
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(test_loader.dataset.classes)),
        yticks=range(len(test_loader.dataset.classes)),
        xticklabels=test_loader.dataset.classes,
        yticklabels=test_loader.dataset.classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # 값 표시
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    out_cm_test = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out_cm_test, format='png', bbox_inches='tight')
    plt.close(fig)

    print(f"[CHECKPOINTS] Test reports saved under: {REPORTS_DIR}")
    print("[CHECKPOINTS] With estimator.checkpoint_s3_uri set, these will sync to your S3 checkpoints prefix.")