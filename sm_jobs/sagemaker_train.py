import os
import sys
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import time
from config import *
from sagemaker.inputs import TrainingInput

def create_training_job():
    """SageMaker Training Job 생성 및 실행"""
    # SageMaker 세션 초기화
    # config.py에서 import한 session을 그대로 사용
    sagemaker_session = sagemaker.Session(boto_session=session)
    # 실행 역할 설정
    try:
        role = get_execution_role()
    except ValueError:
        role = SAGEMAKER_ROLE
    # 하이퍼파라미터
    hyperparameters = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'img_size': 224
    }
    # PyTorch Estimator 생성
    estimator = PyTorch(
        entry_point="train/train_main.py",
        source_dir=".",
        role=role,
        framework_version='1.13',
        py_version='py39',
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        hyperparameters=hyperparameters,
        output_path=f's3://{S3_BUCKET}/sagemaker-output/',
        code_location=f's3://{S3_BUCKET}/sagemaker-code/',
        checkpoint_s3_uri=f"s3://{S3_BUCKET}/sagemaker-checkpoints/{TRAINING_JOB_NAME}/",
        sagemaker_session=sagemaker_session,
        max_run=12*60*60,  # 최대 12시간
        keep_alive_period_in_seconds=1800,
        debugger_hook_config=False,
        disable_profiler=True,
        environment={
            'SAGEMAKER_PROGRAM': 'train/train_main.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
            'SAGEMAKER_REGION': SAGEMAKER_REGION,
            'CUDA_VISIBLE_DEVICES': '0',
            'NVIDIA_VISIBLE_DEVICES': '0',
            'RUN_TEST_AFTER': '1'
        },
        tags=[{'Key': 'project', 'Value': 'pre-6team'}]
    )
    # 데이터 입력 설정
    inputs = {
        "train": TrainingInput(
            s3_data=f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}train/",
            input_mode="FastFile"
        ),
        "val": TrainingInput(
            s3_data=f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}val/",
            input_mode="FastFile"
        ),
        "test": TrainingInput(
            s3_data=f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}test/",
            input_mode="FastFile"
        )
    }
    # Training Job 시작
    print(f"Starting SageMaker Training Job: {TRAINING_JOB_NAME}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"S3 train data path: {inputs['train']}")
    print(f"S3 val data path: {inputs['val']}")
    print("\n[안내] 만약 'No S3 objects found' 또는 'not authorized to perform: s3:ListBucket' 에러가 발생하면:")
    print("  1. SageMaker ExecutionRole에 s3:ListBucket, s3:GetObject 권한이 있는지 확인하세요.")
    print("  2. S3 경로에 실제 데이터가 존재하는지 확인하세요.")
    try:
        estimator.fit(inputs, job_name=TRAINING_JOB_NAME)
    except Exception as e:
        print("\n[ERROR] SageMaker TrainingJob 생성 중 오류 발생:")
        print(e)
        print("\n[조치 방법]")
        print("- SageMaker ExecutionRole의 S3 권한을 확인하세요.")
        print("- S3 경로에 train/val 데이터가 실제로 존재하는지 확인하세요.")
        raise
    return estimator

def monitor_training_job(job_name):
    """Training Job 모니터링"""
    client = boto3.client('sagemaker', region_name=SAGEMAKER_REGION)
    print(f"Monitoring training job: {job_name}")
    while True:
        response = client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        print(f"Status: {status}")
        if status in ['Completed', 'Failed', 'Stopped']:
            if status == 'Completed':
                print("Training job completed successfully!")
                print(f"Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
                print(f"Training time: {response['TrainingTimeInSeconds']} seconds")
                print(f"Billable time: {response['BillableTimeInSeconds']} seconds")
            else:
                print(f"Training job {status.lower()}")
                if 'FailureReason' in response:
                    print(f"Failure reason: {response['FailureReason']}")
            break
        time.sleep(60)

def download_results(job_name, local_dir='results'):
    """Training Job 결과 다운로드"""
    s3_client = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    result_prefix = f'sagemaker-output/{job_name}/output'
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=result_prefix
        )
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith(('.pth', '.tar.gz', '.ckpt', '.pt')):
                    local_path = os.path.join(local_dir, os.path.basename(key))
                    print(f"Downloading {key} to {local_path}")
                    s3_client.download_file(S3_BUCKET, key, local_path)
        print(f"Results downloaded to {local_dir}")
    except Exception as e:
        print(f"Error downloading results: {e}")

def main():
    print("=== SageMaker Training Job ===")
    estimator = create_training_job()
    # 모니터링 예시 (필요시 주석 해제)
    # monitor_training_job(TRAINING_JOB_NAME)
    # 다운로드 예시 (필요시 주석 해제)
    # download_results(TRAINING_JOB_NAME)

if __name__ == "__main__":
    main()
