import os
import boto3
from datetime import datetime
# AWS 자격 증명 설정
session = boto3.Session(
    region_name='ap-northeast-2'
)
# S3 설정
S3_BUCKET = "say1-6team-bucket"
S3_DATA_PREFIX = "x-ray-v2/"  
# SageMaker 설정
SAGEMAKER_ROLE = "arn:aws:iam::666803869796:role/SKKU_SageMaker_Role" 
SAGEMAKER_REGION = "ap-northeast-2"
# Training Job 설정
now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
TRAINING_JOB_NAME = f"aws-say1-team6-effb0-training-{now_str}" 
INSTANCE_TYPE = "ml.g4dn.4xlarge" 
INSTANCE_COUNT = 1