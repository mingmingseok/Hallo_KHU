#!/usr/bin/bash

#SBATCH -J fid-run                   # 작업 이름
#SBATCH --gres=gpu:1                 # GPU 1개 요청
#SBATCH --cpus-per-gpu=8            # GPU당 CPU 8개 요청
#SBATCH --mem-per-gpu=32G           # GPU당 메모리 32GB
#SBATCH -p batch_ce_ugrad           # 사용할 파티션 이름
#SBATCH -w moana-r1                 # 실행할 노드 지정
#SBATCH -t 1-0                      # 최대 실행 시간: 1일
#SBATCH -o logs/slurm-%A.out        # 출력 로그 저장 경로 (%A는 Job ID)

# 작업 환경 정보 출력
pwd
which python
hostname

# FID.py 실행
python3 FID.py

# 작업 완료 메시지
echo "FID Calculation Completed"
