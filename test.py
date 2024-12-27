import os
import subprocess
import time  # 시간 측정을 위한 모듈
from datetime import datetime

local_dataset_dir = "/local_datasets/CelebV_infer"
root_dir = '/data/jms2236/repos/hallo'
ckpt_dir = "/data/jms2236/repos/hallo/exp_output/stage1/checkpoint_sb"

# Step 3: Prepare paths for image and audio directories
image_dir = os.path.join(local_dataset_dir, "Images")  # Adjust as needed after extraction
audio_dir = os.path.join(local_dataset_dir, "Audios")  # Adjust as needed after extraction
output_dir = os.path.join(root_dir, "outputs")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

# Step 4: Run Hallo inference for each image-audio pair
processing_times = []  # 각 작업의 처리 시간 저장
for img in image_files:
    audio_filename = os.path.splitext(img)[0] + ".wav"
    audio_path = os.path.join(audio_dir, audio_filename)

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Audio file for {img} not found, skipping...")
        continue

    # Define paths
    img_path = os.path.join(image_dir, img)
    output_path = os.path.join(output_dir, f"{os.path.splitext(img)[0]}.mp4")

    # 시작 시간 기록
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run the hallo inference command
    command = f"python scripts/inference.py --source_image {img_path} --driving_audio {audio_path} --output {output_path}"
    subprocess.call(command, shell=True)

    # 종료 시간 기록 및 처리 시간 계산
    elapsed_time = time.time() - start_time
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processing_times.append((img, audio_filename, elapsed_time))

    # 로그 출력
    print(f"[{start_time_str}] Started processing {img} with {audio_filename}.")
    print(f"[{end_time_str}] Finished processing {img} with {audio_filename} in {elapsed_time:.2f} seconds.")

# 전체 처리 시간 출력
total_time = sum([x[2] for x in processing_times])
print("\nAll videos processed successfully.")
print(f"Total processing time: {total_time:.2f} seconds.")
print(f"Average processing time: {total_time / len(processing_times):.2f} seconds per video.")

