import os
import cv2
from torch_fidelity import calculate_metrics

FIDs = []

# 영상에서 프레임을 추출하는 함수
def extract_frames(video_path, frame_dir):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()

# FID 계산을 위한 함수
def calculate_fid(real_frames_dir, generated_frames_dir):
    metrics = calculate_metrics(
        input1=real_frames_dir,
        input2=generated_frames_dir,
        cuda=True,  # GPU를 사용하려면 True로 설정
        fid=True    # FID 값을 계산
    )
    return metrics['frechet_inception_distance']

# 여러 영상에 대해 FID를 계산하고 평균을 구하는 함수
def calculate_average_fid(real_folder, generated_folder):
    total_fid = 0
    num_videos = 0

    # Real 폴더와 inference 폴더의 파일명 목록을 가져오기
    real_videos = sorted([f for f in os.listdir(real_folder) if f.endswith('.mp4')])
    generated_videos = sorted([f for f in os.listdir(generated_folder) if f.endswith('.mp4')])

    for real_video in real_videos:
        real_video_base = os.path.splitext(real_video)[0]  # 확장자 제거한 파일명 예: 00001
        corresponding_generated_video = f'{real_video_base}.mp4'  # 생성된 파일 이름 매칭

        generated_video_path = os.path.join(generated_folder, corresponding_generated_video)
        real_video_path = os.path.join(real_folder, real_video)

        # 파일이 존재하는지 확인
        if not os.path.exists(generated_video_path):
            print(f"생성된 영상이 없습니다: {corresponding_generated_video}")
            continue

        real_frames_dir = 'real_frames'
        generated_frames_dir = 'generated_frames'

        # 프레임 추출
        extract_frames(real_video_path, real_frames_dir)
        extract_frames(generated_video_path, generated_frames_dir)

        # FID 계산
        fid_value = calculate_fid(real_frames_dir, generated_frames_dir)
        total_fid += fid_value
        FIDs.append(fid_value)
        num_videos += 1

        print(f'FID for {real_video}: {fid_value}')

        # 프레임 디렉토리 정리
        os.system(f'rm -rf {real_frames_dir}')
        os.system(f'rm -rf {generated_frames_dir}')

    # 평균 FID 계산
    if num_videos > 0:
        average_fid = total_fid / num_videos
    else:
        average_fid = float('inf')

    return average_fid, num_videos

# Real 폴더와 inference 폴더 경로 설정
real_folder = './celebv_original'
generated_folder = './final_output_40step'

# 평균 FID 계산
print ("\n")
average_fid, video_num = calculate_average_fid(real_folder, generated_folder)
for i in range(video_num):
    print(i + 1, ": ", FIDs[i], "\n")
print("==================================")
print(f'Average FID: {average_fid}')
