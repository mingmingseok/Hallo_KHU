
def calculate_average_time(log_file):
    import re

    total_time_values = []

    # 로그 파일 열기 (UTF-8로 열기)
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            # "Total time taken: n"에서 숫자 추출
            match = re.search(r"Total time taken: (\d+(\.\d+)?)", line)
            if match:
                total_time_values.append(float(match.group(1)))

    # 평균 계산
    if total_time_values:
        average_time = sum(total_time_values) / len(total_time_values)
        return average_time
    else:
        return None

# 파일 경로 지정
log_file_path = "slurm-68581.out"  # 로그 파일 경로
average_time = calculate_average_time(log_file_path)

if average_time is not None:
    print(f"평균 시간: {average_time}")
else:
    print("로그에서 시간을 찾을 수 없습니다.")
