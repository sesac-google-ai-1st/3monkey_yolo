# 라이브러리
import numpy as np
import pandas as pd
import os 

# key 생성
def make_key(basic_path):
    # 레이블 폴더 디렉토리 주소를 넣으면,
    # 채널명 + 날짜 + 시간을 합쳐서 고유 key 값을
    # list로 리턴 해줌

    label_path = os.listdir(basic_path)
    keys = []
    for i in label_path :
        split_txt = i.split('_')
        key = split_txt[1] + '-' + split_txt[2] +'-'+ split_txt[3]
        keys.append(key)
    # print(split_txt)
    # print(key)
    return keys


# txt파일 디렉토리 리스트로 txt 파일들의 내용 불러와서 클래스별로 갯수 세어줌

def class_cal(label_path):
    label_path = label_path
    txt_files = os.listdir(label_path)

    file_list_column = []
    for i in range(len(txt_files)):
        # label> 1번째 폴더의 1번째 부터 마지막 txt 파일까지 반복으로 가져오기
        file = open(label_path + '/' + txt_files[i], 'r')
        # 각 파일의 내용읽어 오기
        while True:
            line = file.readline()
            # print(line)
            # 빈 줄이면, 파일의 끝에 도달한 것이므로 반복문 종료
            if not line:
                break
            line = line.strip()
            file_list_column.append(line[0])

    series = pd.Series(file_list_column)
    seriesReturn = [series.value_counts()[0], series.value_counts()[1], series.value_counts()[2]]
    return seriesReturn