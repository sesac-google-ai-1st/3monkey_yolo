import os, json, xmltodict
import numpy as np
import pandas as pd

def yolov8(file_path, save_txt_path):
    # file_path = xml 파일 경로
    # save_txt_path = txt 파일 저장 경로
    
    # xml 파일에서 정보 저장
    file_list = []
    file_col = []
    type = []           # 클래스
    box_x = []          # 박스x
    box_y = []          # 박스 y
    box_w = []
    box_h = []
    json_content = []

    # xml 파일 열기
    with open(file_path, encoding='utf-8') as f:
        a = xmltodict.parse(f.read())    
    
    # xml(dict) -> json 형태 변환 
    json_data = json.loads(json.dumps(a))

    # 해당 정보 추출
    for a in range(0, len(json_data['annotations']['image'])):
        file_name = json_data['annotations']['image'][a]['@name'].split('.')[0]
        file_list.append(file_name)

        # 사용위치 : ( 중앙값 x / 이미지 넓이 ) ( 중앙값 y / 이미지 높이 ) 값이 최종 txt 파일에 사용됨
        img_w = float(json_data['annotations']['image'][a]['@width'])
        img_h = float(json_data['annotations']['image'][a]['@height'])

        for b in range(0, len(json_data['annotations']['image'][a])):
            save_path = './' + file_name + '.txt'
            file_col.append(file_name)

            # 문자형 -> 숫자형 
            class_type = json_data['annotations']['image'][a]['box'][b]['@label']
            if class_type == 'car':
                class_type = 0
            elif class_type == 'bus':
                class_type = 1
            elif class_type == 'truck':
                class_type = 2 

            type.append(class_type)

            spot = [json_data['annotations']['image'][a]['box'][b]['@xtl'], json_data['annotations']['image'][a]['box'][b]['@ytl'], \
            json_data['annotations']['image'][a]['box'][b]['@xbr'], json_data['annotations']['image'][a]['box'][b]['@ybr']]
            
            # yolo 식 json 파일구조 (중심좌표x, y , w, h)
            width = float(spot[2]) - float(spot[0])
            height = float(spot[3]) - float(spot[1])
            senter_x = float(spot[0]) + float(width/2)
            senter_y = float(spot[1]) + float(height/2)

            x_center, y_center, w_box, h_box = round(senter_x/img_w, 5), round(senter_y/img_h, 5), round(width/img_w, 5), round(height/img_h, 5)
            box_x.append(x_center)
            box_y.append(y_center)
            box_w.append(w_box)
            box_h.append(h_box)

            save_note =f'{class_type} {x_center} {y_center} {w_box} {h_box}'
            json_content.append(save_note) 

    # 파일 다루기 편한 데이터 타입생성
    convert_txt = pd.DataFrame({
        'f' : file_col,
        'l' : json_content
    })

    # 파일 저장
    save_txt_path = save_txt_path
    for _ in range(0, len(convert_txt['f'])):
        path = save_txt_path + convert_txt['f'][_]+'.txt'

        if os.path.exists(path) == True:   
            f = open(path, 'a')
            f.write(convert_txt['l'][_] + '\n')
            f.close()

        elif os.path.exists(path) == False:
            f = open(path, 'w')
            f.write(convert_txt['l'][_] + '\n')
            f.close()            