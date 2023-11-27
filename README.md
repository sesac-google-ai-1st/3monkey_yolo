## 3monkey_yolo project : 고속도로 cctv📡 프로젝트

![제목을-입력해주세요_-005 (1)](https://github.com/sesac-google-ai-1st/3monkey_yolo/assets/69001369/9966c3b0-b4eb-4457-80cb-4a8a510133b4)
<br/>
<br/>
<br/>
### Project Description

<center>고속도로 cctv 데이터를 활용하여 Gcp vertex ai로 yolo v8 모델로 학습한 프로젝트 입니다.<br/><br/>
yolo는 아래와 같이 하나의 컨볼루션 네트워크(convolutional network)가 여러 bounding box와 그 bounding box의 클래스 확률을 동시에 계산해 줍니다.   YOLO는 이미지 전체를 학습하여 곧바로 검출 성능(detection performance)을 최적화합니다.   YOLO의 이런 통합된 모델은 기존의 객체 검출 모델에 비해 여러 가지 장점이 있습니다.(1) </center><br/><br/>


![image](https://github.com/sesac-google-ai-1st/3monkey_yolo/assets/69001369/25107f2a-e135-40e8-952e-8179b90b753a)

또한 YOLO는 굉장히 빠릅니다. 왜냐하면 YOLO는 기존의 복잡한 객체 검출 프로세스를 하나의 회귀 문제로 바꾸었기 때문입니다.<br/>
![yolo](https://github.com/sesac-google-ai-1st/3monkey_yolo/assets/69001369/67566326-419a-4190-a41e-e267b7c5de76)  
이러한 여러가지 장점으로 인해서 이번 프로젝트의 알고리즘으로 yolo를 선정하게 되었습니다.   

---
### Preferences

GCP : Vertex AI
인스턴스 노트북 사양 : 
- 환경 : pytorch:1.13
- 머신유형 : cpu 32개 메모리 120gb, gpu v100 4장
- 디스크 : data 200gb, boot 200gb

개발 환경 : 
- python =3.10
- pandas
- numpy
- xmltodict (xml파일 dict형으로 변환)
- ultralytics = 8.0.20 (최신버전은 에폭당 클래스 map 확인 불가. 버전 낮춤.)

데이터(2)
- tarin 데이터
1~4 채널
- validation 데이터
1~4 채널
---
### Experiment

#### 01_download_unzip.ipynb description
버킷에서 파일을 다운로드하기 위해서 gcp jupyterLab에서 명령어를 입력해 준다. 

```
python
# 압축해제하는 함수 설정

import zipfile
import concurrent.futures
import os
from tqdm import tqdm

def unzip(zip_file, output_folder):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        os.makedirs(output_folder, exist_ok=True)
        file_list = zf.namelist()
        with tqdm(total=len(file_list), desc=f'{zip_file} 압축 해제 중') as pbar:
            for file in file_list:
                zf.extract(file, output_folder)
                pbar.update(1)

# 데이터 다운 및 압축해제
!gsutil -m cp gs://sessac-project-05-bucket-01/Train라벨_1.수도권영동선.zip Train라벨_1.수도권영동선.zip

output_folder = '/home/jupyter/datasets/xml_train/'

zip_file_paths = ['Train라벨_1.수도권영동선.zip']


with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(unzip, zip_file_paths, [output_folder] * len(zip_file_paths))

for zip_file in zip_file_paths:
    os.remove(zip_file)
.
.
.
```
---

#### xml_to_txt.ipynb description
coco 데이터셋 annotation 정보를 yolov8 annotation 정보로 바꿔준다. 
```
def to_yolov8(y):
  """
  # change to yolo v8 format
  # [x_top_left, y_top_left, x_bottom_right, y_bottom_right] to
  # [x_center, y_center, width, height]
  """
  width = y[2] - y[0]
  height = y[3] - y[1]

  if width < 0 or height < 0:
      print("ERROR: negative width or height ", width, height, y)
      raise AssertionError("Negative width or height")
  return (y[0] + (width/2)), (y[1] + (height/2)), width, height

```
yolov8 annotaion 정보를 txt파일에 저장해준다.
```
def write_yolov8_txt(folder, annotation):
  #print(annotation[0][:-3])
  out_filename = os.path.join(folder,str(annotation[0][:-3]))
  out_filename = os.path.splitext(out_filename)[0]
  out_filename = out_filename+'.txt'

  f = open(out_filename,"w+")
  for box in annotation[3]:
    f.write("{} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], box[4]))
```
---
####  08_yolov8.ipynb description
🚀yolo nano 모델을 활용하여서 24천 건의 3가지 라벨 데이터를 학습 시켰다.
하이퍼 파라미터는 아래와 같다.

ver.1
```
 !yolo task=detect mode=train model=yolov8n.pt data=ddd.yaml epochs=100 imgsz=640 batch=92 cache=True device=0,1,2,3
```
ver.2 
```
!yolo task=detect mode=train model=yolov8n.pt data=ddd.yaml epochs=50 imgsz=640 batch=128 cache=True device=0,1,2,3
```


배치 크기와 모델 훈련 시간의 관계를 살펴보자. 

100개의 훈련 데이터를 가지고 80에폭 동안 훈련시킨다고 가정해보자.  

배치 크기=1이면, 모델은 1에폭 당 100번 훈련(가중치 업데이트), 총 8000번 훈련  
배치 크기=10이면, 모델은 1에폭 당 10번 훈련, 총 800번 훈련  
배치 크기=100이면, 모델은 1에폭 당 1번 훈련, 총 80번 훈련  
즉, 배치 크기를 키우면 1에폭 당 훈련 수가 감소하고 이로 인해 전체 훈련 횟수가 감소하여 결과적으로 전체 훈련 시간이 감소하게 된다.(3)  

아래 gif를 보면 패턴에 대해서 과하게 인식하여서 교차선 디자인까지 anotation되었다고 추론.   
정성정 평가를 기준으로 해당 상황을 오버피팅이 상황이라고 간주하여 경량 학습을 진행 할 수 있도록 파라마터 조정을 시도하기 위해서 후속연구를 진행 할 예정.

뿐만 아니라, medium 모델을 통해서 성능 개선을 위한 후속 연구 진행 중

 
---

### Results Analysis

#### ver.1 어노테이션 결과
![miss_anotation](https://github.com/sesac-google-ai-1st/3monkey_yolo/assets/69001369/d5a52da2-f43c-43c5-a9b8-731e07e1da35)

교차선 라인을 자동차 패턴으로 인식하여 어노테이션하는 상황이 발생하였다. 
파라미터를 조정하여 ver.2로 다시 학습하였다.

#### ver.2 어노테이션 결과
![right_anotation](https://github.com/sesac-google-ai-1st/3monkey_yolo/assets/69001369/91cc5de4-ed4b-4d45-8872-462989c25127)

조정된 파라미터로 재학습된 결과 교차선을 차로 인식하여 anotation되는 문제를 개선시킬 수 있었다.


#### ver.1 result 
|epochs|recall|mAP50|mAP50-95|
|---|---|---|---|
|96|0.85379|0.89566|0.73628|
|97|0.85377|0.89568|0.7364|
|98|0.85408|0.89556|0.7365|
|99|0.85344|0.89539|0.73649|
|100|0.85461|0.89536|0.73629|


#### ver.2 result 
|epochs|recall|mAP50|mAP50-95|
|---|---|---|---|
|46|0.83281|0.86212|0.71109|
|47|0.8393|0.86266|0.71187|
|48|0.83697|0.86305|0.71207|
|49|0.83767|0.86363|0.7121|
|50|0.83431|0.86254|0.7112|


---
#### Refrence
(1) : [yolo v8](https://github.com/ultralytics/ultralytics)  
(2) : [AI허브 cctv데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=164)
(3) : [epochs,batch](https://otugi.tistory.com/350)

