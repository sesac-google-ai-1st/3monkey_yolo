# 3monkey_yolo project : 고속도로 cctv 프로젝트


고속도로 cctv 데이터를 활용하여 Gcp vertex ai로 yolov8 모델로 학습한 프로젝트 입니다.

---

### 01_download_unzip.ipynb description
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

### xml_to_txt.ipynb description
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
###  08_yolov8.ipynb description
yolo nano 모델을 활용하여서 24천 건의 3가지 라벨 데이터를 학습 시켰다.
하이퍼 파라미터는 아래와 같다.

ver.1
```
 !yolo task=detect mode=train model=yolov8n.pt data=ddd.yaml epochs=100 imgsz=640 batch=92 cache=True device=0,1,2,3
```
ver.2 
```
!yolo task=detect mode=train model=yolov8n.pt data=ddd.yaml epochs=50 imgsz=640 batch=128 cache=True device=0,1,2,3
```
---
