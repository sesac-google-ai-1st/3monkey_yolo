# 3monkey_yolo project : 고속도로 cctv 프로젝트


고속도로 cctv 데이터를 활용하여 vertex ai yolov8 모델로 학습한 프로젝트 입니다.

### 01_download_unzip.ipynb description
버킷에서 파일을 다운로드하기 위해서 gcp jupyterLab에서 명령어를 입력해 준다. 

```
python
# 데이터 다운 및 압축해제
!gsutil -m cp gs://sessac-project-05-bucket-01/Train라벨_1.수도권영동선.zip Train라벨_1.수도권영동선.zip

output_folder = '/home/jupyter/datasets/xml_train/'
```
---

### 04_val_drop_file.ipynb & description
coco data xml 파일에서 bounding box 좌표값을 key-value set을 유지하기 위해서 json 파일로 추출하였다.  
그 후 yolo 학습시키기 위해서 최종적으로 txt 파일로 변환하는 과정을 거쳤다.

---
### 
