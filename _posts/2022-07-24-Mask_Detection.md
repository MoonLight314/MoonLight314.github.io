---
title: "Mask Detection"
date: 2021-10-29 08:26:28 -0400
categories: Deep Learning
---
# Mask Detection

* COVID-19 상황속에서 Deep Learning을 이용하여 RGB Cam.으로 실시간으로 Mask 착용 여부를 확인할 수 있는 Model을 만들어 보겠습니다.

* 우선 사람의 얼굴부분만을 빠르게 Detecting할 수 있는 Model을 찾아보았고, 최종적으로 Tensorflow와 호환이 잘되는 OpenCV DNN Face Detector를 사용하기로 했습니다.

* DNN Face Detector in OpenCV [https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/]
  - Face Detector는 Input으로 Image를 넣어주면, 해당 Image에서 사람 얼굴이라고 판단되는 영역의 정보와 확신(신뢰)도를 값으로 Return해 줍니다.
  - Pre-Trained Model을 받아서 사용하면 됩니다.
  - 다음 2개의 File을 이용합니다.
    - MODEL_FILE : opencv_face_detector_uint8.pb
    - CONFIG_FILE : opencv_face_detector.pbtxt
  - 속도도 빠르고 성능도 좋아서 Face Detection에 이 Module을 사용하도록 하겠습니다.
  - 자세한 사용 방법은 이후 Code로 살펴보겠습니다.

      

* Train에 사용할 Dataset은 아래 2가지를 사용하도록 하겠습니다.      

* Dataset
 - https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset 
 
 - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
     - License(CC0: Public Domain)(https://creativecommons.org/publicdomain/zero/1.0/)

* 순서
  
  1) 2개의 Dataset은 공통적으로 모두 Mask를 쓴 사람들의 사진과 쓰지 않은 사람들의 사진을 가지고 있지만, Mask 착용 여부 / 사람 얼굴 위치등의 정보를 나타내는 방법은 조금 다릅니다.
  
  2) 최종적으로 ResNet으로 분류를 하기 위한 Dataset으로 Preprocessing을 하는 것이 목적이므로 각 Dataset마다 다른 Preprocessing 방법을 적용하도록 하겠습니다.

      

## 0. Preprocess      

* 첫번째 Dataset
 - 이 Dataset은 총 4095장의 Image가 있고, Mask쓴 사람의 Image가 2165장, 쓰지 않는 사람의 Image가 1930장으로 구성되어 있다.
 - Image에는 얼굴만 나오는 사진도 있지만, 사람 몸 전체가 나오는 경우도 있기 때문에 앞서 소개한 OpenCV DNN Face Detector를 이용하여 얼굴부분에 대한 정보만 추출하여 Train에 사용하도록 하겠습니다.
 - Label은 Folder Name으로 알 수 있습니다.
 - 최종적으로 File Path / Mask 착용 여부 / 얼굴부분의 좌표 정보를 추출하여 Pandas Dataframe으로 저장하는 것을 목표로 하겠습니다.

      


```python
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
import xml.etree.ElementTree as et
```

      

      

* OpenCV DNN Face Detector 관련 상수를 정의합니다.
* 'CONFIDENCE_FACE = 0.9'
  - Face Detector가 제공해주는 값으로 추출한 얼굴 부분이 어느 정도 신뢰도가 있는지 나타내주는 값입니다.
* 'MARGIN_RATIO = 0.2'
  - Face Detector는 딱 얼굴 부분만 추출하기 때문에 상하좌우 조금 더 여유를 주기 위한 값입니다.


```python
MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
MARGIN_RATIO = 0.2
```

      

      

* 먼저 Folder내에 있는 Image File들의 Full Path List를 만듭니다.      


```python
def save_file_fist():
    dir_path = "./Dataset/Face-Mask-Detection-master"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            file_path = os.path.join(root, file)
            data_file_path.append( file_path )


    return data_file_path
```

      

* 이후에 이 File List를 가지고 추가 작업을 하기 때문에 이 작업은 가장 먼저 수행되어야 합니다.      


```python
data_file_path = save_file_fist()
```

    3it [00:00, 142.83it/s]
    

      

      

* OpenCV Face Dectector로 각 Image에서 얼굴 부분에 대한 좌표값을 추출하는 부분입니다.      


```python
def get_face_coor_info(file_path):
    
    pass_file_list = []
    lefts = []
    tops = []
    rights = []
    bottoms = []
    masks = []
    low_confidence_cnt = 0

    net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

    for file in tqdm(file_path):

        try:
            img = cv2.imread(file)
            rows, cols, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0)

            net.setInput(blob)
            detections = net.forward()

            detection = detections[0, 0]    
            i = np.argmax(detection[:,2])

            if i != 0:
                print(file , "Max index is not 0")
                continue

            if detection[i,2] < CONFIDENCE_FACE:
                #print(file , "Low CONFIDENCE_FACE" , detection[i,2])
                low_confidence_cnt += 1
                continue
            

            left = detection[i,3] * cols
            top = detection[i,4] * rows
            right = detection[i,5] * cols            
            bottom = detection[i,6] * rows

            left = int(left - int((right - left) * MARGIN_RATIO))
            top = int(top - int((bottom - top) * MARGIN_RATIO))
            right = int(right + int((right - left) * MARGIN_RATIO))
            bottom = int(bottom + int((bottom - top) * MARGIN_RATIO / 2))

            if left < 0:
                left = 0

            if right > cols:
                right = cols

            if top < 0:
                top = 0

            if bottom > rows:
                bottom = rows

            pass_file_list.append(file)
            lefts.append(left)
            tops.append(top)
            rights.append(right)
            bottoms.append(bottom)

            if "with_mask" in file:
                masks.append("with_mask")
            elif "without_mask" in file:
                masks.append("without_mask")
        
        except:
            print(file , " Error")


    print(len(pass_file_list))
    print("No. of Low Confidence : ",low_confidence_cnt)

    result = pd.DataFrame(list(zip(pass_file_list, masks , lefts , tops , rights , bottoms)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    result = result.astype({    'xmin':'int32', 
                                'ymin':'int32',
                                'xmax':'int32', 
                                'ymax':'int32',
                                })

    return result
```

      

      

      


```python
meta_data_01 = get_face_coor_info( data_file_path )
```

     64%|█████████████████████████████████████████████████▎                           | 2622/4095 [00:59<00:14, 103.21it/s]

    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-23 132115.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-23 132400.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 171804.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 172039.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 202509.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 205216.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 215234.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 215615.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 220536.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 222124.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 224833.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 225329.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-24 225427.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150422.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150847.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 150921.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 185823.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0__˙_￠ 2020-02-25 190026.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\0_0_œ￢‘y.png  Error
    

     73%|████████████████████████████████████████████████████████▋                     | 2973/4095 [01:08<00:16, 69.06it/s]

    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 202935.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 215624.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-24 224914.png  Error
    ./Dataset/Face-Mask-Detection-master\with_mask\1_0__˙_￠ 2020-02-25 151918.png  Error
    

    100%|██████████████████████████████████████████████████████████████████████████████| 4095/4095 [03:49<00:00, 17.88it/s]

    3197
    No. of Low Confidence :  875
    

    
    

* 간혹 File Name에 특수 문자가 있는 경우 Open하지 못하는 경우도 있어서 875장은 사용하지 못했습니다.

* 최종적으로 첫번째 Dataset에서는 총 3197장의 유효한 Data를 얻었습니다.

      

      

* 두번째 Dataset
 - 이 Dataset은 총 853장의 Image가 있습니다.
 - 이 Dataset은 Image File이름과 동일한 XML File을 제공해 주고 있으며, 각 XML File에는 Image File Name / 얼굴부분의 좌표 정보 / Mask 착용여부에 대한 정보가 모두 들어 있습니다.
 - 즉, XML File Decoding만 잘 해주면 모든 정보를 얻을 수 있습니다.


```python
def preprocessing_Face_Mask_Detection_Dataset_Kaggle():
    dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/annotations/"
    image_dir_path = "./Dataset/Face_Mask_Detection_Dataset_Kaggle/images/"
    data_file_path = []

    for (root, directories, files) in tqdm(os.walk(dir_path)):
        for file in files:
            if '.xml' in file:
                file_path = os.path.join(root, file)
                data_file_path.append( file_path )

    meta_data = pd.DataFrame({"file_path":[], 
                            "mask":[],
                            "xmin":[],
                            "ymin":[],
                            "xmax":[],
                            "ymax":[]
                            })

    for path in tqdm(data_file_path):

        xtree=et.parse( path )
        xroot=xtree.getroot()

        mask_flag = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []

        for node in xroot:
            
            if node.tag == 'filename':
                fname = os.path.join(image_dir_path , node.text)

            if node.tag == 'object':
                name = node.find("name")
                mask_flag.append( name.text )

                box = node.find("bndbox")
                
                t = box.find("xmin")
                if t != None:
                    xmin.append( t.text )

                t = box.find("ymin")
                if t != None:
                    ymin.append( t.text )

                t = box.find("xmax")
                if t != None:
                    xmax.append( t.text )

                t = box.find("ymax")
                if t != None:
                    ymax.append( t.text )
                


        file_name = [fname] * len(xmin)

        tmp = pd.DataFrame({"file_path":file_name , 
                            "mask":mask_flag,
                            "xmin":xmin,
                            "ymin":ymin,
                            "xmax":xmax,
                            "ymax":ymax
                            })

        meta_data = pd.concat( [meta_data,tmp] )

    meta_data = meta_data.astype({  'xmin':'int32', 
                                    'ymin':'int32',
                                    'xmax':'int32', 
                                    'ymax':'int32',
                                    })

    return meta_data
```

      

      


```python
meta_data_02 = preprocessing_Face_Mask_Detection_Dataset_Kaggle()
```

    1it [00:00, 199.98it/s]
    100%|███████████████████████████████████████████████████████████████████████████████| 853/853 [00:05<00:00, 148.33it/s]
    

      

* 이제 2개의 Dataset에서 Train에 필요한 모든 정보를 얻었기 때문에 하나로 합칩니다.      


```python
meta_data = pd.concat([meta_data_01 , meta_data_02])
```

      

      

      

* 마지막으로 실제로 Train시에 사용할 ResNet에 입력시에 문제가 없는지 확인하도록 하겠습니다.
* 실제 Train을 하다보면 Image를 열지 못하거나 다양한 이유로 Train이 중단되어버리는 경우가 종종 발생하였습니다.
* 이를 미연에 방지하고자 실제 Image Decoding이 제대로 되는지 ResNet Preprocessing을 거치는 작업을 해보도록 하겠습니다.


```python
def verify_image_file(meta_data):

    train_left = meta_data['xmin'].tolist()
    train_right = meta_data['xmax'].tolist()
    train_top = meta_data['ymin'].tolist()
    train_bottom = meta_data['ymax'].tolist()
    train_mask = meta_data['mask'].tolist()
    file_path_train = meta_data['file_path'].tolist()

    new_left = []
    new_right = []
    new_top = []
    new_bottom = []
    new_file_path = []
    new_mask = []

    for idx,image_path in tqdm(enumerate( file_path_train)):
        
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)   
            
            img = tf.image.crop_to_bounding_box( img , train_top[idx] , train_left[idx], train_bottom[idx] - train_top[idx] , train_right[idx] - train_left[idx] )

            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.resnet50.preprocess_input(img)

            new_left.append(train_left[idx])
            new_right.append(train_right[idx])
            new_top.append(train_top[idx])
            new_bottom.append(train_bottom[idx])
            new_file_path.append(image_path)
            new_mask.append(train_mask[idx])
        
        except Exception as e:
            print(e)
            continue
    
    print(len(new_file_path))

    result = pd.DataFrame(list(zip(new_file_path, new_mask , new_left , new_top , new_right , new_bottom)), columns=['file_path','mask','xmin','ymin','xmax','ymax'])

    return result
```


```python
meta_data = verify_image_file(meta_data)
```

    631it [00:09, 76.73it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    706it [00:09, 127.04it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1338it [00:12, 297.58it/s]

    target_width must be > 0.
    

    1415it [00:14, 81.39it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1483it [00:15, 124.41it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1556it [00:15, 183.23it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1634it [00:15, 241.48it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1781it [00:17, 76.46it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1822it [00:17, 102.10it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_height must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1892it [00:18, 154.64it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    1977it [00:18, 231.88it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2075it [00:18, 313.40it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2160it [00:20, 89.39it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2208it [00:20, 120.69it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2301it [00:20, 193.36it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2381it [00:20, 247.72it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2470it [00:21, 316.10it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2549it [00:23, 83.07it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2622it [00:23, 130.51it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2655it [00:23, 139.32it/s]

    Unknown image file format. One of JPEG, PNG, GIF, BMP required. [Op:DecodeImage]
    target_width must be > 0.
    target_width must be > 0.
    Unknown image file format. One of JPEG, PNG, GIF, BMP required. [Op:DecodeImage]
    target_width must be > 0.
    target_width must be > 0.
    Unknown image file format. One of JPEG, PNG, GIF, BMP required. [Op:DecodeImage]
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2684it [00:23, 134.52it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2714it [00:23, 154.79it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2772it [00:25, 59.73it/s] 

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2793it [00:26, 66.51it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2811it [00:26, 60.26it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2825it [00:26, 63.17it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2850it [00:28, 28.19it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2861it [00:28, 33.27it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2886it [00:29, 49.77it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2911it [00:29, 64.68it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2922it [00:29, 70.04it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    

    2949it [00:31, 25.76it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2974it [00:31, 40.83it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    2987it [00:31, 48.04it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3015it [00:31, 72.76it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3045it [00:32, 95.66it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3073it [00:34, 27.88it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_height must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3087it [00:34, 36.46it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3115it [00:34, 57.45it/s]

    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    target_width must be > 0.
    

    3195it [00:34, 155.43it/s]

    target_width must be > 0.
    target_width must be > 0.
    

    3302it [00:37, 58.68it/s] 

    width must be >= target + offset.
    

    3842it [00:56, 30.06it/s]

    width must be >= target + offset.
    

    3997it [01:01, 18.78it/s]

    width must be >= target + offset.
    

    5100it [01:38, 28.36it/s]

    width must be >= target + offset.
    

    5282it [01:43, 39.03it/s]

    width must be >= target + offset.
    width must be >= target + offset.
    

    5971it [02:05, 35.54it/s]

    width must be >= target + offset.
    width must be >= target + offset.
    

    6043it [02:08, 31.23it/s]

    width must be >= target + offset.
    

    6509it [02:22, 66.83it/s]

    width must be >= target + offset.
    

    6904it [02:36, 39.18it/s]

    width must be >= target + offset.
    

    7260it [02:47, 61.09it/s]

    width must be >= target + offset.
    

    7269it [02:47, 43.38it/s]

    6812
    

    
    

      

      

      

      

* 2번째 Dataset에는 Mask 착요여부를 나타내는 값 중에, '제대로 마스크를 쓰지 않음(mask_weared_incorrect)' 값이 있습니다.
* 우선은 이 값을 Mask를 착용했음으로 변경하도록 하겠습니다.
* 최종적으로 얻은 값들을 저장하고, 이 값을 Train시에 사용하도록 하겠습니다.


```python
meta_data = meta_data.replace({'mask':'mask_weared_incorrect'},'with_mask')
meta_data.to_csv("meta_data.csv",index=False)
```





# Train   

* Preprocessing 작업에서 만들어진 Image File List를 가지고 Train을 하도록 하겠습니다.   

   

   

* 필요한 Package들을 Load합니다.   


```python
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization , Dropout , Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler
```

   

   

* Batch Size와 Dropout Rate 설정합니다.   


```python
BATCH_SIZE = 32
DROP_OUT_RATE = 0.2
```

   

* Preprocessing에서 얻은 Meta Data File을 Open합니다.   


```python
dataset_info = pd.read_csv("meta_data.csv")
```

   

   

* Image에 대한 정보와 Label에 대한 정보를 분리합니다.

* Mask 착용여부가 Label이 되고, 이를 Label Encoder로 One-Hot으로 변환합니다.


```python
data_file_path = dataset_info[['file_path' , 'xmin' , 'ymin' , 'xmax' , 'ymax']]
mask = dataset_info['mask'].tolist()

le = LabelEncoder()
le.fit(mask)
print(le.classes_)

le_mask = le.transform(mask)
mask = tf.keras.utils.to_categorical(le_mask , num_classes=2)
```

    ['with_mask' 'without_mask']
    

   

   

* Train & Val. Dataset을 3:1로 나눕니다.


```python
file_path_train, file_path_val, y_train, y_val = train_test_split(data_file_path, mask, 
                                                                  test_size=0.25, 
                                                                  random_state=777, 
                                                                  stratify = mask)

print( len(file_path_train) , len(y_train) , len(file_path_val) , len(y_val) )
```

    5109 5109 1703 1703
    

   

   

* Tensorflow Dataset의 Map Function에서 사용하기 위해서 얼굴 좌표 값들과 File Path를 List로 바꾸어 놓습니다.


```python
train_left = file_path_train['xmin'].tolist()
train_right = file_path_train['xmax'].tolist()
train_top = file_path_train['ymin'].tolist()
train_bottom = file_path_train['ymax'].tolist()
file_path_train = file_path_train['file_path'].tolist()

val_left = file_path_val['xmin'].tolist()
val_right = file_path_val['xmax'].tolist()
val_top = file_path_val['ymin'].tolist()
val_bottom = file_path_val['ymax'].tolist()
file_path_val = file_path_val['file_path'].tolist()
```

   

   

* Dataset Map Function입니다.
* Image File Path를 받아서, 얼굴부분만을 잘라낸 후 Label값과 함께 돌려줍니다.


```python
def load_image( image_path , left , right , top , bottom , label ):
    img = tf.io.read_file(image_path)
    
    img = tf.image.decode_png(img, channels=3)   
    img = tf.image.crop_to_bounding_box( img , top , left, bottom - top , right - left )

    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)    
    
    return img , label
```

   

   

* Dataset을 준비합니다.   
* from_tensor_slices의 Parameter에 얼굴 좌표 값과 File Path를 넣어줍니다.


```python
train_dataset = tf.data.Dataset.from_tensor_slices( (file_path_train , 
                                                     train_left , 
                                                     train_right , 
                                                     train_top , 
                                                     train_bottom , 
                                                     y_train) )

val_dataset = tf.data.Dataset.from_tensor_slices( (file_path_val , 
                                                   val_left , 
                                                   val_right , 
                                                   val_top , 
                                                   val_bottom ,
                                                   y_val) )
```


```python
train_dataset = train_dataset.shuffle(buffer_size=len(file_path_train))\
                                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                                .repeat()\
                                .batch(BATCH_SIZE)\
                                .prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_dataset.shuffle(buffer_size=len(file_path_val))\
                            .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .repeat()\
                            .batch(BATCH_SIZE)\
                            .prefetch(tf.data.experimental.AUTOTUNE)    #
```

   

   

* ResNet50으로 Feature Extraction해서 Dense로 분류하도록 하겠습니다.   


```python
ResNet50 = tf.keras.applications.resnet.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False)
```


```python
model= Sequential()

model.add( ResNet50 )

model.add( GlobalAveragePooling2D() ) 
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 
model.add( Dense(128, activation='relu') )
model.add( Dropout( DROP_OUT_RATE ) ) 
model.add( BatchNormalization() ) 

model.add( Dense(2, activation='softmax') )
```

   

   

* Learning Rate Scheduler 정의합니다.   


```python
initial_learning_rate = 0.01

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * np.math.exp(-k*epoch)

lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)
```

   

   

   

* Tensorboard와 Checkpoint 관련 값들을 정의합니다.   


```python
log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints_Mask_Detection')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                     monitor='val_accuracy',                     
                     save_best_only = True,
                     verbose = 1)
```

   

   

* Model Compile   


```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    #loss='binary_crossentropy',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

   

   

* Train 시작   


```python
hist = model.fit(train_dataset,
                 validation_data=val_dataset,
                 callbacks=[lr_scheduler , cp , tb_callback],
                 steps_per_epoch = 200,
                 validation_steps = 50,
                 epochs = 20,
                 verbose = 1 
)
```

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    Epoch 1/20
    
    Epoch 00001: LearningRateScheduler reducing learning rate to 0.01.
    200/200 [==============================] - 116s 504ms/step - loss: 0.3580 - accuracy: 0.8692 - val_loss: 0.2935 - val_accuracy: 0.8988
    
    Epoch 00001: val_accuracy improved from -inf to 0.89875, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 2/20
    
    Epoch 00002: LearningRateScheduler reducing learning rate to 0.009048374180359595.
    200/200 [==============================] - 95s 476ms/step - loss: 0.2450 - accuracy: 0.9077 - val_loss: 0.3663 - val_accuracy: 0.8525
    
    Epoch 00002: val_accuracy did not improve from 0.89875
    Epoch 3/20
    
    Epoch 00003: LearningRateScheduler reducing learning rate to 0.008187307530779819.
    200/200 [==============================] - 97s 488ms/step - loss: 0.2112 - accuracy: 0.9247 - val_loss: 0.4753 - val_accuracy: 0.8194
    
    Epoch 00003: val_accuracy did not improve from 0.89875
    Epoch 4/20
    
    Epoch 00004: LearningRateScheduler reducing learning rate to 0.007408182206817179.
    200/200 [==============================] - 97s 487ms/step - loss: 0.1983 - accuracy: 0.9278 - val_loss: 0.2442 - val_accuracy: 0.9119
    
    Epoch 00004: val_accuracy improved from 0.89875 to 0.91188, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 5/20
    
    Epoch 00005: LearningRateScheduler reducing learning rate to 0.006703200460356393.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1901 - accuracy: 0.9322 - val_loss: 0.1864 - val_accuracy: 0.9344
    
    Epoch 00005: val_accuracy improved from 0.91188 to 0.93437, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 6/20
    
    Epoch 00006: LearningRateScheduler reducing learning rate to 0.006065306597126334.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1803 - accuracy: 0.9383 - val_loss: 0.2283 - val_accuracy: 0.9050
    
    Epoch 00006: val_accuracy did not improve from 0.93437
    Epoch 7/20
    
    Epoch 00007: LearningRateScheduler reducing learning rate to 0.005488116360940264.
    200/200 [==============================] - 95s 477ms/step - loss: 0.1629 - accuracy: 0.9431 - val_loss: 0.1612 - val_accuracy: 0.9400
    
    Epoch 00007: val_accuracy improved from 0.93437 to 0.94000, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 8/20
    
    Epoch 00008: LearningRateScheduler reducing learning rate to 0.004965853037914095.
    200/200 [==============================] - 95s 476ms/step - loss: 0.1611 - accuracy: 0.9422 - val_loss: 0.1756 - val_accuracy: 0.9369
    
    Epoch 00008: val_accuracy did not improve from 0.94000
    Epoch 9/20
    
    Epoch 00009: LearningRateScheduler reducing learning rate to 0.004493289641172216.
    200/200 [==============================] - 95s 476ms/step - loss: 0.1422 - accuracy: 0.9498 - val_loss: 0.2156 - val_accuracy: 0.9244
    
    Epoch 00009: val_accuracy did not improve from 0.94000
    Epoch 10/20
    
    Epoch 00010: LearningRateScheduler reducing learning rate to 0.004065696597405992.
    200/200 [==============================] - 77s 378ms/step - loss: 0.1497 - accuracy: 0.9503 - val_loss: 0.1187 - val_accuracy: 0.9600
    
    Epoch 00010: val_accuracy improved from 0.94000 to 0.96000, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 11/20
    
    Epoch 00011: LearningRateScheduler reducing learning rate to 0.0036787944117144234.
    200/200 [==============================] - 95s 475ms/step - loss: 0.1374 - accuracy: 0.9513 - val_loss: 0.1832 - val_accuracy: 0.9337
    
    Epoch 00011: val_accuracy did not improve from 0.96000
    Epoch 12/20
    
    Epoch 00012: LearningRateScheduler reducing learning rate to 0.003328710836980796.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1229 - accuracy: 0.9564 - val_loss: 0.1432 - val_accuracy: 0.9525
    
    Epoch 00012: val_accuracy did not improve from 0.96000
    Epoch 13/20
    
    Epoch 00013: LearningRateScheduler reducing learning rate to 0.0030119421191220205.
    200/200 [==============================] - 94s 474ms/step - loss: 0.1266 - accuracy: 0.9583 - val_loss: 0.1450 - val_accuracy: 0.9625
    
    Epoch 00013: val_accuracy improved from 0.96000 to 0.96250, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 14/20
    
    Epoch 00014: LearningRateScheduler reducing learning rate to 0.002725317930340126.
    200/200 [==============================] - 95s 475ms/step - loss: 0.1183 - accuracy: 0.9606 - val_loss: 0.1348 - val_accuracy: 0.9606
    
    Epoch 00014: val_accuracy did not improve from 0.96250
    Epoch 15/20
    
    Epoch 00015: LearningRateScheduler reducing learning rate to 0.0024659696394160645.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1106 - accuracy: 0.9658 - val_loss: 0.1364 - val_accuracy: 0.9619
    
    Epoch 00015: val_accuracy did not improve from 0.96250
    Epoch 16/20
    
    Epoch 00016: LearningRateScheduler reducing learning rate to 0.0022313016014842983.
    200/200 [==============================] - 95s 474ms/step - loss: 0.1120 - accuracy: 0.9619 - val_loss: 0.1379 - val_accuracy: 0.9550
    
    Epoch 00016: val_accuracy did not improve from 0.96250
    Epoch 17/20
    
    Epoch 00017: LearningRateScheduler reducing learning rate to 0.002018965179946554.
    200/200 [==============================] - 95s 476ms/step - loss: 0.0988 - accuracy: 0.9658 - val_loss: 0.1437 - val_accuracy: 0.9569
    
    Epoch 00017: val_accuracy did not improve from 0.96250
    Epoch 18/20
    
    Epoch 00018: LearningRateScheduler reducing learning rate to 0.001826835240527346.
    200/200 [==============================] - 97s 475ms/step - loss: 0.0921 - accuracy: 0.9725 - val_loss: 0.1049 - val_accuracy: 0.9688
    
    Epoch 00018: val_accuracy improved from 0.96250 to 0.96875, saving model to CheckPoints_Mask_Detection
    

    C:\Users\Moon\anaconda3\envs\TF.2.5.0-GPU\lib\site-packages\tensorflow\python\keras\utils\generic_utils.py:494: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
      warnings.warn('Custom mask layers require a config and must override '
    

    INFO:tensorflow:Assets written to: CheckPoints_Mask_Detection\assets
    Epoch 19/20
    
    Epoch 00019: LearningRateScheduler reducing learning rate to 0.0016529888822158654.
    200/200 [==============================] - 95s 475ms/step - loss: 0.0883 - accuracy: 0.9722 - val_loss: 0.1417 - val_accuracy: 0.9613
    
    Epoch 00019: val_accuracy did not improve from 0.96875
    Epoch 20/20
    
    Epoch 00020: LearningRateScheduler reducing learning rate to 0.0014956861922263505.
    200/200 [==============================] - 94s 474ms/step - loss: 0.0928 - accuracy: 0.9686 - val_loss: 0.1166 - val_accuracy: 0.9613
    
    Epoch 00020: val_accuracy did not improve from 0.96875
    

   

   

* Train / Val. 모두 좋은 정확도를 보여줍니다.   


```python
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```


    
![png](output_59_0.png)
    



    
![png](output_59_1.png)
    



```python

```



## Inference   

* 이제 실제로 Cam을 통해서 Train한 Model이 잘 동작하는지 확인해 보도록 하겠습니다.
* 전체적인 방법은 Cam을 통해 들어온 영상을 Preprocess와 동일한 방법으로 전처리를 한 후 Model에 넣은 후에 결과를 출력하도록 하겠습니다.

   

   


```python
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model
import tensorflow_addons as tfa
```

   

   

* Preprocess때와 동일하게 Image를 처리하기 위해서 Face Detector 및 상수 값들도 동일하게 사용하도록 하겠습니다.   


```python
MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9
RESULT = ['with_mask' , 'without_mask']
MARGIN_RATIO = 0.2
```

   

   

* Face Detector를 Load합니다.   


```python
net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )
```

   

   

* Train시킨 Model도 같이 Load합니다.   


```python
print("Loading Saved Model...")

model = load_model("CheckPoints_Mask_Detection")
```

    Loading Saved Model...
    

   

   

* Cam을 연결합니다.   


```python
cap = cv2.VideoCapture(0)
```

   

   

* 아래 부분은 Cam에서 영상을 받아서 Preprocess를 거치고 Model에 입력시키는 과정입니다.

    - **ret, frame = cap.read()**
        * Cam에서 Image를 한 장 받아옵니다.      
      
    - **i = np.argmax(detection[:,2])**
        * Face Detector가 Detect한 얼굴부분에 대한 정보 Index를 얻습니다.        
        
    - **if detection[i,2] < CONFIDENCE_FACE:**
        * Face Detector가 Detect한 얼굴부분의 신뢰도가 특정 값(0.9)이상인 경우에만 얼굴로 판단합니다.
        
    - **left = left - int((right - left) * MARGIN_RATIO)
        top = top - int((bottom - top) * MARGIN_RATIO)
        right = right + int((right - left) * MARGIN_RATIO)
        bottom = bottom + int((bottom - top) * MARGIN_RATIO)**
        * Preprocess때와 마찬가지로 Margin을 두고 얼굴 부분을 Crop합니다.
        
    - **cropped = np.array(cropped).reshape(-1,224,224,3)**
        * 얼굴부분을 Model에 넣기 위해 Numpy Array로 변환하고 Reshape합니다.
        
    - **pred = model.predict( cropped )**
        * Model에 Input시켜서 결과를 받습니다.
        
    - **Result = "Result : {0}".format(RESULT[int(np.argmax(np.reshape( pred , (1,-1) )))])
        cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)**
        * Model은 Mask 착용 여부를 확률로 보여주고, 이를 출력합니다.


```python
while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    rows, cols, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1.0)

    net.setInput(blob)
    detections = net.forward()

    detection = detections[0, 0]    
    i = np.argmax(detection[:,2])

    if i != 0:
        print("Max index is not 0")
        continue

    if detection[i,2] < CONFIDENCE_FACE:
        print("Low CONFIDENCE_FACE" , detection[i,2])
        continue

    if detection[i,3] >= 1.00 or detection[i,4] >= 1.00 or detection[i,5] >= 1.00 or detection[i,6] >= 1.00 or detection[i,3] <= 0 or detection[i,4] < 0 or detection[i,5] <= 0 or detection[i,6] <= 0:
        pass
    else:
        left = int(detection[i,3] * cols)
        top = int(detection[i,4] * rows)
        right = int(detection[i,5] * cols)
        bottom = int(detection[i,6] * rows)

        left = left - int((right - left) * MARGIN_RATIO)
        top = top - int((bottom - top) * MARGIN_RATIO)
        right = right + int((right - left) * MARGIN_RATIO)
        bottom = bottom + int((bottom - top) * MARGIN_RATIO)

        if left < 0:
            left = 0

        if right > cols:
            right = cols

        if top < 0:
            top = 0

        if bottom > rows:
            bottom = rows

        cropped = frame[top:bottom, left:right]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = cv2.resize( cropped , dsize=(224,224) )
        cropped = np.array(cropped).reshape(-1,224,224,3)

        cropped = tf.keras.applications.resnet50.preprocess_input(cropped)

        pred = model.predict( cropped )
        print(pred)

        Result = "Result : {0}".format(RESULT[int(np.argmax(np.reshape( pred , (1,-1) )))])

        cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("VideoFrame", frame)


cap.release()
cv2.destroyAllWindows()
```

    Low CONFIDENCE_FACE 0.14280182
    Low CONFIDENCE_FACE 0.14190249
    Low CONFIDENCE_FACE 0.14129429
    Low CONFIDENCE_FACE 0.14157966
    Low CONFIDENCE_FACE 0.14236318
    Low CONFIDENCE_FACE 0.14125907
    Low CONFIDENCE_FACE 0.14199193
    Low CONFIDENCE_FACE 0.14291707
    Low CONFIDENCE_FACE 0.14235935
    Low CONFIDENCE_FACE 0.1426097
    Low CONFIDENCE_FACE 0.1422901
    Low CONFIDENCE_FACE 0.14139618
    Low CONFIDENCE_FACE 0.14376278
    Low CONFIDENCE_FACE 0.14121968
    Low CONFIDENCE_FACE 0.14125559
    Low CONFIDENCE_FACE 0.14131822
    Low CONFIDENCE_FACE 0.14094456
    Low CONFIDENCE_FACE 0.1419249
    Low CONFIDENCE_FACE 0.14246558
    Low CONFIDENCE_FACE 0.14134978
    Low CONFIDENCE_FACE 0.14168736
    Low CONFIDENCE_FACE 0.1426712
    Low CONFIDENCE_FACE 0.14161922
    Low CONFIDENCE_FACE 0.14150882
    Low CONFIDENCE_FACE 0.1416071
    Low CONFIDENCE_FACE 0.14292993
    Low CONFIDENCE_FACE 0.14311025
    Low CONFIDENCE_FACE 0.1416132
    Low CONFIDENCE_FACE 0.14244516
    Low CONFIDENCE_FACE 0.14291073
    Low CONFIDENCE_FACE 0.14168912
    Low CONFIDENCE_FACE 0.14061464
    Low CONFIDENCE_FACE 0.14211135
    Low CONFIDENCE_FACE 0.14124404
    Low CONFIDENCE_FACE 0.14369263
    Low CONFIDENCE_FACE 0.14264362
    Low CONFIDENCE_FACE 0.14219181
    Low CONFIDENCE_FACE 0.14030434
    Low CONFIDENCE_FACE 0.14164947
    Low CONFIDENCE_FACE 0.14277592
    Low CONFIDENCE_FACE 0.14042556
    Low CONFIDENCE_FACE 0.14322361
    Low CONFIDENCE_FACE 0.14176065
    Low CONFIDENCE_FACE 0.1415934
    Low CONFIDENCE_FACE 0.14275184
    Low CONFIDENCE_FACE 0.14311273
    Low CONFIDENCE_FACE 0.13989192
    Low CONFIDENCE_FACE 0.1421124
    Low CONFIDENCE_FACE 0.14220457
    Low CONFIDENCE_FACE 0.16676228
    Low CONFIDENCE_FACE 0.18839803
    Low CONFIDENCE_FACE 0.20425373
    Low CONFIDENCE_FACE 0.1331807
    Low CONFIDENCE_FACE 0.14578325
    Low CONFIDENCE_FACE 0.13376047
    Low CONFIDENCE_FACE 0.14872557
    Low CONFIDENCE_FACE 0.14934115
    Low CONFIDENCE_FACE 0.14369012
    Low CONFIDENCE_FACE 0.15123594
    Low CONFIDENCE_FACE 0.14166956
    Low CONFIDENCE_FACE 0.16028358
    Low CONFIDENCE_FACE 0.13631478
    Low CONFIDENCE_FACE 0.13824911
    Low CONFIDENCE_FACE 0.14172396
    Low CONFIDENCE_FACE 0.13429609
    Low CONFIDENCE_FACE 0.15828066
    Low CONFIDENCE_FACE 0.13166375
    Low CONFIDENCE_FACE 0.13314725
    Low CONFIDENCE_FACE 0.13404554
    Low CONFIDENCE_FACE 0.1386514
    Low CONFIDENCE_FACE 0.13173284
    Low CONFIDENCE_FACE 0.14980948
    Low CONFIDENCE_FACE 0.13372232
    Low CONFIDENCE_FACE 0.13214064
    Low CONFIDENCE_FACE 0.13299797
    Low CONFIDENCE_FACE 0.25814673
    Low CONFIDENCE_FACE 0.8366865
    Low CONFIDENCE_FACE 0.29836679
    Low CONFIDENCE_FACE 0.8953248
    Low CONFIDENCE_FACE 0.46881586
    Low CONFIDENCE_FACE 0.20059115
    [[9.9993217e-01 6.7791581e-05]]
    [[0.97642094 0.02357905]]
    [[0.97244287 0.02755718]]
    [[0.95950097 0.04049898]]
    [[0.9701627  0.02983732]]
    [[0.9849089  0.01509111]]
    [[0.9650323  0.03496766]]
    [[0.981807   0.01819297]]
    [[0.9684551  0.03154493]]
    [[0.9641147  0.03588523]]
    [[0.97459507 0.0254049 ]]
    [[0.9652043 0.0347957]]
    [[0.951519   0.04848098]]
    [[0.93519586 0.06480411]]
    [[0.71979195 0.280208  ]]
    [[0.48106903 0.518931  ]]
    [[9.9997830e-01 2.1710004e-05]]
    [[9.9990952e-01 9.0474125e-05]]
    [[9.999651e-01 3.491917e-05]]
    [[9.9959773e-01 4.0229305e-04]]
    [[9.9996686e-01 3.3123539e-05]]
    Low CONFIDENCE_FACE 0.85675186
    [[9.9997509e-01 2.4915475e-05]]
    Low CONFIDENCE_FACE 0.89705086
    Low CONFIDENCE_FACE 0.870888
    Low CONFIDENCE_FACE 0.8838204
    Low CONFIDENCE_FACE 0.85938364
    Low CONFIDENCE_FACE 0.8860084
    Low CONFIDENCE_FACE 0.8741514
    [[9.999924e-01 7.634048e-06]]
    Low CONFIDENCE_FACE 0.89449614
    [[9.9998808e-01 1.1954343e-05]]
    [[9.9997783e-01 2.2142174e-05]]
    [[9.9998975e-01 1.0274648e-05]]
    [[9.9995363e-01 4.6324312e-05]]
    [[9.9997783e-01 2.2206334e-05]]
    [[9.9998951e-01 1.0482572e-05]]
    [[9.9998522e-01 1.4789501e-05]]
    [[9.99985456e-01 1.45968215e-05]]
    [[9.9998617e-01 1.3816749e-05]]
    [[9.9998260e-01 1.7353308e-05]]
    [[9.9998224e-01 1.7764520e-05]]
    [[9.9997985e-01 2.0191392e-05]]
    [[9.9996161e-01 3.8407805e-05]]
    [[9.9997580e-01 2.4178973e-05]]
    [[9.9997973e-01 2.0281166e-05]]
    [[9.9997890e-01 2.1129503e-05]]
    [[9.9992263e-01 7.7365279e-05]]
    [[0.9637711  0.03622883]]
    [[0.9655266  0.03447345]]
    [[9.9993944e-01 6.0604056e-05]]
    [[0.96469784 0.03530216]]
    [[0.94482714 0.05517283]]
    [[0.95429456 0.0457054 ]]
    [[0.92537516 0.07462488]]
    [[0.93357944 0.06642057]]
    [[0.9385832  0.06141676]]
    [[0.95551264 0.04448732]]
    [[0.9586996  0.04130036]]
    [[0.95378137 0.04621871]]
    [[0.95848423 0.04151572]]
    [[0.9536278  0.04637217]]
    [[0.94894177 0.05105823]]
    [[0.9266214  0.07337865]]
    [[0.9311236  0.06887641]]
    [[0.94804454 0.0519555 ]]
    [[0.94941777 0.0505823 ]]
    [[0.9477013  0.05229879]]
    [[0.9547586  0.04524139]]
    [[0.9096597  0.09034026]]
    [[0.9556387  0.04436132]]
    [[0.9098226  0.09017736]]
    [[0.92900515 0.07099482]]
    [[0.9429705 0.0570295]]
    [[0.9618354  0.03816456]]
    [[0.92768115 0.07231887]]
    [[0.8891724  0.11082761]]
    [[0.8202518  0.17974819]]
    [[0.52193487 0.4780651 ]]
    [[0.3279555  0.67204446]]
    Low CONFIDENCE_FACE 0.8651485
    Low CONFIDENCE_FACE 0.83407444
    [[0.81551695 0.18448305]]
    Low CONFIDENCE_FACE 0.8126039
    [[0.7756342  0.22436582]]
    [[0.34447944 0.65552056]]
    [[0.75415146 0.24584846]]
    Low CONFIDENCE_FACE 0.8500269
    [[9.9964690e-01 3.5303563e-04]]
    [[0.7604608  0.23953916]]
    [[0.69646204 0.303538  ]]
    [[0.5782617  0.42173833]]
    [[0.58556163 0.41443837]]
    [[0.7136046  0.28639534]]
    [[0.87438816 0.12561187]]
    [[0.9114661 0.0885339]]
    [[0.8629262  0.13707383]]
    [[9.9983549e-01 1.6453935e-04]]
    Low CONFIDENCE_FACE 0.83622754
    Low CONFIDENCE_FACE 0.88209677
    [[9.9923575e-01 7.6426507e-04]]
    [[9.993181e-01 6.818777e-04]]
    [[9.991239e-01 8.761056e-04]]
    [[0.99802434 0.00197565]]
    [[0.9910081  0.00899191]]
    [[0.9909567  0.00904333]]
    [[0.99706036 0.00293958]]
    [[0.9972025  0.00279751]]
    [[9.9911195e-01 8.8802067e-04]]
    [[0.99583614 0.00416382]]
    [[0.9959722  0.00402781]]
    [[0.9932728  0.00672721]]
    [[0.9988059  0.00119418]]
    [[9.992441e-01 7.559025e-04]]
    [[0.9989706 0.0010294]]
    [[0.9984615  0.00153849]]
    [[0.99813133 0.00186871]]
    [[0.9982287  0.00177132]]
    [[0.9987878  0.00121221]]
    [[9.991424e-01 8.575591e-04]]
    [[9.990503e-01 9.496481e-04]]
    [[0.9981408  0.00185919]]
    [[0.9989102  0.00108987]]
    [[0.9975593  0.00244064]]
    [[0.9983102  0.00168985]]
    [[0.99849606 0.00150394]]
    [[0.99870455 0.00129548]]
    Low CONFIDENCE_FACE 0.87674844
    [[0.9955504  0.00444957]]
    [[0.9960835  0.00391643]]
    [[9.9906796e-01 9.3202438e-04]]
    [[9.990427e-01 9.573167e-04]]
    [[9.9937695e-01 6.2313327e-04]]
    [[0.99828726 0.00171267]]
    [[0.9969086  0.00309145]]
    [[0.9960204 0.0039796]]
    [[0.99531317 0.00468677]]
    [[0.994833   0.00516701]]
    [[0.9947961  0.00520388]]
    [[0.9950028  0.00499722]]
    [[0.9950466  0.00495335]]
    [[0.9956655  0.00433454]]
    [[0.9964791  0.00352093]]
    [[0.99441296 0.00558706]]
    [[0.99508876 0.00491125]]
    [[0.99522567 0.00477433]]
    [[0.99509096 0.00490905]]
    [[0.9960588  0.00394118]]
    [[0.996384 0.003616]]
    [[0.9966024  0.00339754]]
    [[0.9960865  0.00391348]]
    [[0.9949626  0.00503743]]
    [[0.9953353  0.00466469]]
    [[0.9945775  0.00542253]]
    [[0.99503136 0.00496866]]
    [[0.9952669  0.00473306]]
    [[0.9950599  0.00494007]]
    [[0.9954541  0.00454585]]
    [[0.99513865 0.00486137]]
    [[0.99376595 0.00623399]]
    [[0.9967334  0.00326664]]
    [[9.994925e-01 5.074363e-04]]
    [[9.9964786e-01 3.5218519e-04]]
    [[0.9984408  0.00155921]]
    [[9.9907815e-01 9.2184462e-04]]
    [[9.9929154e-01 7.0843182e-04]]
    [[9.9960560e-01 3.9442405e-04]]
    [[9.9962986e-01 3.7008020e-04]]
    [[9.9971372e-01 2.8631036e-04]]
    [[9.9938262e-01 6.1740016e-04]]
    [[9.992083e-01 7.918085e-04]]
    [[0.9986766  0.00132342]]
    [[0.9988011  0.00119887]]
    [[9.9912983e-01 8.7020197e-04]]
    [[0.9907658  0.00923418]]
    [[0.9918937  0.00810622]]
    [[0.99087244 0.00912753]]
    [[0.99896586 0.00103418]]
    [[9.9903309e-01 9.6694176e-04]]
    [[9.9936086e-01 6.3915941e-04]]
    [[9.9927753e-01 7.2245754e-04]]
    [[9.994584e-01 5.416377e-04]]
    [[9.994764e-01 5.236629e-04]]
    [[9.9952936e-01 4.7063525e-04]]
    [[9.994978e-01 5.022142e-04]]
    [[0.99890566 0.00109432]]
    [[0.9800745 0.0199255]]
    [[0.9923211  0.00767899]]
    [[9.9965799e-01 3.4198095e-04]]
    [[9.9990845e-01 9.1537542e-05]]
    [[0.9961449  0.00385514]]
    [[9.996001e-01 3.999521e-04]]
    [[9.9945253e-01 5.4750987e-04]]
    [[9.993393e-01 6.607443e-04]]
    [[9.9935776e-01 6.4220640e-04]]
    [[9.9942720e-01 5.7283696e-04]]
    [[9.9941206e-01 5.8798864e-04]]
    [[9.9945039e-01 5.4964155e-04]]
    [[9.9957615e-01 4.2382695e-04]]
    [[9.9973577e-01 2.6416857e-04]]
    [[9.9963856e-01 3.6148631e-04]]
    [[9.9969363e-01 3.0643673e-04]]
    [[9.9966884e-01 3.3120887e-04]]
    [[9.994319e-01 5.680865e-04]]
    [[9.9971265e-01 2.8731447e-04]]
    [[9.9967754e-01 3.2242358e-04]]
    [[0.99823403 0.00176589]]
    [[9.9935240e-01 6.4758735e-04]]
    [[9.9925286e-01 7.4711611e-04]]
    [[9.9938285e-01 6.1706477e-04]]
    [[9.9953747e-01 4.6250279e-04]]
    [[9.9958712e-01 4.1284965e-04]]
    [[9.9957865e-01 4.2130650e-04]]
    [[9.997712e-01 2.287866e-04]]
    Low CONFIDENCE_FACE 0.85542315
    [[9.9969637e-01 3.0366165e-04]]
    Low CONFIDENCE_FACE 0.47966802
    Low CONFIDENCE_FACE 0.58707684
    Low CONFIDENCE_FACE 0.42296803
    Low CONFIDENCE_FACE 0.82349485
    Low CONFIDENCE_FACE 0.39022785
    Low CONFIDENCE_FACE 0.5082578
    Low CONFIDENCE_FACE 0.6949264
    Low CONFIDENCE_FACE 0.2952118
    Low CONFIDENCE_FACE 0.2036937
    Low CONFIDENCE_FACE 0.13645898
    Low CONFIDENCE_FACE 0.17386356
    Low CONFIDENCE_FACE 0.13128085
    Low CONFIDENCE_FACE 0.13417368
    Low CONFIDENCE_FACE 0.15209262
    Low CONFIDENCE_FACE 0.28783774
    Low CONFIDENCE_FACE 0.6745256
    Low CONFIDENCE_FACE 0.26470515
    Low CONFIDENCE_FACE 0.18312697
    Low CONFIDENCE_FACE 0.13624254
    Low CONFIDENCE_FACE 0.17796788
    Low CONFIDENCE_FACE 0.13069569
    Low CONFIDENCE_FACE 0.1310402
    Low CONFIDENCE_FACE 0.13121031
    Low CONFIDENCE_FACE 0.13176244
    Low CONFIDENCE_FACE 0.13099621
    Low CONFIDENCE_FACE 0.13002826
    Low CONFIDENCE_FACE 0.12933245
    Low CONFIDENCE_FACE 0.13300392
    Low CONFIDENCE_FACE 0.1340986
    Low CONFIDENCE_FACE 0.13679013
    Low CONFIDENCE_FACE 0.1414786
    Low CONFIDENCE_FACE 0.13057747
    Low CONFIDENCE_FACE 0.15477897
    Low CONFIDENCE_FACE 0.2815235
    Low CONFIDENCE_FACE 0.5749421
    Low CONFIDENCE_FACE 0.1811666
    Low CONFIDENCE_FACE 0.13516128
    Low CONFIDENCE_FACE 0.16512947
    Low CONFIDENCE_FACE 0.21880317
    Low CONFIDENCE_FACE 0.1296066
    Low CONFIDENCE_FACE 0.1365106
    Low CONFIDENCE_FACE 0.5438485
    Low CONFIDENCE_FACE 0.2512913
    Low CONFIDENCE_FACE 0.1317276
    Low CONFIDENCE_FACE 0.13022307
    Low CONFIDENCE_FACE 0.12880105
    Low CONFIDENCE_FACE 0.12946387
    Low CONFIDENCE_FACE 0.3986798
    Low CONFIDENCE_FACE 0.17132008
    Low CONFIDENCE_FACE 0.14465316
    Low CONFIDENCE_FACE 0.13065624
    Low CONFIDENCE_FACE 0.23391043
    Low CONFIDENCE_FACE 0.18783936
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_4940/1490869148.py in <module>
          6 
          7     net.setInput(blob)
    ----> 8     detections = net.forward()
          9 
         10     detection = detections[0, 0]
    

    KeyboardInterrupt: 


![title](test.png)


```python

```






<p align="center">
  <img src="/assets/Hand_Gesture_Detection_Rev_00/.png">
</p>
