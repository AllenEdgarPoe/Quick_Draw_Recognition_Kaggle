# Quick_Draw_Recognition_Kaggle

### Easy Data Preprocessing
--by Joo Kyung Song, <br>
--Submit to "Programmers, Winter Coding" 

### 데이터 전처리
1. 데이터 규모 확인<br>
2. 320개 주어진 csv 파일 shuffle하여 100개로 나눠 합침. 파일 규모가 너무 크기 때문에 csv.gz 파일로 압축해서 활용 <br>
3. PIL 라이브러리 활용해 64*64 이미지로 convert 하여 그림으로 나타냄.<br>4. one_hot_coding 기법 이용: 324개의 y_label을 np.eye(324)를 사용해서 원-핫인코딩함. <br>

### CNN 학습
1. Tensorflow 활용하려 했으나, tensorflow-gpu 버전과 cuda 버전이 맞지 않는 관계로 계속 해결할 수 없는 문제 발생함.<br>
2. Cuda 문제로 keras또한 사용할 수 없어서 고민 끝에 "밑바닥부터 시작하는 딥러닝" 서적 참고하여 Deep-convnet 구현 <br>
3. 네트워크 구성은 아래와 같음 <br>**** conv - relu - conv - relu - pool - <br> conv - relu - conv - relu - pool - <br> conv - relu - conv - relu - pool - <br>affine - relu - dropout - affine - dropout - softmax****<br>hidden_size = 50 

#### Required Libraries
```
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import tensorflow as tf
```
