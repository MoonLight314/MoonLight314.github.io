---
title: "Deep Face Recognition - A Survey"
categories: Deep Learning
---
# Deep Face Recognition - A Survey

<br>

## Abstract

<br>
<br>

**Deep learning applies multiple processing layers to learn representations of data with multiple levels of feature extraction.**  
Deep learning은 multiple processing layer를 적용하여 representations of data를 학습하고, 여러 단계의 feature extraction을 수행합니다.

<br>

**This emerging technique has reshaped the research landscape of face recognition (FR) since 2014, launched by the breakthroughs of DeepFace and DeepID.**  
이 새로운 기술은 DeepFace와 DeepID를 시작으로 2014년 이후 얼굴 인식(FR) 연구에 새로운 장을 열었습니다.

<br>

**Since then, deep learning technique, characterized by the hierarchical architecture to stitch together pixels into invariant face representation, has dramatically improved the state-of-the-art performance and fostered successful real-world applications.**  
이후로 deep learning technique, 즉 계층적인 구조를 통해 Pixel을 invariant face representation로 조합하는 것이 성능을 크게 향상시키고 성공적인 실제 응용 프로그램을 촉진하였습니다.

<br>

**In this survey, we provide a comprehensive review of the recent developments on deep FR,covering broad topics on algorithm designs, databases, protocols, and application scenes.**  
이 논문에서는 최근 deep FR의 발전에 대한 포괄적인 리뷰를 제공하며, algorithm designs, databases, protocols, application scenes에 대한 광범위한 주제를 다룹니다.

<br>

**First, we summarize different network architectures and loss functions proposed in the rapid evolution of the deep FR methods.**  
첫째, Deep FR 방법론의 빠른 진화에서 제안된 다양한 Network Architecture와 Loss Function를 요약합니다.

<br>

**Second, the related face processing methods are categorized into two classes: “one-to-many augmentation” and “many-to-one normalization”.**  
둘째, 관련 얼굴 처리 방법은 "one-to-many augmentation"과 "many-to-one normalization" 두 가지 클래스로 분류됩니다.

<br>

**Then, we summarize and compare the commonly used databases for both model training and evaluation.**  
그런 다음, Model Train과 Evaluation를 위해 일반적으로 사용되는 Database를 요약하고 비교합니다.

<br>

**Third, we review miscellaneous scenes in deep FR, such as cross-factor, heterogenous, multiple-media and industrial scenes.**  
셋째, 우리는 교차 요인(cross-factor), 이종(heterogenous), 다중 미디어(multiple-media) 및 산업 씬(industrial scenes) 등의 Deep FR에서 다양한 Scene을 검토합니다.

<br>

**Finally, the technical challenges and several promising directions are highlighted.**  
마지막으로, 기술적인 도전과 몇 가지 유망한 방향을 강조합니다.  


<br>
<br>
<br>

## I. INTRODUCTION  

<br>

**Face recognition (FR) has been the prominent biometric technique for identity authentication and has been widely used in many areas, such as military, finance, public security and daily life.**  
얼굴인식(FR)은 신원인증을 위한 대표적인 생체인식 기술로 군사, 금융, 치안, 일상생활 등 다양한 분야에서 널리 활용되고 있다.

<br>

**FR has been a long-standing research topic in the CVPR community.**  
FR은 CVPR 커뮤니티에서 오랜 연구 주제였습니다.

<br>

**In the early 1990s, the study of FR became popular following the introduction of the historical Eigenface approach.**  
1990년대 초, 역사적인 Eigenface 접근 방식이 도입되면서 FR에 대한 연구가 대중화되었습니다.

<br>

**The milestones of feature-based FR over the past years are presented in Fig. 1, in which the times of four major technical streams are highlighted.**  
지난 몇 년 동안 feature-based FR의 이정표가 그림 1에 나와 있으며, 여기에서 네 가지 주요 기술 흐름이 표시됩니다.

<br>

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_01.png">
</p>
<br>
<br>
