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

**The holistic approaches derive the low-dimensional representation through certain distribution assumptions, such as linear subspace, manifold, and sparse representation.**  
전체적인 접근 방식은 linear subspace, manifold, sparse representation과 같은 특정 분포 가정을 통해 low-dimensional representation을 도출합니다.

<br>

**This idea dominated the FR community in the 1990s and 2000s.**  
이 아이디어는 1990년대와 2000년대에 FR 커뮤니티를 지배했습니다.

<br>

**However, a well-known problem is that these theoretically plausible holistic methods fail to address the uncontrolled facial changes that deviate from their prior assumptions.**  
그러나 문제는 이러한 이론적으로 그럴듯한 방법이 이전 가정에서 벗어나는 통제되지 않은 얼굴 변화(uncontrolled facial changes)를 해결하지 못한다는 것입니다.

<br>

**In the early 2000s, this problem gave rise to local-feature-based FR.**  
2000년대 초, 이 문제는 local-feature-based FR을 발생시켰습니다. 

<br>

**Gabor and LBP, as well as their multilevel and high-dimensional extensions, achieved robust performance through some invariant properties of local filtering.**  
Gabor 및 LBP와 그들의 high-dimensional extensions은 invariant properties of local filtering을 통해 강력한 성능을 달성했습니다.

<br>

**Unfortunately, handcrafted features suffered from a lack of distinctiveness and compactness.**  
불행히도 handcrafted features은 독특함과 간결함이 부족했습니다.

<br>

**In the early 2010s, learning-based local descriptors were introduced to the FR community, in which local filters are learned for better distinctiveness and the encoding codebook is learned for better compactness.**  
2010년대 초반에 learning-based local descriptors가 FR 커뮤니티에 도입되었습니다. 여기서 local filters는 더 나은 식별을 위해 학습되고 더 나은 압축을 위해 encoding codebook이 학습됩니다.

<br>

**However, these shallow representations still have an inevitable limitation on robustness against the complex nonlinear facial appearance variations.**  
그러나 이러한 얕은 표현은 여전히 복잡한 비선형 얼굴 모양 변형에 대한 견고성에 불가피한 제한이 있습니다.

<br>

**In general, traditional methods attempted to recognize human face by one or two layer representations, such as filtering responses, histogram of the feature codes, or distribution of the dictionary atoms.**  
일반적으로 전통적인 방법은 필터링 응답, 특징 코드의 히스토그램 또는 사전 원자의 분포와 같은 하나 또는 두 개의 레이어 표현으로 사람의 얼굴을 인식하려고 시도했습니다.

<br>

**The research community studied intensively to separately improve the preprocessing, local descriptors, and feature transformation, but these approaches improved FR accuracy slowly.**  
연구 커뮤니티는 preprocessing, local descriptors 및 feature transformation을 개별적으로 개선하기 위해 집중적으로 연구했지만 이러한 접근 방식은 FR 정확도를 느리게 향상시켰습니다.

<br>

**What’s worse, most methods aimed to address one aspect of unconstrained facial changes only, such as lighting, pose, expression, or disguise.**
설상가상으로, 대부분의 방법은 조명, 포즈, 표정 또는 변장과 같은 제한되지 않은 얼굴 변화의 한 측면만을 다루는 것을 목표로 했습니다.

<br>

**There was no any integrated technique to address these unconstrained challenges integrally.**  
이러한 제한되지 않은 문제를 통합적으로 해결할 수 있는 통합 기술은 없었습니다.

<br>

**As a result, with continuous efforts of more than a decade, “shallow” methods only improved the accuracy of the LFW benchmark to about 95%, which indicates that “shallow” methods are insufficient to extract stable identity feature invariant to real-world changes.**  
그 결과, 10년 이상의 지속적인 노력으로 "shallow" 방법은 LFW benchmark의 정확도를 약 95%까지 향상시켰을 뿐이며, 이는 "shallow" 방법으로는 실제 변화에 영향을 받지 않는 안정적인 식별 특징을 추출하기에 충분하지 않음을 나타냅니다. .

<br>

**Due to the insufficiency of this technical, facial recognition systems were often reported with unstable performance or failures with countless false alarms in real-world applications.**  
이런 기술의 부족으로 인해 안면 인식 시스템은 실제 응용 프로그램에서 무수한 오경보와 함께 불안정한 성능 또는 실패로 보고되는 경우가 많았습니다.

<br>

**But all that changed in 2012 when AlexNet won the ImageNet competition by a large margin using a technique called deep learning.**  
그러나 2012년 AlexNet이 Deep Learning이라는 기술을 사용하여 ImageNet 경쟁에서 큰 차이로 우승하면서 모든 것이 바뀌었습니다.

<br>

**Deep learning methods, such as convolutional neural networks, use a cascade of multiple layers of processing units for feature extraction and transformation.**  
convolutional  neural network과 같은 Deep Learning 방법은 feature extraction and transformation을 위해 여러 계층의 처리 장치를 사용합니다.

<br>

**They learn multiple levels of representations that correspond to different levels of abstraction.**  
다양한 수준의 추상화에 해당하는 여러 수준의 표현을 배웁니다.

<br>

**The levels form a hierarchy of concepts, showing strong invariance to the face pose, lighting, and expression changes, as shown in Fig. 2.**  
레벨은 개념의 계층 구조를 형성하며 그림 2와 같이 얼굴 포즈, 조명 및 표정 변화에 강한 불변성을 나타냅니다.

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_02.png">
</p>
<br>
<br>

**It can be seen from the figure that the first layer of the deep neural network is somewhat similar to the Gabor feature found by human scientists with years of experience.**  
deep neural network의 첫 번째 계층은 오랜 경험을 가진 인간 과학자들이 발견한 Gabor 기능과 다소 유사하다는 것을 그림에서 알 수 있습니다.

<br>

**The second layer learns more complex texture features. The features of the third layer are more complex, and some simple structures have begun to appear, such as high-bridged nose and big eyes.**  
두 번째 레이어는 더 복잡한 텍스처 기능을 학습합니다. 세 번째 층의 특징은 더 복잡하고 높은 콧대와 큰 눈과 같은 단순한 구조가 나타나기 시작했습니다.

<br>

**In the fourth, the network output is enough to explain a certain facial attribute, which can make a special response to some clear abstract concepts such as smile, roar, and even blue eye.**  
네 번째에서 Network 출력은 특정 얼굴 속성을 설명하기에 충분하며 이는 미소, 포효, 파란 눈과 같은 일부 명확한 추상 개념에 대해 특별한 반응을 만들 수 있습니다.

<br>

**In conclusion, in deep convolutional neural networks (CNN), the lower layers automatically learn the features similar to Gabor and SIFT designed for years or even decades (such as initial layers in Fig. 2), and the higher layers further learn higher level abstraction.**  
결론적으로 deep convolutional neural network(CNN)에서 하위 계층은 수년 또는 수십년 동안 설계된 Gabor 및 SIFT와 유사한 기능(예: 그림 2의 초기 계층)을 자동으로 학습하고 상위 계층은 상위 수준 추상화를 추가로 학습합니다. .

<br>

**Finally, the combination of these higher level abstraction represents facial identity with unprecedented stability.**  
마지막으로 이러한 상위 수준 추상화의 조합은 전례 없는 안정성으로 얼굴의 정체성을 나타냅니다.    

<br>
<br>

**In 2014, DeepFace achieved the SOTA accuracy on the famous LFW benchmark, approaching human performance on the unconstrained condition for the first time (DeepFace: 97.35% vs. Human: 97.53%), by training a 9-layer model on 4 million facial images.**  
2014년에 DeepFace는 유명한 LFW benchmark에서 SOTA 정확도를 달성했으며, 400만 개의 얼굴 이미지에 대한 9계층 Model을 Train하여 제약 없는 조건에서 처음으로 인간의 성능에 접근했습니다(DeepFace: 97.35% 대 인간: 97.53%). .

<br>

**Inspired by this work, research focus has shifted to deep-learning-based approaches,and the accuracy was dramatically boosted to above 99.80% in just three years.**  
이 작업에 영감을 받아 연구 초점은 Deep Learning 기반 접근 방식으로 전환되었으며 정확도는 불과 3년 만에 99.80% 이상으로 크게 향상되었습니다.

<br>

**Deep learning technique has reshaped the research landscape of FR in almost all aspects such as algorithm designs, training/test datasets, application scenarios and even the evaluation protocols.**  
Deep Learning 기술은 알고리즘 설계, training/test datasets, application scenarios 및 Evaluation protocol과 같은 거의 모든 측면에서 FR의 연구 환경을 재구성했습니다.

<br>

**Therefore, it is of great significance to review the breakthrough and rapid development process in recent years.**  
따라서 최근 몇 년간의 혁신적이고 급속한 발전 과정을 검토하는 것은 큰 의미가 있습니다.

<br>

**There have been several surveys on FR and its subdomains, and they mostly summarized and compared a diverse set of techniques related to a specific FR scene, such as illumination-invariant FR, 3D FR, pose-invariant FR.**  
FR과 그 하위 영역에 대한 여러 조사가 있었고, 그들은 주로 조명 불변 FR, 3D FR, 포즈 불변 FR과 같은 특정 FR 장면과 관련된 다양한 기술 세트를 요약하고 비교했습니다.

<br>

**Unfortunately, due to their earlier publication dates, none of them covered the deep learning methodology that is most successful nowadays.**  
불행하게도, 그들의 발표 날짜가 더 빨랐기 때문에 그들 중 누구도 오늘날 가장 성공적인 Deep Learning 방법론을 다루지 않았습니다.

<br>

**This survey focuses only on recognition problem.**  
이 조사는 인식 문제에만 초점을 맞추고 있다.

<br>

**one can refer to Ranjan et al. for a brief review of a full deep FR pipeline with detection and alignment, or refer to Jin et al. for a survey of face alignment.**  
brief review of a full deep FR pipeline with detection and alignment에 관련된 내용은 Ranjan et al.을 참고하면 되고, survey of face alignment 관련 내용은 Jin et al.을 참고하면 된다.

<br>

**Specifically, the major contributions of this survey are as follows:**  
구체적으로 이 조사의 주요 기여는 다음과 같습니다.

<br>

**• A systematic review on the evolution of the network architectures and loss functions for deep FR is provided.**  
• 심층 FR에 대한 Network Architecture 및 loss functions의 진화에 대한 체계적인 검토가 제공됩니다.

<br>

**Various loss functions are categorized into Euclideandistance-based loss, angular/cosine-margin-based loss and softmax loss and its variations.**  
다양한 Loss Function는 Euclideandistance-based loss, angular/cosine-margin-based loss 및 softmax loss and its variations으로 분류됩니다.

<br>

**Both the mainstream network architectures, such as Deepface, DeepID series, VGGFace, FaceNet, and VGGFace2, and other architectures designed for FR are covered.**  
Deepface, DeepID 시리즈, VGGFace, FaceNet 및 VGGFace2와 같은 주류 Network Architecture와 FR용으로 설계된 기타 Architecture를 모두 다룹니다.

<br>

**• We categorize the new face processing methods based on deep learning, such as those used to handle recognition difficulty on pose changes, into two classes: “one-tomany augmentation” and “many-to-one normalization”, and discuss how emerging generative adversarial network(GAN) facilitates deep FR.**  
• 포즈 변화에 대한 인식 어려움을 처리하는 데 사용되는 것과 같은 Deep Learning에 기반한 새로운 얼굴 처리 방법을 "one-tomany augmentation"와 "many-to-one normalization"의 두 가지 클래스로 분류하고, generative adversarial network(GAN)이 어떻게 deep FR을 용이하게 하는지도 알아봅니다.

<br>

**• We present a comparison and analysis on public available databases that are of vital importance for both model training and testing.**  
• Model training and testing 모두에 매우 중요한 공개 Database에 대한 비교 및 분석을 제공합니다.

<br>

**Major FR benchmarks, such as LFW, IJB-A/B/C, Megaface, and MSCeleb-1M, are reviewed and compared, in term of the four aspects: training methodology, evaluation tasks and metrics, and recognition scenes, which provides an useful reference for training and testing deep FR.**  
LFW, IJB-A/B/C, Megaface 및 MSCeleb-1M과 같은 주요 FR benchmark를 training methodology, Evaluation Task 및 metrics, recognition scenes의 네 가지 측면에서 검토하고 비교합니다. Deep FR Train 및 테스트에 유용한 참고 자료입니다.

<br>

**• Besides the general purpose tasks defined by the major databases, we summarize a dozen scenario-specific databases and solutions that are still challenging for deep learning, such as anti-attack, cross-pose FR, and cross-age FR.**  
• 주요 Database에서 정의한 범용 작업 외에도 anti-attack, cross-pose FR 및 cross-age FR과 같이 Deep Learning에 여전히 도전적인 12개의 시나리오별 Database 및 솔루션을 요약합니다.

<br>

**By reviewing specially designed methods for these unsolved problems, we attempt to reveal the important issues for future research on deep FR, such as adversarial samples, algorithm/data biases, and model interpretability.**  
이러한 미해결 문제를 위해 특별히 설계된 방법을 검토함으로써 우리는 adversarial samples, algorithm/data biases 그리고 model interpretability과 같은 deep FR에 대한 향후 연구의 중요한 문제를 밝히려고 시도합니다.

<br>

**The remainder of this survey is structured as follows.**  
이 조사의 나머지 부분은 다음과 같이 구성됩니다.

<br>

**In Section II, we introduce some background concepts and terminologies, and then we briefly introduce each component of FR.**  
II장에서는 몇 가지 background concepts과 terminologies를 소개하고 FR의 각 구성 요소를 간략하게 소개합니다.

<br>

**In Section III, different network architectures and loss functions are presented. Then, we summarize the face processing algorithms and the datasets.**  
섹션 III에서는 다양한 Network Architecture와 Loss Function을 소개합니다. 그런 다음 face processing algorithms과 datasets를 요약합니다.

<br>

**In Section V, we briefly introduce several methods of deep FR used for different scenes.**  
섹션 V에서는 다양한 scenes에 사용되는 몇 가지 deep FR 방법을 간략하게 소개합니다.

<br>

**Finally, the conclusion of this paper and discussion of future works are presented in Section VI.**  
마지막으로 본 논문의 결론과 향후 연구에 대한 논의는 Ⅵ장에서 제시한다.  

<br>

## II. OVERVIEW  

<br>
<br>
<br>
<br>

### A. Components of Face Recognition

**As mentioned in [32], there are three modules needed for FR system, as shown in Fig. 3.**  
Fig. 3에 보여지는 것처럼 FR 시스템에는 세 가지 Module이 필요합니다.

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_03.png">
</p>
<br>
<br>

**First, a face detector is used to localize faces in images or videos.**  
첫째, face detector는 이미지나 비디오에서 얼굴을 찾아냅니다.

<br>

**Second, with the facial landmark detector, the faces are aligned to normalized canonical coordinates.**  
둘째, facial landmark detector를 통해 normalized canonical coordinates로 정렬됩니다.

<br>

**Third, the FR module is implemented with these aligned face images.**  
셋째, 이 aligned 얼굴 이미지들로 FR Module이 구현됩니다.

<br>

**We only focus on the FR module throughout the remainder of this paper.**  
이 논문의 나머지 부분에서는 FR Module에만 집중합니다.

<br>

**Before a face image is fed to an FR module, face antispoofing, which recognizes whether the face is live or spoofed, is applied to avoid different types of attacks.**  
얼굴 이미지가 FR Module로 전달되기 전에, 얼굴이 실제인지 가짜인지를 인식하는 face antispoofing이 적용되어 다양한 유형의 공격을 방지합니다.

<br>

**Then, recognition can be performed.**  
그런 다음, 인식이 수행될 수 있습니다.

<br>

**As shown in Fig. 3(c), an FR module consists of face processing, deep feature extraction and face matching, and it can be described as follows:**  
Fig. 3(c)에서 보여지는 것처럼, FR Module은 얼굴 처리, Deep feature extraction, face matching으로 구성되며, 다음과 같이 설명될 수 있습니다:  

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_01.png">
</p>
<br>
<br>

**where Ii and Ij are two face images, respectively.**  
Ii와 Ij는 각각 두 얼굴 이미지를 나타냅니다.

<br>

**P stands for face processing to handle intra-personal variations before training and testing, such as poses, illuminations, expressions and occlusions.**  
P는 학습과 테스트 전에 자세, 조명, 표정 및 가림 현상과 같은 개인 내 변동을 처리하기 위한 얼굴 처리를 의미합니다.

<br>

**F denotes feature extraction, which encodes the identity information.**  
F는 신원 정보를 Encoding하는 feature extraction을 나타냅니다.

<br>

**The feature extractor is learned by loss functions when training, and is utilized to extract features of faces when testing.**  
feature extractor는 학습 시 Loss Function를 통해 학습되며, 테스트 시 features of faces를 추출하는데 사용됩니다.

<br>

**M means a face matching algorithm used to compute similarity scores of features to determine the specific identity of faces.**  
M은 얼굴의 특정 신원을 결정하기 위해 피처의 similarity scores를 계산하는데 사용되는 face matching algorithm을 의미합니다.

<br>

**Different from object classification, the testing identities are usually disjoint from the training data in FR, which makes the learned classifier cannot be used to recognize testing faces.**  
object classification와 달리, 얼굴 인식(FR)에서 테스트 신원은 대개 학습 데이터와 분리되어 있어, 학습된 classifier는 테스트 얼굴을 인식하는데 사용될 수 없습니다.

<br>

**Therefore, face matching algorithm is an essential part in FR.**  
따라서, face matching algorithm은 FR에서 필수적인 부분입니다.  

<br>
<br>
<br>

## 1) Face Processing  

<br>
<br>

**Although deep-learning-based approaches have been widely used, Mehdipour et al. [46] proved that various conditions, such as poses, illuminations, expressions and occlusions, still affect the performance of deep FR.**  
Deep Learning 기반의 접근법들이 널리 사용되고 있지만, Mehdipour 등[46]은 포즈, 조명, 표정, 가림 등 다양한 조건들이 여전히 Deep FR의 성능에 영향을 미친다는 것을 증명했습니다.

<br>

**Accordingly, face processing is introduced to address this problem.**  
따라서, 이 문제를 해결하기 위해 face processing이 도입되었습니다.

<br>

**The face processing methods are categorized as “one-to-many augmentation” and “many-to-one normalization”, as shown in Table I.**  
face processing method들은 Table I에 보여진 것과 같이 "one-to-many augmentation"과 "many-to-one normalization"로 분류됩니다.

<br>

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Table_01.png">
</p>
<br>
<br>

**• “One-to-many augmentation”. These methods generate many patches or images of the pose variability from a single image to enable deep networks to learn poseinvariant representations.**  
• "One-to-many augmentation". 이 방법들은 single image에서 포즈의 변동성에 대한 많은 패치나 이미지를 생성하여 Deep Network가 포즈에 불변한 표현을 학습하도록 합니다.

<br>

**• “Many-to-one normalization”. These methods recover the canonical view of face images from one or many images of a nonfrontal view; then, FR can be performed as if it were under controlled conditions.**  
• "Many-to-one normalization". 이 방법들은 하나 또는 많은 수의 비정면 이미지(nonfrontal view)들로부터 얼굴 이미지의 정규화된 뷰(canonical view of face images)를 복구하고, 그런 다음에 FR을 제어된 조건하에서 수행된 것처럼 실행할 수 있습니다.

<br>

**Note that we mainly focus on deep face processing method designed for pose variations in this paper, since pose is widely regarded as a major challenge in automatic FR applications and other variations can be solved by the similar methods.**  
주의할 점은 우리가 본 논문에서 주로 포즈 변동성에 대해 설계된 Deep face processing 방법에 집중한다는 것인데, 이는 포즈가 자동 FR 응용 프로그램에서 주요 도전 과제로 널리 인식되고 있으며, 다른 변동성들은 비슷한 방법들로 해결될 수 있기 때문입니다.      

<br>
<br>
<br>


## 2) Deep Feature Extraction: Network Architecture. 

<br>
<br>

**The architectures can be categorized as backbone and assembled networks, as shown in Table II.**  
Architecture들은 표 II에 표시된 대로 backbone 및 assembled Network로 분류할 수 있습니다.

<br>

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Table_02.png">
</p>
<br>
<br>

**Inspired by the extraordinary success on the ImageNet [74] challenge, the typical CNN architectures, e.g. AlexNet, VGGNet, GoogleNet, ResNet and SENet, are introduced and widely used as the baseline models in FR (directly or slightly modified).**  
ImageNet [74] 챌린지의 놀라운 성공에 영감을 받아 전형적인 CNN Architecture, 예를 들어 AlexNet, VGGNet, GoogleNet, ResNet 및 SENet이 도입되어 FR의 기본 Model로 널리 사용됩니다(직접 또는 약간 수정됨).

<br>

**In addition to the mainstream, some assembled networks, e.g. multi-task networks and multi-input networks, are utilized in FR.**  
주류 외에도 일부 assembled networks, 예를 들어, multi-task networks 및 multi-input networks는 FR에서 활용됩니다. 

<br>

**Hu et al. shows that accumulating the results of assembled networks provides an increase in performance compared with an individual network.**  
Huet al.은 assembled network의 결과를 누적하면 개별 Network에 비해 성능이 향상됨을 보여줍니다.

<br>
<br>
<br>

## Loss Function 

<br>
<br>

**The softmax loss is commonly used as the supervision signal in object recognition, and it encourages the separability of features.**  
softmax loss는 일반적으로 객체 인식에서 supervision signal로 사용되며 특징의 분리성을 장려합니다.

<br>

**However, the softmax loss is not sufficiently effective for FR because intra-variations could be larger than inter-differences and more discriminative features are required when recognizing different people.**  
그러나 softmax loss는 inter-difference보다 intra-variation이 클 수 있고 다른 사람을 인식할 때 더 많은 discriminative 특징이 필요하기 때문에 FR에 대해 충분히 효과적이지 않습니다.

<br>

**Many works focus on creating novel loss functions to make features not only more separable but also discriminative, as shown in Table III.**  
많은 작업은 표 III에 나와 있는 것처럼 features를 더 분리 가능할 뿐만 아니라 차별적으로 만들기 위해 새로운 손실 기능을 만드는 데 중점을 둡니다.    

<br>
<br>
<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Table_03.png">
</p>
<br>
<br>

## 3) Face Matching by Deep Features 

<br>
<br>
<br>

**FR can be categorized as face verification and face identification.**  
FR은 face verification과 face identification로 분류할 수 있습니다.

<br>

**In either scenario, a set of known subjects is initially enrolled in the system (the gallery), and during testing, a new subject (the probe) is presented.**  
두 시나리오 모두 set of known subjects가 처음에 시스템(Gallery)에 등록되고 테스트 중에 new subject (the probe)가 표시됩니다.

<br>

**After the deep networks are trained on massive data with the supervision of an appropriate loss function, each of the test images is passed through the networks to obtain a deep feature representation.**  
Deep Network가 적절한 Loss Function의 감독 하에 방대한 데이터에 대해 학습된 후 각 테스트 이미지가 Network를 통과하여 deep feature representation을 얻습니다.

<br>

**Using cosine distance or L2 distance, face verification computes one-to-one similarity between the gallery and probe to determine whether the two images are of the same subject, whereas face identification computes one-to-many similarity to determine the specific identity of a probe face.**  
cosine distance or L2 distance를 사용하여 얼굴 확인(face verification)은 gallery와 probe 간의 one-to-one similarity을 계산하여 두 이미지가 동일한 대상인지 여부를 결정하는 반면, 얼굴 식별(face identification)은 one-to-many similarity을 계산하여 specific identity of a probe face를 결정합니다.

<br>

**In addition to these, other methods are introduced to postprocess the deep features such that the face matching is performed efficiently and accurately, such as metric learning, sparse-representation-based classifier (SRC), and so forth.**  
이 외에도 metric learning, sparse-representation-based classifier(SRC) 등과 같이 얼굴 매칭이 효율적이고 정확하게 수행되도록 deep features을 postprocess하는 다른 방법이 도입됩니다.

<br>

**To sum up, we present FR modules and their commonlyused methods in Fig. 4 to help readers to get a view of the whole FR.**  
요약하면 독자가 전체 FR을 볼 수 있도록 그림 4에 FR Module과 일반적으로 사용되는 방법을 제시합니다.

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_04.png">
</p>
<br>
<br>

**In deep FR, various training and testing face databases are constructed, and different architectures and losses of deep FR always follow those of deep object classification and are modified according to unique characteristics of FR.**  
deep FR에서는 다양한 Train 및 테스트 얼굴 Database가 구축되며 deep FR의 다양한 Architecture와 손실은 항상 deep object classification을 따르고 FR의 고유한 특성에 따라 수정됩니다.

<br>

**Moreover, in order to address unconstrained facial changes, face processing methods are further designed to handle poses, expressions and occlusions variations.**  
또한 제약 없는 얼굴 변화를 처리하기 위해 포즈, 표정 및 폐색 변화를 처리하도록 얼굴 처리 방법이 추가로 설계되었습니다.

<br>

**Benefiting from these strategies, deep FR system significantly improves the SOTA and surpasses human performance.**  
이러한 전략의 이점을 활용하여 Deep FR 시스템은 SOTA를 크게 개선하고 인간을 능가합니다.

<br>

**When the applications of FR becomes more and more mature in general scenario, recently, different solutions are driven for more difficult specific scenarios, such as cross-pose FR, crossage FR, video FR.**  
일반적인 시나리오에서 FR의 적용이 점점 더 성숙해짐에 따라 최근에는 교차 포즈 FR, 교차 FR, 비디오 FR과 같이 더 어려운 특정 시나리오에 대해 서로 다른 솔루션이 구동됩니다.      

<br>
<br>
<br>
<br>

## III. NETWORK ARCHITECTURE AND TRAINING LOSS

<br>
<br>
<br>

**For most applications, it is difficult to include the candidate faces during the training stage, which makes FR become a “zero-shot” learning task.**  
대부분의 애플리케이션에서 Train 단계 동안 후보 얼굴을 포함하는 것은 어렵기 때문에 FR은 "제로 샷" 학습 작업이 됩니다.

<br>

**Fortunately, since all human faces share a similar shape and texture, the representation learned from a small proportion of faces can generalize well to the rest.**  
다행스럽게도 모든 사람의 얼굴은 비슷한 모양과 질감이기 때문에 얼굴의 작은 부분에서 학습된 표현은 나머지 얼굴에 잘 일반화될 수 있습니다.

<br>

**Based on this theory, a straightforward way to improve generalized performance is to include as many IDs as possible in the training set.**
이 이론에 따라 일반화된 성능을 향상시키는 간단한 방법은 Training Set에 가능한 한 많은 ID를 포함하는 것입니다.

<br>

**For example, Internet giants such as Facebook and Google have reported their deep FR system trained by 106 − 107 IDs.**  
예를 들어, Facebook 및 Google과 같은 거대 인터넷 기업은 106 - 107 ID로 Train된 Deep FR 시스템을 보고했습니다.

<br>

**Unfortunately, these personal datasets, as well as prerequisite GPU clusters for distributed model training, are not accessible for academic community.**  
안타깝게도 이러한 개인 Dataset와 분산 Model 교육을 위한 필수 GPU 클러스터는 학계에서 액세스할 수 없습니다.

<br>

**Currently, public available training databases for academic research consist of only 103−105 IDs.**  
현재 학술 연구를 위해 공개적으로 사용 가능한 Train Database는 103-105개의 ID로만 구성됩니다.

<br>

**Instead, academic community makes effort to design effective loss functions and adopts efficient architectures to make deep features more discriminative using the relatively small training data sets.**  
대신 학계에서는 효과적인 Loss Function를 설계하기 위해 노력하고 상대적으로 작은 Train Dataset를 사용하여 깊은 특성을 보다 식별할 수 있도록 효율적인 Architecture를 채택합니다.

<br>

**For instance, the accuracy of most popular LFW benchmark has been boosted from 97% to above 99.8% in the pasting four years, as enumerated in Table IV.**  
예를 들어, 가장 인기 있는 LFW benchmark의 정확도는 지난 4년 동안 97%에서 99.8% 이상으로 향상되었으며 표 IV에 열거되어 있습니다.

<br>

**In this section, we survey the research efforts on different loss functions and network architectures that have significantly improved deep FR methods.**
이 섹션에서는 Deep FR 방법을 크게 개선한 다양한 loss functions 및 Network Architecture에 대한 연구 노력을 조사합니다.  

<br>
<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Table_04.png">
</p>
<br>
<br>

### A. Evolution of Discriminative Loss Functions

<br>
<br>
<br>

**Inheriting from the object classification network such as AlexNet, the initial Deepface and DeepID adopted cross-entropy based softmax loss for feature learning.**  
AlexNet과 같은 객체 분류 Network에서 상속받은 초기 Deepface 및 DeepID는 feature learning을 위해 cross-entropy based softmax loss를 채택했습니다.

<br>

**After that, people realized that the softmax loss is not sufficient by itself to learn discriminative features, and more researchers began to explore novel loss functions for enhanced generalization ability.**  
그 후 사람들은 softmax loss만으로는 판별 기능을 학습하기에 충분하지 않다는 것을 깨달았고 더 많은 연구자들이 일반화 능력을 향상시키기 위해 새로운 Loss Function를 탐색하기 시작했습니다.

<br>

**This becomes the hottest research topic in deep FR research, as illustrated in Fig. 5.**
이는 그림 5에서 볼 수 있듯이 Deep FR 연구에서 가장 뜨거운 연구 주제가 됩니다.

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_05.png">
</p>
<br>
<br>

**Before 2017, Euclidean-distance-based loss played an important role; In 2017, angular/cosine-margin-based loss as well as feature and weight normalization became popular.**  
2017년 이전에는 Euclidean-distance-based loss가 중요한 역할을 했습니다. 2017년에는 angular/cosine-margin-based loss와 feature and weight normalization가 인기를 끌었습니다.

<br>

**It should be noted that, although some loss functions share the similar basic idea, the new one is usually designed to facilitate the training procedure by easier parameter or sample selection.**  
일부 Loss Function은 유사한 기본 아이디어를 공유하지만 새로운 Loss Function는 일반적으로 더 쉬운 매개변수 또는 샘플 선택을 통해 Train 절차를 용이하게 하도록 설계되었습니다.    

<br>
<br>
<br>
<br>

### 1) Euclidean-distance-based Loss  

<br>

**Euclidean-distancebased loss is a metric learning method that embeds images into Euclidean space in which intra-variance is reduced and inter-variance is enlarged.**  
Euclidean-distance-based loss는 intra-variance를 줄이고 inter-variance를 확대한 Euclidean 공간에 이미지를 삽입하는 메트릭 학습 방법입니다.

<br>

**The contrastive loss and the triplet loss are the commonly used loss functions.**  
Contrastive loss와 triplet loss는 일반적으로 사용되는 loss function이다.

<br>

**The contrastive loss requires face image pairs, and then pulls together positive pairs and pushes apart negative pairs.**  
대조적인 손실에는 얼굴 이미지 쌍이 필요하며 양수 쌍을 함께 당기고 음수 쌍을 밀어냅니다.  

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_02.png">
</p>
<br>
<br>

**where yij = 1 means xi and xj are matching samples and yij = 0 means non-matching samples. f(·) is the feature embedding,  + and 
− control the margins of the matching and non-matching pairs respectivel**  

<br>

**DeepID2 combined the face identification (softmax) and verification (contrastive loss) supervisory signals to learn a discriminative representation, and joint Bayesian (JB) was applied to obtain a robust embedding space.**  
DeepID2는 얼굴 식별(softmax)과 검증(contrastive loss) 감독 신호를 결합하여 차별적인 표현을 학습하고 joint Bayesian(JB)을 적용하여 강력한 임베딩 공간을 확보했습니다.

<br>

**Extending from DeepID2, DeepID2+ increased the dimension of hidden representations and added supervision to early convolutional layers.**  
DeepID2에서 확장된 DeepID2+는 숨겨진 표현의 차원을 높이고 초기 convolutional 레이어에 감독 기능을 추가했습니다.

<br>

**DeepID3 further introduced VGGNet and GoogleNet to their work.**  
DeepID3는 VGGNet과 GoogleNet을 작업에 추가로 도입했습니다.

<br>

**However, the main problem with the contrastive loss is that the margin parameters are often difficult to choose.**  
그러나 대조 손실(contrastive loss )의 주요 문제는 마진 매개변수(margin parameters )를 선택하기 어려운 경우가 많다는 것입니다.

<br>

**Contrary to contrastive loss that considers the absolute distances of the matching pairs and non-matching pairs, triplet loss considers the relative difference of the distances between them.**  
일치하는 쌍과 일치하지 않는 쌍의 절대 거리를 고려하는 Contrastive loss와 달리 triplet loss는 그들 사이의 거리의 상대적인 차이를 고려합니다.

<br>

**Along with FaceNet proposed by Google, Triplet loss was introduced into FR.**  
Google에서 제안한 FaceNet과 함께 Triplet loss가 FR에 도입되었습니다.

<br>

**It requires the face triplets, and then it minimizes the distance between an anchor and a positive sample of the same identity and maximizes the distance between the anchor and a negative sample of a different identity.**  
그것은 얼굴 삼중항(face triplets)을 필요로 하고 앵커와 동일한 신원의 positive sample 사이의 거리를 최소화하고 앵커와 다른 신원의 negative sample 사이의 거리를 최대화합니다.  

<br>
<br>  

**Inspired by FaceNet, TPE and TSE learned a linear projection W to construct triplet loss.**  
FaceNet에서 영감을 받은 TPE와 TSE는 triplet loss을 구성하기 위해 선형 프로젝션 W를 학습했습니다.

<br>

**Other methods optimize deep models using both triplet loss and softmax loss.**  
다른 방법은 triplet loss와 softmax loss를 모두 사용하여 deep Model을 최적화합니다.

<br>

**They first train networks with softmax and then fine-tune them with triplet loss.**  
그들은 먼저 softmax로 Network를 Train시킨 다음 triplet loss로 미세 조정합니다.

<br>
<br>

**However, the contrastive loss and triplet loss occasionally encounter training instability due to the selection of effective training samples, some paper begun to explore simple alternatives.**  
그러나 Contrastive loss와 triplet loss는 때때로 효과적인 Train 샘플의 선택으로 인해 Train 불안정성에 직면하며, 일부 논문에서는 간단한 대안을 탐색하기 시작했습니다.

<br>

**Center loss and its variants are good choices for reducing intra-variance.**  
중심 손실(Center loss )과 그 variants은 내부 분산(intra-variance)을 줄이기 위한 좋은 선택입니다.

<br>

**The center loss learned a center for each class and penalized the distances between the deep features and their corresponding class centers.**  
중심 손실은 각 클래스의 중심을 학습하고 deep features과 해당 class centers 사이의 거리에 불이익을 줍니다.

<br>

**This loss can be defined as follows:**  
이 손실은 다음과 같이 정의할 수 있습니다.  

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_03.png">
</p>
<br>
<br>

**where xi denotes the i-th deep feature belonging to the yi-th class and cyi denotes the yi-th class center of deep features.**  
여기서 xi는 yi번째 클래스에 속하는 i번째 Deep feature를 나타내고 cyi는 Deep features의 yi번째 클래스 중심을 나타냅니다.

<br>

**To handle the long-tailed data, a range loss, which is a variant of center loss, is used to minimize k greatest range’s harmonic mean values in one class and maximize the shortest interclass distance within one batch.**  
long-tailed 데이터를 처리하기 위해 하나의 클래스에서 k greatest range’s harmonic mean 값을 최소화하고 한 배치 내에서 shortest interclass distance를 최대화하기 위해 중심 손실의 변형인 범위 손실을 사용합니다.

<br>

**Wu et al. proposed a center-invariant loss that penalizes the difference between each center of classes.**  
Wuet al.은 클래스의 각 센터 사이의 차이에 페널티를 주는 center-invariant loss을 제안했습니다.

<br>

**Deng et al. selected the farthest intraclass samples and the nearest inter-class samples to compute a margin loss.**  
Deng et al.은 마진 손실을 계산하기 위해 가장 먼 내부 클래스 샘플과 가장 가까운 클래스 간 샘플을 선택했습니다.

<br>

**However, the center loss and its variants suffer from massive GPU memory consumption on the classification layer, and prefer balanced and sufficient training data for each identity.**  
그러나 center loss과 그 변형은 분류 계층에서 막대한 GPU 메모리 소비로 인해 어려움을 겪고 있으며 각 신원에 대해 균형 있고 충분한 Train 데이터를 선호합니다.  

<br>
<br>
<br>
<br>

### 2) Angular/cosine-margin-based Loss  

<br>
<br>

**In 2017, people had a deeper understanding of loss function in deep FR and thought that samples should be separated more strictly to avoid misclassifying the difficult samples.**  
2017년에 사람들은 Deep FR의 Loss Function에 대해 더 깊이 이해했고 어려운 샘플을 잘못 분류하지 않도록 샘플을 더 엄격하게 분리해야 한다고 생각했습니다.

<br>

**Angular/cosinemargin-based loss is proposed to make learned features potentially separable with a larger angular/cosine distance.**  
Angular/cosinemargin-based loss은 larger angular/cosine distance로 잠재적으로 분리할 수 있는 학습된 기능을 만들기 위해 제안됩니다.

<br>

**The decision boundary in softmax loss is (W1 − W2) x + b1 − b2 = 0, where x is feature vector, Wi and bi are weights and bias in softmax loss, respectively.**  
softmax 손실의 결정 경계는 (W1 − W2) x + b1 − b2 = 0입니다. 여기서 x는 특징 벡터이고 Wi와 bi는 각각 softmax 손실의 가중치와 편향입니다.

<br>

**Liu et al. reformulated the original softmax loss into a large-margin softmax (L-Softmax) loss.**  
Liu et al.은 원래의 softmax loss를 마진이 큰 Softmax(L-Softmax) 손실로 재공식화했습니다.

<br>

**They constrain b1 = b2 = 0, so the decision boundaries for class 1 and class 2 become kxk (kW1k cos (mθ1) − kW2k cos (θ2)) = 0 and kxk (kW1k kW2k cos (θ1) − cos (mθ2)) = 0, respectively, where m is a positive integer introducing an angular margin, and θi is the angle between Wi and x.**  
b1 = b2 = 0으로 제한하므로 클래스 1과 클래스 2에 대한 결정 경계는 kxk(kW1k cos(mθ1) − kW2k cos(θ2)) = 0 및 kxk(kW1k kW2k cos(θ1) − cos(mθ2))가 됩니다. = 0, 여기서 m은 각도 마진을 도입하는 양의 정수이고 θi는 Wi와 x 사이의 각도입니다.

<br>

**Due to the nonmonotonicity of the cosine function, a piece-wise function is applied in L-softmax to guarantee the monotonicity.**  
cosine function의 비단조성(nonmonotonicity )으로 인해 단조성을 보장하기 위해 L-softmax에서 구간 함수를 적용합니다.

<br>

**The loss function is defined as follows:**  
Loss Function는 다음과 같이 정의됩니다.  

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_04.png">
</p>
<br>
<br>

where  

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_05.png">
</p>
<br>
<br>

**Based on L-Softmax, A-Softmax loss further normalized the weight W by L2 norm (kWk = 1) such that the normalized vector will lie on a hypersphere, and then the discriminative face features can be learned on a hypersphere manifold with an angular margin (Fig. 6).**  
L-Softmax를 기반으로 A-Softmax 손실은 가중치 W를 L2 표준(kWk = 1)으로 더 정규화하여 정규화된 벡터가 하이퍼스피어에 놓이게 한 다음 각이 있는 하이퍼스피어 매니폴드에서 식별 가능한 얼굴 특징을 학습할 수 있습니다. (그림 6).

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_06.png">
</p>
<br>
<br>

**Liu et al. [108] introduced a deep hyperspherical convolution network (SphereNet) that adopts hyperspherical convolution as its basic convolution operator and is supervised by angular-margin-based loss.**  
Liu et al. [108]은 기본 convolutional 연산자로 hyperspherical convolutional 을 채택하고 각도-마진 기반 손실에 의해 감독되는 심층 하이퍼스페리컬 convolutional  Network(SphereNet)를 도입했습니다.

<br>

**To overcome the optimization difficulty of L-Softmax and A-Softmax, which incorporate the angular margin in a multiplicative manner, ArcFace and CosFace, AMS loss respectively introduced an additive angular/cosine margin cos(θ + m) and cosθ − m.**  
각도 마진을 곱셈 방식으로 통합하는 L-Softmax 및 A-Softmax, ArcFace 및 CosFace의 최적화 어려움을 극복하기 위해 AMS 손실은 각각 additive angular/cosine margin cos(θ + m) 및 cosθ − m을 도입했습니다.

<br>

**They are extremely easy to implement without tricky hyperparameters λ, and are more clear and able to converge without the softmax supervision.**  
까다로운 hyperparameters λ 없이 구현하기가 매우 쉽고 softmax 감독 없이 더 명확하고 수렴할 수 있습니다.

<br>

**The decision boundaries under the binary classification case are given in Table V.**  
이진 분류 사례 아래의 decision boundaries는 표 V에 나와 있습니다.

<br>
<br>

<br>
<br>
<p align="center">
  <img src="/assets/DeepFaceRecognitionSurvey/Table_05.png">
</p>
<br>
<br>

**Based on large margin, FairLoss and AdaptiveFace further proposed to adjust the margins for different classes adaptively to address the problem of unbalanced data.**  
큰 마진을 기반으로 FairLoss 및 AdaptiveFace는 불균형 데이터 문제를 해결하기 위해 서로 다른 클래스의 마진을 적응적으로 조정할 것을 추가로 제안했습니다.

<br>

**Compared to Euclidean-distance-based loss, angular/cosinemargin-based loss explicitly adds discriminative constraints on a hypershpere manifold, which intrinsically matches the prior that human face lies on a manifold.**  
Euclidean-distance-based loss과 비교할 때, angular/cosinemargin-based loss은 인간의 얼굴이 다양체에 놓이는 이전과 본질적으로 일치하는 초광각 다양체에 차별적 제약을 명시적으로 추가합니다.

<br>

**However, Wang et al. showed that angular/cosine-margin-based loss can achieve better results on a clean dataset, but is vulnerable to noise and becomes worse than center loss and softmax in the high-noise region as shown in Fig. 7.**  
그러나 Wang et al.은 angular/cosine-margin-based loss은 깨끗한 Dataset에서 더 나은 결과를 얻을 수 있지만 노이즈에 취약하고 그림 7과 같이 노이즈가 많은 영역에서 center loss 및 softmax보다 나빠집니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_07.png">
<p align="center">
</p>
<br>
<br>

### 3) Softmax Loss and its Variations

<br>
<br>

**In 2017, in addition to reformulating softmax loss into an angular/cosine-marginbased loss as mentioned above, some works tries to normalize the features and weights in loss functions to improve the model performance, which can be written as follows:**  
2017년에는 위에서 언급한 대로 softmax 손실을 angular/cosine-marginbased loss로 재공식화하는 것 외에도 일부 작업에서는 Model 성능을 개선하기 위해 Loss Function의 기능과 가중치를 정규화하려고 시도했습니다. 이는 다음과 같이 작성할 수 있습니다.    

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_06.png">
<p align="center">
</p>
<br>
<br>

**where α is a scaling parameter, x is the learned feature vector, W is weight of last fully connected layer.**  
여기서 α는 스케일링 파라미터, x는 학습된 feature vector, W는 마지막 완전 연결 레이어의 가중치입니다.

<br>

**Scaling x to a fixed radius α is important, as Wang et al. proved that normalizing both features and weights to 1 will make the softmax loss become trapped at a very high value on the training set.**  
Wang et al.처럼 x를 고정 반경 α로 스케일링하는 것이 중요합니다. 특징과 가중치를 모두 1로 정규화하면 Softmax 손실이 Train 세트에서 매우 높은 값에 갇히게 된다는 것을 증명했습니다.

<br>

**After that, the loss function, e.g. softmax, can be performed using the normalized features and weights.**  
그 후 Loss Function, 예를 들면 다음과 같습니다. softmax는 정규화된 특징과 가중치를 사용하여 수행할 수 있습니다.

<br>

**Some papers first normalized the weights only and then added angular/cosine margin into loss functions to make the learned features be discriminative.**  
일부 논문에서는 먼저 가중치만 정규화한 다음 Loss Function에 angular/cosine margin을 추가하여 학습된 features을 구별하도록 했습니다.

<br>

**In contrast, some works, such as, adopted feature normalization only to overcome the bias to the sample distribution of the softmax.**  
대조적으로, 일부 작업은 softmax의 샘플 분포에 대한 편향을 극복하기 위해서만 기능 정규화를 채택했습니다.

<br>

**Based on the observation of [125] that the L2-norm of features learned using the softmax loss is informative of the quality of the face, L2-softmax enforced all the features to have the same L2-norm by feature normalization such that similar attention is given to good quality frontal faces and blurry faces with extreme pose.**  
softmax 손실을 사용하여 학습한 특징의 L2-norm이 얼굴의 품질에 대한 정보를 제공한다는 [125]의 관찰을 기반으로 L2-softmax는 특징 정규화를 통해 모든 특징이 동일한 L2-norm을 갖도록 강제했습니다. 좋은 품질의 정면 얼굴과 극단적인 포즈의 흐릿한 얼굴에 부여됩니다.  

<br>
<br>

**Ring loss encouraged the norm of samples being value R (a learned parameter) rather than explicit enforcing through a hard normalization operation.**  
Ring loss은 hard normalization operation을 통해 명시적으로 적용하기보다는 값 R(학습된 매개변수)이 되는 샘플의 표준을 장려했습니다.

<br>

**Moreover, normalizing both features and weights has become a common strategy.**  
또한 features와 weights를 모두 정규화하는 것이 일반적인 전략이 되었습니다.

<br>

**Wang et al. [110] explained the necessity of this normalization operation from both analytic and geometric perspectives.**  
Wang et al. [110]은 분석적 관점과 기하학적 관점 모두에서 이 정규화 작업의 필요성을 설명했습니다.

<br>

**After normalizing features and weights, CoCo loss [112] optimized the cosine distance among data features, and Hasnat et al. used the von MisesFisher (vMF) mixture model as the theoretical basis to develop a novel vMF mixture loss and its corresponding vMF deep features.**  
features 와 weights를 정규화한 후 CoCo loss[112]는 data features간의 cosine distance를 최적화했으며 Hasnat et al. von MisesFisher(vMF)은 혼합 Model을 이론적 기반으로 사용하여 새로운 vMF 혼합 손실 및 해당 vMF Deep Features을 개발했습니다.  

<br>
<br>
<br>

### B. Evolution of Network Architecture

<br>
<br>

### 1) Backbone Network

<br>
<br>

#### Mainstream architectures. 

<br>
<br>

**The commonly used network architectures of deep FR have always followed those of deep object classification and evolved from AlexNet to SENet rapidly.**  
Deep FR의 일반적으로 사용되는 Network Architecture는 항상 Deep Object Classification을 따랐으며 AlexNet에서 SENet으로 빠르게 발전했습니다.

<br>

**We present the most influential architectures of deep object classification and deep face recognition in chronological order 1 in Fig. 8.**  
deep object classification 및 deep face recognition 의 가장 영향력 있는 Architecture를 그림 8의 연대순 1로 제시합니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_08.png">
<p align="center">
</p>
<br>
<br>

**In 2012, AlexNet was reported to achieve the SOTA recognition accuracy in the ImageNet large-scale visual recognition competition (ILSVRC) 2012, exceeding the previous best results by a large margin.**  
2012년 AlexNet은 ImageNet 대규모 시각 인식 대회(ILSVRC) 2012에서 SOTA 인식 정확도를 달성하여 이전 최고 결과를 크게 뛰어넘는 것으로 보고되었습니다.

<br>

**AlexNet consists of five convolutional layers and three fully connected layers, and it also integrates various techniques, such as rectified linear unit (ReLU), dropout, data augmentation, and so forth.**  
AlexNet은 5개의 Convolutional Layer와 3개의 Fully Connected Layer로 구성되어 있으며 ReLU(Rectified Linear Unit), Dropout, Data Augmentation 등의 다양한 기술을 통합하고 있습니다.

<br>

**ReLU was widely regarded as the most essential component for making deep learning possible.**  
ReLU는 Deep Learning을 가능하게 하는 데 가장 필수적인 구성 요소로 널리 간주되었습니다.

<br>

**Then, in 2014, VGGNet proposed a standard network architecture that used very small 3 × 3 convolutional filters throughout and doubled the number of feature maps after the 2×2 pooling.**  
그리고 2014년에 VGGNet은 매우 작은 3×3 컨벌루션 필터를 전체적으로 사용하고 2×2 풀링 후 특징 맵의 수를 두 배로 늘린 표준 Network Architecture를 제안했습니다.

<br>

**It increased the depth of the network to 16-19 weight layers, which further enhanced the flexibility to learn progressive nonlinear mappings by deep architectures.**  
그것은 Network의 깊이를 16-19개의 가중치 레이어로 증가시켰고, 이는 심층 Architecture에 의한 점진적 비선형 매핑을 학습할 수 있는 유연성을 더욱 향상시켰습니다.

<br>

**In 2015, the 22-layer GoogleNet introduced an “inception module” with the concatenation of hybrid feature maps, as well as two additional intermediate softmax supervised signals.**  
2015년에 22계층 GoogleNet은 hybrid feature maps을 연결하는 "inception Module"과 2개의 추가 중간 softmax supervised signals를 도입했습니다.

<br>

**It performs several convolutions with different receptive fields (1 × 1, 3 × 3 and 5 × 5) in parallel, and concatenates all feature maps to merge the multi-resolution information.**  
서로 다른 수용 필드(1 × 1, 3 × 3 및 5 × 5)로 여러 convolutional 을 병렬로 수행하고 모든 feature maps을 연결하여 다중 해상도 정보를 병합합니다.        

<br>
<br>

**In 2016, ResNet proposed to make layers learn a residual mapping with reference to the layer inputs F(x) := H(x) − x rather than directly learning a desired underlying mapping H(x) to ease the training of very deep networks (up to 152 layers).**  
2016년에 ResNet은 계층이 원하는 기본 매핑 H(x)를 직접 학습하는 대신 계층 입력 F(x) := H(x) − x를 참조하여 residual 매핑을 학습하도록 제안하여 매우 깊은 Network의 Train을 용이하게 했습니다. (최대 152개 레이어).

<br>

**The original mapping is recast into F(x) + x and can be realized by “shortcut connections”.**  
원래 매핑은 F(x) + x로 재구성되며 "shortcut connections"로 실현될 수 있습니다.

<br>

**As the champion of ILSVRC 2017, SENet introduced a “Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.**  
ILSVRC 2017의 챔피언인 SENet은 채널 간의 상호 의존성을 명시적으로 modelling하여 채널별 기능 응답을 적응적으로 재보정하는 "Squeeze-and-Excitation"(SE) 블록을 도입했습니다.

<br>

**These blocks can be integrated with modern architectures, such as ResNet, and improves their representational power.**  
이러한 블록은 ResNet과 같은 최신 Architecture와 통합될 수 있으며 표현력을 향상시킵니다.

<br>

**With the evolved architectures and advanced training techniques, such as batch normalization (BN), the network becomes deeper and the training becomes more controllable.**  
batch normalization (BN)과 같은 진화된 Architecture와 고급 Train 기술을 통해 Network가 더 깊어지고 Train을 더 잘 제어할 수 있습니다.

<br>

**Following these architectures in object classification, the networks in deep FR are also developed step by step, and the performance of deep FR is continually improving.**  
객체 분류에서 이러한 Architecture에 따라 deep FR의 Network도 단계적으로 개발되며 deep FR의 성능은 지속적으로 향상되고 있습니다.

<br>

**We present these mainstream architectures of deep FR in Fig. 9.**  
그림 9에서 이러한 심층 FR의 주류 Architecture를 제시합니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_09.png">
<p align="center">
</p>
<br>
<br>

**In 2014, DeepFace was the first to use a nine-layer CNN with several locally connected layers.**  
2014년에 DeepFace는 로컬로 연결된 여러 계층이 있는 9계층 CNN을 처음으로 사용했습니다.

<br>

**With 3D alignment for face processing, it reaches an accuracy of 97.35% on LFW. In 2015, FaceNet used a large private dataset to train a GoogleNet.**  
얼굴 처리를 위한 3D 정렬로 LFW에서 97.35%의 정확도에 도달합니다. 2015년에 FaceNet은 대규모 개인 Dataset를 사용하여 GoogleNet을 교육했습니다.

<br>

**It adopted a triplet loss function based on triplets of roughly aligned matching/nonmatching face patches generated by a novel online triplet mining method and achieved good performance of 99.63%.**  
새로운 온라인 triplet mining 방법으로 생성된 대략적으로 정렬된 일치/비일치 안면 패치의 삼중항을 기반으로 triplet loss Function를 채택했으며 99.63%의 좋은 성능을 달성했습니다.

<br>

**In the same year, VGGface designed a procedure to collect a large-scale dataset from the Internet.**  
같은 해에 VGGface는 인터넷에서 대규모 Dataset를 수집하는 절차를 설계했습니다.

<br>

**It trained the VGGNet on this dataset and then fine-tuned the networks via a triplet loss function similar to FaceNet.**  
이 Dataset에서 VGGNet을 교육한 다음 FaceNet과 유사한 triplet Loss Function를 통해 Network를 미세 조정했습니다.

<br>

**VGGface obtains an accuracy of 98.95%.**  
VGGface는 98.95%의 정확도를 얻습니다.

<br>

**In 2017, SphereFace [84] used a 64-layer ResNet architecture and proposed the angular softmax (A-Softmax) loss to learn discriminative face features with angular margin.**  
2017년에 SphereFace[84]는 64계층 ResNet Architecture를 사용하고 각도 마진을 가진 차별적인 얼굴 특징을 학습하기 위해 angular softmax (A-Softmax) 손실을 제안했습니다.

<br>

**It boosts the achieves to 99.42% on LFW. In the end of 2017, a new largescale face dataset, namely VGGface2, was introduced, which consists of large variations in pose, age, illumination, ethnicity and profession.**  
LFW에서 달성률을 99.42%로 높입니다. 2017년 말에 새로운 대규모 얼굴 Dataset인 VGGface2가 도입되었습니다. 이 Dataset는 포즈, 연령, 조명, 민족 및 직업의 큰 변화로 구성됩니다.

<br>

**Cao et al. first trained a SENet with MS-celeb-1M dataset and then fine-tuned the model with VGGface2, and achieved the SOTA performance on the IJB-A and IJB-B.**  
Caoet al.은 먼저 MS-celeb-1M Dataset로 SENet을 교육한 다음 VGGface2로 Model을 미세 조정하고 IJB-A 및 IJB-B에서 SOTA 성능을 달성했습니다.   

<br>
<br>
<br>

### Light-weight networks. 

<br>
<br>
<br>

**Using deeper neural network with hundreds of layers and millions of parameters to achieve higher accuracy comes at cost.**  
더 높은 정확도를 달성하기 위해 수백 개의 레이어와 수백만 개의 매개변수가 있는 심층 neural network을 사용하면 비용이 발생합니다.

<br>

**Powerful GPUs with larger memory size are needed, which makes the applications on many mobiles and embedded devices impractical.**  
더 큰 메모리 크기의 강력한 GPU가 필요하므로 많은 모바일 및 임베디드 장치의 응용 프로그램이 비실용적입니다.

<br>

**To address this problem, light-weight networks are proposed.**  
이 문제를 해결하기 위해 경량 Network가 제안됩니다.

<br>

**Light CNN proposed a max-feature-map (MFM) activation function that introduces the concept of maxout in the fully connected layer to CNN.**  
Light CNN은 Fully Connected Layer의 maxout 개념을 CNN에 도입한 MFM(max-feature-map) 활성화 함수를 제안했습니다.

<br>

**The MFM obtains a compact representation and reduces the computational cost. Sun et al. proposed to sparsify deep networks iteratively from the previously learned denser models based on a weight selection criterion.**  
MFM은 간결한 표현을 얻고 계산 비용을 줄입니다. Sun et al. 가중치 선택 기준을 기반으로 이전에 학습된 밀도가 높은 Model에서 Deep Network를 반복적으로 희소화하도록 제안되었습니다.

<br>

**MobiFace adopted fast downsampling and bottleneck residual block with the expansion layers and achieved high performance with 99.7% on LFW database.**  
MobiFace는 확장 레이어와 함께 fast downsampling 및 bottleneck residual block을 채택했으며 LFW Database에서 99.7%의 높은 성능을 달성했습니다.

<br>

**Although some other light-weight CNNs, such as SqueezeNet, MobileNet, ShuffleNet and Xception, are still not widely used in FR, they deserve more attention.**  
SqueezeNet, MobileNet, ShuffleNet 및 Xception과 같은 일부 다른 light-weight CNN은 아직 FR에서 널리 사용되지 않지만 더 많은 관심을 받을 가치가 있습니다.    

<br>
<br>
<br>

### Adaptive-architecture networks

<br>
<br>
<br>

**Considering that designing architectures manually by human experts are timeconsuming and error-prone processes, there is growing interest in adaptive-architecture networks which can find well-performing architectures, e.g. the type of operation every layer executes (pooling, convolution, etc) and hyper-parameters associated with the operation (number of filters, kernel size and strides for a convolutional layer, etc), according to the specific requirements of training and testing data.**  
인간 전문가가 수동으로 Architecture를 설계하는 것은 시간이 많이 걸리고 오류가 발생하기 쉬운 프로세스라는 점을 고려할 때 성능이 좋은 Architecture를 찾을 수 있는 적응형 Architecture Network에 대한 관심이 높아지고 있습니다. Train 및 Test Data의 특정 요구 사항에 따라 모든 레이어가 실행하는 작업 유형(풀링, convolutional  등) 및 작업과 관련된 hyper-parameters(number of filters, kernel size and strides for a convolutional layer, etc).

<br>

**Currently, neural architecture search (NAS) is one of the promising methodologies, which has outperformed manually designed architectures on some tasks such as image classification or semantic segmentation.**  
현재 NAS(Neural Architecture Search)는 유망한 방법론 중 하나로 이미지 분류 또는 의미 분할과 같은 일부 작업에서 수동으로 설계된 Architecture를 능가합니다.

<br>

**Zhu et al. integrated NAS technology into face recognition.**  
Zhuet al. NAS 기술을 얼굴 인식에 통합했습니다.

<br>

**They used reinforcement learning algorithm (policy gradient) to guide the controller network to train the optimal child architecture.**  
그들은 컨트롤러 Network가 최적의 child architecture를 Train하도록 안내하기 위해 reinforcement learning algorithm(policy gradient)을 사용했습니다.

<br>

**Besides NAS, there are some other explorations to learn optimal architectures adaptively.**  
NAS 외에도 최적의 Architecture를 적응적으로 학습하기 위한 몇 가지 다른 탐구가 있습니다.

<br>

**For example, conditional convolutional neural network (c-CNN) dynamically activated sets of kernels according to modalities of samples;**  
예를 들어, c-CNN(Conditional Convolutional Neural Network)은 샘플 양식에 따라 커널 세트를 동적으로 활성화했습니다.

<br>

**Han et al. proposed a novel contrastive convolution consisted of a trunk CNN and a kernel generator, which is beneficial owing to its dynamistic generation of contrastive kernels based on the pair of faces being compared.**  
Han et al.은 트렁크 CNN과 kernel generator로 구성된 novel contrastive convolution을 제안했습니다.    

<br>
<br>
<br>

### Joint alignment-recognition networks

<br>
<br>
<br>

**Recently, an endto-end system was proposed to jointly train FR with several modules (face detection, alignment, and so forth) together.**
최근에는 FR을 여러 Module(얼굴 감지, 정렬 등)과 함께 공동으로 Train하기 위한 end-to-end 시스템이 제안되었습니다.

<br>

**Compared to the existing methods in which each module is generally optimized separately according to different objectives, this end-to-end system optimizes each module according to the recognition objective, leading to more adequate and robust inputs for the recognition model.**  
일반적으로 각 Module이 서로 다른 목표에 따라 개별적으로 최적화되는 기존 방법과 비교하여 이 end-to-end 시스템은 인식 목표에 따라 각 Module을 최적화하여 인식 Model에 대한 더 적절하고 강력한 입력을 유도합니다.

<br>

**For example, inspired by spatial transformer, Hayat et al. proposed a CNN-based data-driven approach that learns to simultaneously register and represent faces (Fig.10), while Wu et al. designed a novel recursive spatial transformer (ReST) module for CNN allowing face alignment and recognition to be jointly optimized.**  
예를 들어, 공간 변환기에서 영감을 얻은 Hayat et al. Wu et al.은 얼굴을 동시에 등록하고 표현하는 방법을 배우는 CNN 기반 데이터 기반 접근 방식을 제안했습니다(그림 10). 얼굴 정렬 및 인식을 함께 최적화할 수 있도록 CNN을 위한 새로운 재귀 공간 변환기(ReST) Module을 설계했습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_10.png">
<p align="center">
</p>
<br>
<br>

## 2) Assembled Networks 

<br>
<br>
<br>

### Multi-input networks

<br>
<br>
<br>

**In “one-to-many augmentation”, multiple images with variety are generated from one image in order to augment training data.**  
일대다 증강(one-to-many Augmentation)은 Train 데이터를 증강시키기 위해 하나의 이미지에서 다양한 이미지를 여러 개 생성하는 것입니다.

<br>

**Taken these multiple images as input, multiple networks are also assembled together to extract and combine features of different type of inputs, which can outperform an individual network.**  
이러한 여러 이미지를 입력으로 사용하면 여러 Network도 함께 조립되어 개별 Network를 능가할 수 있는 다양한 입력 유형의 기능을 추출하고 결합합니다.

<br>

**Assembled networks are built after different face patches are cropped, and then different types of patches are fed into different sub-networks for representation extraction.**  
조립된 Network는 서로 다른 얼굴 패치가 잘린 후 구축된 다음 representation extraction을 위해 서로 다른 유형의 패치가 서로 다른 하위 Network에 공급됩니다.

<br>

**By combining the results of subnetworks, the performance can be improved. Other papers used assembled networks to recognize images with different poses.**  
서브 Network의 결과를 결합하여 성능을 향상시킬 수 있습니다. 다른 논문에서는 assembled Network를 사용하여 포즈가 다른 이미지를 인식했습니다.      

<br>
<br>
<br>

**A multi-view deep network (MvDN) [95] consists of view-specific subnetworks and common subnetworks; the former removes view-specific variations, and the latter obtains common representations.**  
MvDN(multi-view deep network)[95]은 view-specific subnetworks와 common subnetworks로 구성됩니다. 전자는 view-specific variations을 제거하고 후자는 common representations을 얻습니다.

<br>

**Multi-task networks. FR is intertwined with various factors, such as pose, illumination, and age. To solve this problem, multitask learning is introduced to transfer knowledge fromother relevant tasks and to disentangle nuisance factors.**  
Multi-task Network. FR은 포즈, 조명, 연령 등 다양한 요소와 얽혀 있습니다. 이 문제를 해결하기 위해 multitask learning을 도입하여 다른 관련 작업에서 지식을 이전하고 방해 요소를 분리합니다.

<br>

**In multi-task networks, identity classification is the main task and the side tasks are pose, illumination, and expression estimations, among others.**  
multi-task Network에서 identity classification가 주요 작업이고 부수 작업은 무엇보다도 포즈, 조명 및 표정 추정입니다.

<br>

**The lower layers are shared among all the tasks, and the higher layers are disentangled into different sub-networks to generate the task-specific outputs.**  
하위 계층은 모든 작업 간에 공유되며 상위 계층은 서로 다른 하위 Network로 분리되어 작업별 출력을 생성합니다.

<br>

**The task-specific sub-networks are branched out to learn face detection, face alignment, pose estimation, gender recognition, smile detection, age estimation and FR.**  
작업별 하위 Network는 얼굴 감지, 얼굴 정렬, 자세 추정, 성별 인식, 미소 감지, 연령 추정 및 FR을 학습하기 위해 분기됩니다.

<br>

**Yin et al. proposed to automatically assign the dynamic loss weights for each side task.**  
Yin et al.은 각 부업에 대한 동적 손실 가중치를 자동으로 할당하도록 제안되었습니다.

<br>

**Peng et al. used a feature reconstruction metric learning to disentangle a CNN into subnetworks for jointly learning the identity and non-identity features as shown in Fig. 11.**  
Peng et al.은 그림 11과 같이 신원 및 비신원 특징을 함께 학습하기 위해 CNN을 하위 Network로 분리하기 위해 특징 재구성 메트릭 학습을 사용했습니다. 

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_11.png">
<p align="center">
</p>
<br>
<br>
