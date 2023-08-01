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

