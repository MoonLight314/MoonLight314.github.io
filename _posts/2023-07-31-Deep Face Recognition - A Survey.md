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

### C. Face Matching by deep features

<br>
<br>
<br>

**During testing, the cosine distance and L2 distance are generally employed to measure the similarity between the deep features x1 and x2; then, threshold comparison and the nearest neighbor (NN) classifier are used to make decision for verification and identification.**  
테스트 중에 cosine distance와 L2 distance는 일반적으로 Deep features x1과 x2 사이의 유사성을 측정하는 데 사용됩니다. 그런 다음 임계값 비교 및 가장 가까운 이웃(NN) 분류기를 사용하여 확인 및 식별을 위한 결정을 내립니다.

<br>

**In addition to these common methods, there are some other explorations.**  
이러한 일반적인 방법 외에도 몇 가지 다른 탐색이 있습니다.

<br>
<br>

### 1) Face verification

<br>
<br>

**Metric learning, which aims to find a new metric to make two classes more separable, can also be used for face matching based on extracted deep features.**  
두 클래스를 더 분리할 수 있도록 새로운 메트릭을 찾는 것을 목표로 하는 메트릭 학습은 추출된 deep features을 기반으로 얼굴 매칭에도 사용할 수 있습니다.

<br>

**The JB model is a well-known metric learning method, and Hu et al. proved that it can improve the performance greatly.**  
JB Model은 잘 알려진 메트릭 학습 방법이며 Hu et al.은 성능을 크게 향상시킬 수 있음을 입증했습니다.

<br>

**In the JB model, a face feature x is modeled as x = µ+ε, where µ and ε are identity and intra-personal variations, respectively.**  
JB Model에서 얼굴 특징 x는 x = µ+ε로 Modeling되며, 여기서 µ와 ε는 각각 identity과 intra-personal variations입니다.

<br>

**The similarity score r(x1, x2) can be represented as follows:**  

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Formula_07.png">
<p align="center">
</p>
<br>
<br>

where P(x1, x2|HI ) is the probability that two faces belong to the same identity and P(x1, x2|HE) is the probability that two faces belong to different identities.  

<br>
<br>
<br>

### 2) Face identification

<br>
<br>
<br>

**After cosine distance was computed,Cheng et al. proposed a heuristic voting strategy at the similarity score level to combine the results of multiple CNN models and won first place in Challenge 2 of MSceleb-1M 2017.**  
cosine distance를 계산한 후 Cheng et al.은 여러 CNN Model의 결과를 결합하기 위해 유사성 점수 수준에서 heuristic voting strategy을 제안하고 MSceleb-1M 2017의 Challenge 2에서 1위를 차지했습니다.

<br>

**Yang et al. extracted the local adaptive convolution features from the local regions of the face image and used the extended SRC for FR with a single sample per person.**  
Yang et al.은 얼굴 이미지의 로컬 영역에서 local adaptive convolution features을 추출하고 1인당 단일 샘플로 FR용 extended SRC를 사용했습니다.

<br>

**Guo et al. combined deep features and the SVM classifier to perform recognition.**  
Guoet al.은 deep features과 SVM 분류기를 결합하여 인식을 수행합니다.

<br>

**Wang et al. first used product quantization (PQ) to directly retrieve the topk most similar faces and re-ranked these faces by combining similarities from deep features and the COTS matcher.**  
Wang et al.은 먼저 product quantization(PQ)를 사용하여 가장 유사한 얼굴을 직접 검색하고 Deep Features과 COTS 매처의 유사성을 결합하여 이러한 얼굴의 순위를 다시 매겼습니다.

<br>

**In addition, Softmax can be also used in face matching when the identities of training set and test set overlap.**  
또한 Softmax는 Train 세트와 test 세트의 ID가 중복되는 경우 얼굴 매칭에도 사용할 수 있습니다.

<br>

**For example, in Challenge 2 of MS-celeb-1M, Ding et al. [142] trained a 21,000-class softmax classifier to directly recognize faces of one-shot classes and normal classes after augmenting feature by a conditional GAN; Guo et al. trained the softmax classifier combined with underrepresented-classes promotion (UP) loss term to enhance the performance on one-shot classes.**  
예를 들어 MS-celeb-1M의 Challenge 2에서 Ding et al. [142]은 conditional GAN으로 Features을 보강한 후 원샷 클래스와 일반 클래스의 얼굴을 직접 인식하도록 21,000 클래스의 Softmax 분류기를 Train했습니다. Guoet al.은 원샷 클래스의 성능을 향상시키기 위해 underrepresented-classes promotion (UP) loss term과 결합된 Softmax classifier를 Train했습니다.

<br>

**When the distributions of training data and testing data are the same, the face matching methods mentioned above are effective.**  
training data와 testing data 의 분포가 같을 때 위에서 언급한 얼굴 매칭 방법이 효과적입니다.

<br>

**However, there is always a distribution change or domain shift between two data domains that can degrade the performance on test data.**  
그러나 Test Data의 성능을 저하시킬 수 있는 두 데이터 도메인 사이에는 항상 distribution change 또는 domain shift이 있습니다.

<br>

**Transfer learning has recently been introduced into deep FR to address the problem of domain shift.**  
domain shift 문제를 해결하기 위해 Transfer learning이 최근 심층 FR에 도입되었습니다.

<br>

**It learns transferable features using a labeled source domain (training data) and an unlabeled target domain (testing data) such that domain discrepancy is reduced and models trained on source domain will also perform well on target domain.**  
레이블이 지정된 소스 도메인(training data)과 레이블이 지정되지 않은 대상 도메인(testing data)을 사용하여 이전 transferable features을 학습하므로 domain discrepancy가 줄어들고 소스 도메인에서 Train된 Model이 대상 도메인에서도 잘 수행됩니다.

<br>

**Sometimes, this technology is applied to face matching. For example, Crosswhite et al. and Xiong et al. adopted template adaptation to the set of media in a template by combining CNN features with template-specific linear SVMs.**  
때때로 이 기술은 얼굴 매칭에 적용됩니다. 예를 들어 Crosswhite et al. 및 Xiong et al.은 CNN features과 template-specific linear SVMs을 결합하여 template의 미디어 집합에 대한 template 적응을 채택했습니다.

<br>

**But most of the time, it is not enough to do transfer learning only at face matching stage.**  
그러나 대부분의 경우 얼굴 매칭 단계에서만 transfer learning을 하는 것으로는 충분하지 않습니다.

<br>

**Transfer learning should be embedded in deep models to learn more transferable representations. Kan et al. proposed a bi-shifting autoencoder network (BAE) for domain adaptation across view angle, ethnicity, and imaging sensor; while Luo et al. utilized the multi-kernels maximum mean discrepancy (MMD) to reduce domain discrepancies.**  
transfer learning은 더 많은 전이 가능한 표현을 학습하기 위해 Deep Model에 포함되어야 합니다. Kanet al.은 시야각, 인종 및 이미징 센서에 걸친 도메인 적응을 위한 BAE(bi-shifting autoencoder network)를 제안했습니다. 반면 Luo et al.은 domain discrepancies를 줄이기 위해 multi-kernels maximum mean discrepancy(MMD)를 활용했습니다.

<br>

**Sohn et al. used adversarial learning [150] to transfer knowledge from still image FR to video FR.**  
Sohn et al.은 정지 이미지 FR에서 비디오 FR로 지식을 전달하기 위해 적대적 학습(adversarial learning )[150]을 사용했습니다.

<br>

**Moreover, finetuning the CNN parameters from a prelearned model using a target training dataset is a particular type of transfer learning, and is commonly employed by numerous methods.**  
또한 대상 target training dataset를 사용하여 prelearned Model에서 CNN 매개변수를 미세 조정하는 것은 특정 유형의 Transfer Learning이며 일반적으로 다양한 방법에서 사용됩니다.    

<br>
<br>
<br>
<br>

# IV. FACE PROCESSING FOR TRAINING AND RECOGNITION

<br>
<br>
<br>

**We present the development of face processing methods in chronological order in Fig. 12.**  
우리는 그림 12에서 얼굴 처리 방법의 개발을 연대순으로 제시합니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_12.png">
<p align="center">
</p>
<br>
<br>

**As we can see from the figure, most papers attempted to perform face processing by autoencoder model in 2014 and 2015**  
그림에서 볼 수 있듯이 대부분의 논문은 2014년과 2015년에 Autoencoder Model로 얼굴 처리를 시도했습니다. 

<br>

**while 3D model played an important role in 2016.**  
3D Model은 2016년에 중요한 역할을 했습니다. 

<br>

**GAN [40] has drawn substantial attention from the deep learning and computer vision community since it was first proposed by Goodfellow et al.**  
GAN[40]은 Goodfellow 등이 처음 제안한 이후 Deep Learning 및 computer vision community에서 상당한 관심을 끌었습니다.

<br>

**It can be used in different fields and was also introduced into face processing in 2017.**  
다양한 분야에서 활용이 가능하며 2017년에는 face processing에도 도입되었습니다.

<br>

**GAN can be used to perform “one-tomany augmentation” and “many-to-one normalization”, and it broke the limit that face synthesis should be done under supervised way.**  
GAN은 "one-to-many augmentation" 및 "many-to-one normalization"를 수행하는 데 사용할 수 있으며 감독 방식으로 얼굴 합성을 수행해야 하는 한계를 깨뜨렸습니다.

<br>

**Although GAN has not been widely used in face processing for training and recognition, it has great latent capacity for preprocessing, for example, Dual-Agent GANs (DA-GAN) won the 1st places on verification and identification tracks in the NIST IJB-A 2017 FR competitions.**  
GAN은 Train 및 recognition을 위한 얼굴 처리에 널리 사용되지는 않았지만 preprocessing를 위한 잠재 능력이 큽니다. 예를 들어 DA-GAN(Dual-Agent GAN)은 NIST IJB-A의 2017년 프랑스 대회에서 verification and identification 부문에서 1위를 차지했습니다.      

<br>
<br>
<br>

### A. One-to-Many Augmentation

<br>
<br>
<br>

**Collecting a large database is extremely expensive and time consuming.**  
대규모 Database 수집은 비용과 시간이 많이 소요됩니다.

<br>

**The methods of “one-to-many augmentation” can mitigate the challenges of data collection, and they can be used to augment not only training data but also the gallery of test data.**  
"one-to-many augmentation" 방법은 데이터 수집 문제를 완화할 수 있으며 Train 데이터뿐만 아니라 Test Data Gallery도 확대하는 데 사용할 수 있습니다.

<br>

**we categorized them into four classes: data augmentation, 3D model, autoencoder model and GAN model.**  
One-to-Many Augmentation은 data augmentation, 3D Model, autoencoder model 및 GAN Model의 네 가지 클래스로 분류할 수 있습니다.    

<br>
<br>
<br>

#### Data augmentation

<br>
<br>
<br>

**Common data augmentation methods consist of photometric transformations and geometric transformations, such as oversampling (multiple patches obtained by cropping at different scales), mirroring, and rotating the images.**  
일반적인 data augmentation 방법은 oversampling(서로 다른 축척으로 잘라서 얻은 여러 패치), 미러링 및 이미지 회전과 같은 광도 변환 및 기하학적 변환으로 구성됩니다.

<br>

**Recently, data augmentation has been widely used in deep FR algorithms.**  
최근 데이터 증가는 심층 FR 알고리즘에서 널리 사용되었습니다.

<br>

**for example, Sun et al. cropped 400 face patches varying in positions, scales, and color channels and mirrored the images.**  
예를 들어 Sun et al.은 위치, 크기 및 색상 채널이 다른 400개의 얼굴 패치를 자르고 이미지를 미러링했습니다.

<br>

**Liu et al. generated seven overlapped image patches centered at different landmarks on the face region and trained them with seven CNNs with the same structure.**  
Liu et al.은 얼굴 영역의 서로 다른 랜드마크를 중심으로 7개의 중첩된 이미지 패치를 생성하고 동일한 구조를 가진 7개의 CNN으로 Train했습니다.

<br>
<br>
<br>

#### 3D model

<br>
<br>
<br> 

**3D face reconstruction is also a way to enrich the diversity of training data.**  
3D 얼굴 재구성은 Train 데이터의 다양성을 풍부하게 하는 방법이기도 합니다.

<br>

**They utilize 3D structure information to model the transformation between poses.**  
3D 구조 정보를 활용하여 포즈 간의 변형을 Model링합니다.

<br>

**3D models first use 3D face data to obtain morphable displacement fields and then apply them to obtain 2D face data in different pose angles.**  
3D Model은 먼저 3D 얼굴 데이터를 사용하여 변형 가능한 변위 필드를 얻은 다음 이를 적용하여 다양한 포즈 각도에서 2D 얼굴 데이터를 얻습니다.

<br>

**There is a large number of papers about this domain, but we only focus on the 3D face reconstruction using deep methods or used for deep FR.**  
이 영역에 대한 많은 논문이 있지만 우리는 deep method를 사용하거나 deep FR에 사용되는 3D 얼굴 재구성에만 집중합니다.

<br>

**Masi et al. generated face images with new intra-class facial appearance variations, including pose, shape and expression, and then trained a 19-layer VGGNet with both real and augmented data.**  
Masi et al.은 포즈, 모양 및 표정을 포함한 새로운 클래스 내 얼굴 모양 변형으로 얼굴 이미지를 생성한 다음 실제 데이터와 증강 데이터를 모두 사용하여 19계층 VGGNet을 Train했습니다.

<br>

**Masi et al. used generic 3D faces and rendered fixed views to reduce much of the computational effort.**  
Masi et al.은 일반 3D faces를 사용하고 고정된 뷰를 렌더링하여 많은 계산 작업을 줄였습니다.

<br>

**Richardson et al. employed an iterative 3D CNN by using a secondary input channel to represent the previous network’s output as an image for reconstructing a 3D face as shown in Fig. 13.**  
Richardson et al.은 그림 13과 같이 3D 얼굴을 재구성하기 위한 이미지로 이전 Network의 출력을 표현하기 위해 보조 입력 채널을 사용하여 반복 3D CNN을 사용했습니다.

<br>

**Dou et al. used a multi-task CNN to divide 3D face reconstruction into neutral 3D reconstruction and expressive 3D reconstruction.**  
Dou et al.은 multi-task CNN을 사용하여 3D 얼굴 재구성을 neutral 3D reconstruction과 expressive 3D reconstruction으로 나누었습니다.

<br>

**Tran et al. directly regressed 3D morphable face model (3DMM) [155] parameters from an input photo by a very deep CNN architecture.**  
Tran et al.은 3DMM(3D morphable face model)[155] 매개변수를 very deep CNN Architecture에 의해 입력 사진에서 직접 회귀(regress)했습니다.

<br>

**An et al. synthesized face images with various poses and expressions using the 3DMM method, then reduced the gap between synthesized data and real data with the help of MMD.**  
An et al.은 다양한 포즈와 표정의 얼굴 이미지를 3DMM 방식으로 합성한 후 MMD를 통해 synthesized data 와 real data 의 격차를 줄였습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_13.png">
<p align="center">
</p>
<br>
<br>

### Autoencoder model

<br>
<br>
<br>

**Rather than reconstructing 3D models from a 2D image and projecting it back into 2D images of different poses, autoencoder models can generate 2D target images directly.**  
2D 이미지에서 3D Model을 재구성하고 다른 포즈의 2D 이미지로 다시 투영하는 대신 Autoencoder Model은 2D 대상 이미지를 직접 생성할 수 있습니다.

<br>

**Taken a face image and a pose code encoding a target pose as input, an encoder first learns pose-invariant face representation, and then a decoder generates a face image with the same identity viewed at the target pose by using the pose-invariant representation and the pose code.**  
얼굴 이미지와 목표 포즈를 Encoding한 포즈 코드를 입력받아 Encoder는 먼저 포즈 불변 얼굴 표현(pose-invariant face representation)을 학습하고, Decoder는 pose-invariant representation과 pose code을 이용하여 목표 포즈에서 본 동일한 정체성을 가진 얼굴 이미지를 생성한다.

<br>

**For example, given the target pose codes, multi-view perceptron (MVP) trained some deterministic hidden neurons to learn pose-invariant face representations, and simultaneously trained some random hidden neurons to capture pose features, then a decoder generated the target images by combining poseinvariant representations with pose features.**  
예를 들어, 대상 포즈 코드가 주어지면 multi-view perceptron(MVP)은 pose-invariant face representation을 학습하기 위해 일부 결정론적 숨겨진 뉴런을 Train시키고 동시에 포즈 특징을 캡처하기 위해 일부 임의의 숨겨진 뉴런을 Train시킨 다음 Decoder는 다음을 결합하여 대상 이미지를 생성했습니다.

<br>

**As shown in Fig. 14, Yim et al. and Qian et al. introduced an auxiliary CNN to generate better images viewed at the target poses.**  
그림 14에 도시된 바와 같이, Yim et al. 및 Qian et al.은 target poses에서 더 나은 이미지를 생성하기 위해 보조 CNN을 도입했습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_14.png">
<p align="center">
</p>
<br>
<br>

**First, an autoencoder generated the desired pose image, then the auxiliary CNN reconstructed the original input image back from the generated target image, which guarantees that the generated image is identity-preserving.**  
먼저 Autoencoder가 원하는 포즈 이미지를 생성한 다음 보조 CNN이 생성된 대상 이미지에서 원본 입력 이미지를 다시 재구성하여 생성된 이미지가 ID를 보존하도록 보장합니다.

<br>

**Two groups of units are embedded between encoder and decoder.**  
두 그룹의 units가 Encoder와 Decoder 사이에 내장되어 있습니다.

<br>

**The identity units remain unchanged and the rotation of images is achieved by taking actions to pose units at each time step.**  
identity units은 변경되지 않고 이미지 회전은 각 시간 단계에서 포즈 단위에 대한 조치를 취함으로써 달성됩니다.      

<br>
<br>
<br>

### GAN model

<br>
<br>
<br>

**In GAN models, a generator aims to fool a discriminator through generating images that resemble the real images, while the discriminator aims to discriminate the generated samples from the real ones.**  
GAN Model에서 Generator는 실제 이미지와 유사한 이미지를 생성하여 판별자를 속이는 것을 목표로 하고, Discriminator는 생성된 샘플과 실제 샘플을 구별하는 것을 목표로 합니다.

<br>

**By this minimax game between generator and discriminator, GAN can successfully generate photo-realistic images with different poses.**  
GAN은 generator와 discriminator 사이의 이 미니맥스 게임을 통해 다양한 포즈로 사진과 같은 이미지를 성공적으로 생성할 수 있습니다.

<br>

**After using a 3D model to generate profile face images, DA-GAN[56] refined the images by a GAN, which combines prior knowledge of the data distribution and knowledge of faces (pose and identity perception loss).**  
3D Model을 사용하여 프로필 얼굴 이미지를 생성한 후 DA-GAN[56]은 데이터 분포에 대한 data distribution와 knowledge of faces (pose and identity perception loss)을 결합하는 GAN으로 이미지를 정제했습니다.

<br>

**CVAE-GAN [159] combined a variational auto-encoder with a GAN for augmenting data, and took advantages of both statistic and pairwise feature matching to make the training process converge faster and more stably.**  
CVAE-GAN[159]은 variational auto-encoder를 GAN과 결합하여 데이터를 보강하고 통계 및 쌍별 특징 매칭을 모두 활용하여 학습 프로세스가 더 빠르고 안정적으로 수렴되도록 했습니다.

<br>

**In addition to synthesizing diverse faces from noise, some papers also explore to disentangle the identity and variation, and synthesize new faces by exchanging identity and variation from different people.**  
노이즈에서 다양한 얼굴을 합성하는 것 외에도 일부 논문에서는 정체성과 변이를 풀고 다른 사람의 정체성과 변이를 교환하여 새로운 얼굴을 합성하는 방법을 모색합니다.

<br>

**In CG-GAN, a generator directly resolves each representation of input image into a variation code and an identity code and regroups these codes for cross-generating, simultaneously, a discriminator ensures the reality of generated images.**  
CG-GAN에서 generator는 입력 이미지의 각 표현을 변형 코드와 식별 코드로 직접 해결하고 이러한 코드를 다시 그룹화하여 교차 생성하는 동시에 discriminator는 생성된 이미지의 사실성을 보장합니다.

<br>

**Bao et al. extracted identity representation of one input image and attribute representation of any other input face image, then synthesized new faces by recombining these representations.**  
Bao et al.은 하나의 입력 이미지의 신원 표현과 다른 입력 얼굴 이미지의 속성 표현을 추출한 다음 이러한 표현을 재결합하여 새로운 얼굴을 합성합니다.

<br>

**This work shows superior performance in generating realistic and identity preserving face images, even for identities outside the training dataset.**  
이 작업은 training dataset 외부의 ID에 대해서도 사실적이고 ID를 보존하는 얼굴 이미지를 생성하는 데 탁월한 성능을 보여줍니다.

<br>

**Unlike previous methods that treat classifier as a spectator, FaceID-GAN [162] proposed a three-player GAN where the classifier cooperates together with the discriminator to compete with the generator from two different aspects, i.e. facial identity and image quality respectively.**  
classifier를 관중으로 취급하는 이전 방법과 달리 FaceID-GAN[162]은 classifier가 discriminator와 협력하여 두 가지 다른 측면, 즉 각각 얼굴 정체성과 이미지 품질에서 생성자와 경쟁하는 three-player GAN을 제안했습니다.

<br>
<br>
<br>

## B. Many-to-One Normalization

<br>
<br>
<br>

**In contrast to “one-to-many augmentation”, the methods of “many-to-one normalization” produce frontal faces and reduce appearance variability of test data to make faces align and compare easily. It can be categorized as autoencoder model, CNN model and GAN model.**  
"one-to-many augmentation"와 달리 "many-to-one normalization" 방법은 정면 얼굴을 생성하고 Test Data의 모양 변동성을 줄여 얼굴을 쉽게 정렬하고 비교할 수 있도록 합니다. Autoencoder Model, CNN Model 및 GAN Model로 분류할 수 있습니다.    

<br>
<br>
<br>

### Autoencoder model

<br>
<br>
<br>


**Autoencoder can also be applied to “many-to-one normalization”.**  
Autoencoder는 "many-to-one normalization"에도 적용될 수 있습니다.

<br>

**Different from the autoencoder model in “one-to-many augmentation” which generates the desired pose images with the help of pose codes, autoencoder model here learns pose-invariant face representation by an encoder and directly normalizes faces by a decoder without pose codes.**  
포즈 코드의 도움으로 원하는 포즈 이미지를 생성하는 "one-to-many augmentation"의 Autoencoder Model과 달리, 여기서 Autoencoder Model은 포즈 코드 없이 포즈 불변 얼굴 표현을 학습하고 Decoder로 얼굴을 직접 정규화합니다. 

<br>

**Zhu et al. selected canonicalview images according to the face images’ symmetry and sharpness and then adopted an autoencoder to recover the frontal view images by minimizing the reconstruction loss error.**  
Zhuet al.은 얼굴 영상의 대칭성과 선명도에 따라 canonicalview 영상을 선택하고 autoencoder를 채택하여 재구성 손실 오류를 최소화하여 정면 영상을 복구합니다.

<br>

**The proposed stacked progressive autoencoders (SPAE) progressively map the nonfrontal face to the frontal face through a stack of several autoencoders.**  
제안된 SPAE(Stacked Progressive Autoencoders)는 여러 자동 Encoder 스택을 통해 nonfrontal face를 frontal face로 점진적으로 매핑합니다.

<br>

**Each shallow autoencoders of SPAE is designed to convert the input face images at large poses to a virtual view at a smaller pose, so the pose variations are narrowed down gradually layer by layer along the pose manifold.**  
SPAE의 각각의 shallow Autoencoder는 큰 포즈의 입력 얼굴 이미지를 더 작은 포즈의 virtual view로 변환하도록 설계되어 포즈 변형이 포즈 매니폴드를 따라 레이어별로 점진적으로 좁혀집니다. 

<br>

**Zhang et al. built a sparse many-to-one encoder to enhance the discriminant of the pose free feature by using multiple random faces as the target values for multiple encoders.**  
Zhang et al.은 여러 Encoder의 대상 값으로 여러 임의의 얼굴을 사용하여 포즈 없는 기능의 판별을 향상시키기 위해 희소한 many-to-one encoder를 구축했습니다.    

<br>
<br>
<br>

### CNN model

<br>
<br>
<br>

**CNN models usually directly learn the 2D mappings between non-frontal face images and frontal images, and utilize these mapping to normalize images in pixel space.**  
CNN Model은 일반적으로 non-frontal face images와 frontal images간의 2D 매핑을 직접 학습하고 이러한 매핑을 활용하여 이미지를 픽셀 공간에서 정규화합니다.

<br>

**The pixels in normalized images are either directly the pixels or the combinations of the pixels in non-frontal images.**  
정규화된 이미지의 픽셀은 바로 픽셀이거나 non-frontal images의 픽셀 조합입니다.

<br>

**In LDF-Net, the displacement field network learns the shifting relationship of two pixels, and the translation layer transforms the input non-frontal face image into a frontal one with this displacement field.**  
LDF-Net에서 displacement field Network는 두 픽셀의 이동 관계를 학습하고 변환 레이어는 이 변위 필드를 사용하여 입력된 non-frontal face image를 frontal 이미지로 변환합니다.

<br>

**In GridFace shown in Fig. 15, first, the rectification network normalizes the images by warping pixels from the original image to the canonical one according to the computed homography matrix,**  
그림 15의 GridFace에서 먼저 보정 Network는 계산된 호모그래피 행렬에 따라 원본 이미지에서 표준 이미지로 픽셀을 워핑하여 이미지를 정규화한 다음 정규화된 출력을 암시적 표준 뷰 페이스에 의해 정규화합니다. 

<br>

**then the normalized output is regularized by an implicit canonical view face prior, finally, with the normalized faces as input, the recognition network learns discriminative face representation via metric learning.**  
그런 다음,정규화된 얼굴을 입력으로 사용하여 인식 Network는 메트릭 학습을 통해 차별적인 얼굴 표현을 학습합니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_15.png">
<p align="center">
</p>
<br>
<br>

### GAN model

<br>
<br>
<br>

**Huang et al. proposed a two-pathway generative adversarial network (TP-GAN) that contains four landmark-located patch networks and a global encoderdecoder network.**  
Huang et al.은 4개의 랜드마크에 위치한 패치 Network와 글로벌 Encoder/Decoder Network를 포함하는 two-pathway generative adversarial network(TP-GAN)를 제안했습니다.

<br>

**Through combining adversarial loss, symmetry loss and identity-preserving loss, TP-GAN generates a frontal view and simultaneously preserves global structures and local details as shown in Fig. 16.**  
adversarial loss, symmetry loss 및 identity-preserving loss을 결합하여 TP-GAN은 정면 뷰를 생성하고 동시에 그림 16과 같이 전역 구조 및 로컬 세부 정보를 보존합니다.

<br>

**In a disentangled representation learning generative adversarial network (DR-GAN), the generator serves as a face rotator, in which an encoder produces an identity representation, and a decoder synthesizes a face at the specified pose using this representation and a pose code.**  
DR-GAN(disentangled representation learning generative adversarial network)에서 generator는 Encoder가 ID 표현을 생성하고 Decoder가 이 표현과 포즈 코드를 사용하여 지정된 포즈에서 얼굴을 합성하는 얼굴 회전기 역할을 합니다.

<br>

**And the discriminator is trained to not only distinguish real vs. synthetic images, but also predict the identity and pose of a face.**  
그리고 discriminator는 실제 이미지와 합성 이미지를 구별할 뿐만 아니라 얼굴의 정체성과 포즈를 예측하도록 Train됩니다.

<br>

**Yin et al. incorporated 3DMM into the GAN structure to provide shape and appearance priors to guide the generator to frontalization.**  
Yin et al.은 3DMM을 GAN 구조에 통합하여 generator를 전면화(frontalization)로 안내하기 전에 모양과 모양을 제공합니다.    

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_16.png">
<p align="center">
</p>
<br>
<br>
<br>
<br>
<br>

## V. FACE DATABASES AND EVALUATION PROTOCOLS

<br>
<br>
<br>

**In the past three decades, many face databases have been constructed with a clear tendency from small-scale to largescale, from single-source to diverse-sources, and from labcontrolled to real-world unconstrained condition, as shown in Fig. 17.**  
지난 30년 동안 그림 17과 같이 소규모에서 대규모로, 단일 소스에서 다양한 소스로, 실험실 제어(labcontrolled)에서 실제 비제약 조건(real-world unconstrained condition)으로 많은 얼굴 Database가 명확한 경향으로 구축되었습니다.

<br>

**As the performance of some simple databases become saturated, e.g. LFW, more and more complex databases were continually developed to facilitate the FR research.**
일부 단순 Database의 성능이 포화됨에 따라, 예를 들어. LFW, 점점 더 복잡한 Database가 FR 연구를 용이하게 하기 위해 지속적으로 개발되었습니다.

<br>

**It can be said without exaggeration that the development process of the face databases largely leads the direction of FR research.**  
얼굴 Database의 개발 과정은 FR 연구의 방향성을 크게 이끌어간다고 해도 과언이 아니다.

<br>

**In this section, we review the development of major training and testing academic databases for the deep FR.**  
이 섹션에서는 Deep FR을 위한 주요 training and testing academic databases의 개발을 검토합니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_17.png">
<p align="center">
</p>
<br>
<br>

### A. Large-scale training data sets

<br>
<br>
<br>

**The prerequisite of effective deep FR is a sufficiently large training dataset.**  
효과적인 Deep FR의 전제 조건은 충분히 큰 Train Dataset입니다

<br>

**Zhou et al. [59] suggested that large amounts of data with deep learning improve the performance of FR.**  
Zhou et al.은 [59]는 Deep Learning을 통해 많은 양의 데이터가 FR의 성능을 향상시킨다고 제안했습니다.

<br>

**The results of Megaface Challenge also revealed that premier deep FR methods were typically trained on data larger than 0.5M images and 20K people.**  
Megaface Challenge의 결과는 또한 최고의 deep FR 방법이 일반적으로 0.5M 이미지와 20,000명보다 큰 데이터에 대해 Train되었음을 밝혔습니다.

<br>

**The early works of deep FR were usually trained on private training datasets.**  
Deep FR의 초기 작업은 일반적으로 개인 Train Dataset에서 Train되었습니다.

<br>

**Facebook’s Deepface model was trained on 4M images of 4K people; Google’s FaceNet was trained on 200M images of 3M people; DeepID serial models were trained on 0.2M images of 10K people.**  
Facebook의 Deepface Model은 4,000명에 대한 4백만 개의 이미지로 Train되었습니다. Google의 FaceNet은 3백만 명의 이미지 2억 개로 Train되었습니다. DeepID serial Model은 10,000명에 대한 0.2M 이미지로 Train되었습니다.

<br>

**Although they reported ground-breaking performance at this stage, researchers cannot accurately reproduce or compare their models without public training datasets.**  
이 단계에서 획기적인 성능을 보고했지만 연구원은 public training datasets 없이는 Model을 정확하게 재현하거나 비교할 수 없습니다.

<br>

**To address this issue, CASIA-Webface [120] provided the first widely-used public training dataset for the deep model training purpose, which consists of 0.5M images of 10K celebrities collected from the web.**  
이 문제를 해결하기 위해 CASIA-Webface[120]는 웹에서 수집한 10,000명의 유명인사 이미지 0.5M로 구성된 Deep Model 교육 목적으로 널리 사용되는 최초의 widely-used public training dataset를 제공했습니다.

<br>

**Given its moderate size and easy usage, it has become a great resource for fair comparisons for academic deep models.**  
적당한 크기와 쉬운 사용법을 감안할 때 학문적 심층 Model에 대한 공정한 비교를 위한 훌륭한 리소스가 되었습니다.

<br>

**However, its relatively small data and ID size may not be sufficient to reflect the power of many advanced deep learning methods.**  
그러나 상대적으로 작은 데이터와 ID 크기는 많은 고급 Deep Learning 방법의 성능을 반영하기에 충분하지 않을 수 있습니다.

<br>

**Currently, there have been more databases providing public available large-scale training dataset (Table VI), especially three databases with over 1M images, namely MS-Celeb-1M, VGGface2, and Megaface, and we summary some interesting findings about these training sets, as shown in Fig. 18.**  
현재 공개된 대규모 교육 Dataset(표 VI)를 제공하는 더 많은 Database, 특히 MS-Celeb-1M, VGGface2 및 Megaface와 같은 1M 이상의 이미지가 포함된 3개의 Database가 있으며 이러한 교육 세트에 대한 몇 가지 흥미로운 결과를 그림 18에 도시된 바와 같이 요약합니다.    

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_06.png">
<p align="center">
</p>
<br>
<br>
<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_18.png">
<p align="center">
</p>
<br>
<br>

### Depth v.s. breadth

<br>
<br>
<br>

**These large training sets are expanded from depth or breadth. VGGface2 provides a large-scale training dataset of depth, which have limited number of subjects but many images for each subjects.**  
이러한 대규모 Train 세트는 깊이 또는 폭에서 확장됩니다. VGGface2는 피험자의 수는 제한되어 있지만 각 피험자에 대한 이미지가 많은 대규모 학습 Dataset를 제공합니다.

<br>

**The depth of dataset enforces the trained model to address a wide range intraclass variations, such as lighting, age, and pose.**  
Dataset의 깊이는 Train된 Model이 조명, 연령 및 포즈와 같은 광범위한 클래스 내 변형을 처리하도록 합니다.

<br>

**In contrast, MS-Celeb-1M and Mageface (Challenge 2) offers large-scale training datasets of breadth, which contains many subject but limited images for each subjects.**  
대조적으로 MS-Celeb-1M 및 Mageface(Challenge 2)는 많은 주제를 포함하지만 각 주제에 대해 제한된 이미지를 포함하는 폭넓은 대규모 Train Dataset를 제공합니다.

<br>

**The breadth of dataset ensures the trained model to cover the sufficiently variable appearance of various people.**  
Dataset의 폭이 넓기 때문에 Train된 Model이 다양한 사람들의 충분히 다양한 모습을 다룰 수 있습니다. 

<br>

**Cao et al. [39] conducted a systematic studies on model training using VGGface2 and MSCeleb-1M, and found an optimal model by first training on MS-Celeb-1M (breadth) and then fine-tuning on VGGface2 (depth).**  
Caoet al. [39]는 VGGface2와 MSCeleb-1M을 사용한 Model Train에 대한 체계적인 연구를 수행했으며, MS-Celeb-1M(breadth)에서 먼저 Train한 다음 VGGface2(depth)에서 미세 조정하여 최적의 Model을 찾았습니다.    

<br>
<br>
<br>

### Long tail distribution

<br>
<br>
<br>

**The utilization of long tail distribution is different among datasets.**  
롱테일 분포의 활용은 Dataset마다 다릅니다. 

<br>

**For example, in Challenge 2 of MS-Celeb-1M, the novel set specially uses the tailed data to study low-shot learning**  
예를 들어, MS-Celeb-1M의 챌린지 2에서 새로운 세트는 특히 꼬리 데이터를 사용하여 로우 샷 학습을 연구합니다. 

<br>

**central part of the long tail distribution is used by the Challenge 1 of MS-Celeb1M and images’ number is approximately limited to 100 for each celebrity;**  
롱테일 분포의 중앙 부분은 MS-Celeb1M의 Challenge 1에서 사용되며 이미지 수는 유명인당 약 100개로 제한됩니다. 

<br>

**VGGface and VGGface2 only use the head part to construct deep databases;**  
VGGface 및 VGGface2는 헤드 부분만 사용하여 심층 Database를 구성합니다. 

<br>

**Megaface utilizes the whole distribution to contain as many images as possible, the minimal number of images is 3 per person and the maximum is 2469.**  
Megaface는 가능한 한 많은 이미지를 포함하기 위해 전체 분포를 활용하며 최소 이미지 수는 1인당 3개, 최대 2469개입니다.    

<br>
<br>
<br>

### Data engineering

<br>
<br>
<br>

**Several popular benchmarks, such as LFW unrestricted protocol, Megaface Challenge 1, MS-Celeb1M Challenge 1&2, explicitly encourage researchers to collect and clean a large-scale data set for enhancing the capability of deep neural network.**
LFW 무제한 Protocol, Megaface Challenge 1, MS-Celeb1M Challenge 1&2와 같은 몇 가지 인기 있는 benchmark는 연구원이 심층 neural network의 기능을 향상시키기 위해 대규모 Dataset를 수집하고 정리하도록 명시적으로 권장합니다.

<br>

**Although data engineering is a valuable problem to computer vision researchers, this protocol is more incline to the industry participants.**  
데이터 엔지니어링은 컴퓨터 비전 연구자에게 중요한 문제이지만 이 Protocol은 업계 참여자에게 더 적합합니다.

<br>

**As evidence, the leaderboards of these experiments are mostly occupied by the companies holding invincible hardwares and data scales.**  
증거로, 이러한 실험의 순위표는 대부분 무적의 하드웨어와 데이터 규모를 보유한 회사가 차지하고 있습니다.

<br>

**This phenomenon may not be beneficial for developments of new models in academic community.**  
이러한 현상은 학계에서 새로운 Model을 개발하는 데 도움이 되지 않을 수 있습니다.

<br>
<br>
<br>

### Data noise

<br>
<br>
<br>

**Owing to data source and collecting strategies, existing large-scale datasets invariably contain label noises.**  
데이터 소스 및 수집 전략으로 인해 기존 대규모 Dataset에는 항상 레이블 노이즈가 포함됩니다.

**Wang et al. [124] profiled the noise distribution in existing datasets in Fig. 19 and showed that the noise percentage increases dramatically along the scale of data.**  
Wang et al. [124]는 그림 19의 기존 Dataset에서 노이즈 분포를 프로파일링했으며 노이즈 비율이 데이터 규모에 따라 극적으로 증가함을 보여주었습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_19.png">
<p align="center">
</p>
<br>
<br>

**Moreover, they found that noise is more lethal on a 10,000-class problem of FR than on a 10-class problem of object classification and that label flip noise severely deteriorates the performance of a model, especially the model using A-softmax [84].**  
또한 객체 분류의 10개 클래스 문제보다 FR의 10,000개 클래스 문제에서 노이즈가 더 치명적이며 레이블 플립 노이즈가 Model, 특히 A-softmax를 사용하는 Model의 성능을 심각하게 저하시킨다는 것을 발견했습니다[84].

<br>

**Therefore, building a sufficiently large and clean dataset for academic research is very meaningful.**  
따라서 학술 연구를 위해 충분히 크고 깨끗한 Dataset를 구축하는 것은 매우 의미가 있습니다.

<br>

**Deng et al. [106] found there are serious label noise in MS-Celeb-1M [45], and they cleaned the noise of MS-Celeb-1M, and made the refined dataset public available.**  
Deng et al. [106]은 MS-Celeb-1M[45]에서 심각한 레이블 노이즈가 있음을 발견했으며 MS-Celeb-1M의 노이즈를 정리하고 정제된 Dataset를 공개했습니다.

<br>

**Microsoft and Deepglint jointly released the largest public data set [163] with cleaned labels, which includes 4M images cleaned from MS-Celeb-1M dataset and 2.8M aligned images of 100K Asian celebrities.**  
Microsoft와 Deepglint는 MS-Celeb-1M 데이터세트에서 정리된 4백만 개의 이미지와 100,000명의 아시아 유명인사의 정렬된 이미지 280만 개를 포함하는 정리된 레이블이 포함된 가장 큰 공개 Dataset[163]를 공동으로 발표했습니다.

<br>

**Moreover, Zhan et al. [167] shifted the focus from cleaning the datasets to leveraging more unlabeled data.**  
또한, Zhan et al. [167]은 Dataset 정리에서 레이블이 지정되지 않은 더 많은 데이터 활용으로 초점을 이동했습니다. 

<br>

**Through automatically assigning pseudo labels to unlabeled data with the help of relational graphs, they obtained competitive or even better results over the fullysupervised counterpart.**  
관계형 그래프의 도움으로 레이블이 지정되지 않은 데이터에 의사 레이블을 자동으로 할당함으로써 fullysupervised counterpart보다 경쟁력 있거나 더 나은 결과를 얻었습니다.      

<br>
<br>
<br>

### Data bias

<br>
<br>
<br>

**Large-scale training datasets, such as CASIAWebFace [120], VGGFace2 [39] and MS-Celeb-1M [45], are typically constructed by scraping websites like Google Images, and consist of celebrities on formal occasions: smiling, makeup, young, and beautiful.**  
CASIAWebFace [120], VGGFace2 [39] 및 MS-Celeb-1M [45]과 같은 대규모 교육 Dataset는 일반적으로 Google 이미지와 같은 웹 사이트를 스크랩하여 구성되며 공식 행사에서 웃고, 화장하고 젊고 아름다운 유명인사의 이미지로 구성됩니다.

<br>

**They are largely different from databases captured in the daily life (e.g. Megaface).**  
일상 생활에서 캡처한 Database(예: 메가페이스)와는 크게 다릅니다.

<br>

**The biases can be attributed to many exogenous factors in data collection, such as cameras, lightings, preferences over certain types of backgrounds, or annotator tendencies.**  
편향은 카메라, 조명, 특정 유형의 배경에 대한 선호도 또는 주석 작성자 경향과 같은 데이터 수집의 많은 외생적 요인에 기인할 수 있습니다.

<br>

**Dataset biases adversely affect cross-dataset generalization; that is, the performance of the model trained on one dataset drops significantly when applied to another one.**  
Dataset biases는 cross-dataset generalization에 악영향을 미칩니다. 즉, 한 Dataset에서 Train된 Model의 성능은 다른 Dataset에 적용될 때 크게 떨어집니다.

<br>

**One persuasive evidence is presented by P.J. Phillips’ study [168] which conducted a cross benchmark assessment of VGGFace model [37] for face recognition.**  
P.J. Phillips의 연구[168]는 얼굴 인식을 위한 VGGFace Model[37]의 교차 benchmark Evaluation를 수행한 설득력 있는 증거 중 하나입니다.

<br>

**The VGGFace model achieves 98.95% on LFW [23] and 97.30% on YTF [169], but only obtains 26%, 52% and 85% on Ugly, Bad and Good partition of GBU database [170].**  
VGGFace Model은 LFW[23]에서 98.95%, YTF[169]에서 97.30%를 달성했지만 GBU Database[170]의 Ugly, Bad 및 Good 파티션에서는 26%, 52% 및 85%만 얻었습니다.

<br>

**Demographic bias (e.g., race/ethnicity, gender, age) in datasets is a universal but urgent issue to be solved in data bias field.**  
데이터셋의 인구학적 편향(예: 인종/민족, 성별, 연령)은 보편적이지만 데이터 편향 분야에서 해결해야 할 시급한 문제입니다.

<br>

**In existing training and testing datasets, the male, White, and middle-aged cohorts always appear more frequently, as shown in Table VII, which inevitably causes deep learning models to replicate and even amplify these biases resulting in significantly different accuracies when deep models are applied to different demographic groups.**  
기존 Train 및 Test Dataset에서 남성, 백인 및 중년 코호트는 표 VII에 표시된 것처럼 항상 더 자주 나타납니다. 이는 필연적으로 Deep Learning Model이 다양한 인구 통계 그룹에 적용될 때 상당히 다른 정확도를 초래합니다. 

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_07.png">
<p align="center">
</p>
<br>
<br>

**Some researches showed that the female, Black, and younger cohorts are usually more difficult to recognize in FR systems trained with commonly-used datasets.**  
일부 연구에 따르면 여성, 흑인 및 젊은 코호트는 일반적으로 일반적으로 사용되는 Dataset로 Train된 FR 시스템에서 인식하기 더 어렵습니다.

<br>

**For example, Wang et al. proposed a Racial Faces in-the-Wild (RFW) database and proved that existing commercial APIs and the SOTA algorithms indeed work unequally for different races and the maximum difference in error rate between the best and worst groups is 12%, as shown in Table VIII.**  
예를 들어, Wang et al. RFW(Racial Faces in-the-Wild)은 Database를 제안하고 기존 상용 API와 SOTA 알고리즘이 실제로 서로 다른 인종에 대해 불평등하게 작동하고 최고 그룹과 최악 그룹 간의 최대 오류율 차이가 12%임을 입증했습니다(표 참조). VIII.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_08.png">
<p align="center">
</p>
<br>
<br>

**Hupont et al. showed that SphereFace has a TAR of 0.87 for White males which drops to 0.28 for Asian females, at a FAR of 1e − 4.**  
Hupont et al.은 SphereFace는 백인 남성의 경우 TAR이 0.87이고 FAR이 1e - 4일 때 아시아 여성의 경우 0.28로 떨어지는 것으로 나타났습니다.

<br>

**Such bias can result in mistreatment of certain demographic groups, by either exposing them to a higher risk of fraud, or by making access to services more difficult.**  
이러한 편향은 특정 인구 집단을 더 높은 사기 위험에 노출시키거나 서비스에 대한 접근을 더 어렵게 함으로써 이들을 학대하는 결과를 초래할 수 있습니다.

<br>

**Therefore, addressing data bias and enhancing fairness of FR systems in real life are urgent and necessary tasks.**  
따라서 실생활에서 data bias를 해소하고 FR 시스템의 공정성을 높이는 것이 시급하고 필요한 과제입니다. 

<br>

**Collecting balanced data to train a fair model or designing some debiasing algorithms are effective way.**  
공정한 Model을 Train하기 위해 균형 잡힌 데이터를 수집하거나 편향성 제거 알고리즘을 설계하는 것이 효과적인 방법입니다.        

<br>
<br>
<br>

### B. Training protocols

<br>
<br>
<br>

**In terms of training protocol, FR can be categorized into subject-dependent and subject-independent settings, as illustrated in Fig. 20.**  
Train Protocol의 관점에서 FR은 그림 20과 같이 subject-dependent settings과 subject-dependent settings으로 분류할 수 있습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_20.png">
<p align="center">
</p>
<br>
<br>

### Subject-dependent protocol

<br>
<br>
<br>

**For subject-dependent protocol, all testing identities are predefined in training set, it is natural to classify testing face images to the given identities.**  
subject-dependent protocol의 경우 모든 테스트 ID는 Train 세트에 미리 정의되어 있으므로 테스트 얼굴 이미지를 주어진 ID로 분류하는 것이 자연스럽습니다.

<br>

**Therefore, subject-dependent FR can be well addressed as a classification problem, where features are expected to be separable.**  
따라서 subject-dependent FR은 features가 분리될 것으로 예상되는 분류 문제로 잘 다룰 수 있습니다.

<br>

**The protocol is mostly adopted by the early-stage (before 2010) FR studies on FERET [177], AR [178], and is suitable only for some small-scale applications.**  
이 Protocol은 대부분 FERET [177], AR [178]에 대한 초기 단계(2010년 이전) FR 연구에서 채택되었으며 일부 소규모 응용 프로그램에만 적합합니다.

<br>

**The Challenge 2 of MS-Celeb-1M is the only large-scale database using subject-dependent training protocol.**  
MS-Celeb-1M의 Challenge 2는 subject-dependent Train Protocol을 사용하는 유일한 대규모 Database입니다.

<br>
<br>
<br>

### Subject-independent protocol

<br>
<br>
<br>

**For subject-independent protocol, the testing identities are usually disjoint from the training set, which makes FR more challenging yet close to practice.**  
subject-independent protocol의 경우 테스트 ID는 일반적으로 Train 세트와 분리되어 FR을 더 도전적이면서도 현실에 가깝게 만듭니다.

<br>

**Because it is impossible to classify faces to known identities in training set, generalized representation is essential.**  
Train 세트에서 얼굴을 알려진 신원으로 분류하는 것은 불가능하기 때문에 일반화된 표현이 필수적입니다.

<br>

**Due to the fact that human faces exhibit similar intrasubject variations, deep models can display transcendental generalization ability when training with a sufficiently large set of generic subjects, where the key is to learn discriminative large-margin deep features.**  
인간의 얼굴이 유사한 개체 내 변형을 나타내기 때문에 심층 Model은 충분히 큰 일반 개체 집합으로 Train할 때 초월적인 일반화 능력을 표시할 수 있습니다. 여기에서 핵심은 차별적이고 마진이 큰 Deep Features을 학습하는 것입니다.

<br>

**This generalization ability makes subject-independent FR possible.**  
이 일반화 능력은 subject-independent FR을 가능하게 합니다.

<br>

**Almost all major face-recognition benchmarks, such as LFW [23], PaSC [179], IJB-A/B/C and Megaface [44], [164], require the tested models to be trained under subject-independent protocol.**  
LFW [23], PaSC [179], IJB-A/B/C 및 Megaface [44], [164]와 같은 거의 모든 주요 안면 인식 benchmark는 테스트된 Model이 subject-independent Protocol에 따라 Train되어야 합니다.

<br>
<br>
<br>

### C. Evaluation tasks and performance metrics

<br>
<br>
<br>

**In order to evaluate whether our deep models can solve the different problems of FR in real life, many testing datasets are designed to evaluate the models in different tasks, i.e. face verification, close-set face identification and open-set face identification.**  
deep Model이 실생활에서 FR의 다양한 문제를 해결할 수 있는지 여부를 Evaluation하기 위해 많은 Test Data 세트가 다양한 작업, 즉 얼굴 검증, 근접 얼굴 식별 및 열린 얼굴 식별에서 Model을 Evaluation하도록 설계되었습니다.

<br>

**In either task, a set of known subjects is initially enrolled in the system (the gallery), and during testing, a new subject (the probe) is presented.**  
어느 작업에서든 일련의 알려진 주제가 처음에 system (the gallery)에 등록되고 테스트 중에 새로운 subject (the probe)가 제시됩니다.

<br>

**Face verification computes one-to-one similarity between the gallery and probe to determine whether the two images are of the same subject, whereas face identification computes one-to-many similarity to determine the specific identity of a probe face.**  
Face verification은 Gallery와 Probe 간의 일대일 유사성을 계산하여 두 이미지가 동일한 대상인지 여부를 결정하는 반면, face identification은 일대다 유사성을 계산하여 Probe 얼굴의 특정 ID를 결정합니다.

<br>

**When the probe appears in the gallery identities, this is referred to as closed-set identification; when the probes include those who are not in the gallery, this is open-set identification.**  
probe가 gallery identities에 나타나면 이를 폐쇄 세트 식별(closed-set identification)이라고 합니다. Probe에 Gallery에 없는 사람들이 포함되는 경우 이는 개방형 식별(open-set identification)입니다.

<br>

**Face verification is relevant to access control systems, re-identification, and application independent evaluations of FR algorithms.**  
Face verification은 액세스 제어 시스템, 재식별 및 FR 알고리즘의 애플리케이션 독립적 Evaluation와 관련이 있습니다.

<br>

**It is classically measured using the receiver operating characteristic (ROC) and estimated mean accuracy (Acc).**  
receiver operating characteristic (ROC) 및 estimated mean accuracy (Acc)를 사용하여 고전적으로 측정됩니다.

<br>

**At a given threshold (the independent variable), ROC analysis measures the true accept rate (TAR), which is the fraction of genuine comparisons that correctly exceed the threshold, and the false accept rate (FAR), which is the fraction of impostor comparisons that incorrectly exceed the threshold.**  
주어진 임계값(독립 변수)에서 ROC 분석은 임계값을 올바르게 초과하는 실제 비교의 비율인 true accept rate (TAR)과 사기꾼 비교의 비율인 false accept rate (FAR)을 측정합니다. 

<br>

**And Acc is a simplified metric introduced by LFW [23], which represents the percentage of correct classifications.**  
그리고 Acc는 LFW[23]에 의해 도입된 단순화된 메트릭으로 올바른 분류의 백분율을 나타냅니다.

<br>

**With the development of deep FR, more accurate recognitions are required.**  
deep FR의 발달로 더 정확한 인식이 필요합니다.

<br>

**Customers concern more about the TAR when FAR is kept in a very low rate in most security certification scenario.**  
고객은 대부분의 보안 인증 시나리오에서 FAR이 매우 낮은 비율로 유지될 때 TAR에 대해 더 우려합니다.    

<br>
<br>
<br>

### Close-set face identification

<br>
<br>
<br>

**Close-set face identification is relevant to user driven searches (e.g., forensic identification), rank-N and cumulative match characteristic (CMC) is commonly used metrics in this scenario.**  
Close-set face identification은 user driven searches (e.g., forensic identification)과 관련이 있으며, rank-N 및 CMC(cumulative match characteristic)는 이 시나리오에서 일반적으로 사용되는 메트릭입니다.

<br>

**Rank-N is based on what percentage of probe searches return the probe’s gallery mate within the top k rank-ordered results.**  
Rank-N은 상위 k 순위 결과 내에서 Probe의 Gallery mate를 반환하는 Probe 검색의 백분율을 기반으로 합니다.

<br>

**The CMC curve reports the percentage of probes identified within a given rank (the independent variable).**  
CMC 곡선은 주어진 순위(독립 변수) 내에서 식별된 Probe의 백분율을 보고합니다.

<br>

**IJB-A/B/C concern on the rank-1 and rank-5 recognition rate.**  
IJB-A/B/C는 1등급 및 5등급 인식률에 대한 관심입니다. 

<br>

**The MegaFace challenge systematically evaluates rank-1 recognition rate function of increasing number of gallery distractors (going from 10 to 1 Million), the results of the SOTA evaluated on MegaFace challenge are listed in Table IX.**  
MegaFace 챌린지는 Gallery 선택 항목의 증가(10에서 100만)에 따른 랭크-1 인식률 함수를 체계적으로 Evaluation하며, MegaFace 챌린지에서 Evaluation된 SOTA의 결과는 표 IX에 나열되어 있습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_09.png">
<p align="center">
</p>
<br>
<br>

**Rather than rank-N and CMC, MS-Celeb-1M [45] further applies a precision-coverage curve to measure identification performance under a variable threshold t.**  
rank-N 및 CMC 대신 MS-Celeb-1M [45]은 가변 임계값 t에서 식별 성능을 측정하기 위해 precision-coverage curve을 추가로 적용합니다.

<br>

**The probe is rejected when its confidence score is lower than t.**  
신뢰도 점수가 t보다 낮으면 Probe가 거부됩니다.

<br>

**The algorithms are compared in term of what fraction of passed probes, i.e. coverage, with a high recognition precision, e.g. 95% or 99%, the results of the SOTA evaluated on MS-Celeb-1M challenge are listed in Table X.**  
알고리즘은 통과된 Probe의 비율, 즉 커버리지, 높은 인식 정밀도(예: 95% 또는 99%, MS-Celeb-1M 챌린지에서 Evaluation된 SOTA의 결과는 표 X에 나열되어 있습니다.      

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_10.png">
<p align="center">
</p>
<br>
<br>

### Open-set face identification

<br>
<br>
<br>

**Open-set face identification is relevant to high throughput face search systems (e.g., de-duplication, watch list identification), where the recognition system should reject unknown/unseen subjects (probes who do not present in gallery) at test time.**  
Open-set face identification은 인식 시스템이 테스트 시간에 알 수 없거나 보이지 않는 주제(Gallery에 없는 Probe)를 거부해야 하는 고처리량 얼굴 검색 시스템(예: 중복 제거, 감시 목록 식별)과 관련이 있습니다.

<br>

**At present, there are very few databases covering the task of open-set FR.**  
현재 개방형 세트 FR 작업을 다루는 Database가 거의 없습니다. 

<br>

**IJB-A/B/C benchmarks introduce a decision error tradeoff (DET) curve to characterize the the false negative identification rate (FNIR) as function of the false positive identification rate (FPIR).**  
IJB-A/B/C benchmark는 FPIR(False Positive Identification Rate)의 함수로 FNIR(False Negative Identification Rate)을 특성화하기 위해 DET(Decision Error Tradeoff) 곡선을 도입합니다.

<br>

**FPIR measures what fraction of comparisons between probe templates and non-mate gallery templates result in a match score exceeding T.**  
FPIR은 Probe template과 non-mate gallery template 간의 비교에서 일치 점수가 T를 초과하는 비율을 측정합니다.

<br>

**At the same time, FNIR measures what fraction of probe searches will fail to match a mated gallery template above a score of T.**  
동시에 FNIR은 T점수 이상으로 결합된 Gallery template과 일치하지 않는 Probe 검색의 비율을 측정합니다.

<br>

**The algorithms are compared in term of the FNIR at a low FPIR, e.g. 1% or 10%, the results of the SOTA evaluated on IJB-A dataset as listed in Table XI.**  
알고리즘은 낮은 FPIR(Ex. 1% 또는 10%)에서 FNIR 측면에서 비교됩니다. 표 XI에 나열된 IJB-A Dataset에서 Evaluation된 SOTA의 결과입니다.    

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_11.png">
<p align="center">
</p>
<br>
<br>

### D. Evaluation Scenes and Data

<br>
<br>
<br>

**Public available training databases are mostly collected from the photos of celebrities due to privacy issue, it is far from images captured in the daily life with diverse scenes.**  
공개된 트레이닝 Database는 사생활 보호 문제로 유명인 사진을 주로 수집하며, 일상에서 다양한 장면을 포착한 이미지와는 거리가 멀다.

<br>

**In order to study different specific scenarios, more difficult and realistic datasets are constructed accordingly, as shown in Table XII.**  
서로 다른 특정 시나리오를 연구하기 위해 표 XII와 같이 더 어렵고 현실적인 Dataset가 그에 따라 구성됩니다.

<br>

**According to their characteristics, we divide these scenes into four categories: cross-factor FR, heterogenous FR, multiple (or single) media FR and FR in industry (Fig. 21).**  
그 특성에 따라 이러한 장면을 cross-factor FR, heterogenous FR, multiple (or single) media FR 및 industry FR의 네 가지 범주로 나눕니다(그림 21).

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Table_12.png">
<p align="center">
</p>
<br>
<br>
<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_21.png">
<p align="center">
</p>
<br>
<br>

<br>
<br>

### Cross-factor FR

<br>
<br>
<br>

**Due to the complex nonlinear facial appearance, some variations will be caused by people themselves, such as cross-pose, cross-age, make-up, and disguise.**  
복잡한 비선형 얼굴 모양으로 인해 cross-pose, cross-age, make-up 및 disguise과 같은 일부 변형이 사람 자체에 의해 발생합니다.

<br>

**For example, CALFW, MORPH, CACD and FG-NET are commonly useddatasets with different age range;**  
예를 들어 CALFW, MORPH, CACD 및 FG-NET은 연령대가 다른 일반적으로 사용되는 Dataset입니다. 

<br>

**CFP [182] only focuses on frontal and profile face, CPLFW [181] is extended from LFW and contains different poses.**  
CFP[182]는 정면과 옆얼굴에만 초점을 맞추고, CPLFW[181]는 LFW에서 확장되어 다양한 포즈를 포함합니다.

<br>

**Disguised faces in the wild (DFW) [214] evaluates face recognition across disguise.**  
DFW(Disguised Faces in the Wild)[214]는 변장을 통해 얼굴 인식을 Evaluation합니다.

<br>
<br>
<br>

### Heterogenous FR

<br>
<br>
<br>

**It refers to the problem of matching faces across different visual domains.**  
서로 다른 시각적 영역에서 일치하는 얼굴의 문제를 나타냅니다.

<br>

**The domain gap is mainly caused by sensory devices and cameras settings, e.g. visual light vs. near-infrared and photo vs. sketch.**  
domain gap는 주로 감각 장치 및 카메라 설정으로 인해 발생합니다. 예를 들면, 가시광 대 근적외선 및 사진 대 스케치 등이 있습니다.

<br>

**For example, CUFSF [201] and CUFS [199] are commonly used photo-sketch datasets and CUFSF dataset is harder due to lighting variation and shape exaggeration.**  
예를 들어 CUFSF [201] 및 CUFS [199]는 일반적으로 사용되는 사진 스케치 Dataset이며 CUFSF Dataset는 조명 변화 및 모양 과장으로 인해 더 어렵습니다.

<br>
<br>
<br>

### Multiple (or single) media FR

<br>
<br>
<br>

**Ideally, in FR, many images of each subject are provided in training datasets and image-to-image recognitions are performed when testing.**  
이상적으로 FR에서는 Train Dataset에 각 주제의 많은 이미지가 제공되고 테스트 시 이미지 대 이미지 인식이 수행됩니다.

<br>

**But the situation will be different in reality.**  
그러나 현실에서는 상황이 다를 것이다.

<br>

**Sometimes, the number of images per person in training set could be very small, such as MS-Celeb-1M challenge 2.**  
경우에 따라 MS-Celeb-1M 챌린지 2와 같이 training set의 1인당 이미지 수가 매우 적을 수 있습니다.

<br>

**This challenge is often called low- shot or fewshot FR.**  
이 챌린지는 종종 low-shot 또는 fewshot FR이라고 합니다.

<br>

**Moreover, each subject face in test set may be enrolled with a set of images and videos and set-to-set recognition should be performed, such as IJB-A and PaSC.**  
또한 test set의 각 피사체 얼굴은 일련의 이미지 및 비디오로 등록될 수 있으며 IJB-A 및 PaSC와 같은 세트 간 인식이 수행되어야 합니다.        

<br>
<br>
<br>

### FR in industry

<br>
<br>
<br>

**Although deep FR has achieved beyond human performance on some standard benchmarks, but some other factors should be given more attention rather than accuracy when deep FR is adopted in industry, e.g. anti-attack (CASIA-FASD) and 3D FR (Bosphorus, BU-3DFE and FRGCv2).**  
deep FR은 일부 표준 benchmark에서 인간의 성능을 넘어섰지만 deep FR이 업계에서 채택될 때 정확성보다는 공격 방지(CASIA-FASD) 및 3D FR(Bosphorus, BU-3DFE 및 FRGCv2)와 같은 다른 요인에 더 주의를 기울여야 합니다.

<br>

**Compared to publicly available 2D face databases, 3D scans are hard to acquire, and the number of scans and subjects in public 3D face databases is still limited, which hinders the development of 3D deep FR.**  
공개적으로 사용 가능한 2D 얼굴 Database와 비교할 때 3D 스캔은 획득하기 어렵고 공개 3D 얼굴 Database의 스캔 및 대상 수는 여전히 제한되어 3D deep FR 개발을 방해합니다.    

<br>
<br>
<br>

## VI. DIVERSE RECOGNITION SCENES OF DEEP LEARNING

<br>
<br>
<br>

**Despite the high accuracy in the LFW and Megaface, benchmarks, the performance of FR models still hardly meets the requirements in real-world application.**  
LFW 및 Megaface, benchmark의 높은 정확도에도 불구하고 FR Model의 성능은 여전히 실제 응용 프로그램의 요구 사항을 거의 충족하지 못합니다.

<br>

**A conjecture in industry is made that results of generic deep models can be improved simply by collecting big datasets of the target scene.**  
대상 장면의 큰 Dataset를 수집하는 것만으로 일반 심층 Model의 결과를 개선할 수 있다는 업계의 추측이 있습니다.

<br>

**However, this holds only to a certain degree.**  
그러나 이것은 어느 정도만 적용됩니다.

<br>

**More and more concerns on privacy may make the collection and human-annotation of face data become illegal in the future.**  
프라이버시에 대한 우려가 점점 더 커짐에 따라 얼굴 데이터의 수집 및 인간 주석이 향후 불법이 될 수 있습니다.

<br>

**Therefore, significant efforts have been paid to design excellent algorithms to address the specific problems with limited data in these realistic scenes.**  
따라서 이러한 현실적인 장면에서 제한된 데이터로 특정 문제를 해결하기 위해 우수한 알고리즘을 설계하기 위해 많은 노력을 기울였습니다.

<br>

**In this section, we present several special algorithms of FR.**  
이 섹션에서는 FR의 몇 가지 특수 알고리즘을 제시합니다.

<br>
<br>
<br>

### A. Cross-Factor Face Recognition

<br>
<br>
<br>

### 1) Cross-Pose Face Recognition

<br>
<br>
<br>

**As [182] shows that many existing algorithms suffer a decrease of over 10% from frontalfrontal to frontal-profile verification, cross-pose FR is still an extremely challenging scene.**  
[182]에서 많은 기존 알고리즘이 frontalfrontal에서 frontal-profile 검증까지 10% 이상의 감소를 겪고 있음을 보여주듯이 교차 포즈 FR은 여전히 매우 어려운 장면입니다.

<br>

**In addition to the aforementioned methods, including “one-to-many augmentation”, “many-to-one normalization” and assembled networks (Section IV and III-B.2), there are some other algorithms designed for crosspose FR.**  
"one-to-many augmentation", "many-to-one normalization" 및 assembled Network(섹션 IV 및 III-B.2)를 포함하여 앞서 언급한 방법 외에도 교차 FR을 위해 설계된 몇 가지 다른 알고리즘이 있습니다.

<br>

**Considering the extra burden of above methods, Cao et al. [215] attempted to perform frontalization in the deep feature space rather than the image space.**  
위의 방법의 추가 부담을 고려하여 Cao et al. [215]는 이미지 공간이 아닌 깊은 특징 공간에서 정면화를 시도했다.

<br>

**A deep residual equivariant mapping (DREAM) block dynamically added residuals to an input representation to transform a profile face to a frontal image.**  
DREAM(Deep Residual Equivariant Mapping) 블록은 프로필 얼굴을 정면 이미지로 변환하기 위해 입력 표현에 잔차를 동적으로 추가했습니다.

<br>

**Chen et al. proposed to combine feature extraction with multi-view subspace learning to simultaneously make features be more pose-robust and discriminative.**  
Chenet al.은 feature extraction과 multi-view subspace 학습을 결합하여 features을 보다 견고하고 차별적으로 만드는 것을 제안했습니다.

<br>

**Pose Invariant Model (PIM) jointly performed face frontalization and learned pose invariant representations end-to-end to allow them to mutually boost each other, and further introduced unsupervised cross-domain adversarial training and a learning to learn strategy to provide high-fidelity frontal reference face images.**  
PIM(Pose Invariant Model)은 얼굴 정면화를 공동으로 수행하고 포즈 불변 표현을 end-to-end로 학습하여 서로를 부스트할 수 있게 했으며, 추가로 감독되지 않은 도메인 간 adversarial Train 및 high-fidelity frontal reference face images를 제공하기 위한 학습 전략을 도입했습니다.      

<br>
<br>
<br>

### 2) Cross-Age Face Recognition

<br>
<br>
<br>

**Cross-age FR is extremely challenging due to the changes in facial appearance by the aging process over time.**  
Cross-age FR은 시간이 지남에 따라 노화 과정에 따른 얼굴 모양의 변화로 인해 매우 어렵습니다.

<br>

**One direct approach is to synthesize the desired image with target age such that the recognition can be performed in the same age group.**  
직접적인 접근 방법 중 하나는 원하는 이미지를 대상 연령과 합성하여 동일한 연령대에서 인식이 가능하도록 하는 것입니다.

<br>

**A generative probabilistic model was used by to model the facial aging process at each short-term stage.**  
각 단기 단계에서 얼굴 노화 과정을 Modeling하기 위해 generative probabilistic model이 사용되었습니다.

<br>

**The identity-preserved conditional generative adversarial networks (IPCGANs) framework utilized a conditional-GAN to generate a face in which an identity-preserved module preserved the identity information and an age classifier forced the generated face with the target age.**  
identity-preserved conditional generative adversarial networks (IPCGANs) 프레임워크는 conditional-GAN을 활용하여 ID 보존 Module이 ID 정보를 보존하고 연령 분류기가 생성된 얼굴에 목표 연령을 강제하는 얼굴을 생성했습니다.

<br>

**Antipov et al. proposed to age faces by GAN, but the synthetic faces cannot be directly used for face verification due to its imperfect preservation of identities.**  
Antipovet al.은 GAN이 얼굴을 노화시키기 위해 제안했지만 합성 얼굴은 신원 보존이 불완전하기 때문에 얼굴 확인에 직접 사용할 수 없습니다.

<br>

**Then, they used a local manifold adaptation (LMA) approach to solve the problem of [220].**  
그런 다음 [220]의 문제를 해결하기 위해 local manifold adaptation(LMA) 접근 방식을 사용했습니다.

<br>

**In [222], high-level age-specific features conveyed by the synthesized face are estimated by a pyramidal adversarial discriminator at multiple scales to generate more lifelike facial details.**  
[222]에서, 합성된 얼굴에 의해 전달되는 높은 수준의 연령별 특징은 더 생생한 얼굴 세부 사항을 생성하기 위해 여러 척도에서 피라미드형 적대적 판별기에 의해 추정됩니다.

<br>

**An alternative to address the crossage problem is to decompose aging and identity components separately and extract age-invariant representations.**  
교차 문제를 해결하기 위한 대안은 노화 및 정체성 구성 요소를 별도로 분해하고 연령 불변 표현(age-invariant representations)을 추출하는 것입니다.

<br>

**Wen et al. [192] developed a latent identity analysis (LIA) layer to separate these two components, as shown in Fig. 22.**  
Wen et al. [192]는 그림 22와 같이 이 두 구성 요소를 분리하기 위해 LIA(Latent Identity Analysis) 레이어를 개발했습니다.

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_22.png">
<p align="center">
</p>
<br>
<br>

**In [193], age-invariant features were obtained by subtracting agespecific factors from the representations with the help of the age estimation task.**  
[193]에서 연령 추정 작업의 도움으로 표현에서 연령별 요인을 빼서 연령 불변 특징을 얻었습니다.

<br>

**In [124], face features are decomposed in the spherical coordinate system, in which the identity-related components are represented with angular coordinates and the age-related information is encoded with radial coordinate.**  
[124]에서는 얼굴 특징을 구형 좌표계로 분해하여 신원 관련 요소는 각도 좌표로, 나이 관련 정보는 방사 좌표로 부호화하였다.

<br>

**Additionally, there are other methods designed for cross-age FR.**  
또한 교차 연령 FR을 위해 설계된 다른 방법이 있습니다.

<br>

**For example, Bianco ett al. [223] and El et al. [224] finetuned the CNN to transfer knowledge across age.**  
예를 들어, Bianco et al. [223] 및 El et al. 나이에 걸쳐 지식을 전달하기 위해 CNN을 미세 조정했습니다. 

<br>

**Wang et al. [225] proposed a siamese deep network to perform multi-task learning of FR and age estimation.**  
Wang et al. [225]는 FR 및 나이 추정의 멀티태스크 학습을 수행하기 위해 siamese Deep Network를 제안했습니다.

<br>

**Li et al. [226] integrated feature extraction and metric learning via a deep CNN.**  
Li et al.은 deep CNN을 통한 feature extraction 및 metric learning을 통합하였습니다.

<br>
<br>
<br>

### 3) Makeup Face Recognition

<br>
<br>
<br>

**Makeup is widely used by the public today, but it also brings challenges for FR due to significant facial appearance changes.**  
메이크업은 오늘날 대중에 의해 널리 사용되지만 상당한 얼굴 모양 변화로 인해 FR에 문제를 야기하기도 합니다.

<br>

**The research on matching makeup and nonmakeup face images is receiving increasing attention.**  
메이크업과 노메이크업 얼굴 이미지의 매칭에 대한 연구가 주목을 받고 있다.

<br>

**Li et al. [208] generated nonmakeup images from makeup ones by a bi-level adversarial network (BLAN) and then used the synthesized nonmakeup images for verification as shown in Fig. 23.**  
Li et al.은 BLAN(bi-level adversarial network)에 의해 nonmakeup 이미지에서 메이크업 이미지를 생성한 다음 합성된 메이크업 이미지를 검증에 사용했습니다(그림 23 참조).

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_23.png">
<p align="center">
</p>
<br>
<br>

**Sun et al. [227] pretrained a triplet network on videos and fine-tuned it on a small makeup datasets.**  
Sun et al. [227]은 비디오에서 triplet Network를 사전 Train하고 작은 메이크업 Dataset에서 미세 조정했습니다.

<br>

**Specially, facial disguise [214], [228], [229] is a challenging research topic in makeup face recognition.**  
특히 얼굴 변장[214], [228], [229]은 메이크업 얼굴 인식에서 도전적인 연구 주제이다.

<br>

**By using disguise accessories such as wigs, beard, hats, mustache, and heavy makeup, disguise introduces two variations:**  
가발, 턱수염, 모자, 콧수염, 짙은 화장과 같은 disguise 액세서리를 사용하여 disguise는 두 가지 변형을 선보입니다.

<br>

**(i) when a person wants to obfuscate his/her own identity, and (ii) another individual impersonates someone else’s identity.**  
(i) 자신의 신원을 난독화하려는 경우 및 (ii) 다른 사람이 다른 사람의 신원을 사칭하는 경우.

<br>

**Obfuscation increases intra-class variations whereas impersonation reduces the inter-class dissimilarity, thereby affecting face recognition/verification task.**  
난독화(Obfuscation)는 클래스 내 변형을 증가시키는 반면 가장(impersonation)은 클래스 간 비 유사성을 줄여 얼굴 인식/검증 작업에 영향을 미칩니다.

<br>

**To address this issue, a variety of methods are proposed.**  
이 문제를 해결하기 위해 다양한 방법이 제안됩니다.

<br>

**Zhang et al. [230] first trained two DCNNs for generic face recognition and then used Principal Components Analysis (PCA) to find the transformation matrix for disguised face recognition adaptation.**  
장 외. [230]은 먼저 일반 얼굴 인식을 위해 두 개의 DCNN을 교육한 다음 위장된 얼굴 인식 적응을 위한 변환 매트릭스를 찾기 위해 PCA(Principal Components Analysis)를 사용했습니다.

<br>

**Kohli et al. [231] finetuned models using disguised faces.**  
Kohliet al.은 위장된 얼굴을 사용한 Model을 미세 조정하였습니다.

<br>

**Smirnov et al. [232] proposed a hard example mining method benefitted from class-wise (Doppelganger Mining [233]) and example-wise mining to learn useful deep embeddings for disguised face recognition.**  
Smirnov et al. [232]는 위장된 얼굴 인식을 위한 유용한 Deep 임베딩을 학습하기 위해 클래스별(Doppelganger Mining [233]) 및 예제별 마이닝의 혜택을 받는 하드 예제 마이닝 방법을 제안했습니다.

<br>

**Suri et al. [234] learned the representations of images in terms of colors, shapes, and textures (COST) using an unsupervised dictionary learning method, and utilized the combination of COST features and CNN features to perform recognition.**  
Suriet al.은 무감독 사전 학습 방법을 사용하여 색상, 모양 및 질감(COST) 측면에서 이미지 표현을 학습하고 COST 기능과 CNN 기능의 조합을 활용하여 인식을 수행했습니다.      

<br>
<br>
<br>

### B. Heterogenous Face Recognition

<br>
<br>
<br>

### 1) NIR-VIS Face Recognition

<br>
<br>
<br>

**Due to the excellent performance of the near-infrared spectrum (NIS) images under lowlight scenarios, NIS images are widely applied in surveillance systems.**  
저조도 시나리오에서 NIS(근적외선 스펙트럼,near-infrared spectrum) 이미지의 뛰어난 성능으로 인해 NIS 이미지는 감시 시스템에 널리 적용됩니다.

<br>

**Because most enrolled databases consist of visible light (VIS) spectrum images, how to recognize a NIR face from a gallery of VIS images has been a hot topic.**  
대부분의 등록된 Database는 가시광선(VIS) 스펙트럼 이미지로 구성되어 있기 때문에 VIS 이미지 Gallery에서 NIR 얼굴을 인식하는 방법이 뜨거운 주제였습니다.

<br>

**Saxena et al. [235] and Liu et al. [236] transferred the VIS deep networks to the NIR domain by fine-tuning.**  
Saxenaet al. [235] 및 Liu et al. [236]은 미세 조정을 통해 VIS Deep Network를 NIR 도메인으로 이전했습니다.

<br>

**Lezama et al. [237] used a VIS CNN to recognize NIR faces by transforming NIR images to VIS faces through cross-spectral hallucination and restoring a low-rank structure for features through low-rank embedding.**  
Lezama et al. [237]은 교차 스펙트럼 환각(cross-spectral hallucination )을 통해 근적외선 이미지를 VIS 얼굴로 변환하고 하위 임베딩을 통해 기능의 하위 구조를 복원하여 근적외선 얼굴을 인식하기 위해 VIS CNN을 사용했습니다.

<br>

**Reale et al. [198] trained a VISNet (for visible images) and a NIRNet (for near-infrared images), and coupled their output features by creating a siamese network.**  
Realeet al. [198]은 VISNet(가시 이미지용)과 NIRNet(근적외선 이미지용)을 교육하고 siamese Network를 생성하여 출력 기능을 결합했습니다.

<br>

**He et al. [238], [239] divided the high layer of the network into a NIR layer, a VIS layer and a NIR-VIS shared layer, then, a modality-invariant feature can be learned by the NIR-VIS shared layer.**  
He et al.은 Network의 상위 계층을 NIR 계층, VIS 계층 및 NIR-VIS 공유 계층으로 나눈 다음 NIR-VIS 공유 계층에서 모달리티 불변 특징을 학습할 수 있습니다.

<br>

**Song et al. [240] embedded cross-spectral face hallucination and discriminative feature learning into an end-to-end adversarial network.**  
Song et al.은 cross-spectral face hallucination 및 discriminative feature learning을 end-to-end adversarial network에 임베디드 시켰습니다.

<br>

**In [196], the low-rank relevance and cross-modal ranking were used to alleviate the semantic gap.**  
[196]에서는 의미론적 격차를 완화하기 위해 낮은 순위 관련성과 교차 모달 순위를 사용했습니다.

<br>
<br>
<br>

### 2) Low-Resolution Face Recognition

<br>
<br>
<br>

**Although deep networks are robust to low resolution to a great extent, there are still a few studies focused on promoting the performance of low-resolution FR.**  
Deep Network는 저해상도에 대해 상당히 견고하지만 저해상도 FR의 성능을 촉진하는 데 초점을 맞춘 연구는 여전히 몇 가지 있습니다.

<br>

**For example, Zangeneh et al. [241] proposed a CNN with a two-branch architecture (a superresolution network and a feature extraction network) to map the high- and low-resolution face images into a common space where the intra-person distance is smaller than the interperson distance.**  
예를 들어, Zangeneh 외. [241]은 고해상도 및 저해상도 얼굴 이미지를 사람 간 거리보다 사람 간 거리가 더 작은 공통 공간에 매핑하기 위해 2개 분기 Architecture(초해상도 Network 및 특징 추출 Network)가 있는 CNN을 제안했습니다.

<br>

**Shen et al. [242] exploited the face semantic information and local structural constraints to better restore the shape and detail of face images.**  
Shenet al. [242]는 얼굴 이미지의 모양과 세부 사항을 더 잘 복원하기 위해 얼굴 의미론적 정보와 로컬 구조 제약 조건을 활용했습니다.

<br>

**In addition, they optimized the network with perceptual and adversarial losses to produce photo-realistic results.**  
또한 그들은 사진과 같은 결과를 생성하기 위해 perceptual 및 적대적 손실(adversarial losses)로 Network를 최적화했습니다.    

<br>
<br>
<br>

### 3) Photo-Sketch Face Recognition

<br>
<br>
<br>

**The photo-sketch FR may help law enforcement to quickly identify suspects.**  
photo-sketch FR은 법 집행 기관이 용의자를 신속하게 식별하는 데 도움이 될 수 있습니다.

<br>

**The commonly used methods can be categorized as two classes.**  
일반적으로 사용되는 방법은 두 가지 클래스로 분류할 수 있습니다.

<br>

**One is to utilize transfer learning to directly match photos to sketches.**  
하나는 Transfer Learning을 활용하여 사진과 스케치를 직접 일치시키는 것입니다.

<br>

**Deep networks are first trained using a large face database of photos and are then fine-tuned using small sketch database [243], [244].**  
Deep Network는 먼저 사진의 큰 얼굴 Database를 사용하여 Train된 다음 작은 스케치 Database를 사용하여 미세 조정됩니다[243], [244].

<br>

**The other is to use the image-to-image translation, where the photo can be transformed to a sketch or the sketch to a photo; then, FR can be performed in one domain.**  
다른 하나는 사진을 스케치로 변환하거나 스케치를 사진으로 변환할 수 있는 이미지 대 이미지 변환을 사용하는 것입니다. 그런 다음 FR은 하나의 도메인에서 수행할 수 있습니다.

<br>

**Zhang et al. [200] developed a fully convolutional network with generative loss and a discriminative regularizer to transform photos to sketches.**  
장 외. [200]은 사진을 스케치로 변환하기 위해 Generative Loss와 discriminative regularizer를 사용하여 완전히 합성곱 Network를 개발했습니다.

<br>

**Zhang et al. [245] utilized a branched fully convolutional neural network (BFCN) to generate a structure-preserved sketch and a texture-preserved sketch, and then they fused them together via a probabilistic method.**  
장 외. [245]는 BFCN(branched fully convolutional neural network)을 활용하여 구조 보존 스케치와 텍스처 보존 스케치를 생성한 다음 확률론적 방법을 통해 융합했습니다.

<br>

**Recently, GANs have achieved impressive results in image generation.**  
최근 GAN은 이미지 생성에서 인상적인 결과를 얻었습니다.

<br>

**Yi et al. [246], Kim et al. [247] and Zhu et al. [248] used two generators, GA and GB, to generate sketches from photos and photos from sketches, respectively (Fig. 24).**  
Yiet al. [246], Kim et al. [247] 및 Zhu et al. [248]은 사진에서 스케치를 생성하고 스케치에서 사진을 생성하기 위해 각각 GA와 GB라는 두 개의 생성기를 사용했습니다(그림 24).

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_24.png">
<p align="center">
</p>
<br>
<br>

**Based on [248], Wang et al. [202] proposed a multi-adversarial network to avoid artifacts by leveraging the implicit presence of feature maps of different resolutions in the generator subnetwork.**  
[248], Wang et al. [202]는 generator subnetwork에서 서로 다른 해상도의 피쳐 맵의 암시적 존재를 활용하여 아티팩트를 피하기 위해 multi-adversarial network를 제안했습니다.

<br>

**Similar to photo-sketch FR, photocaricature FR is one kind of heterogenous FR scenes which is challenging and important to understanding of face perception.**  
포토 스케치 FR과 유사하게 포토캐리커처 FR은 이질적인 FR 장면의 한 종류로 얼굴 인식을 이해하는 데 도전적이고 중요합니다.

<br>

**Huo et al. [213] built a large dataset of caricatures and photos, and provided several evaluation protocols and their baseline performances for comparison.**  
Huo et al. [213] 캐리커처와 사진의 대규모 Dataset를 구축하고 비교를 위해 여러 Evaluation Protocol과 기본 성능을 제공했습니다.

<br>
<br>
<br>

### C. Multiple (or single) media Face Recognition

<br>
<br>
<br>

### 1) Low-Shot Face Recognition

<br>
<br>
<br>

**For many practical applications, such as surveillance and security, the FR system should recognize persons with a very limited number of training samples or even with only one sample.**  
감시 및 보안과 같은 많은 실제 응용 분야에서 FR 시스템은 매우 제한된 수의 Train 샘플 또는 심지어 단 하나의 샘플로도 사람을 인식해야 합니다. 

<br>

**The methods of lowshot learning can be categorized as 1) synthesizing training data and 2) learning more powerful features.**  
로우샷 학습 방법은 1) synthesizing training data 2) learning more powerful features로 분류할 수 있다.

<br>

**Hong et al. [249] generated images in various poses using a 3D face model and adopted deep domain adaptation to handle other variations, such as blur, occlusion, and expression (Fig. 25).**  
Hong et al. [249] 3D 얼굴 Model을 사용하여 다양한 포즈의 이미지를 생성하고 흐림, 폐색 및 표현과 같은 다른 변형을 처리하기 위해 심층 도메인 적응을 채택했습니다(그림 25).

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_25.png">
<p align="center">
</p>
<br>
<br>

**Choe et al. [250] used data augmentation methods and a GAN for pose transition and attribute boosting to increase the size of the training dataset.**  
Choe et al. [250]은 Train Dataset의 크기를 늘리기 위해 포즈 전환 및 속성 부스팅을 위해 data augmentation 방법과 GAN을 사용했습니다.

<br>

**Wu et al. [176] proposed a framework with hybrid classifiers using a CNN and a nearest neighbor (NN) model.**
Wuet al. [176]은 CNN과 최근접 이웃(NN) Model을 사용하는 하이브리드 분류기가 있는 프레임워크를 제안했습니다.

<br>

**Guo et al. [143] made the norms of the weight vectors of the one-shot classes and the normal classes aligned to address the data imbalance problem.**  
Guoet al. [143]은 데이터 불균형 문제를 해결하기 위해 원샷 클래스와 일반 클래스의 가중치 벡터의 규범을 정렬했습니다.

<br>

**Cheng et al. [137] proposed an enforced softmax that contains optimal dropout, selective attenuation, L2 normalization and modellevel optimization.**  
Cheng et al. [137]은 최적의 드롭아웃, 선택적 감쇠, L2 정규화 및 Model 수준 최적화를 포함하는 강제 Softmax를 제안했습니다.

<br>

**Yin et al. [251] augmented feature space of low-shot classes by transferring the principal components from regular to low-shot classes to encourage the variance of low-shot classes to mimic that of regular classes.**  
Yin et al. 로우샷 클래스의 분산이 정규 클래스의 분산을 모방하도록 장려하기 위해 주성분을 정규 클래스에서 로우샷 클래스로 이전함으로써 로우샷 클래스의 feature space을 확장합니다.        

<br>
<br>
<br>

### 2) Set/Template-Based Face Recognition

<br>
<br>
<br>

**Different from traditional image-to-image recognition, set-to-set recognition takes a set (heterogeneous contents containing both images and videos) as the smallest unit of representation.**  
기존의 이미지간 인식과 달리 세트간 인식은 세트(이미지와 동영상이 모두 포함된 이종 콘텐츠)를 최소 표현 단위로 삼는다.

<br>

**This kind of setting does reflect the real-world biometric scenarios, thereby attracting a lot of attention.**  
이러한 종류의 설정은 실제 생체 인식 시나리오를 반영하므로 많은 관심을 끌고 있습니다.

<br>

**After learning face representations of media in each set, two strategies are generally adopted to perform set-to-set matching.**  
각 세트에서 미디어의 얼굴 표현을 학습한 후 세트 간 매칭을 수행하기 위해 일반적으로 두 가지 전략이 채택됩니다.

<br>

**One is to use these representations to perform pair-wise similarity comparison of two sets and aggregate the results into a single and final score by max score pooling [96], average score pooling [252] and its variations [253], [254].**  
하나는 이러한 표현을 사용하여 두 세트의 쌍별 유사성 비교를 수행하고 최대 점수 풀링[96], 평균 점수 풀링[252] 및 그 변형[253], [254]에 의해 결과를 단일 및 최종 점수로 집계하는 것입니다.

<br>

**The other strategy is feature pooling [96], [103], [81] which first aggregates face representations into a single representation for each set and then performs a comparison between two sets.**  
다른 전략은 feature pooling[96], [103], [81]으로 먼저 얼굴 표현을 각 세트에 대한 단일 표현으로 집계한 다음 두 세트 간의 비교를 수행합니다.

<br>

**In addition to the commonly used strategies, there are also some novel methods proposed for set/template-based FR.**  
일반적으로 사용되는 전략 외에도 세트/template 기반 FR에 대해 제안된 몇 가지 새로운 방법이 있습니다.

<br>

**For example, Hayat et al. [255] proposed a deep heterogeneous feature fusion network to exploit the features’ complementary information generated by different CNNs.**  
예를 들어, 하야트 외. [255]는 서로 다른 CNN에서 생성된 기능의 보완 정보를 활용하기 위해 deep heterogeneous feature fusion network를 제안했습니다. 

<br>

**Liu et al. [256] introduced the actorcritic reinforcement learning for set-based FR.**  
Liu et al. [256]는 set-based FR에 대한 배우 비판적 강화 학습을 도입했습니다.

<br>

**They casted the inner-set dependency modeling to a Markov decision process in the latent space, and trained a dependency-aware attention control agent to make attention control for each image in each step.**  
그들은 내부 집합 종속성 Modeling(inner-set dependency modeling)을 잠재 공간의 Markov decision process에 캐스팅하고 각 단계에서 각 이미지에 대한 주의 제어를 수행하도록 종속성 인식 주의 제어 에이전트를 Train했습니다.    

<br>
<br>
<br>

### 3) Video Face Recognition

<br>
<br>
<br>

**There are two key issues in video FR: one is to integrate the information across different frames together to build a representation of the video face, and the other is to handle video frames with severe blur, pose variations, and occlusions.**  
비디오 FR에는 두 가지 주요 문제가 있습니다. 하나는 서로 다른 프레임의 정보를 통합하여 비디오 얼굴의 표현을 구축하는 것이고 다른 하나는 심각한 블러, 포즈 변형 및 폐색이 있는 비디오 프레임을 처리하는 것입니다.

<br>

**For frame aggregation, Yang et al. [83] proposed a neural aggregation network (NAN) in which the aggregation module, consisting of two attention blocks driven by a memory, produces a 128-dimensional vector representation (Fig. 26).**  
프레임 집계(frame aggregation)의 경우 Yang et al. [83]은 메모리에 의해 구동되는 두 개의 어텐션 블록으로 구성된 aggregation Module이 128차원 벡터 표현을 생성하는 신경 집계 Network(NAN)를 제안했습니다(그림 26).

<br>
<br>
<br>
  <img src="/assets/DeepFaceRecognitionSurvey/Fig_26.png">
<p align="center">
</p>
<br>
<br>

**Rao et al. [187] aggregated raw video frames directly by combining the idea of metric learning and adversarial learning.**  
Raoet al.은 metric learning과 adversarial learning의 아이디어를 결합하여 직접 원시 비디오 프레임을 집계했습니다.

<br>

**For dealing with bad frames, Rao et al. [185] discarded the bad frames by treating this operation as a Markov decision process and trained the attention model through a deep reinforcement learning framework.**  
불량 프레임을 처리하기 위해 Rao et al.은 이 작업을 Markov decision process로 처리하여 불량 프레임을 폐기하고 deep reinforcement learning framework를 통해 주의 Model을 Train했습니다.

<br>

**Ding et al. [257] artificially blurred clear images for training to learn blurrobust face representations.**  
딩 외.은 흐릿한 얼굴 표현을 학습하기 위한 Train을 위한 인위적으로 선명한 이미지를 흐려지게 만들었습니다.

<br>

**Parchami et al. [258] used a CNN to reconstruct a lower-quality video into a high-quality face.**  
Parchamiet al. [258]은 CNN을 사용하여 저품질 비디오를 고품질 얼굴로 재구성했습니다.

<br>
<br>
<br>

## D. Face Recognition in Industry

<br>
<br>
<br>

### 1) 3D Face Recognition

<br>
<br>
<br>

**3D FR has inherent advantages over 2D methods, but 3D deep FR is not well developed due to the lack of large annotated 3D data.**  
3D FR은 2D 방법에 비해 고유한 장점이 있지만 3D deep FR은 주석이 달린 큰 3D data가 부족하여 잘 개발되지 않았습니다.

<br>

**To enlarge 3D training datasets, most works use the methods of “one-to-many augmentation” to synthesize 3D faces.**  
3D Train Dataset를 확대하기 위해 대부분의 작업은 3D 얼굴을 합성하기 위해 "one-to-many augmentation" 방법을 사용합니다.

<br>

**However, the effective methods for extracting deep features of 3D faces remain to be explored.**  
그러나 3D 얼굴의 깊은 특징을 추출하는 효과적인 방법은 아직 연구되지 않았습니다.

<br>

**Kim et al. [204] fine-tuned a 2D CNN with a small amount of 3D scans for 3D FR. average prediction log-loss.**  
Kim et al. [204]은 3D FR에 대한 소량의 3D 스캔으로 2D CNN을 미세 조정했습니다. 

<br>

**Zulqarnain et al. [259] used a three-channel (corresponding to depth, azimuth and elevation angles of the normal vector) image as input and minimized the**  
Zulqarnain et al. [259]는 3채널(법선 벡터의 깊이, 방위각 및 앙각에 해당) 영상을 입력으로 사용하고 평균 예측 로그 손실을 최소화했습니다.

<br>

**Zhang et al. [260] first selected 30 feature points from the Candide-3 face model to characterize faces, then conducted the unsupervised pretraining of face depth data, and finally performed the supervised fine-tuning.**  
장 외. [260]은 먼저 Candide-3 얼굴 Model에서 30개의 특징점을 선택하여 얼굴을 특성화한 다음 얼굴 깊이 데이터의 무감독 사전 Train을 수행하고 마지막으로 지도 미세 조정을 수행했습니다.      

<br>
<br>
<br>

### 2) Partial Face Recognition

<br>
<br>
<br>

**Partial FR, in which only arbitrary-size face patches are presented, has become an emerging problem with increasing requirements of identification from CCTV cameras and embedded vision systems in mobile devices, robots and smart home facilities.**  
임의의 크기의 안면 패치만 제시되는 부분 FR은 모바일 장치, 로봇 및 스마트 홈 시설에서 CCTV 카메라 및 임베디드 비전 시스템의 식별 요구 사항이 증가함에 따라 새로운 문제가 되었습니다.

<br>

**He et al. [261] divided the aligned face image into several multiscale patches, and the dissimilarity between two partial face images is calculated as the weighted L2 distance between corresponding patches.**  
He et al. [261]은 정렬된 얼굴 영상을 여러 개의 멀티스케일 패치로 나누고 두 부분 얼굴 영상 간의 상이도를 해당 패치 간의 가중치 L2 거리로 계산한다.

<br>

**Dynamic feature matching (DFM) [262] utilized a sliding window of the same size as the probe feature maps to decompose the gallery feature maps into several gallery sub-feature maps, and the similarity-guided constraint imposed on sparse representation classification (SRC) provides an alignment-free matching.**  
DFM(Dynamic Feature Matching)[262]은 Probe feature maps과 동일한 크기의 슬라이딩 윈도우를 활용하여 Gallery 특징 맵을 여러 Gallery sub-feature maps으로 분해하고 SRC(Sparse Representation Classification)에 부과된 유사성 기반 제약 조건을 사용했습니다. 얼라인먼트 없는 매칭을 제공합니다.  

<br>
<br>
<br>

### 3) Face Recognition for Mobile Devices

<br>
<br>
<br>

**With the emergence of mobile phones, tablets and augmented reality, FR has been applied in mobile devices.**  
휴대폰, 태블릿, 증강현실의 등장으로 FR은 모바일 기기에 적용되고 있습니다.

<br>

**Due to computational limitations, the recognition tasks in these devices need to be carried out in a light but timely fashion.**  
계산상의 한계로 인해 이러한 장치의 인식 작업은 가볍지만 시기 적절한 방식으로 수행되어야 합니다.

<br>

**MobiFace [87] required efficient memory and low cost operators by adopting fast downsampling and bottleneck residual block, and achieves 99.7% on LFW database and 91.3% on Megaface database.**  
MobiFace[87]는 fast downsampling과 bottleneck residual block을 채택하여 효율적인 메모리와 저비용 연산자가 필요했으며 LFW Database에서 99.7%, Megaface Database에서 91.3%를 달성했습니다.

<br>

**Tadmor et al. [263] proposed a multibatch method that first generates signatures for a minibatch of k face images and then constructs an unbiased estimate of the full gradient by relying on all k 2 − k pairs from the minibatch.**  
Tadmoret al. [263]은 먼저 k 얼굴 이미지의 미니배치에 대한 서명을 생성한 다음 미니배치의 모든 k 2 - k 쌍에 의존하여 전체 기울기의 편향되지 않은 추정치를 구성하는 다중배치 방법을 제안했습니다.

<br>

**As mentioned in Section III-B.1, light-weight deep networks perform excellently in the fundamental tasks of image classification and deserve further attention in FR tasks.**  
섹션 III-B.1에서 언급했듯이 light-weight deep networks는 이미지 분류의 기본 작업에서 탁월하게 수행되며 FR 작업에서 더 많은 관심을 받을 만합니다.

<br>

**Moreover, some well-known compressed networks such as Pruning, BinaryNets, Mimic Networks [271], [272], also have potential to be introduced into FR.**  
또한 Pruning, BinaryNets, Mimic Networks[271], [272]와 같은 일부 잘 알려진 압축 Network도 FR에 도입될 가능성이 있습니다.

<br>
<br>
<br>

### 4) Face Anti-attack

<br>
<br>
<br>

**With the success of FR techniques, various types of attacks, such as face spoofing and adversarial perturbations, are becoming large threats.**  
FR 기술의 성공으로 face spoofing , adversarial perturbations 등 다양한 유형의 공격이 큰 위협이 되고 있습니다.

<br>

**Face spoofing involves presenting a fake face to the biometric sensor using a printed photograph, worn mask, or even an image displayed on another electronic device.**  
Face spoofing은 인쇄된 사진, 착용한 마스크 또는 다른 전자 장치에 표시된 이미지를 사용하여 생체 인식 센서에 가짜 얼굴을 제시하는 것입니다.

<br>

**In order to defense this type of attack, several methods are proposed.**  
이러한 유형의 공격을 방어하기 위해 몇 가지 방법이 제안됩니다.

<br>

**Atoum et al. [211] proposed a novel two-stream CNN in which the local features discriminate the spoof patches that are independent of the spatial face areas, and holistic depth maps ensure that the input live sample has a face-like depth.**  
Atoum et al. [211]은 local features가 공간 얼굴 영역과 독립적인 스푸핑 패치를 구별하고 전체적인 depth maps이 입력 라이브 샘플이 얼굴과 같은 깊이를 갖도록 보장하는 새로운 two-stream CNN을 제안했습니다.

<br>

**Yang et al. [273] trained a CNN using both a single frame and multiple frames with five scales as input, and using the live/spoof label as the output.**  
Yang et al. [273]는 입력으로 5개의 스케일이 있는 단일 프레임과 다중 프레임을 모두 사용하고 출력으로 live/spoof label을 사용하여 CNN을 교육했습니다.

<br>

**Taken the sequence of video frames as input, Xu et al. [274] applied LSTM units on top of CNN to obtain end-to-end features to recognize spoofing faces which leveraged the local and dense property from convolution operation and learned the temporal structure using LSTM units.**  
Xu et al.은 일련의 비디오 프레임을 입력으로 사용했으며, [274] CNN 위에 LSTM 유닛을 적용하여 convolutional 연산에서 local and dense property을 활용하고 LSTM 유닛을 사용하여 시간 구조를 학습한 스푸핑 얼굴을 인식하기 위한 종단간 특징을 얻었습니다.

<br>

**Li et al. [275] and Patel et al. [276] finetuned their networks from a pretrained model by training sets of real and fake images.**  
Li et al. [275] 및 Patel et al. [276]은 실제 이미지와 가짜 이미지의 Train 세트를 통해 사전 Train된 Model에서 Network를 미세 조정했습니다.

<br>

**Jourabloo et al. [277] proposed to inversely decompose a spoof face into the live face and the spoof noise pattern.**  
Jourablooet al. [277] spoof face를 live face와 spoof noise pattern으로 역분해하는 것을 제안했습니다.

<br>

**Adversarial perturbation is the other type of attack which can be defined as the addition of a minimal vector r such that with addition of this vector into the input image x, i.e. (x + r), the deep learning models misclassifies the input while people will not.**  
적대적 섭동(Adversarial perturbation)은 최소 벡터 r을 입력 이미지 x에 추가하는 것으로 정의할 수 있는 다른 유형의 공격입니다. 즉, (x + r), Deep Learning Model은 입력을 잘못 분류하지만 사람들은 아니다.

<br>

**Recently, more and more work has begun to focus on solving this perturbation of FR.**  
최근 점점 더 많은 작업이 FR의 이러한 섭동(perturbation,(특성·행동·움직임의) 작은 변화)을 해결하는 데 집중하기 시작했습니다.

<br>

**Goswami et al. [280] proposed to detect adversarial samples by characterizing abnormal filter response behavior in the hidden layers and increase the network’s robustness by removing the most problematic filters.**  
Goswamiet al.은 숨겨진 레이어에서 비정상적인 필터 응답 동작을 특성화하여 적대적 샘플을 탐지하고 가장 문제가 되는 필터를 제거하여 Network의 견고성을 높일 것을 제안했습니다.

<br>

**Goel et al. [281] provided an open source implementation of adversarial detection and mitigation algorithms.**  
Goel et al.은 적대적 탐지 및 완화 알고리즘의 오픈 소스 구현을 제공했습니다.

<br>

**Despite of progresses of anti-attack algorithms, attack methods are updated as well and remind us the need to further increase security and robustness in FR systems,**  
공격 방지 알고리즘의 발전에도 불구하고 공격 방법도 업데이트되어 FR 시스템의 보안 및 견고성을 더욱 향상시킬 필요성을 상기시킵니다. 

<br>

**for example, Mai et al. [282] proposed a neighborly deconvolutional neural network (NbNet) to reconstruct a fake face using the stolen deep templates.**  
예를 들어 Mai et al. [282]는 훔친 Deep template을 사용하여 가짜 얼굴을 재구성하기 위해 NbNet(neighborly deconvolutional neural network)을 제안했습니다.

<br>
<br>
<br>

### 5) Debiasing face recognition

<br>
<br>
<br>

**As described in Section V-A, existing datasets are highly biased in terms of the distribution of demographic cohorts, which may dramatically impact the fairness of deep models.**  
섹션 V-A에서 설명한 바와 같이, 기존 Dataset는 인구통계학적 코호트 분포 측면에서 매우 편향되어 심층 Model의 공정성에 큰 영향을 미칠 수 있습니다.

<br>

**To address this issue, there are some works that seek to introduce fairness into face recognition and mitigate demographic bias, e,g. unbalancedtraining [283], attribute removal [284], [285], [286] and domain adaptation.**  
이 문제를 해결하기 위해 얼굴 인식에 공정성을 도입하고 인구통계학적 편향을 완화하려는 일부 작업이 있습니다. ( e.g. unbalancedtraining[283], 속성 제거[284], [285], [286] 및 도메인 적응 )

<br>
<br>

**1) Unbalanced-training methods mitigate the bias via model regularization, taking into consideration of the fairness goal in the overall model objective function.**  
1) Unbalanced-training methods은 전체 Model 목적 함수의 공정성 목표를 고려하여 Model regularization를 통해 편향을 완화합니다.

<br>

**For example, RL-RBN [283] formulated the process of finding the optimal margins for non-Caucasians as a Markov decision process and employed deep Q-learning to learn policies based on large margin loss.**  
예를 들어, RL-RBN[283]은 백인이 아닌 사람들을 위한 최적의 마진을 찾는 프로세스를 Markov decision process로 공식화하고 큰 마진 손실을 기반으로 정책을 학습하기 위해 deep Q-learning을 사용했습니다.

<br>
<br>

**2) Attribute removal methods confound or remove demographic information of faces to learn attribute-invariant representations.**  
2) 속성 제거 방법은 속성 불변 표현을 학습하기 위해 얼굴의 인구 통계학적 정보를 혼동하거나 제거합니다.

<br>

**For example, Alvi et al. [284] applied a confusion loss to make a classifier fail to distinguish attributes of examples so that multiple spurious variations are removed from the feature representation.**  
예를 들어 Alvi et al.은 classifier가 attributes of examples을 구별하지 못하도록 혼동 손실을 적용하여 여러 개의 스퓨리어스 변형이 feature representation에서 제거되었습니다.

<br>

**SensitiveNets [288] proposed to introduce sensitive information into triplet loss.**  
SensitiveNets [288]는 triplet loss에 민감한 정보를 도입할 것을 제안했습니다.

<br>

**They minimized the sensitive information, while maintaining distances between positive and negative embeddings.**  
포지티브 임베딩과 네거티브 임베딩 사이의 거리를 유지하면서 민감한 정보를 최소화했습니다.

<br>
<br>

**3) Domain adaptation methods propose to investigate data bias problem from a domain adaptation point of view and attempt to design domain-invariant feature representations to mitigate bias across domains.**  
3) 도메인 적응 방법은 도메인 적응 관점에서 데이터 바이어스 문제를 조사하고 도메인 간 바이어스를 완화하기 위해 도메인 불변 기능 표현을 설계하려고 제안합니다.

<br>

**IMAN [173] simultaneously aligned global distribution to decrease race gap at domain-level, and learned the discriminative target representations at cluster level.**  
IMAN [173]은 도메인 수준에서 인종 격차를 줄이기 위해 글로벌 분포를 동시에 정렬하고 클러스터 수준에서 차별적인 대상 표현을 학습했습니다.

<br>

**Kan [147] directly converted the Caucasian data to non-Caucasian domain in the image space with the help of sparse reconstruction coefficients learnt in the common subspace.**  
Kan [147]은 common subspace에서 학습된 희소 재구성 계수의 도움을 받아 이미지 공간에서 백인 데이터를 비백인 도메인으로 직접 변환했습니다.   

<br>
<br>
<br>

## VII. TECHNICAL CHALLENGES

<br>
<br>
<br>

**In this paper, we provide a comprehensive survey of deep FR from both data and algorithm aspects.**  
이 백서에서는 데이터 및 알고리즘 측면 모두에서 심층 FR에 대한 포괄적인 조사를 제공합니다.

<br>

**For algorithms, mainstream and special network architectures are presented.**  
알고리즘의 경우 mainstream 및 special Network Architecture가 제시됩니다.

<br>

**Meanwhile, we categorize loss functions into Euclideandistance-based loss, angular/cosine-margin-based loss and variable softmax loss.**  
한편 Loss Function는 Euclidean distance-based loss, angular/cosine-margin-based loss 및 variable Softmax loss로 분류됩니다.

<br>

**For data, we summarize some commonly used datasets.**  
데이터의 경우 일반적으로 사용되는 일부 Dataset를 요약합니다.

<br>

**Moreover, the methods of face processing are introduced and categorized as “one-to-many augmentation” and “many-to-one normalization”.**  
또한 얼굴 처리 방법을 도입하여 “one-to-many augmentation” and “many-to-one normalization”로 분류합니다.

<br>

**Finally, the special scenes of deep FR, including video FR, 3D FR and cross-age FR, are briefly introduced.**  
마지막으로 영상 FR, 3D FR, 크로스 에이지 FR 등 Deep FR의 특별한 장면을 간략하게 소개한다.

<br>

**Taking advantage of big annotated data and revolutionary deep learning techniques, deep FR has dramatically improved the SOTA performance and fostered successful real-world applications.**  
주석이 달린 큰 데이터와 혁신적인 Deep Learning 기술을 활용하는 Deep FR은 SOTA 성능을 극적으로 개선하고 성공적인 실제 응용 프로그램을 육성했습니다.

<br>

**With the practical and commercial use of this technology, many ideal assumptions of academic research were broken, and more and more real-world issues are emerging.**  
이 기술의 실용적이고 상업적인 사용으로 학술 연구의 많은 이상적인 가정이 깨졌고 점점 더 많은 실제 문제가 등장하고 있습니다.

<br>

**To the best our knowledge, major technical challenges include the following aspects.**  
우리가 아는 한, 주요 기술 과제에는 다음과 같은 측면이 포함됩니다.

<br>
<br>
<br>

### • Security issues

<br>
<br>
<br>

**Presentation attack [289], adversarial attack [280], [281], [290], template attack [291] and digital manipulation attack [292], [293] are developing to threaten the security of deep face recognition systems.**  
보안 문제들. Presentation attack [289], adversarial attack [280], [281], [290], template attack [291] 및 digital manipulation attack [292], [293]은 심층 얼굴 인식 시스템의 보안을 위협하기 위해 개발되고 있습니다.

<br>

**1) Presentation attack with 3D silicone mask, which exhibits skin-like appearance and facial motion, challenges current anti-sproofing methods [294].**  
1) 피부와 같은 모양과 안면 움직임을 나타내는 3D 실리콘 마스크를 사용한 프리젠테이션 공격은 현재의 anti-sproofing methods에 도전합니다[294].

<br>

**2) Although adversarial perturbation detection and mitigation methods are recently proposed [280][281], the root cause of adversarial vulnerability is unclear and thus new types of adversarial attacks are still upgraded continuously [295], [296].**  
2) 최근 adversarial perturbation detection 및 완화 방법이 제안되었지만[280][281], 적대적 취약성의 근본 원인이 불분명하여 새로운 유형의 적대적 공격이 계속 업그레이드되고 있다[295,296].

<br>

**3) The stolen deep feature template can be used to recover its facial appearance, and how to generate cancelable template without loss of accuracy is another important issue.**  
3) stolen deep feature template을 사용하여 얼굴 모습을 복구할 수 있으며 정확도 손실 없이 취소 가능한 template을 생성하는 방법은 또 다른 중요한 문제입니다.

<br>

**4) Digital manipulation attack, made feasible by GANs, can generate entirely or partially modified photorealistic faces by expression swap, identity swap, attribute manipulation and entire face synthesis, which remains a main challenge for the security of deep FR.**  
4) GAN에 의해 실현 가능한 디지털 조작 공격은 표정 교환, 신원 교환, 속성 조작 및 전체 얼굴 합성을 통해 전체 또는 부분적으로 수정된 사실적인 얼굴을 생성할 수 있으며 이는 deep FR의 보안에 대한 주요 과제로 남아 있습니다.

<br>
<br>
<br>

### Privacy-preserving face recognition

<br>
<br>
<br>

**With the leakage of biological data, privacy concerns are raising nowadays.**  
최근 생체 데이터 유출로 개인정보 보호에 대한 우려가 높아지고 있습니다.

<br>

**Facial images can predict not only demographic information such as gender, age, or race, but even the genetic information [297].**  
얼굴 이미지는 성별, 연령, 인종과 같은 인구학적 정보뿐만 아니라 유전 정보까지 예측할 수 있다[297].

<br>

**Recently, the pioneer works such as Semi-Adversarial Networks [298], [299], [285] have explored to generate a recognizable biometric templates that can hidden some of the private information presented in the facial images.**  
최근에 Semi-Adversarial Networks[298], [299], [285]와 같은 선구적인 작업은 얼굴 이미지에 표시된 일부 개인 정보를 숨길 수 있는 인식 가능한 생체 인식 template을 생성하는 방법을 모색했습니다.

<br>

**Further research on the principles of visual cryptography, signal mixing and image perturbation to protect users’ privacy on stored face templates are essential for addressing public concern on privacy.**  
저장된 얼굴 template에서 사용자의 개인 정보를 보호하기 위한 시각적 암호화, 신호 혼합 및 이미지 교란의 원칙에 대한 추가 연구는 개인 정보 보호에 대한 대중의 우려를 해결하는 데 필수적입니다.    

<br>
<br>
<br>

### Understanding deep face recognition

<br>
<br>
<br>

**Deep face recognition systems are now believed to surpass human performance in most scenarios [300].**  
Deep face recognition systems은 이제 대부분의 시나리오에서 인간의 성능을 능가하는 것으로 여겨집니다[300].

<br>

**There are also some interesting attempts to apply deep models to assist human operators for face verification [183][300].**  
얼굴 확인을 위해 인간 조작자를 돕기 위해 deep Model을 적용하려는 몇 가지 흥미로운 시도도 있습니다[183][300].

<br>

**Despite this progress, many fundamental questions are still open, such as what is the “identity capacity” of a deep representation [301]?**  
이러한 진전에도 불구하고 deep representation[301]의 "identity capacity"은 무엇인가와 같은 많은 근본적인 질문이 여전히 열려 있습니다.

<br>

**Why deep neural networks, rather than humans, are easily fooled by adversarial samples?**  
인간이 아닌 deep neural network이 적대적 샘플에 쉽게 속는 이유는 무엇입니까?

<br>

**While bigger and bigger training dataset by itself cannot solve this problem, deeper understanding on these questions may help us to build robust applications in real world.**  
더 크고 더 큰 Train Dataset만으로는 이 문제를 해결할 수 없지만 이러한 질문에 대한 더 깊은 이해는 실제 세계에서 강력한 응용 프로그램을 구축하는 데 도움이 될 수 있습니다.

<br>

**Recently, a new benchmark called TALFW has been proposed to explore this issue [93].**  
최근에 이 문제를 탐구하기 위해 TALFW라는 새로운 benchmark가 제안되었습니다[93].

<br>
<br>
<br>

### Remaining challenges defined by non-saturated benchmark datasets

<br>
<br>
<br>

**Three current major datasets, namely, MegaFace [44], [164] , MS-Celeb-1M [45] and IJB-A/B/C [41], [42], [43], are corresponding to largescale FR with a very large number of candidates, low/oneshot FR and large pose-variance FR which will be the focus of research in the future.**  
MegaFace [44], [164], MS-Celeb-1M [45] 및 IJB-A/B/C [41], [42], [43]의 세 가지 현재 주요 Dataset는 다음과 같은 매우 많은 수의 후보, 낮은/원샷 FR 및 향후 연구의 초점이 될 큰 포즈 분산 FR등의 대규모 FR에 해당합니다. 

<br>

**Although the SOTA algorithms can be over 99.9 percent accurate on LFW [23] and Megaface [44], [164] databases, fundamental challenges such as matching faces cross ages [181], poses [188], sensors, or styles still remain.**  
SOTA 알고리즘은 LFW [23] 및 Megaface [44], [164] Database에서 99.9% 이상 정확할 수 있지만, 연령에 따른 얼굴 일치[181], 포즈[188], 센서 또는 스타일과 같은 근본적인 문제는 여전히 남아 있습니다.

<br>

**For both datasets and algorithms, it is necessary to measure and address the racial/gender/age biases of deep FR in future research.**  
Dataset와 알고리즘 모두에 대해 향후 연구에서 deep FR의 인종/성별/연령 편향을 측정하고 해결하는 것이 필요합니다.

<br>
<br>
<br>

### Ubiquitous face recognition across applications and scenes

<br>
<br>
<br>

**Deep face recognition has been successfully applied on many user-cooperated applications, but the ubiquitous recognition applications in everywhere are still an ambitious goal.**  
Deep face recognition은 많은 사용자 협력 응용 프로그램에 성공적으로 적용되었지만 모든 곳에서 유비쿼터스 인식 응용 프로그램은 여전히 야심 찬 목표입니다.

<br>

**In practice, it is difficult to collect and label sufficient samples for innumerable scenes in real world.**  
실제로 현실 세계의 수많은 장면에 대해 충분한 샘플을 수집하고 레이블을 지정하는 것은 어렵습니다.

<br>

**One promising solution is to first learn a general model and then transfer it to an application-specific scene.**  
유망한 솔루션 중 하나는 먼저 일반 Model을 학습한 다음 이를 애플리케이션별 장면으로 전송하는 것입니다.

<br>

**While deep domain adaptation [145] has recently been applied to reduce the algorithm bias on different scenes [148], different races [173], general solution to transfer face recognition is largely open.**  
최근에 서로 다른 장면[148], 서로 다른 인종[173]에서 알고리즘 편향을 줄이기 위해 심층 도메인 적응[145]이 적용되었지만 얼굴 인식을 전송하는 일반적인 솔루션은 크게 열려 있습니다.

<br>
<br>
<br>

### Pursuit of extreme accuracy and efficiency

<br>
<br>
<br>

**Many killer-applications, such as watch-list surveillance or financial identity verification, require high matching accuracy at very low alarm rate, e.g. 10−9**  
감시 목록 감시 또는 금융 신원 확인과 같은 많은 킬러 애플리케이션은 매우 낮은 경보율(예: 10-9)이 요구됩니다.

<br>

**It is still a big challenge even with deep learning on massive training data.**  
방대한 Train 데이터에 대한 Deep Learning으로도 여전히 큰 과제입니다.

<br>

**Meanwhile, deploying deep face recognition on mobile devices pursues the minimum size of feature representation and compressed deep network.**  
한편, 모바일 디바이스에 심층 얼굴 인식을 배치하는 것은 minimum size of feature representation와 compressed deep network를 추구합니다.

<br>

**It is of great significance for both industry and academic to explore this extreme face-recognition performance beyond human imagination.**  
인간의 상상을 뛰어넘는 이 극한의 얼굴 인식 성능을 탐구하는 것은 산업계와 학계 모두에게 큰 의미가 있습니다.

<br>

**It is also exciting to constantly push the performance limits of the algorithm after it has already surpassed human.**  
이미 인간을 능가한 알고리즘의 성능 한계를 지속적으로 밀어붙이는 것도 흥미진진합니다.

<br>
<br>
<br>

### Fusion issues

<br>
<br>
<br>

**Face recognition by itself is far from sufficient to solve all biometric and forensic tasks, such as distinguishing identical twins and matching faces before and after surgery [302].**  
안면 인식만으로는 일란성 쌍둥이를 구별하고 수술 전후의 얼굴을 일치시키는 것과 같은 모든 생체 인식 및 법의학 작업을 해결하기에 충분하지 않습니다[302].

<br>

**A reliable solution is to consolidate multiple sources of biometric evidence [303].**  
신뢰할 수 있는 솔루션은 생체 증거의 여러 소스를 통합하는 것입니다[303].

<br>

**These sources of information may correspond to different biometric traits (e.g., face + hand [304]), sensors (e.g., 2D + 3D face cameras), feature extraction and matching techniques, or instances (e.g., a face sequence of various poses).**  
이러한 정보 소스는 서로 다른 생체 특성(예: 얼굴 + 손[304]), 센서(예: 2D + 3D 얼굴 카메라), 특징 추출 및 일치 기술 또는 인스턴스(예: 다양한 포즈의 얼굴 시퀀스)에 해당할 수 있습니다.

<br>

**It is beneficial for face biometric and forensic applications to perform information fusion at the data level, feature level, score level, rank level, and decision level [305].**  
데이터 수준, 특징 수준, 점수 수준, 순위 수준, 결정 수준에서 정보 융합을 수행하는 것이 얼굴 생체 인식 및 법의학 응용에 유리합니다[305].

