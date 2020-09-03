---
title: "Ubuntu에 GPU & Tensorflow 사용환경 설정"
date: 2020-09-03 08:26:28 -0400
categories: DeepLearning
---
# Ubuntu에 GPU & Tensorflow 사용환경 설정
<br>
<br>
<br>

이번 Posting에서는 GPU를 이용한 Deep Learning 환경 구축할 때 가장 일반적이며 많이 사용되는 Configuration인, Linux & nVidia GPU를 사용한 환경 설정을 하는 방법을 알아보도록 하겠습니다.

제가 사용할 Linux와 GPU를 아래와 같습니다.
* GPU : **GTX 1050 Ti 6GB**
* Ubuntu : **18.04**

<br>
<br>
<br>
<br>
<br>
<br>

## 0. Prepare Ubuntu Installation

먼저 설치할 Ubuntu 배포한의 Image File을 Download하도록 하겠습니다.

[Ubuntu Image File Download](https://releases.ubuntu.com/?_ga=2.140264563.1504976568.1598255557-1939476017.1598255557)


저는 18.04 Version을 사용하도록 하겠습니다.

설치를 편리하게 하기 위해서 방금 Download한 Ubuntu 설치 ISO Image를 USB Drive에 굽도록 하겠습니다.

적절한 크기의 USB를 하나 준비하셔서 ISO Image를 굽는 Tool로 굽도록 하겠습니다.

저는 Rufus라는 Tool을 이용해 ISO Image를 구웠습니다.

아래 Link에서 Download할 수 있으니, 참고하시기 바랍니다.

[Rufus Download](https://rufus.ie/)

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_00.png">
</p>
<br>
<br>

<br>
<br>
<br>
Image를 구울 때, Rufus Option은 아래와 같이 하면 무난하게 구울 수 있을 겁니다.   

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_01.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
<br>
<br>

## 1. Ubuntu 설치 & Update

Ubuntu 설치 후, Update & Upgrade를 실시합니다.

**apt update**

**apt upgrade**

<br>
<br>
<br>
<br>
<br>
<br>

## 2. GPU Driver 설치

Ubuntu를 설치하면 기본 GPU Driver가 설치됩니다. 이 Driver는 최적화가 잘 되어 있지 않기 때문에 GPU 성능을 최대로 끌어내지 못합니다.

그래서, 최신 nVidia GPU Driver를 다시 설치하도록 하겠습니다.

향상된 성능의 nVidia GPU Driver를 설치할 수 있는 방법은 몇 가지 있습니다.


* **Nvidia PPA**
  
  성능이 뛰어나고, PPA에 포함된 드라이버를 사용하여 대부분의 nVidia GPU에서 즉시 사용할 수 있습니다.

&nbsp;

* **Ubuntu Default Recommended Driver(추천)**

  Ubuntu에서 제공해주는 Driver. 현재 사용중인 GPU 종류에 따라서 다양한 Driver를 제공해줍니다.

&nbsp;

* **Nouveau**
  
  nVidia Driver의 Open Source 구현입니다. 훌륭하긴 합니다만 최신 GPU Driver일수록 성능 면에서는 뒤떨어집니다. 

&nbsp;

* **Official Nvidia Site**

  이것은 공식 드라이버 (PPA의 드라이버와 동일)이지만 차이점은 자동으로 업그레이드되지 않고 업데이트, 제거 및 설치할 때 몇 가지 문제가 있다는 것입니다 (매우 드물지만 발생 함).
  
<br>  
<br>
어떤 형태의 GPU Driver를 설치하는 것이 가장 좋을지 살펴본 후, 저는 Ubuntu Default Recommended Driver를 사용하기로 했습니다.

Official Nvidia Driver([Link](https://www.nvidia.com/Download/index.aspx?lang=en-us))는 X Server에 알려진 문제([Link](https://askubuntu.com/questions/149206/how-to-install-nvidia-run))가 있었으며 작동하려면 더 많은 노력이 필요했습니다. 

기본 Ubuntu 드라이버는 성능이 부족합니다. 

그리고 Nouveau 드라이버는 오픈 소스라는 장점이 있지만, 성능은 떨어집니다. 

Ubuntu Default Recommended Driver를 추가하는 것은 최신 Nvidia 공식 드라이버가 있고 Ubuntu에서 테스트 되었으므로 원활한 설치도 가능하므로 가장 좋은 선택이라고 생각합니다.

<br>
<br>

이제 Ubuntu Default Recommended Driver Repository를 추가합니다. 

아래 명령은 Ubuntu의 패키징 시스템에 PPA 저장소를 추가합니다. 

이렇게 하면 apt를 이용하여 관리가 가능함을 의미합니다. 그렇게 되면 훨씬 더 편리하게 관리할 수 있겠죠?

&nbsp;

**add-apt-repository ppa:graphics-drivers/ppa**

**apt update**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_02.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
이제 어떤 Version의 Driver를 설치할 지 결정해야 합니다. 

앞의 Message들을 보면, 430이 괜찮다고 추천해 주네요.

그럼 저도 430 설치 하겠습니다.

**apt install nvidia-driver-430**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_03.png">
</p>
<br>
<br>

### **설치 완료 후 재부팅을 한 번 해줍니다.**

<br>
<br>
재부팅 후에 Driver가 제대로 설치되었는지 확인 한 번 해 보도록 하겠습니다.


**nvidia-smi**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_04.png">
</p>
<br>
<br>

어? 난 분명히 430을 설치했는데, 왜 440이 나올까요..? 


음… 일단 진행하도록 하죠

<br>
<br>
이 부분은 선택적인 사항인데, 우리가 설치한 Driver를 고정하는 작업입니다.


Linux에서 GPU가 잘 동작하도록 하는 작업은 다양한 HW & SW가 잘 조화가 되어야 하며, 각각 서로간의 호환성도 좋아야 합니다.


우리가 선택한 Driver가 다른 어떤 동작으로 인해서 변경되는 일이 없도록 고정하는 작업입니다. 아래 명령어로 고정할 수 있습니다.

&nbsp;

**apt-mark hold nvidia-driver-430**


<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_05.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 3. CUDA Toolkit 설치   

이제 CUDA Toolkit을 설치할 순서입니다.

어떤 CUDA Toolkit Version을 설치할 것이냐는 자신이 방금 설치한 GPU Driver와 연관이 있는데, 아래 Link에 가 보시면 Version 선택하시는데 
도움이 될 정보가 있습니다.

[CUDA Toolkit Version Check](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

저는 430을 설치하였으니, 10.0 혹은 10.1을 설치하면 되는데, 10.0을 설치하도록 하겠습니다.

아래 Link에 10.0을 Download할 수 있는 곳이니 가서 Download하도록 하겠습니다.

[CUDA Toolkit 10.0 Download](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)

<br>
<br>
자신의 환경에 맞는 Option을 선택하면 Download Link가 만들어 집니다.


편하네요. Installer와 Patch를 받도록 합시다.

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_06.png">
</p>
<br>
<br>

<br>
<br>
다음 command로 CUDA Toolkit을 설치합니다.

&nbsp;

**sh cuda_10.0.130_410.48_linux.run**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_07.png">
</p>
<br>
<br>

<br>
<br>
설치하실 때 주의할 점은 **Driver 설치는 No**를 선택하시기 바랍니다.

기존 Driver와 충돌이 발생하는지는 몰라도, 이후에 잘 진행이 되지 않더라구요.
<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_08.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
설치가 완료되면 bashrc에 Path를 추가합니다. 

Editor로 가장 아래쪽에 다음 항목을 추가해 줍니다.

**nano ~/.bashrc**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_09.png">
</p>
<br>
<br>

수정후에는 다시 Load합니다.

&nbsp;

**source ~/.bashrc**

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 4. cuDNN 설치

CUDA Toolkit 설치를 끝냈으면 다음에는 cuDNN을 설치해 보도록 하겠습니다.

아래 Link의 cuDNN Archive에 가서 앞서 설치한 CUDA Version과 맞는 cuDNN을 찾아서 설치하도록 하겠습니다.

&nbsp;

[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_10.png">
</p>
<br>
<br>

cuDNN을 Download한 위치에 가서 아래 명령으로 해당 cuDNN을 설치하면 됩니다.

**sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb**

**sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb** 

**sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_11.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
<br>
<br>

## 5. Anaconda 설치

이제 GPU 사용관련 Driver는 모두 설치가 끝났습니다. 

이제부터는 Virtual Env. 설정 관련 항목인데, 사실 Deep Learning 하다보면 가상환경 사용은 선택이 아닌 거의 필수입니다. 

다양한 Project들을 Test해 볼 필요가 있고, 개별 Project마다 사용하는 Package Version이 다를수도 있으니

이를 하나의 환경으로 다 관리한다는 것은 거의 불가능에 가깝다고 볼 수 있습니다.

&nbsp;

가상 환경 관리에 어떤 도구를 사용해도 문제는 없습니다.

저는 예전부터 사용하던 Anaconda를 설치해서 사용하도록 하겠습니다.

<br>
<br>

Anaconda 설치파일은 아래 Link에서 Download할 수 있습니다.

[Anaconda 설치 파일 Download](https://www.anaconda.com/products/individual)


<br>
<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_12.png">
</p>
<br>
<br>

아래 명령으로 설치하면 됩니다.

<br>
<br>

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_13.png">
</p>
<br>
<br>
설치 마지막에 yes해서 Path 등록 해줍시다.

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_14.png">
</p>
<br>
<br>

Path 등록을 했으니 아래 명령으로 다시 Load해야겠죠?

&nbsp;

**source ~/.bashrc**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_15.png">
</p>
<br>
<br>

<br>
<br>
<br>
<br>
<br>
<br>

## 6. Tensorflow GPU 설치

자~! 이제 GPU관련 Driver 설치도 했고, 
가상 환경 설정도 했으니 시험삼아 Tensorflow GPU Version을 설치해 보도록 할까요?

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_16.png">
</p>
<br>
<br>

어? 안되네… 1.4.0 Version이 없을리가 없는데…

구글링 해 본 결과, Anaconda와 같이 설치된 Python 버전이 3.8이어서 안되는 것이었습니다.  

<br>
<br>

**Python Version에 따라 설치 가능한 Tensorflow Version이 달라진다고 하네요.**

현재 선택된 가상환경(base)의 Python Version을 변경해 보도록 하겠습니다.

아래 Command로 Python Ver.을 3.6.2로 변경합니다

**conda install python==3.6.2**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_17.png">
</p>
<br>
<br>

자, 그럼 이제 다시 한 번 Tensorflow GPU Version을 설치해보록 하겠습니다.

**pip3 install tensorflow-gpu==1.14.0**

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_18.png">
</p>
<br>
<br>

와우~! 잘 되네요. 다행입니다.

<br>
<br>
실제 Code에서 Tensorflow를 사용하면 GPU를 사용하는지 알아보죠.

<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_19.png">
</p>
<br>
<br>
<p align="center">
  <img src="/assets/Tensorflow_Ubuntu/pic_20.png">
</p>
<br>
<br>

설치된 GPU를 사용하는 모습이 보이네요.
