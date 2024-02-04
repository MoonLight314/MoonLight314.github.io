---
title: "Tensorflow GPU Docker Install on WSL2"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# Tensorflow GPU Docker Install on WSL2
<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/Title_01.png">

<br>

안녕하세요, MoonLight입니다.

이번 Post 주제는 Tensorflow GPU Version Docker Image를 WSL2에 설치한 후에, Visual Studio Code에서 이를 활용하여 Train을 하는 방법까지 진행해 보려고 합니다.

먼저, 제가 WSL2에 Tensorflow Docker Image로 설치하려는 이유는
<br>

**1) Windows 환경이 친숙하고 편합니다.**

**2) Tensorflow 2.10부터는 Native Windows에서 설치가 되지 않는다고 합니다.**

**3) Docker를 이용하면 복잡한 설치과정을 거치지 않아도 되기 때문에 매우 편합니다.**

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/00-1.png">

최종적으로 완성된 전체 구조는 아래와 같을 것입니다.

<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/00.png">

전체적으로 아래의 순서로 진행할 예정입니다.
<br>

### 1. WSL2 설치
### 2. Ubuntu 설치
### 3. nVidia GPU Driver for WSL 설치
### 4. Docker 설치
### 5. nVidia CUDA Toolkit Container 설치
### 6. Tensorflow GPU Container 설치
### 7. Visual Studo Code 준비
### 8. Visual Studo Code와 Tensorflow Container 연결
### 9. Image Commit
### 10. 최종 Test


자, 그럼 하나씩 진행해 보도록 하겠습니다.

<br>
<br>
<br>

## 1. WSL2 설치

WSL (Windows Subsystem for Linux)이란 Windows에서 직접 Linux 배포판을 실행할 수 있도록 하는 기능으로써,
이를 통해서 사용자는 Linux 명령어, 스크립트, 도구들을 Windows 환경에서 바로 사용할 수 있게 됩니다.
기본적으로 개발자가 Windows에서 Linux 기반의 개발 환경을 구축하고 사용할 수 있도록 해줍니다.

WSL은 가상 머신이나 듀얼 부팅 없이도 Linux 환경을 Windows 내에서 손쉽게 접근하고 사용할 수 있게 해줍니다.
예전 초창기 WSL에서는 Linux를 설치할 수는 있지만, GPU를 지원하지는 못해서 Deep Learning 개발 환경 구축에
크게 유용하지는 않았습니다.

하지만, WSL2가 나오고 GPU를 지원할 수 있게 되면서 이제는 상당히 쓸만해 졌습니다.

<br>
<br>

### 1.1. WSL2 설치 가능 조건

WSL2를 설치하기 위해서는 Windows 10 버전 2004 이상(빌드 19041 이상) 또는 Windows 11이 필요합니다.

https://learn.microsoft.com/ko-kr/windows/wsl/install



본인의 Windows 정보를 확인하셔서 설치 가능한지 확인해 보시기 바랍니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/01.png">

<br>
<br>

### 1.2. Windows Subsystem for Linux 기능 활성화

먼저 PowerShell을 관리자 권한으로 실행합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/02.png">

그리고, Windows Subsystem for Linux(WSL) 기능을 활성화 시켜주어야 합니다.

아래 명령어를 입력합니다.

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

<br>
<br>

### 1.3 Virtual Machine feature 활성화

추가로, Virtual Machine 기능도 활성화 시켜주어야 합니다. 아래 명령어를 입력합니다.

```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

<br>
<br>

### 1.4. 재부팅

반드시 재부팅해주셔야 변경사항이 적용됩니다. 꼭 재부팅 해주세요.

<br>
<br>

### 1.5. WSL 설치

재부팅 후에, 다시 PowerShell을 관리자 권한으로 실행하여, 다음 명령어를 입력합니다.

```bash
wsl --install
```

<br>
<br>

### 1.6 Linux Update Kernel Package 설치

아래 링크를 클릭하여 Linux Update Kernel Package를 설치합니다.

https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi

<br>
<br>

### 1.7 WSL2를 기본으로 설정

WSL에는 WSL 1 & WSL 2가 있는데, 우리는 GPU를 지원하는 WSL 2를 사용할 예정이기 때문에 기본버전을 WSL2로 설정하도록 하겠습니다.

​

powersehll 관리자 모드에서 아래 명령어 입력해 줍니다.

```bash
wsl --set-default-version 2
```

<br>
<br>
<br>

## 2. Ubuntu 설치

이제 WSL2를 설치했으니, WSL2에서 실행할 Linux를 설치하도록 하겠습니다. 가장 많이 사용하는 Linux 배포판인 Ubuntu 20.04 LTS를 설치해 보겠습니다.

<br>
<br>

### 2.1 Microsoft Store에서 Ubuntu 설치

Microsoft Store에서 Ubuntu 20.04.4 LTS를 설치합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/03.png">

<br>
<br>

#### Ubuntu 20.04.4 LTS
https://apps.microsoft.com/detail/9MTTCL66CPXJ?hl=en-us&gl=US
​
​

아래 Link는 Ubuntu 22.04.4 LTS입니다. 좀 더 최신 배포판을 설치하고자 하시는 분은 사용해 주세요.

https://apps.microsoft.com/detail/9PN20MSR04DW?hl=en-us&gl=US

<br>
<br>

### 2.2. 계정 생성

Ubuntu 설치가 완료되면 계정을 생성하고 나면, Ubuntu 설치는 완료됩니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/04.png">

<br>
<br>
<br>

## 3. nVidia GPU Driver for WSL 설치

이번에는 GPU Driver를 설치해 주어야 합니다. ​

아마 이미 Driver가 설치가 되어 있겠지만, WSL을 사용하기 위해서는 nVidia에서 제공하는 WSL용 Driver를 설치해 주어야 합니다. GPU Driver는 아래 Link에서 Download가 가능합니다.

https://developer.nvidia.com/cuda/wsl

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/05.png">


자신의 GPU와 OS 선택하고 다운로드하고 설치까지 진행해 주시면 됩니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/06.png">

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/07.png">

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/08.png">

<br>
<br>
<br>

## 4. Docker 설치

저는 WSL2에 설치된 Ubuntu안에 Docker를 설치하고, WSL2안에서 Tensorflow Container를 실행할 예정입니다.

그래서, 먼저 WSL2에 Ububtu에 Docker 설치를 진행하도록 하겠습니다.

<br>
<br>

### 4.1. Service 명령 권한 설정

먼저 visudo를 사용하여 docker command 앞에 일일이 'sudo'를 붙이지 않고 docker command를 실행할 수 있도록 하겠습니다. docker command를 사용할 일이 많은데, sudo를 항상 붙이려면 은근히 번거롭습니다.

<br>
<br>

#### 4.1.1. visudo 실행

WSL2 Ubuntu에서 아래 명령어 입력합니다.

```bash
sudo visudo
```
​

그 후에,service 권한 모든 사람이 사용할 수 있도록 설정합니다.

파일이 열리면, 가장 아래에 아래 코드 삽입해줍니다.

```bash
%sudo ALL=NOPASSWD: /usr/sbin/service
```

​그리고 나서, ctrl 와 o를 함께 누르고 enter key눌러 저장하고, ctrl 와 x를 함께 눌러서 빠져나옵니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/09.png">

<br>
<br>

#### 4.1.2. 다른 방법

혹시 위와 같은 방법을 사용해도 sudo 없이 docker command가 수행되지 않는다면, 아래 Link의 방법을 따라하면 해결됩니다.

저같은 경우에도 위와 같은 방법으로 되지 않아서, 아래 Link를 따라하면서 해결되었습니다.

https://docs.docker.com/engine/install/linux-postinstall/

<br>
<br>

### 4.2. Docker 설치

이제 Ubuntu에 Docker를 설치하도록 하겠습니다. 아래 Command들을 순서대로 입력하면 됩니다.

<br>
<br>

#### 4.2.1. 패키지 업데이트

```bash
sudo apt-get update
```

<br>
<br>

#### 4.2.2. https로 데이터를 받아서 설치하는 패키지들을 위한 준비

```bash
sudo apt-get install -y \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common
```

<br>
<br>

#### 4.2.3. Docker의 GPG key 추가

GPG란 툴이나 소스 등을 배포 할 때 원본이 수정되었나를 확인하는 SW인데, GPG key는 이를 확인하는데 사용하는 값입니다.

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

<br>
<br>

#### 4.2.4. Stable Docker Repository 설정

```bash
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
```

<br>
<br>

#### 4.2.5. Docker Engine 설치

이제, 마지막으로 Docker Engine을 설치합니다. 아래 Command들을 순서대로 입력해 줍니다.

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo service docker start
```
​

아래 Command로 Docker Engine이 잘 설치 되었는지 확인할 수 있습니다. 

```bash
sudo docker run hello-world
```
​

잘 설치되었다면 환영 Message(?)를 볼 수 있습니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/10.png">

<br>
<br>
<br>

## 5. nVidia CUDA Toolkit Container 설치

WSL에서 CUDA Toolkit을 사용할 수 있도록 해주는 특별한 Docker Container를 먼저 설치해 줍니다.

<br>
<br>

### 5.1. Package Repository 추가

Ubuntu에 아래 명령어 입력합니다.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
```

​<br>
<br>

### 5.2. nVidia-Docker GPGKey 추가

위에서 설명했던 GPGKey를 추가합니다.

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```

<br>
<br>

### 5.3.  Package Repository 추가

Ubuntu에 아래 명령어 입력

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

<br>
<br>​

### 5.4.  Package Update

```bash
sudo apt-get update
```

<br>
<br>

### 5.5.  CUDA Toolkit 설치

Ubuntu에 아래 명령어 입력

```bash
sudo apt-get install -y nvidia-container-toolkit
```

<br>
<br>​

### 5.6.  확인

성공적으로 설치 되었는지 확인하기 위하여 도커 시스템 재시작 후 CUDA Toolkit Container를 실행하여 GPU 정보를 출력해 봅니다.

```bash
sudo service docker restart
```
​

아래 명령어를 입력하면, 정상적으로 설치가 된 경우라면 GPU에 대한 정보가 출력될 것입니다.

```bash
sudo docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

​
<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/11.png">

<br>
<br>
<br>

## 6. Tensorflow GPU Container 설치

이제 Ubuntu에 Tensorflow GPU Version Docker Container를 설치하도록 하겠습니다.

​

일반적으로 Docker Container를 실행하는 순서는 다음과 같습니다.

### 원하는 Docker Image를 Download(docker pull)
### 받은 Image로 Conatiner 생성(docker create)
### 생성된 Conatiner 실행(docker start)
### 실행된 Conatiner 접속을 하여야 한다.(docker attach)

​

하지만, docker run 명령어는 위의 순서를 모두 한 번에 처리합니다.  nVidia Toolkit Container 설치할 때 한 번 사용했었죠.

​

다양한 종류의 Tensorflow Docker Image가 있고, 어떤 것들이 있는 확인해보고 선택할 수 있습니다.

현재 전체 List는 다음 Link에서 확인해 보실 수 있습니다.


https://hub.docker.com/r/tensorflow/tensorflow/tags/

저는 안정적인 Tensorflow GPU Docker중에 최신 Version으로 받아서 사용할 예정입니다. 2.15.0이 현 시점에 최종같네요. 

​

​Tensorflow GPU Container를 실행하기 전에 몇가지 고려해야 할 사항이 있습니다.

​<br>
<br>


### 1. GPU 사용량

실행할 Container가 GPU를 어느 정도 사용할 지 결정합니다. 보통 전체를 다 사용하기 때문에 다음과 같은 Option으로 씁니다.

--gpus all

<br>
<br>

### 2. Ubuntu Directory Mapping

실제 물리적인 GPU가 실행되는 Container는 별도의 환경으로 운영되기 때문에, GPU에 Storage에 있는 Train Data를 보내기 위해서는 WSL에서 실행되는 Ubuntu에 Mount된 Drive를 Tensorflow GPU Container에 Mapping 시켜주어야 합니다.

​

하지만, 실제로 WSL의 Ubuntu에 Mapping된 Drive는 실제로는 Windows에 있는 물리 Stroage입니다.

좀 복잡하지만, 2중으로 Mapping되는 상황입니다.

이 작업을 위해서 docker는 -v Option을 지원해 줍니다. 사용법은 아래와 같습니다.

   -v [WSL Ubuntu의 Directory Path]:[Container의 Directory Path]

   Ex) -v $(pwd):/moonlight

 

위와 같이 입력하면, WSL의 사용자 계정 Directory를 Container에 /moonlight Directory로 Mapping합니다.

-v Option은 여러개 사용가능합니다. 즉, Directory Mapping을 여러개 할 수 있으니, 필요한 만큼 사용하면 됩니다.

<br>
<br>

### 3. 종료시 Container 삭제 여부

Container를 종료시키고 나올 때 보통 exit 명령어로 빠져나오는데, 이렇게 빠져나와도 docker images 명령어로 보면 Container는 그대로 남아 있습니다.

​

하지만, --rm Option을 사용하면 종료와 동시에 Container도 삭제합니다. 저는 이 Option을 사용하지 않을 예정인데, 그 이유는 Tensorflow Container는 Deep Learning Model Train이 필수적인 여러 Package들, 예를 들면, Pandas, sci-kit learn등과 같은 Package가 설치되어 있지 않습니다.

​

그래서 Container를 실행하고 필요한 Package들을 설치해 주어야 하는데, --rm Option을 사용하고 종료를 한 후에 다시 Container를 실행하면 이전에 열심히 설치한 Package들이 전부 다 없어져 있습니다. 참 허무한 일이죠.

​

이런 일을 막기 위해서 Package 설치 후 종료 한 후에, 현재의 Container를 Image로 저장하는 작업을 해야 합니다. 이를 Commit이라고 하는데, 뒤쪽에 다시 다루도록 하겠습니다.

​

​

앞서 언급한 내용들을 모두 적용한 docker run command line은 다음과 같습니다.

```bash
sudo docker run --gpus all -it -v $(pwd):/moonlight tensorflow/tensorflow:2.15.0-gpu
```

<br>
<br>
<br>

## 7. Visual Studo Code 준비

WSL , CUDA Toolkit , Tensorflow GPU까지 모두 성공적으로 설치를 마무리했습니다.

이제 Code Editor를 설치하고 이를 Tensorflow Container와 연결해 보도록 하겠습니다.

저는 주로 Visual Studo Code를 사용하고 있어서 이를 설치하겠습니다.

이 부분은 특별한 내용이 없으니 넘어가도록 하겠습니다.

<br>
<br>
<br>

## 8. Visual Studo Code와 Tensorflow Container 연결

아래 그림을 잘 보면, 우리는 Visual Studo Code를 이용해서 WSL안에서 돌고 있는 Tensorflow Container에 연결해야 하는 상황입니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/00.png">


2중으로 타고 들어가서 작업을 해야 하는 상황인데, 저는 처음에 '이게 가능할까...?'라고 생각했는데, 이런 상황을 고려한 Visual Studo Code Extension이 있습니다 !!

​<br>
<br>

### 8.1. Remote Development Extension 설치

Visual Studo Code의 'Extensions'을 클릭해서 'remote'를 검색하면 **Remote Development**가 나오는데, 이것을 설치해 줍니다.


<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/12.png">

<br>
<br>

### 8.2. Container 연결

이제 Visual Studo Code와 Tensorflow Container를 연결해 보도록 하겠습니다.

​

WSL을 실행하고 Tensorflow Container도 run 시킵니다.

그 후에 Visual Studo Code로 와서 왼쪽 아이콘 중에 아래 그림과 같은 아이콘을 클릭하면 'Remote Explorer'가 열립니다.

여기에서 'WSL Targets'를 선택합니다.


<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/13.png">

아래와 같이 실행중인 WSL Ubuntu 20.04가 보이고, 왼쪽 아래에 조그맣게 파란색으로 'WSL: Ubuntu 20.04'가 있습니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/14.png">

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/15.png">


이것을 클릭하면 아래와 같은 메뉴가 쭉~ 나오는데, 그 중에서 'Attach to Running Containers...'를 클릭합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/16.png">


그러면, 아래와 같이 현재 실행중인 Tensorflow Container가 보입니다. 클릭해서 연결합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/17.png">


연결이 되면 아래 그림과 같이, Visual Studo Code 왼쪽 아래에 연결된 Tensorflow Container 정보가 나옵니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/18.png">

이제 모든 준비가 끝났습니다. 아래와 같이 간단한 예제 Code를 실행해 보면 정상적으로 실행되는 것을 확인할 수 있습니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/20.png">

<br>
<br>
<br>

## 9. Image Commit

실제로 Model Train을 시키기 위해서는 Tensorflow뿐만 아니라, 다른 Package들도 필연적으로 필요합니다.

앞에서 설명했듯이, Container 종료 후에 Commit을 해주지 않으면 설치한 Package들이 다음 Container 실행시에 제대로 보이지 않습니다.

이를 미연에 방지하고자, 수정된 Container를 Image로 저장(Commit)하는 방법을 알려드리겠습니다.

<br>
<br>

### 9.1. docker commit 사용법

docker commit은 아래와 같이 사용합니다.

#### docker commit [원래 Container ID] [새로 저장할 Container Name]

​

Container ID는 docker ps -a 를 입력하면 현재 Container들의 정보가 나오는데, 여기에서 확인할 수 있습니다.

​

우리가 여러 Package들을 설치한 Container의 ID가 6164b18a5d78이고, 새로운 tensorflow/tensorflow:2.15.0-gpu-with-package라는 Image로 저장하려면 아래와 같이 사용하면 됩니다.

```bash
docker commit 6164b18a5d78 tensorflow/tensorflow:2.15.0-gpu-with-package
```


다음부터는 tensorflow/tensorflow:2.15.0-gpu-with-package Image로 Container를 실행하면 이전에 설치된 Package가 모두 들어가 있는 Container가 실행이 됩니다.

​<br>
<br>
<br>

## 10. 최종 Test

이제 실제로 Train시켜 보도록 하겠습니다.

간단하게 이전에 만들어 놓은 Dog & Cat Classification 예제를 실행해 보도록 하겠습니다.

아래와 같이 실행이 잘 되네요.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/19.png">


Linux에 Docker를 설치하고 사용하면 좀 더 편하게 Setting이 가능하겠지만, Windows가 편한 분들에게 이 Post들이 도움이 되셨으면 좋겠습니다.

​

긴 글 읽어주셔서 감사합니다 !!

<br>
