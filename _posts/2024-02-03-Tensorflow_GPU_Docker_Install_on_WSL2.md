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



## 1. WSL2 설치

WSL (Windows Subsystem for Linux)이란 Windows에서 직접 Linux 배포판을 실행할 수 있도록 하는 기능으로써,
이를 통해서 사용자는 Linux 명령어, 스크립트, 도구들을 Windows 환경에서 바로 사용할 수 있게 됩니다.
기본적으로 개발자가 Windows에서 Linux 기반의 개발 환경을 구축하고 사용할 수 있도록 해줍니다.

WSL은 가상 머신이나 듀얼 부팅 없이도 Linux 환경을 Windows 내에서 손쉽게 접근하고 사용할 수 있게 해줍니다.
예전 초창기 WSL에서는 Linux를 설치할 수는 있지만, GPU를 지원하지는 못해서 Deep Learning 개발 환경 구축에
크게 유용하지는 않았습니다.

하지만, WSL2가 나오고 GPU를 지원할 수 있게 되면서 이제는 상당히 쓸만해 졌습니다.



### 1.1. WSL2 설치 가능 조건

WSL2를 설치하기 위해서는 Windows 10 버전 2004 이상(빌드 19041 이상) 또는 Windows 11이 필요합니다.

https://learn.microsoft.com/ko-kr/windows/wsl/install



본인의 Windows 정보를 확인하셔서 설치 가능한지 확인해 보시기 바랍니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/01.png">


### 1.2. Windows Subsystem for Linux 기능 활성화

먼저 PowerShell을 관리자 권한으로 실행합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/02.png">

그리고, Windows Subsystem for Linux(WSL) 기능을 활성화 시켜주어야 합니다.

아래 명령어를 입력합니다.

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```


### 1.3 Virtual Machine feature 활성화

추가로, Virtual Machine 기능도 활성화 시켜주어야 합니다. 아래 명령어를 입력합니다.

```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

​

### 1.4. 재부팅

반드시 재부팅해주셔야 변경사항이 적용됩니다. 꼭 재부팅 해주세요.

​

​

### 1.5. WSL 설치

재부팅 후에, 다시 PowerShell을 관리자 권한으로 실행하여, 다음 명령어를 입력합니다.

```bash
wsl --install
```
​


### 1.6 Linux Update Kernel Package 설치

아래 링크를 클릭하여 Linux Update Kernel Package를 설치합니다.

https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi

​​

### 1.7 WSL2를 기본으로 설정

WSL에는 WSL 1 & WSL 2가 있는데, 우리는 GPU를 지원하는 WSL 2를 사용할 예정이기 때문에 기본버전을 WSL2로 설정하도록 하겠습니다.

​

powersehll 관리자 모드에서 아래 명령어 입력해 줍니다.

```bash
wsl --set-default-version 2
```



​

​

## 2. Ubuntu 설치

이제 WSL2를 설치했으니, WSL2에서 실행할 Linux를 설치하도록 하겠습니다. 가장 많이 사용하는 Linux 배포판인 Ubuntu 20.04 LTS를 설치해 보겠습니다.

​


### 2.1 Microsoft Store에서 Ubuntu 설치

Microsoft Store에서 Ubuntu 20.04.4 LTS를 설치합니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/03.png">


#### Ubuntu 20.04.4 LTS
https://apps.microsoft.com/detail/9MTTCL66CPXJ?hl=en-us&gl=US
​
​

아래 Link는 Ubuntu 22.04.4 LTS입니다. 좀 더 최신 배포판을 설치하고자 하시는 분은 사용해 주세요.

https://apps.microsoft.com/detail/9PN20MSR04DW?hl=en-us&gl=US
​



### 2.2. 계정 생성

Ubuntu 설치가 완료되면 계정을 생성하고 나면, Ubuntu 설치는 완료됩니다.

<br>
<img src="https://moonlight314.github.io/assets/TensorflowGPUDockerInstallonWSL2/04.png">






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




## 4. Docker 설치

저는 WSL2에 설치된 Ubuntu안에 Docker를 설치하고, WSL2안에서 Tensorflow Container를 실행할 예정입니다.

그래서, 먼저 WSL2에 Ububtu에 Docker 설치를 진행하도록 하겠습니다.

​

​

### 4.1. Service 명령 권한 설정

먼저 visudo를 사용하여 docker command 앞에 일일이 'sudo'를 붙이지 않고 docker command를 실행할 수 있도록 하겠습니다. docker command를 사용할 일이 많은데, sudo를 항상 붙이려면 은근히 번거롭습니다.

​

​

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


#### 4.1.2. 다른 방법

혹시 위와 같은 방법을 사용해도 sudo 없이 docker command가 수행되지 않는다면, 아래 Link의 방법을 따라하면 해결됩니다.

저같은 경우에도 위와 같은 방법으로 되지 않아서, 아래 Link를 따라하면서 해결되었습니다.

https://docs.docker.com/engine/install/linux-postinstall/


### 4.2. Docker 설치

이제 Ubuntu에 Docker를 설치하도록 하겠습니다. 아래 Command들을 순서대로 입력하면 됩니다.

​

#### 4.2.1. 패키지 업데이트

```bash
sudo apt-get update
```
​

#### 4.2.2. https로 데이터를 받아서 설치하는 패키지들을 위한 준비

```bash
sudo apt-get install -y \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common
```
​

#### 4.2.3. Docker의 GPG key 추가

GPG란 툴이나 소스 등을 배포 할 때 원본이 수정되었나를 확인하는 SW인데, GPG key는 이를 확인하는데 사용하는 값입니다.

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
​

#### 4.2.4. Stable Docker Repository 설정

```bash
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
```
​

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





## 5. nVidia CUDA Toolkit Container 설치

WSL에서 CUDA Toolkit을 사용할 수 있도록 해주는 특별한 Docker Container를 먼저 설치해 줍니다.

​

### 5.1. Package Repository 추가

Ubuntu에 아래 명령어 입력합니다.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
```
​

### 5.2. nVidia-Docker GPGKey 추가

위에서 설명했던 GPGKey를 추가합니다.

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```
​

### 5.3.  Package Repository 추가

Ubuntu에 아래 명령어 입력

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
​

### 5.4.  Package Update

```bash
sudo apt-get update
```

### 5.5.  CUDA Toolkit 설치

Ubuntu에 아래 명령어 입력

```bash
sudo apt-get install -y nvidia-container-toolkit
```
​

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
