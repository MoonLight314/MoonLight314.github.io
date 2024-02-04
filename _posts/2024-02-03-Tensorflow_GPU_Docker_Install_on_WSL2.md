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
