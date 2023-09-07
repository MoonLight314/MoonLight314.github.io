---
title: "Conda Commands"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# Conda Commands

<br>
<br>
<br>

## 0. Introduction   

<br>

* Anaconda는 Data Scient와 Machine / Deep Learning에서 사용하는 다양한 Package들을 관리해 주는 Open Source Platform입니다.  

<br>

* GUI를 사용할 수도 있지만, 보다 세밀하고 다양한 기능을 사용하기 위해서는 Console에서 사용하는 Command들을 알아둘 필요가 있습니다.  

<br>

* 이번 Post에서는 Anaconda의 Command들을 알아보도록 하겠습니다.   

<br>
<br>
<br>

## 1. Commands

<br>
<br>
<br>

### 1.0. Conda 정보 출력

### **conda info**

  - 이 명령어는 Conda Package 관리자와 관련된 정보를 출력하는 명령어입니다.    

<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_00.png">
</p>
   
<br>
<br>
<br>   

   

   

   

### 1.1. Conda Package 관리자 Update
### **conda update conda**
  - Conda Package 관리자 자체를 최신 Version으로 Update하는 명령어입니다.  
  - 이 명령어를 실행하면 Conda는 현재 설치된 Version과 저장소에서 사용 가능한 최신 Version을 비교하여 필요한 경우 Update를 수행합니다.

   

<br>
<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_01.png">
</p>
   
<br>
<br>
<br>

   

   

### 1.2. Package 설치
### conda install "Package Name"
  - Conda Package 관리자를 사용하여 지정된 "Package Name"를 설치하는 명령어입니다
<br>
<br>
<br>

### 1.3. 특정 가상 환경에 Package 설치
### conda install --name "Environment Name" "Package Name"
  - 특정 가상 환경("Environment Name")에 지정된 Package("Package Name")를 설치합니다.
<br>
<br>
<br>


### 1.4. 특정 Channel에서 Package 설치
### conda install --channel "Channel Name" "Package Name"
  - 특정 채널("Channel Name")에서 지정된 Package("Package Name")를 설치하기 위해 사용됩니다.
  - Conda는 다양한 Package와 Version을 제공하는 여러 채널을 지원합니다. 
  - 따라서 특정 채널에서 Package를 설치하려는 경우, --channel (또는 -c로도 축약됨) Option을 사용하여 해당 채널을 지정할 수 있습니다.
<br>
<br>
<br>


### 1.5. pip로 Package 설치
### pip install "Package Name"
  - Python의 Package 관리 시스템인 pip를 사용하여 지정된 Package("Package Name")를 설치하기 위해 사용됩니다.
  - pip는 Python의 표준 Package 관리 도구이며, PyPI (Python Package Index)라는 저장소에서 수천 개의 Package를 설치할 수 있게 해줍니다.
  - 그러나, Python 환경을 관리하는 데에는 conda와 같은 도구를 사용하는 것이 좋습니다. 특히 여러 가상 환경을 동시에 관리해야하는 경우나, Package 간의 의존성이 복잡한 경우에 유용합니다.
<br>
<br>
<br>


### 1.6. Package Update
### conda update "Package Name"
  - Conda Package 관리자를 사용하여 지정된 Package를 최신 Version으로 Update하는 명령어입니다
<br>
<br>
<br>


### 1.7. Command 도움말
### conda "Command" --help
  - "Command"에 해당하는 명령어에 대한 도움말과 사용 가능한 Option들을 볼 수 있습니다.
<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_02.png">
</p>
   
<br>
<br>
<br>


### 1.8. 가상 환경 생성
### conda create --name "Environment Name" python="Python Version"
  - Conda Package 및 환경 관리자를 사용하여 특정 Python Version을 가진 새 가상 환경을 생성합니다.
  - "Environment Name" : 생성할 가상 환경의 이름입니다.
  - "Python Version"
     * 설치할 파이썬의 Version입니다. 예를 들어, 3.7로 설정하면 파이썬 3.7 Version이 해당 가상 환경에 설치됩니다.
     * 특정 Package들은 특정 Python Version을 요구하는 경우가 흔한데, 이런 경우에 유용하게 사용할 수 있습니다.
     
<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_03.png">
</p>
   
<br>
<br>
<br>


### 1.9. 가상 환경 활성화
### conda activate "Environment Name"
  - Conda Package 및 환경 관리자를 사용하여 특정 가상 환경을 활성화하는 데 사용됩니다.
<br>
<br>
<br>


### 1.10. 전체 가상 환경 목록 출력
### conda env list
  - 설치된 모든 가상 환경의 목록을 출력합니다. ( conda info --envs와 동일한 결과 )
  - 현재 활성화되어 있는 가상 환경의 이름 앞에는 * 가 붙어 있습니다.

<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_04.png">
</p>
   
<br>
<br>
<br>


### 1.11. 가상 환경 복사
### conda create --clone "Source Environment Name" --name "New Environment Name"
  - 기존의 가상 환경을 복사하여 동일한 구성을 가신 새로운 환경을 생성하는 명령어입니다.
  - "Source Environment Name" : 기존에 있던 가상 환경 이름
  - "New Environment Name" : 새롭게 복사되어서 만들어질 가상 환경 이름
<br>
<br>
<br>


### 1.12. 설치된 Package 정보 확인
### conda list
  - 현재 활성화된 가상 환경에 설치된 모든 Package와 그 Version을 나열합니다.

<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_07.png">
</p>
   
<br>
<br>
<br>


### 1.13. 가상 환경 History(Revision) 출력
### conda list --revisions
  - 현재 conda 환경의 변경 내역(리비전)을 보여줍니다. 
  - 각 리비전은 환경에 대한 변경사항(예: Package 추가, Update, 제거)을 나타냅니다.
  - 각 리비전은 고유한 번호와 함께 표시됩니다.

<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_06.png">
</p>
   
<br>
<br>
<br>


### 1.14. 특정 Revision으로 Restore(Rollback)
### conda install --revision "Rev. No"
  - 특정 리비전 번호("Rev. No")로 가상 환경을 복원하는 데 사용됩니다.
  - 이 기능은 실수로 중요한 Package를 제거하거나 Update 후 문제가 발생한 경우 이전 상태로 쉽게 복원할 수 있게 도와줍니다.
<br>
<br>
<br>


### 1.15. 설치 Package List 저장
### conda list --explicit > "Text File Name"
  - 현재 conda 환경에 설치된 모든 Package의 명확한(specified) Version들을 나열하고, 그 리스트를 지정한 텍스트 파일("Text File Name")에 저장합니다.
  - 이렇게 생성된 목록은 다른 시스템이나 환경에서 동일한 Package와 Version들을 설치하는 데 사용될 수 있습니다  
<br>
<br>
<br>


### 1.16. 가상 환경 삭제
### conda env remove --name "Environment Name"
  - 지정한 conda 환경을 제거합니다.
  
<br>
<br>   

<p align="center">
  <img src="/assets/Conda_Commands/pic_05.png">
</p>
   
<br>
<br>
<br>


### 1.17. 가상 환경 비활성화
### conda deactivate
  - 현재 활성화된 가상 환경을 비활성화합니다.
<br>
<br>
<br>


### 1.18. 파일로 부터 가상 환경 생성
### conda env create --file "Text File"
  - 주어진 텍스트 파일("Text File")에 명시된 Package와 종속성을 사용하여 새로운 conda 가상 환경을 생성합니다.
  - 이 "Text File"은 주로 environment.yml 또는 .yml 확장자를 갖는 YAML 파일을 의미하며, 해당 파일은 conda 환경의 스펙(spec)을 명시하고 있습니다.
  - 이러한 파일을 사용하면 동일한 환경을 여러 시스템에서 쉽게 재생성할 수 있습니다.
<br>
<br>
<br>


### 1.19. Package 삭제
### conda remove --name "Environment Name" "Package Name 1" "Package Name 2"
  - 지정된 가상 환경("Environment Name")에서 하나 이상의 Package("Package Name 1", "Package Name 2", ...)를 제거하기 위해 사용됩니다.
<br>
<br>
<br>


### 1.20. Package 정보 검색
### conda search "Package Name"
  - 특정 Package에 대해 사용 가능한 모든 Version들을 검색합니다. 이를 통해 해당 Package의 이전 Version이나 최신 Version 등을 확인할 수 있습니다.
  - 특정 채널에서 Package를 검색하려면 --channel "Channel Name"을 이용하면 됩니다.
<br>
<br>
<br>


### 1.21. Cache 삭제
### conda clean
  - Conda Package 관리자의 캐시와 불필요한 파일들을 제거함으로써 저장 공간을 확보하는 데 사용됩니다. 
  - 사용되는 Option은 다음과 같습니다:
    * -a 또는 --all  :  모든 불필요한 파일과 캐시를 제거합니다.
    * -p 또는 --packages  :  더 이상 사용되지 않는 Package 통합 파일들을 제거합니다.
    * -t 또는 --tarballs  :  Conda Package의 .tar.bz2 파일들을 제거합니다. 이 파일들은 Package가 설치된 후 더 이상 필요하지 않습니다.
    * -i 또는 --index-cache  :  Package 메타데이터 인덱스 캐시를 제거합니다.
    * -s 또는 --source-cache  :  소스 캐시를 제거합니다.
<br>
<br>
<br>


### 1.22. 설정 관리
### conda config
  - Conda의 설정을 관리하는 데 사용되는 명령어입니다. 이 명령어를 사용하여 Conda의 설정 파일을 읽고, 수정하며, 추가할 수 있습니다.
    * --show  :  현재 Conda 설정을 표시합니다.
    * --show-sources  :  설정 값의 소스 위치를 표시합니다.
    * --get "key_name"  :  하나 이상의 설정 키의 값을 가져옵니다.
    * --add channels "channel_name"  :  새로운 채널을 추가합니다.
    * --remove channels "channel_name"  :  특정 채널을 제거합니다.
    * --set "key_name" "value"  :  특정 키의 값을 설정합니다.
    * --prepend channels "channel_name"  :  채널을 채널 목록의 맨 앞에 추가합니다.
    * --append channels "channel_name" : 채널을 채널 목록의 맨 뒤에 추가합니다.
<br>
<br>
<br>


### 1.23. 진단
### conda doctor
  - conda의 문제를 진단하고 디버깅 정보를 제공하는 도구입니다. 
  - 이 도구는 여러가지 conda 서브시스템에 대한 정보를 출력하며, 현재 설치된 conda의 상태나 문제점에 대한 진단을 제공합니다.  
<br>
<br>
<br>


### 1.24. Package 관리
### conda package
  - conda Package를 관리하고 조작하는 데 도움이 되는 도구입니다. 
  - 이 명령어를 사용하면 현재 환경에서 설치된 Package에 대한 정보를 얻거나 Package를 tarball 형식으로 변환할 수 있습니다.  
<br>
<br>
<br>
<br>
<br>
<br>


## 2. Specifying Version Numbers

* Package들을 설치하다 보면, 특정 Version을 지정해야 할 필요가 있는 경우가 있습니다.  

* 이런 경우에 유용한 Version 지정 방법입니다.

Constraint type  | Specification | Result
:------------- | :------------- | :-------------
Fuzzy  | numpy=1.11 | 1.11.0, 1.11.1, 1.11.2, 1.11.18 etc.
Exact  | numpy==1.11 | 1.11.0
Greater than or equal to  | "numpy>=1.11" | 1.11.0 or higher
OR  | "numpy=1.11.1|1.11.3" | 1.11.1, 1.11.3
AND  | "numpy>=1.8,<2"            | 1.8, 1.9, not 2.0   

   
<br>
<br>
<br>
   
