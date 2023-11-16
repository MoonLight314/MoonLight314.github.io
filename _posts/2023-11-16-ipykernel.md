---
title: "Jupyter Kernel 관리 - ipykernel 사용법"
date: 2023-09-06 08:26:28 -0400
categories: Deep Learning
---

# ipykernel

<br>
<br>

안녕하세요, MoonLight입니다.

이번 Post에서는 Jupyter Notebook을 사용할 때 많이 사용하는 ipykernel Package에 대해서 알아보도록 하겠습니다.   

<br>
<br>

## 0. Jupyter   

<br>

최초에는 OS Shell에서 Python Code를 입력하면서 실행할 수 있는 프로젝트가 있었는데, 이를 ipython notebook이라고 했습니다.  

Command창에서 Python Code를 한줄한줄 입력하면서 결과를 확인할 수 있는 방식이었죠  .


Python이 Compile방식이 아닌 Interpreter 방식이라서 가능한입니다.   위한 package입니다.

<br>   

현재의 Jupyter의 Text Version이라고 할 수 있겠습니다.  

그 후에 지원하는 언어가 R이나 Ruby등과 같은 것들이 추가되면서 지원 폭이 넓어지면서 이름을 Jupyter로 바꾸고 UI를 Console에서 Browser로 변경합니다.  
하지만, UI가 console에서 Browswer로 바뀌었지만, 이를 말그대로 UI뿐이고 여전히 ipython은 필요했습니다.

Jupyter로 바뀌면서 Kernel이란 것도 같이 생겼는데, Jupyter Kernel은 Jupyter가 각 Language와의 연결을 해주고
추가로 Python 가상 환경과 Jupyter를 연결하는 역할도 해 줍니다.

<br>

가상 환경과 Jupyter를 연결하는 역할도 하기 때문에 가상 환경을 여러개 만들경우에는 이 Kernel도 각 가상환경마다 만들어줘야 합니다.

가상환경을 다시 만들어서 Jupyter를 실행했는데, 새로 만든 가상환경의 Kernerl이 없거나 혹은 이전에 만들어 놨던 Kernel이 계속 보이거나 하면 찝찝하죠.

ipykernel은 이 Jupyter Kernel을 관리하기 위한 package입니다.      

<br>
<br>
<br>

## 1. 설치   

설치하는 방법은 간단합니다.   


```python
pip install ipykernel
```

<br>
<br>

## 2. 사용법   

설치 후에 Kernel을 사용하는 방법을 알아보겠습니다.   

<br>
<br>

### 2.1 설치된 Kernel 확인   

현재 어떤 Kernel이 설치되어 있는지 확인하기 위해서 Kernel 목록을 볼 때 사용합니다.   


```python
jupyter kernelspec list
```

<img src="https://moonlight314.github.io/assets/ipykernel/pic_00.png">   

<br>
<br>   

### 2.2. Kernel 삭제

사용하지 않거나 혹은 삭제하고 싶은 Kernel이 있는 경우에 사용합니다.   


```python
jupyter kernelspec uninstall <Kernal Name>
```

<br>

<img src="https://moonlight314.github.io/assets/ipykernel/pic_03.png">   

<br>
<br>

### 2.3. Kernel 추가   


```python
python -m ipykernel install --user --name=<추가할 Kernel Name>
```

<br>

<img src="https://moonlight314.github.io/assets/ipykernel/pic_01.png">

<br>

<img src="https://moonlight314.github.io/assets/ipykernel/pic_02.png">   

<br>

**-m**
 - 뒤에 나오는 package를 실행하라는 의미입니다. 여기서는 ipykernel을 실행하라는 의미입니다.     

**--name**
 - 가상환경의 이름을 입력합니다. '-'가 2개 입니다. 빠뜨리면 안됩니다. 주의하세요.  

**--display-name**
 - 겉으로 보이는 Kernel Name입니다. 이 Option없으면 가상환경 이름이랑 같은 이름으로 보입니다.
 - 이 Option도 '-'가 2개 입니다. 빠뜨리면 안됩니다. 주의하세요.

<br>
<br>

오늘은 ipykernel에 대해서 알아보았는데요, 도움이 되셨으면 좋겠네요.

다른 유용한 정보로 다시 오겠습니다.

<br>
