---
title: "Gensim Text Summarizer"
date: 2020-10-23 08:26:28 -0400
categories: DeepLearning
---
# Gensim Text Summarizer
<br>
<br>
<br>

* Gensim(https://radimrehur다.com/genism/about.html)은 2008년에 Czech Digital Mathematics Library가 다양한 자연어 처리 Python 스크립트를 모은 것으로부터 시작되어, 현재는 다양한 자연어 처리 기능을 제공하는 Library입니다.

* 설치 방법은 이 Link를 참조하도록 하세요 : [https://pypi.org/project/gensim/](https://pypi.org/project/gensim/)

* Gensim의 다양한 기능중에는 Text Summarizing 기능도 포함되어 있습니다.

* 이번 Post에서는 Gensim의 Text Summarizing 기능을 사용하는 방법에 대해서 알아보도록 하겠습니다.

<br>
<br>
<br>

## 0. Dataset   

* Text Summarizing을 하기 위해서는 Summarizing을 하기 위한 Text 모음(Dataset)이 필요한 것은 당연한 일입니다.

* 다양한 공개된 Dataset이 있지만, 제가 사용하기로 한 Dataset은 많은 사람들이 사용하는 Dataset인 CNN / Daily Mail Dataset을 사용하도록 하겠습니다.

* Nallapati(https://www.aclweb.org/anthology/K16-1028/) 등이 처리한 CNN / Daily Mail 데이터 세트 (2016)은 Text Summarization의 평가에 사용되어 왔습니다.

* 이 Dataset에는 온라인 News Article과 Mail들로 구성되어 있으며,각 항목에는 해당 글을 요약한 문장들도 포함되어 있습니다.

* News Article은 총 92579개,  Daily Mail은 219506개로 구성되어 있습니다.

<br>
<br>
<br>

### 0.1. CNN / DM Dataset Preprocessing   

* 일반적으로, Text Summarizer는 입력으로 Summarizing할 문장들을 입력을 받고 출력으로 Summarizing된 문장을 출력으로 내놓게 됩니다.

* CNN / DM Dataset은 안타깝게도 Text Summarizer의 입력으로 바로 입력할 수 있도록 깔끔한 구조로 되어있지 않습니다. 

* 약간의 Preprocessing을 거쳐야 합니다. 

* 앞으로 Gensim Package 뿐만 아니라 다양한 Text Summarizer들의 Test를 진행할 예정이므로 여기서 미리 앞으로 사용하기 쉽도록 PreProcessing을 해 놓도록 하겠습니다.

<br>
<br>
<br>

### 0.2. CNN / DM Dataset Download
* CNN / Daily Mail Dataset은 다음 Link에서 Download할 수 있습니다.
[https://cs.nyu.edu/~kcho/DMQA/](https://cs.nyu.edu/~kcho/DMQA/)
  
* CNN / DM이 양쪽에 나뉘어져 있고, Dataset을 Download할 수 있는 Link가 있는데 여기서 'Stories'를 Download하시면 됩니다.

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_00.png">
</p>
<br>
<br>
<br>
<br>
<br>
<br>

### 0.3. CNN / DM Dataset 구조   

* Download를 마친 후에 각각의 크기를 보면, CNN이 154MB , DM이 367MB 정도 됩니다.

* 각각 압축을 풀면 stories라는 Folder가 있고, 수많은 이상한 이름들의 File들이 있습니다.

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_01.png">
</p>
<br>
<br>
<br>
<br>
<br>

* 각각의 File 하나가 하나의 문장들과 그 문장들을 Summarizing한 값들이 저장되어 있습니다.

* CNN File에는 이런 개별 File들이 전부 92579개 있고, DM File에는 219506개가 있는 것입니다.

* 제가 하려고 하는 것은 앞으로 Test할 다양한 Summarizer들의 Test를 위해서 이 FIle들을 하나하나 읽어서 불필요한 내용들은 없앤 후에 하나의 File로 만들고, Summarizing된 값들도 따로 뽑아서 하나의 File로 저장하는 일을 할 것입니다.

* 자, 우선 구조를 알아야 어떻게 전처리를 하면 좋을지 Idea가 떠오를것 같습니다. 아무 File이나 하나 열어보도록 하겠습니다.   
<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_02.png">
</p>
<br>
<br>
<br>
<br>
<br>

* 다행히도 일반 Text File이네요. 

* 그리고, 가장 앞쪽에 '(CNN)'이라는 단어가 보이고, 각 줄 사이에 공백줄이 하나씩 들어가 있네요. 이 두 가지는 나중에 없애야 할 것 같습니다.

* 그리고, 긴 문장이 끝난 후에, '@highlight'라는 단어 뒤에 위의 문장을 요약한 요약문이 나와 있는 형식이네요.

* 이제 앞의 문장들을 모두 모은 File들과 '@highlight' 뒤에 나오는 문장들을 모은 File들을 분리하는 작업을 진행하도록 하겠습니다.

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

### 0.4. Preprocessing


* 필요한 Package를 Load합니다.


```python
from os import listdir
import string
from tqdm.notebook import tqdm
import pickle
import gzip
```
<br>
<br>
<br>
<br>

* 하나의 File을 읽어서 내용을 Return해주는 함수   

```python
def load_doc( filename ):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    
    # read all text
    text = file.read()
    
    # close the file
    file.close()
    
    return text
```
<br>
<br>
<br>
<br>

* 읽은 File에서 요약할 문장과 요약문을 구분하여 저장합니다.

* 요약문은 하나의 문장에서 여러개가 있을 수도 있기 때문에 이를 고려한 처리도 합니다.


```python
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    
    return story, highlights
```
<br>
<br>
<br>
<br>

* Folder 안에 있는 모든 File들에 대해서 위의 처리를 합니다.   


```python
def load_stories(directory):
    
    stories = list()
    
    for name in listdir(directory):
        filename = directory + '/' + name
    
        # load document
        doc = load_doc(filename)
        
        # split into story and highlights
        story, highlights = split_story(doc)
        
        # store
        stories.append({'story':story, 'highlights':highlights})
    
    return stories
```

<br>
<br>
<br>
<br>

* 문장들을 전처리 합니다.

* 필요없는 단어를 삭제하거나, 모두 소문자로 바꾸는 등의 처리를 합니다.

```python
def clean_lines(lines , is_story):
    cleaned = list()
    
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        
        if index > -1:
            line = line[index+len('(CNN)'):]
            
        # tokenize on white space
        line = line.split()
        
        # convert to lower case
        line = [word.lower() for word in line]
        
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        
        # store as string
        cleaned.append(' '.join(line))
        
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    
    if is_story is True:
        ret = ''
        for s in cleaned:
            ret = ret + '\n' + s

        return ret
    else:
        return cleaned
```

<br>
<br>
<br>
<br>

* 우선 CNN Data부터 Load하고 전처리 작업을 하겠습니다.

```python
directory = 'cnn/stories/'

stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))
```

    Loaded Stories 92579
    
<br>
<br>
<br>
<br>

* 전처리한 Data를 본문과 요약문을 따로 File로 저장하기 위해 분리하겠습니다.   


```python
story = []
highlights = []

for example in tqdm( stories ):
    story.append( clean_lines(example['story'].split('\n') , True) )
    highlights.append( clean_lines(example['highlights'] , False) )

print(len(story) , len(highlights))
```


    
    92579 92579
    
<br>
<br>
<br>
<br>


```python
with gzip.open('CNN_stories.pickle', 'wb') as f:
    pickle.dump(story, f)
```


```python
with gzip.open('CNN_highlights.pickle', 'wb') as f:
    pickle.dump(highlights, f)
```


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

* 이제 DM Data도 Load하고 전처리 작업을 하겠습니다.   


```python
directory = 'dailymail/stories/'

stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))
```

    Loaded Stories 219506

<br>
<br>
<br>
<br>

```python
story = []
highlights = []

for example in tqdm( stories ):
    story.append( clean_lines(example['story'].split('\n') , True) )
    highlights.append( clean_lines(example['highlights'] , False) )

print(len(story) , len(highlights))
```


    219506 219506
    
<br>
<br>
<br>
<br>

```python
with gzip.open('DM_stories.pickle', 'wb') as f:
    pickle.dump(story, f)
```


```python
with gzip.open('DM_highlights.pickle', 'wb') as f:
    pickle.dump(highlights, f)
```

<br>
<br>
<br>
<br>

* 다음과 같이 결과 File이 이쁘게 저장이 되어 있네요.

* 앞으로 다양한 Text Summarization 기법들을 Test할 때 유용하게 사용할 것입니다.

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_03.png">
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
<br>

## 1. Evaluation

### 1.1. ROUGE   
* ROUGE란 Recall-Oriented Understudy for Gisting Evaluation의 줄임말이며, Text Summarization이나 Machine Translation의 평가를 위한 기준으로 많이 사용됩니다.

* 예를 들어 보겠습니다.   

* 만약 Model이 생성한 문장이 다음과 같다고 가정해 보겠습니다.
  - **the cat was found under the bed**
  
* 그리고, Reference Summary(Golden Standard, 보통 사람이 생성한 문장이 아래와 같다고 가정해 보죠
  - **the cat was under the bed**
  
* 각각의 개별 단어만으로 겹치는 단어를 세어 본다면 겹치는 단어의 수는 6이지만, 이 수치는 별로 중요한 정보를 알려주지는 않습니다.

* 이를 보완하기 위해 다양한 평가 Metric이 존재합니다.

<br>
<br>
<br>

### 1.2. Precision & Recall   

* Text Summarization에서 Recall의 의미는 생성된 문장이 원래 문장의 얼마만큼을 Cover하는가를 나타냅니다. 즉,    

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_04.png">
</p>
<br>
<br>

* 위와 같은 수식이 되고, 예제의 Recall값을 계산하면   

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_05.png">
</p>
<br>
<br>

* 1이 됩니다. ... 뭔가 찝찝합니다.

* 수식을 잘 살펴보면, Model이 생성한 문장이 아주 길어져서 매우 다양한 단어를 포함하고 있는 경우에도 Recall값은 좋아질 수 있습니다.

* 이런 단점을 보완하기 위해서 Precision값이 사용됩니다.


* Precision값의 정의는 다음과 같습니다.

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_06.png">
</p>
<br>
<br>

* 위의 정의에 따라 Precision 값을 계산하면,   

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_07.png">
</p>
<br>
<br>

* 이제, 다음과 같은 Model이 생성한 문장이 있다고 생각해 봅시다.
  - **the tiny little cat was found under the big funny bed**
  
  
* 이 문장의 Precision값은 다음과 같습니다.   

<br>
<br>
<p align="center">
  <img src="/assets/Gensim/pic_08.png">
</p>
<br>
<br>

* 이 결과는 썩 좋진 않습니다. 결과 문장에 불필요한 단어가 많이 들어가 있기 때문입니다.

* Precision은 간결한 문장을 생성하려고 할 때 중요한 Metric이 됩니다. 

* 따라서 요약문을 평가할 때는 Recall과 Precision을 모두 고려한 F-Score가 중요합니다.

<br>
<br>
<br>
<br>

### 1.3. ROUGE-N, ROUGE-S & ROUGE-L

* ROUGE 측정에는 크게 3가지 종류가 있는데, 각각의 특징은 아래와 같습니다.

  1. ROUGE-N
     - N은 1,2,3 등의 자연수이며, unigram, bigram, trigram 이라고 불립니다.
     - 연속된 N개의 단어가 겹치는 비율을 이용한 측정 방법입니다.
     
  2. ROUGE-L
     - LCS(Longest Common Subsequence)를 사용하여 일치하는 가장 긴 단어를 이용한 측정 방법입니다.
     
  3. ROUGE-S
     - Skip-bigram을 기반한 동시 발생 통계 측정 방법입니다.. Skip-bigram은 문장 순서의 단어 쌍을 말합니다.
     
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## 2. Gensim Text Summarizer 사용

### 2.1. Gensim Text Summarizer 사용하기

* 이제 평가에 사용할 Dataset도 준비가 되었고, 객관적인 평가를 할 방법도 마련되었으니 이제 본격적으로 Gensim Text Summarizer를 사용해 보도록 하겠습니다.

* 필요한 Package를 Load하도록 하겠습니다.


```python
import pickle
import gzip
from tqdm.notebook import tqdm
from gensim.summarization.summarizer import summarize
from rouge_score import rouge_scorer
import pandas as pd
```

<br>
<br>
<br>
<br>

**from gensim.summarization.summarizer import summarize**
: Gensim Text Summarizer를 사용하기 위한 Package입니다.

**from rouge_score import rouge_scorer**
: Rouge Score를 계산하기 위한 Package입니다.


* 앞에서 전처리를 마친 CNN / DM Dataset의 원문과 요약문을 Load합니다.   


```python
with gzip.open('../CNN_stories.pickle','rb') as f:
    story = pickle.load(f)
```


```python
with gzip.open('../CNN_highlights.pickle','rb') as f:
    highlights = pickle.load(f)
```

<br>
<br>
<br>
<br>

* ROUGE Socre 측정을 위해서 RougeScorer를 하나 만듭니다.

* 측정할 항목은 ROUGE-1 , ROUGE-2 , ROUGE-L로 하도록 하겠습니다.

```python
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2' , 'rougeL'], use_stemmer=True)
```

<br>
<br>
<br>
<br>

* 이제 Summarize를 하고, ROUGE 값도 구해보도록 하겠습니다.

* Gensim Text Summarizer를 사용하는 방법은 간단합니다.

* summarize() 함수의 인자로 요약할 문장을 넣어주면 됩니다.


* ROUGE score함수는 list 형태로 3개의 값을 return 해 주는데, 순서대로 Precision , Recall , F-Score 값을 Return해 줍니다.

* 우리는 평가에서 F-Score 값을 사용하도록 하겠습니다.


```python
rouge_1 = []
rouge_2 = []
rouge_L = []
ext_result = []
highlight = []
ExceptionCount = 0

for idx,val in tqdm( enumerate(story) ):
    try:
        gensim_ext_result = summarize( val )
        
        ext_result.append( gensim_ext_result )        

        scores = scorer.score( gensim_ext_result , highlights[idx][0])
        
        highlight.append( highlights[idx][0] )

        rouge_1.append( scores['rouge1'][2] * 100 ) # F-Score
        rouge_2.append( scores['rouge2'][2] * 100 )
        rouge_L.append( scores['rougeL'][2] * 100 )
        
        
    except ValueError:
        ExceptionCount += 1
        
        ext_result.append( "" )        
        
        rouge_1.append( 0 )
        rouge_2.append( 0 )
        rouge_L.append( 0 )
        
print( "ExceptionCount : " , ExceptionCount)
```

    
    ExceptionCount :  29
    
<br>
<br>
<br>
<br>

* CNN에 대해서 Summarize 및 ROUGE F-Score 값을 구했습니다.         

* 실제로 어떻게 요약했는지 한 번 살펴볼까요?   
* 아래는 원래 문장입니다. 오바마 대통령에 관란 CNN News 기사인것 같네요.


```python
story[0]
```

<br>
<br>


    '\nits official us president barack obama wants lawmakers to weigh in on whether to use military force in syria\nobama sent a letter to the heads of the house and senate on saturday night hours after announcing that he believes military action against syrian targets is the right step to take over the alleged use of chemical weapons\nthe proposed legislation from obama asks congress to approve the use of military force to deter disrupt prevent and degrade the potential for future uses of chemical weapons or other weapons of mass destruction\nits a step that is set to turn an international crisis into a fierce domestic political battle\nthere are key questions looming over the debate what did un weapons inspectors find in syria what happens if congress votes no and how will the syrian government react\nin a televised address from the white house rose garden earlier saturday the president said he would take his case to congress not because he has to but because he wants to\nwhile i believe i have the authority to carry out this military action without specific congressional authorization i know that the country will be stronger if we take this course and our actions will be even more effective he said we should have this debate because the issues are too big for business as usual\nobama said top congressional leaders had agreed to schedule a debate when the body returns to washington on september the senate foreign relations committee will hold a hearing over the matter on tuesday sen robert menendez said\ntranscript read obamas full remarks\nsyrian crisis latest developments\nun inspectors leave syria\nobamas remarks came shortly after un inspectors left syria carrying evidence that will determine whether chemical weapons were used in an attack early last week in a damascus suburb\nthe aim of the game here the mandate is very clear and that is to ascertain whether chemical weapons were used and not by whom un spokesman martin nesirky told reporters on saturday\nbut who used the weapons in the reported toxic gas attack in a damascus suburb on august has been a key point of global debate over the syrian crisis\ntop us officials have said theres no doubt that the syrian government was behind it while syrian officials have denied responsibility and blamed jihadists fighting with the rebels\nbritish and us intelligence reports say the attack involved chemical weapons but un officials have stressed the importance of waiting for an official report from inspectors\nthe inspectors will share their findings with un secretarygeneral ban kimoon ban who has said he wants to wait until the un teams final report is completed before presenting it to the un security council\nthe organization for the prohibition of chemical weapons which nine of the inspectors belong to said saturday that it could take up to three weeks to analyze the evidence they collected\nit needs time to be able to analyze the information and the samples nesirky said\nhe noted that ban has repeatedly said there is no alternative to a political solution to the crisis in syria and that a military solution is not an option\nbergen syria is a problem from hell for the us\nobama this menace must be confronted\nobamas senior advisers have debated the next steps to take and the presidents comments saturday came amid mounting political pressure over the situation in syria some us lawmakers have called for immediate action while others warn of stepping into what could become a quagmire\nsome global leaders have expressed support but the british parliaments vote against military action earlier this week was a blow to obamas hopes of getting strong backing from key nato allies\non saturday obama proposed what he said would be a limited military action against syrian president bashar alassad any military attack would not be openended or include us ground forces he said\nsyrias alleged use of chemical weapons earlier this month is an assault on human dignity the president said\na failure to respond with force obama argued could lead to escalating use of chemical weapons or their proliferation to terrorist groups who would do our people harm in a world with many dangers this menace must be confronted\nsyria missile strike what would happen next\nmap us and allied assets around syria\nobama decision came friday night\non friday night the president made a lastminute decision to consult lawmakers\nwhat will happen if they vote no\nits unclear a senior administration official told cnn that obama has the authority to act without congress even if congress rejects his request for authorization to use force\nobama on saturday continued to shore up support for a strike on the alassad government\nhe spoke by phone with french president francois hollande before his rose garden speech\nthe two leaders agreed that the international community must deliver a resolute message to the assad regime and others who would consider using chemical weapons that these crimes are unacceptable and those who violate this international norm will be held accountable by the world the white house said\nmeanwhile as uncertainty loomed over how congress would weigh in us military officials said they remained at the ready\nkey assertions us intelligence report on syria\nsyria who wants what after chemical weapons horror\nreactions mixed to obamas speech\na spokesman for the syrian national coalition said that the opposition group was disappointed by obamas announcement\nour fear now is that the lack of action could embolden the regime and they repeat his attacks in a more serious way said spokesman louay safi so we are quite concerned\nsome members of congress applauded obamas decision\nhouse speaker john boehner majority leader eric cantor majority whip kevin mccarthy and conference chair cathy mcmorris rodgers issued a statement saturday praising the president\nunder the constitution the responsibility to declare war lies with congress the republican lawmakers said we are glad the president is seeking authorization for any military action in syria in response to serious substantive questions being raised\nmore than legislators including of obamas fellow democrats had signed letters calling for either a vote or at least a full debate before any us action\nbritish prime minister david cameron whose own attempt to get lawmakers in his country to support military action in syria failed earlier this week responded to obamas speech in a twitter post saturday\ni understand and support barack obamas position on syria cameron said\nan influential lawmaker in russia which has stood by syria and criticized the united states had his own theory\nthe main reason obama is turning to the congress the military operation did not get enough support either in the world among allies of the us or in the united states itself alexei pushkov chairman of the internationalaffairs committee of the russian state duma said in a twitter post\nin the united states scattered groups of antiwar protesters around the country took to the streets saturday\nlike many other americanswere just tired of the united states getting involved and invading and bombing other countries said robin rosecrans who was among hundreds at a los angeles demonstration\nwhat do syrias neighbors think\nwhy russia china iran stand by assad\nsyrias government unfazed\nafter obamas speech a military and political analyst on syrian state tv said obama is embarrassed that russia opposes military action against syria is crying for help for someone to come to his rescue and is facing two defeats on the political and military levels\nsyrias prime minister appeared unfazed by the saberrattling\nthe syrian armys status is on maximum readiness and fingers are on the trigger to confront all challenges wael nader alhalqi said during a meeting with a delegation of syrian expatriates from italy according to a banner on syria state tv that was broadcast prior to obamas address\nan anchor on syrian state television said obama appeared to be preparing for an aggression on syria based on repeated lies\na top syrian diplomat told the state television network that obama was facing pressure to take military action from israel turkey some arabs and rightwing extremists in the united states\ni think he has done well by doing what cameron did in terms of taking the issue to parliament said bashar jaafari syrias ambassador to the united nations\nboth obama and cameron he said climbed to the top of the tree and dont know how to get down\nthe syrian government has denied that it used chemical weapons in the august attack saying that jihadists fighting with the rebels used them in an effort to turn global sentiments against it\nbritish intelligence had put the number of people killed in the attack at more than\non saturday obama said all told well over people were murdered us secretary of state john kerry on friday cited a death toll of more than of them children no explanation was offered for the discrepancy\niran us military action in syria would spark disaster\nopinion why strikes in syria are a bad idea'

<br>
<br>
<br>
<br>

* 아래는 Gensim Text Summarizer가 요약한 문장입니다.

* 그럭저럭 요약을 한 것 같네요.

* 사실, 뉴스 기사는 대부분의 문장이 중요한 내용을 담고 있긴 하죠

```python
ext_result[0]
```
<br>
<br>

    'its official us president barack obama wants lawmakers to weigh in on whether to use military force in syria\nobama sent a letter to the heads of the house and senate on saturday night hours after announcing that he believes military action against syrian targets is the right step to take over the alleged use of chemical weapons\nthere are key questions looming over the debate what did un weapons inspectors find in syria what happens if congress votes no and how will the syrian government react\nin a televised address from the white house rose garden earlier saturday the president said he would take his case to congress not because he has to but because he wants to\nobamas remarks came shortly after un inspectors left syria carrying evidence that will determine whether chemical weapons were used in an attack early last week in a damascus suburb\nbut who used the weapons in the reported toxic gas attack in a damascus suburb on august has been a key point of global debate over the syrian crisis\nobamas senior advisers have debated the next steps to take and the presidents comments saturday came amid mounting political pressure over the situation in syria some us lawmakers have called for immediate action while others warn of stepping into what could become a quagmire\nsome global leaders have expressed support but the british parliaments vote against military action earlier this week was a blow to obamas hopes of getting strong backing from key nato allies\non saturday obama proposed what he said would be a limited military action against syrian president bashar alassad any military attack would not be openended or include us ground forces he said\nsyrias alleged use of chemical weapons earlier this month is an assault on human dignity the president said\nunder the constitution the responsibility to declare war lies with congress the republican lawmakers said we are glad the president is seeking authorization for any military action in syria in response to serious substantive questions being raised\nbritish prime minister david cameron whose own attempt to get lawmakers in his country to support military action in syria failed earlier this week responded to obamas speech in a twitter post saturday\nthe syrian government has denied that it used chemical weapons in the august attack saying that jihadists fighting with the rebels used them in an effort to turn global sentiments against it'

<br>
<br>

* 아래의 문장은 CNN Dataset에 저장된 사람이 요약한 문장입니다.
* 정말 핵심만 한 문장으로 요약해 놓았네요. ㅎㅎ
<br>
<br>

```python
highlight[0]
```
<br>
<br>

    'syrian official obama climbed to the top of the tree doesnt know how to get down'
    
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

* 자~! 이제 나중에 참고할 수 있도록 ROUGE 값들을 CSV File로 저장해 놓읍시다.   


```python
df = pd.DataFrame()
```


```python
df['rouge_1'] = rouge_1
df['rouge_2'] = rouge_2
df['rouge_L'] = rouge_L
```


```python
df.to_csv('gensim_CNN_result.csv')
```


<br>
<br>
<br>
<br>

* CNN은 다 확인했으니 이제, DM Dataset도 한 번 확인해 볼까요?
* 모든 내용은 다 동일하고, 처음에 읽어오는 Data만 CNN에서 DM으로 바뀝니다.

```python
with gzip.open('../DM_stories.pickle','rb') as f:
    story = pickle.load(f)   
```


```python
with gzip.open('../DM_highlights.pickle','rb') as f:
    highlights = pickle.load(f)
```
<br>
<br>

```python
rouge_1 = []
rouge_2 = []
rouge_L = []
ext_result = []
highlight = []
ExceptionCount = 0

for idx,val in tqdm( enumerate(story) ):
    try:
        gensim_ext_result = summarize( val )
        
        ext_result.append( gensim_ext_result )        

        scores = scorer.score( gensim_ext_result , highlights[idx][0])
        
        highlight.append( highlights[idx][0] )

        rouge_1.append( scores['rouge1'][2] * 100 ) # F-Score
        rouge_2.append( scores['rouge2'][2] * 100 )
        rouge_L.append( scores['rougeL'][2] * 100 )
        
        
    except ValueError:
        ExceptionCount += 1
        
        ext_result.append( "" )        
        
        rouge_1.append( 0 )
        rouge_2.append( 0 )
        rouge_L.append( 0 )
        
print( "ExceptionCount : " , ExceptionCount)
```
    
    ExceptionCount :  3
    
<br>
<br>
<br>
<br>

* 어떻게 요약했는지 한 번 살펴보죠
* 아래는 원래 문장입니다. 메이웨더와 파퀴아오 경기에 관한 내용같네요.


```python
story[0]
```
<br>
<br>

    '\nsky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday\nthe richest fight of all time will not come cheap either for sky sports or their subscribers even though sky are keeping faith with their core following by keeping the base price below\nit has taken what is described by industry insiders as very substantial for sky to fend off fierce competition from frank boxnation\nfloyd mayweathers hotlyanticipated bout with manny pacquiao will be shown on sky sports\npacquiao headed for the playground after working out in los angeles previously\nthe price for the fight has been set at until midnight of friday may\nthe cost will remain the same for those paying via remote control or online but will be if booked via phone after friday\nsky are flirting with their threshold of by charging a buy on their sports box office channel until midnight on may rising to on may the day of the fight in las vegas since they are understood to have broken past protocol by offering the us promoters a cut of that revenue as well as a hefty upfront payment it is expected they will have to shatter the payperview record in this country to break even\nthe current sky record stands at buys for ricky vegas loss to mayweather in\nwarren is believed to have offered a higher lump sum than sky in the hope of attracting another two million customers to his subscription channel\nit is doubtful if sky can reach that number at per sale at on a sunday morning but if they get buys they should be out of the red\nmayweather continued to work on the pads in his las vegas gym as he prepares for the fight\npacquiao will take on mayweather at the mgm grand in las vegas on may in one of the biggest fights ever'

<br>
<br>
<br>
<br>

* Gensim이 요약한 문장입니다.   


```python
ext_result[0]
```
<br>
<br>


    'sky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday\nsky are flirting with their threshold of by charging a buy on their sports box office channel until midnight on may rising to on may the day of the fight in las vegas since they are understood to have broken past protocol by offering the us promoters a cut of that revenue as well as a hefty upfront payment it is expected they will have to shatter the payperview record in this country to break even'

<br>
<br>
<br>
<br>

* 아래는 원래 문장의 사람이 만든 요약문입니다.


```python
highlight[0]
```
<br>
<br>


    'sky has been in fierce competition with frank warrens boxnation'
    
<br>
<br>
<br>
<br>

* DM 결과도 같이 저장해 놓겠습니다.   


```python
df = pd.DataFrame()
```


```python
df['rouge_1'] = rouge_1
df['rouge_2'] = rouge_2
df['rouge_L'] = rouge_L
```



```python
df.to_csv('gensim_DM_result.csv')
```

<br>
<br>
<br>
<br>

* 자, 이것으로 Gensim Text Summarizer 사용을 해 봤습니다.

* ROUGE 값을 저장해 놓았으니, 다른 Text Summarizer 비교시에 사용할 수 있을 것입니다.

* 다음 Post에서는 다른 Text Summarizer를 사용해 보도록 하겠습니다.
