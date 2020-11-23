---
title: "Extractive Text Summarizers with BERT , GPT2 , XLNet"
date: 2020-11-23 08:26:28 -0400
categories: DeepLearning
---
# Extractive Text Summarizers with BERT , GPT2 , XLNet
<br>
<br>
<br>
이번 Post에서는 지난번 Gensim에 이어서 다양한 Language Model을 이용한 Text Summarizer에 대해서 알아보도록 하겠습니다.   
<br>
오늘 소개해 드릴 모든 Summarizer들은 **Extractive Summarizers** 이니, 참고하시기 바랍니다.
<br>
<br>
<br>
<br>
<br>
<br>

## 1. BERT Extractive Summarizer

   

* BERT는 Bidirectional Encoder Representations from Transformers의 약자로 2018년 10월에 논문이 공개됐고, 11월에 오픈소스로 코드까지 공개된 Google의 새로운 Language Representation Model입니다.
   
   
* 그간 높은 성능을 보이며 좋은 평가를 받아온 ELMo를 의식한 이름에, 무엇보다 NLP 11개 태스크에 state-of-the-art를 기록하며 당시 가장 치열한 분야인 SQuAD의 기록마저 갈아치우며 혜성처럼 등장한 Model이었습니다.   
   
   
* BERT는 Sebastian Ruder가 언급한 NLP's ImageNet에 해당하는 가장 최신 모델 중 하나로, 대형 코퍼스에서 Unsupervised Learning으로 General-Purpose Language Understanding 모델을 구축하고 Pre-training Supervised Learning으로 Fine-tuning 해서 QA, STS등의 하위 Downstream NLP 태스크에 적용하는 Semi-supervised Learning Model입니다.
   
   
* ULMFiT이 가능성을 보여주었고, 이후 ELMo, OpenAI GPT등이 놀라운 성능을 보여주면서 그 진가를 인정받았다. BERT의 경우 무려 11개의 NLP 태스크에서 state-of-the-art를 기록하면서 뉴욕 타임즈의 지면을 장식하기도 했습니다.

* 관련 논문 : https://arxiv.org/pdf/1810.04805.pdf

* Package 설명은 다음의 Link에서 확인하실 수 있습니다.
  
  https://pypi.org/project/bert-extractive-summarizer/   
  
  
<br>
<br>
<br>
<br>
<br>
<br>

## 2. Transformer Summarizer   

* BERT Model을 이용하지 않고, 다른 Model을 이용해서 Text Summarizer를 만들 수도 있습니다.

* 이번 예제에서는 Transformer Model(BERT도 Transformer 기반이긴 합니다)을 기반으로 한, GPT2 & XLNet Model을 이용해 보도록 하겠습니다.   

* Transformer Model
  - RNN 이나 LSTM을 사용하지 않는 매우 혁신적인 Model입니다.
        
  - 자세한 설명은 여기를 한 번 읽어보시기 바랍니다.
    https://blog.pingpong.us/transformer-review/
        
  - 논문 : https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

* GPT2 Model : https://openai.com/blog/better-language-models/   

  논문 : https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  
  Github : https://github.com/openai/gpt-2

   

* XLNet
  - 설명 : https://blog.pingpong.us/xlnet-review/
  
  
<br>
<br>
<br>
<br>
<br>
<br>

## 3. Summarization

* 이제 위에서 살펴본 3가지의 Model을 이용하여 실제 Text Summarization을 해보도록 하겠습니다.

* Dataset은 이전에 사용했던 CNN / DM Dataset을 그대로 사용하도록 하겠습니다.
  
  미리 전처리도 다 해놓았기 때문에 시간을 많이 줄일 수 있을 것입니다.

* 앞서 살펴본 Model들은 이론적으로 매우 복잡하지만 실제 사용법은 간단합니다.   

   
<br>
<br>
<br>   

   

* 먼저 다음 명령어로 필요한 Package들을 설치하도록 하겠습니다.   


```python
!pip install bert-extractive-summarizer
```


    Collecting bert-extractive-summarizer
      Downloading https://files.pythonhosted.org/packages/7b/e3/c8b820d8c0a96a9318a423a38242275f2e862425793b5b8287e982324ffc/bert_extractive_summarizer-0.5.1-py3-none-any.whl
    Collecting spacy (from bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/9b/ce/ddac37d457ae17152bc7e15164a11bf8236fc4e8a05cabb94d922f58ea23/spacy-2.3.2-cp37-cp37m-win_amd64.whl (9.3MB)
    Collecting transformers (from bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/3a/83/e74092e7f24a08d751aa59b37a9fc572b2e4af3918cb66f7766c3affb1b4/transformers-3.5.1-py3-none-any.whl (1.3MB)
    Requirement already satisfied: scikit-learn in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from bert-extractive-summarizer) (0.21.3)
    Collecting murmurhash<1.1.0,>=0.28.0 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/12/70/ee00202a818e8df318fa0738babf642e651a583ff7b541d42ea94fe267e7/murmurhash-1.0.4-cp37-cp37m-win_amd64.whl
    Collecting thinc==7.4.1 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/8b/6f/7cd666630afeb9b85cbd23c75da76b61580e45a9fe1c19145e3f7675ffc8/thinc-7.4.1-cp37-cp37m-win_amd64.whl (2.0MB)
    Collecting catalogue<1.1.0,>=0.0.7 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/6c/f9/9a5658e2f56932e41eb264941f9a2cb7f3ce41a80cb36b2af6ab78e2f8af/catalogue-1.0.0-py2.py3-none-any.whl
    Requirement already satisfied: setuptools in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from spacy->bert-extractive-summarizer) (41.4.0)
    Collecting srsly<1.1.0,>=1.0.2 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/41/58/76032df5afe0774d677e07014afc835b6ce0cfc2edcd16b73008125d7648/srsly-1.0.4-cp37-cp37m-win_amd64.whl (285kB)
    Collecting blis<0.5.0,>=0.4.0 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/d5/7e/1981d5389b75543f950026de40a9d346e2aec7e860b2800e54e65bd46c06/blis-0.4.1-cp37-cp37m-win_amd64.whl (5.0MB)
    Collecting wasabi<1.1.0,>=0.4.0 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/1b/10/55f3cf6b52cc89107b3e1b88fcf39719392b377a3d78ca61da85934d0d10/wasabi-0.8.0-py3-none-any.whl
    Collecting preshed<3.1.0,>=3.0.2 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/66/34/9020adcdabdf7b5e32f280786e9a98df7616c30efb2d09ffe9f48c320367/preshed-3.0.4-cp37-cp37m-win_amd64.whl (266kB)
    Collecting tqdm<5.0.0,>=4.38.0 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/93/3a/96b3dc293aa72443cf9627444c3c221a7ba34bb622e4d8bf1b5d4f2d9d08/tqdm-4.51.0-py2.py3-none-any.whl (70kB)
    Collecting cymem<2.1.0,>=2.0.2 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/08/a0/6210b235a731799ed80ba991b4cdfd9d2fc3a876f8fbad20e97b1169b85a/cymem-2.0.4-cp37-cp37m-win_amd64.whl
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from spacy->bert-extractive-summarizer) (2.22.0)
    Collecting plac<1.2.0,>=0.9.6 (from spacy->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/86/85/40b8f66c2dd8f4fd9f09d59b22720cffecf1331e788b8a0cab5bafb353d1/plac-1.1.3-py2.py3-none-any.whl
    Requirement already satisfied: numpy>=1.15.0 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from spacy->bert-extractive-summarizer) (1.16.5)
    Collecting sentencepiece==0.1.91 (from transformers->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/78/c7/fb817b7f0e8a4df1b1973a8a66c4db6fe10794a679cb3f39cd27cd1e182c/sentencepiece-0.1.91-cp37-cp37m-win_amd64.whl (1.2MB)
    Collecting regex!=2019.12.17 (from transformers->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/4f/3f/40c8db23e022ccc9eb9fc0f39202af49c8614b22990b2e7129c2543f2da5/regex-2020.11.13-cp37-cp37m-win_amd64.whl (269kB)
    Collecting tokenizers==0.9.3 (from transformers->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/c4/eb/7391faa9651b568a233379d93e0754d4fc94498191e23d77d9ab8274a3e7/tokenizers-0.9.3-cp37-cp37m-win_amd64.whl (1.9MB)
    Requirement already satisfied: protobuf in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from transformers->bert-extractive-summarizer) (3.11.2)
    Collecting sacremoses (from transformers->bert-extractive-summarizer)
      Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    Requirement already satisfied: packaging in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from transformers->bert-extractive-summarizer) (19.2)
    Requirement already satisfied: filelock in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from transformers->bert-extractive-summarizer) (3.0.12)
    Requirement already satisfied: scipy>=0.17.0 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from scikit-learn->bert-extractive-summarizer) (1.3.1)
    Requirement already satisfied: joblib>=0.11 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from scikit-learn->bert-extractive-summarizer) (0.13.2)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (0.23)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (2019.9.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (1.24.2)
    Requirement already satisfied: idna<2.9,>=2.5 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (2.8)
    Requirement already satisfied: six>=1.9 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from protobuf->transformers->bert-extractive-summarizer) (1.12.0)
    Requirement already satisfied: click in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from sacremoses->transformers->bert-extractive-summarizer) (7.0)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from packaging->transformers->bert-extractive-summarizer) (2.4.2)
    Requirement already satisfied: zipp>=0.5 in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (0.6.0)
    Requirement already satisfied: more-itertools in c:\users\csyi\appdata\local\continuum\anaconda3\lib\site-packages (from zipp>=0.5->importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (7.2.0)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py): started
      Building wheel for sacremoses (setup.py): finished with status 'done'
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp37-none-any.whl size=893262 sha256=1e898ee64d901e37a629844abe795b6e423c6df2292d27f857fe8b454afbbd8b
      Stored in directory: C:\Users\csyi\AppData\Local\pip\Cache\wheels\29\3c\fd\7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sacremoses
    Installing collected packages: murmurhash, catalogue, tqdm, blis, srsly, wasabi, cymem, preshed, plac, thinc, spacy, sentencepiece, regex, tokenizers, sacremoses, transformers, bert-extractive-summarizer
      Found existing installation: tqdm 4.36.1
        Uninstalling tqdm-4.36.1:
          Successfully uninstalled tqdm-4.36.1
    Successfully installed bert-extractive-summarizer-0.5.1 blis-0.4.1 catalogue-1.0.0 cymem-2.0.4 murmurhash-1.0.4 plac-1.1.3 preshed-3.0.4 regex-2020.11.13 sacremoses-0.0.43 sentencepiece-0.1.91 spacy-2.3.2 srsly-1.0.4 thinc-7.4.1 tokenizers-0.9.3 tqdm-4.51.0 transformers-3.5.1 wasabi-0.8.0

<br>
<br>
<br>
<br>
<br>
<br>


* 이제 사용할 Package들을 Load하겠습니다.   


```python
from summarizer import Summarizer
from summarizer import TransformerSummarizer
import pickle
import gzip
from tqdm.notebook import tqdm
from rouge_score import rouge_scorer
import pandas as pd
```


<br>
<br>
<br>
<br>
<br>
<br>

* 먼저 CNN 부터 Test해 보도록 하죠
* 이전에 미리 전처리 해둔 CNN / DM Data를 여기서도 그대로 사용하도록 하겠습니다.
* 미리 전처리 해 둔것이 매우 유용하게 사용되네요.

<br>
<br>
<br>
<br>   

* 원문을 읽어옵니다.   


```python
with gzip.open('../CNN_stories.pickle','rb') as f:
    story = pickle.load(f)
```
<br>
<br>
<br>

* 전체 길이를 한 번 확인해 보도록 하죠.   


```python
print( len( story ) )
```

    92579
    

* 92579 개가 맞네요. 제대로 읽어왔습니다.

<br>
<br>
<br>
<br>
<br>
<br>

* 이제 각 Model들을 Load하도록 하겠습니다.   


```python
bert_model = Summarizer()
```


```python
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
```


```python
XLNet_model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
```
<br>
<br>
<br>

* 최초로 Model을 Load하는 것이면 Pre-Trained Model을 Network으로 받아오는데 시간이 걸릴 수도 있습니다.   

<br>
<br>
<br>   
<br>
<br>
<br>   

* 자, 이제 실제로 Summarization을 하겠습니다.
* 하나의 문장에 대해서 3개의 각 Model들이 Summarization을 하고, 그 결과를 모아둡니다.
* 일정 갯수(STEP)만큼 모이면 하나의 결과 File로 저장하는 것을 전체에 반복합니다.

<br>
<br>
<br>

```python
STEP = 5
count = 0

for idx in range(0,92579,STEP):
    short_story = story[idx : idx+STEP]
    
    df = pd.DataFrame(columns=["bert_result","GPT2","XLNet"])
    bert_result = []
    GPT2_result = []
    XLNet_result = []

    for i in range(0,STEP):
        s = short_story[i]
        s = s.replace("\n",'. ')
        s = s+'.'
        s = s[2:]
        
        bert_summary = ''.join(bert_model(s , ratio=0.01))
        bert_result.append( bert_summary )
        
        GPT2_model_sum = ''.join(GPT2_model(s, ratio=0.01))
        GPT2_result.append( GPT2_model_sum )
        
        XLNet_model_sum = ''.join(XLNet_model(s, ratio=0.01))
        XLNet_result.append( XLNet_model_sum )
        
        print(count)
        count += 1
        
    df['bert_result'] = bert_result
    df['GPT2'] = GPT2_result
    df['XLNet'] = XLNet_result

    filename = 'Result_{}_{}.csv'.format(idx,idx+STEP-1)
    df.to_csv( filename )
```

    0
    1
    2
    3
    4
    

<br>
<br>
<br>


* CPU 만을 사용하기 때문에 시간이 매우 많이 걸립니다.
* 걸어놓고 다른 일 보시기 바랍니다.

   
<br>
<br>
<br>   
<br>
<br>
<br>  

* 여기서는 시간 관계상 5개만 요약하겠습니다.

<br>
<br>
<br>
<br>
<br>
<br>
   
* 결과를 하나만 살펴볼까요?   

* 먼저 원래 문장부터 보고 각 Model이 요약한 요약문을 보도록 하겠습니다.

* 원래 문장입니다. '오바마','시리아', 음...

<br>
<br>
<br>

```python
print( short_story[0] )
```

    
    its official us president barack obama wants lawmakers to weigh in on whether to use military force in syria
    obama sent a letter to the heads of the house and senate on saturday night hours after announcing that he believes military action against syrian targets is the right step to take over the alleged use of chemical weapons
    the proposed legislation from obama asks congress to approve the use of military force to deter disrupt prevent and degrade the potential for future uses of chemical weapons or other weapons of mass destruction
    its a step that is set to turn an international crisis into a fierce domestic political battle
    there are key questions looming over the debate what did un weapons inspectors find in syria what happens if congress votes no and how will the syrian government react
    in a televised address from the white house rose garden earlier saturday the president said he would take his case to congress not because he has to but because he wants to
    while i believe i have the authority to carry out this military action without specific congressional authorization i know that the country will be stronger if we take this course and our actions will be even more effective he said we should have this debate because the issues are too big for business as usual
    obama said top congressional leaders had agreed to schedule a debate when the body returns to washington on september the senate foreign relations committee will hold a hearing over the matter on tuesday sen robert menendez said
    transcript read obamas full remarks
    syrian crisis latest developments
    un inspectors leave syria
    obamas remarks came shortly after un inspectors left syria carrying evidence that will determine whether chemical weapons were used in an attack early last week in a damascus suburb
    the aim of the game here the mandate is very clear and that is to ascertain whether chemical weapons were used and not by whom un spokesman martin nesirky told reporters on saturday
    but who used the weapons in the reported toxic gas attack in a damascus suburb on august has been a key point of global debate over the syrian crisis
    top us officials have said theres no doubt that the syrian government was behind it while syrian officials have denied responsibility and blamed jihadists fighting with the rebels
    british and us intelligence reports say the attack involved chemical weapons but un officials have stressed the importance of waiting for an official report from inspectors
    the inspectors will share their findings with un secretarygeneral ban kimoon ban who has said he wants to wait until the un teams final report is completed before presenting it to the un security council
    the organization for the prohibition of chemical weapons which nine of the inspectors belong to said saturday that it could take up to three weeks to analyze the evidence they collected
    it needs time to be able to analyze the information and the samples nesirky said
    he noted that ban has repeatedly said there is no alternative to a political solution to the crisis in syria and that a military solution is not an option
    bergen syria is a problem from hell for the us
    obama this menace must be confronted
    obamas senior advisers have debated the next steps to take and the presidents comments saturday came amid mounting political pressure over the situation in syria some us lawmakers have called for immediate action while others warn of stepping into what could become a quagmire
    some global leaders have expressed support but the british parliaments vote against military action earlier this week was a blow to obamas hopes of getting strong backing from key nato allies
    on saturday obama proposed what he said would be a limited military action against syrian president bashar alassad any military attack would not be openended or include us ground forces he said
    syrias alleged use of chemical weapons earlier this month is an assault on human dignity the president said
    a failure to respond with force obama argued could lead to escalating use of chemical weapons or their proliferation to terrorist groups who would do our people harm in a world with many dangers this menace must be confronted
    syria missile strike what would happen next
    map us and allied assets around syria
    obama decision came friday night
    on friday night the president made a lastminute decision to consult lawmakers
    what will happen if they vote no
    its unclear a senior administration official told cnn that obama has the authority to act without congress even if congress rejects his request for authorization to use force
    obama on saturday continued to shore up support for a strike on the alassad government
    he spoke by phone with french president francois hollande before his rose garden speech
    the two leaders agreed that the international community must deliver a resolute message to the assad regime and others who would consider using chemical weapons that these crimes are unacceptable and those who violate this international norm will be held accountable by the world the white house said
    meanwhile as uncertainty loomed over how congress would weigh in us military officials said they remained at the ready
    key assertions us intelligence report on syria
    syria who wants what after chemical weapons horror
    reactions mixed to obamas speech
    a spokesman for the syrian national coalition said that the opposition group was disappointed by obamas announcement
    our fear now is that the lack of action could embolden the regime and they repeat his attacks in a more serious way said spokesman louay safi so we are quite concerned
    some members of congress applauded obamas decision
    house speaker john boehner majority leader eric cantor majority whip kevin mccarthy and conference chair cathy mcmorris rodgers issued a statement saturday praising the president
    under the constitution the responsibility to declare war lies with congress the republican lawmakers said we are glad the president is seeking authorization for any military action in syria in response to serious substantive questions being raised
    more than legislators including of obamas fellow democrats had signed letters calling for either a vote or at least a full debate before any us action
    british prime minister david cameron whose own attempt to get lawmakers in his country to support military action in syria failed earlier this week responded to obamas speech in a twitter post saturday
    i understand and support barack obamas position on syria cameron said
    an influential lawmaker in russia which has stood by syria and criticized the united states had his own theory
    the main reason obama is turning to the congress the military operation did not get enough support either in the world among allies of the us or in the united states itself alexei pushkov chairman of the internationalaffairs committee of the russian state duma said in a twitter post
    in the united states scattered groups of antiwar protesters around the country took to the streets saturday
    like many other americanswere just tired of the united states getting involved and invading and bombing other countries said robin rosecrans who was among hundreds at a los angeles demonstration
    what do syrias neighbors think
    why russia china iran stand by assad
    syrias government unfazed
    after obamas speech a military and political analyst on syrian state tv said obama is embarrassed that russia opposes military action against syria is crying for help for someone to come to his rescue and is facing two defeats on the political and military levels
    syrias prime minister appeared unfazed by the saberrattling
    the syrian armys status is on maximum readiness and fingers are on the trigger to confront all challenges wael nader alhalqi said during a meeting with a delegation of syrian expatriates from italy according to a banner on syria state tv that was broadcast prior to obamas address
    an anchor on syrian state television said obama appeared to be preparing for an aggression on syria based on repeated lies
    a top syrian diplomat told the state television network that obama was facing pressure to take military action from israel turkey some arabs and rightwing extremists in the united states
    i think he has done well by doing what cameron did in terms of taking the issue to parliament said bashar jaafari syrias ambassador to the united nations
    both obama and cameron he said climbed to the top of the tree and dont know how to get down
    the syrian government has denied that it used chemical weapons in the august attack saying that jihadists fighting with the rebels used them in an effort to turn global sentiments against it
    british intelligence had put the number of people killed in the attack at more than
    on saturday obama said all told well over people were murdered us secretary of state john kerry on friday cited a death toll of more than of them children no explanation was offered for the discrepancy
    iran us military action in syria would spark disaster
    opinion why strikes in syria are a bad idea
    

<br>
<br>
<br>
<br>
<br>
<br>

* 아래는 각 Model들이 요약한 문장들입니다.
* 모두 비슷한 문장을 생성해 내고 있네요.


```python
print(bert_result[0])
```

    its official us president barack obama wants lawmakers to weigh in on whether to use military force in syria. the aim of the game here the mandate is very clear and that is to ascertain whether chemical weapons were used and not by whom un spokesman martin nesirky told reporters on saturday.
    

<br>
<br>
<br>   

```python
print(GPT2_result[0])
```

    its official us president barack obama wants lawmakers to weigh in on whether to use military force in syria. obama sent a letter to the heads of the house and senate on saturday night hours after announcing that he believes military action against syrian targets is the right step to take over the alleged use of chemical weapons.
    
<br>
<br>
<br>

```python
print(XLNet_result[0])
```

    its official us president barack obama wants lawmakers to weigh in on whether to use military force in syria. obamas remarks came shortly after un inspectors left syria carrying evidence that will determine whether chemical weapons were used in an attack early last week in a damascus suburb.
    
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
   

* DM Dataset에서도 동일하게 수행하면 됩니다.   

* CNN에서 했던 것과 동일하게 이전에 전처리해둔 DM Data를 읽어옵니다.


```python
with gzip.open('../DM_stories.pickle','rb') as f:
    story = pickle.load(f)
```
<br>
<br>
<br>
   

* 전체 갯수를 확인해 보니, 219506개가 맞네요.   


```python
print( len( story ) )
```

    219506
    

<br>
<br>
<br>


* Summarization하는 Code는 동일합니다.

<br>
<br>
<br>

```python
STEP = 5
count = 0

for idx in range(0,219506,STEP):
    short_story = story[idx : idx+STEP]
    
    df = pd.DataFrame(columns=["bert_result","GPT2","XLNet"])
    bert_result = []
    GPT2_result = []
    XLNet_result = []

    for i in range(0,STEP):
        s = short_story[i]
        s = s.replace("\n",'. ')
        s = s+'.'
        s = s[2:]
        
        bert_summary = ''.join(bert_model(s , ratio=0.01))
        bert_result.append( bert_summary )
        
        GPT2_model_sum = ''.join(GPT2_model(s, ratio=0.01))
        GPT2_result.append( GPT2_model_sum )
        
        XLNet_model_sum = ''.join(XLNet_model(s, ratio=0.01))
        XLNet_result.append( XLNet_model_sum )
        
        print(count)
        count += 1
        
    df['bert_result'] = bert_result
    df['GPT2'] = GPT2_result
    df['XLNet'] = XLNet_result

    filename = 'Result_{}_{}.csv'.format(idx,idx+STEP-1)
    df.to_csv( filename )
```

    0
    1
    2
    3
    4
    

<br>
<br>
<br>
<br>
<br>
<br>
   

* 요약문을 한 번 살펴볼까요?
   
   
* 먼저 원래 문장입니다. 메이웨더와 파퀴아오의 경기에 관한 내용이네요.   

<br>
<br>
<br>

```python
print( short_story[0] )
```
    
    sky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday
    the richest fight of all time will not come cheap either for sky sports or their subscribers even though sky are keeping faith with their core following by keeping the base price below
    it has taken what is described by industry insiders as very substantial for sky to fend off fierce competition from frank boxnation
    floyd mayweathers hotlyanticipated bout with manny pacquiao will be shown on sky sports
    pacquiao headed for the playground after working out in los angeles previously
    the price for the fight has been set at until midnight of friday may
    the cost will remain the same for those paying via remote control or online but will be if booked via phone after friday
    sky are flirting with their threshold of by charging a buy on their sports box office channel until midnight on may rising to on may the day of the fight in las vegas since they are understood to have broken past protocol by offering the us promoters a cut of that revenue as well as a hefty upfront payment it is expected they will have to shatter the payperview record in this country to break even
    the current sky record stands at buys for ricky vegas loss to mayweather in
    warren is believed to have offered a higher lump sum than sky in the hope of attracting another two million customers to his subscription channel
    it is doubtful if sky can reach that number at per sale at on a sunday morning but if they get buys they should be out of the red
    mayweather continued to work on the pads in his las vegas gym as he prepares for the fight
    pacquiao will take on mayweather at the mgm grand in las vegas on may in one of the biggest fights ever
    
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

* 각 요약문을 보면, BERT와 GPT2는 동일하게 요약한 반면, XLNet은 약간 더 긴 문장으로 요약했네요.   


```python
print(bert_result[0])
```

    sky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday.

<br>
<br>
<br>
<br>
<br>
<br>

```python
print(GPT2_result[0])
```

    sky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday.
    

   
<br>
<br>
<br>
   


```python
print(XLNet_result[0])
```

    sky have won the bidding war for the rights to screen floyd mayweather v manny pacquiao in the uk as revealed by sportsmail last friday. warren is believed to have offered a higher lump sum than sky in the hope of attracting another two million customers to his subscription channel.
    

<br>
<br>
<br>
   

* 내용의 대부분의 대동소이 하네요.
