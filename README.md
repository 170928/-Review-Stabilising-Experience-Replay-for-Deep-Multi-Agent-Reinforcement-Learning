# -Review-Stabilising-Experience-Replay-for-Deep-Multi-Agent-Reinforcement-Learning
[Review]
> Jakob Foerster, Nantas Nardelli,  Gregory Farquhar, Triantafyllos Afouras, Philip. H. S. Torr, Pushmeet Kohli, Shimon Whiteson

## [Motivation]
(1) 대부분의 real-world 문제는 multi-agent reinforcement learning으로 모델링 되어야 합니다.  
그러나, 기존에 존재하는 multi-agent RL 방법들은 문제에 따라서 확장성이 좋지 못한 상태입니다.  
그러므로, 현재의 성공적인 single-agent RL 기법을 multi-agent RL로 성공적인 확장을 이루는 것이 중요한 연구 분야로써 고려되어지고 있습니다.  

(2) 1993년 independent Q-learning (IQL) 의 기법이 발표되면서 multi-agent-RL에 대한 연구의 기반이 되었습니다.  
그러나, 이 논문에서는 각자 학습을 독립적으로 수행하는 agent들에 의해서 학습 과정에서의 environment가 non-stationary해지는 문제로 인해  
학습이 convergence 되지 못하는 문제에 대해서 거론하였습니다. non-stationary 문제로 인해 replay buffer를 사용할 수 없게 되었으며, 이는 DQN 과 같은 최근 사용되는 효과적인 기법을 적용하지 못하게 됨을 의미합니다.  

## [Methodology]
(1) importance sampling을 사용하여 오래된 data를 replay buffer에서 자연스럽게 사라지도록 합니다.  
(2) agent들의 value function을 기반으로 replay buffer로 부터 샘플링 된 데이터의 나이 (age) 를 모호하게 하는 fingerprint를 만듭니다.  

## [Related Work]
(1) Leibo et al., 2017  
이 연구에서는, non-stationary로 인해 발생하는 문제를 해결하기 위해서, replay buffer의 사용 기한을 짧게 만듭니다.  
(2) Foerster et al., 2016  
replay buffer 내의 데이터에서 오래된 데이터를 disable 합니다.  

## [Details]
(1) Replay memory에 있는 experience를 "off-environment data" 로써 해석합니다. (Ciosek & Whiteson, 2017)  
(2) Replay memory의 각 튜플을 해당 튜플에서 joint action의 확률로 보완하면, 해당 시간에 사용중인 정책에 따라 튜플을 나중에 샘플링하여 training 할 때 important sampling 보정을 계산할 수 있습니다.  
(3) Tesauro, 2003 "hyper Q-learning" 기법을 적용  
다른 agent들의 behaviour에서 해당 agent들의 policy를 추론합니다.  

## [Multi Agent Reinforcement Learning]
### [Environment]
이 논문에서는 다음과 같은 Setting을 통해서 multi-agent 환경을 다룹니다.  
(1) n개의 agents : ![image](https://user-images.githubusercontent.com/40893452/45855672-247c5800-bd8c-11e8-854d-385c0ea1a0e9.png)  
(2) Stochastic game : ![image](https://user-images.githubusercontent.com/40893452/45855688-352cce00-bd8c-11e8-9f02-568688f71e8a.png)  
(3) action & joint action : ![image](https://user-images.githubusercontent.com/40893452/45855711-61e0e580-bd8c-11e8-9245-373e49569970.png) & ![image](https://user-images.githubusercontent.com/40893452/45855721-6d341100-bd8c-11e8-9c33-761844a891fb.png)  
(4) state transition probability : ![image](https://user-images.githubusercontent.com/40893452/45855721-6d341100-bd8c-11e8-9c33-761844a891fb.png) ![image](https://user-images.githubusercontent.com/40893452/45855758-92c11a80-bd8c-11e8-8558-98c4855297c4.png)  
(5) shared reward function : ![image](https://user-images.githubusercontent.com/40893452/45855773-a3719080-bd8c-11e8-9eac-d83cd3125889.png)  
(6) observation : ![image](https://user-images.githubusercontent.com/40893452/45855789-bbe1ab00-bd8c-11e8-9518-04c95cc4d63c.png) ![image](https://user-images.githubusercontent.com/40893452/45855800-c7cd6d00-bd8c-11e8-99ed-1c20159ff7e7.png)   
(7) action-observation history : ![image](https://user-images.githubusercontent.com/40893452/45855865-1975f780-bd8d-11e8-8c29-7d716ce79eb5.png)  
(8) policy : ![image](https://user-images.githubusercontent.com/40893452/45855963-812c4280-bd8d-11e8-9656-d159750500c4.png)

### [Methods]
> IQL (Tan, 1993) 에서 각각의 agent는 (state, action) 에 대한 정보를 각자 기록합니다.   
이 논문에서의 환경은 "partially observable"한 환경이기 때문에, IQL은 action-observation history를 (partially observable, action) 정보를 기록합니다.   
그러므로, 각각의 agent가 수행하는 DQN의 기반 모델을 "recurrent neural network"로 구성합니다.   
여전히 생겨나는 문제는, neural network의 "experience buffer"를 사용한 접근법에서 "non-stationary" 문제가 발생합니다.  
> 이는 agent들이 생성한 replay memory내의 데이터가 생성되었을 때의 "dynamics" 가 current training time의 dynamics와 맞지 않다는 것입니다. 즉, current dynamics를 반영하지 못합니다.   
> 그래서, replay memory 때문에 발생하는 학습과정에서의 agent의 obsolete 문제를 해결하기 위해, replay memory가 없는 방법을 사용하기도 합니다.  
### [Multi-Agent Important Sampling]
이 논문에서는 multi agent setting을 위해서 "importance sampling"을 이용합니다.  

[Case 1 - full-observable case]
1. fully-observable multi-agent setting.  
2. other agent들의 policy를 모두 알 수 있다고 가정하는 경우.  

Bellman optimality equation을 다음과 같이 사용할 수 있습니다.   
![image](https://user-images.githubusercontent.com/40893452/45858330-e422d700-bd97-11e8-84b6-db423fccd0b9.png)  

위의 식에서 "non-stationary"의 문제가 있는 부분은 다음과 같습니다.   
![image](https://user-images.githubusercontent.com/40893452/45858423-47ad0480-bd98-11e8-9bb6-3643178f82e5.png)   
이 식은 시간이 지남에 따라 다른 agent들의 정책이 변하는 것으로 인해서 변하게 됩니다.  

그러므로, replay memory를 만드는 시점 tc에서 다음과 같은 정보로 기록합니다.  
![image](https://user-images.githubusercontent.com/40893452/45858459-7925d000-bd98-11e8-9728-559fedbf6329.png)  

그 후, replay time tr 때는 다음과 같은 "importance weighted loss function"을 최소화 시키는 "off-environment"를 훈련시킵니다.  
![image](https://user-images.githubusercontent.com/40893452/45858481-a1adca00-bd98-11e8-8717-d1b190448e83.png)

그러나, 위의 방법들은 "partially observable" setting에서는 observation & transition function 뿐만 아니라 agent들의 policy에 영향을 받는 복잡한 환경과 밀접하게 연관되는 observation history들에 의해 훨씬 더 복잡해집니다.   

[Case 2 - Partially Observable Case]
1. augmented state space를 다음과 같이 정의합니다.   
![image](https://user-images.githubusercontent.com/40893452/45858731-db330500-bd99-11e8-8785-08f1603fc0ce.png)  
이 state space에는  
(1) "original state" s   
(2) other agents의 action-observation history 
를 포함합니다.  

2. new observation function 은 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/45859150-c48dad80-bd9b-11e8-88dd-b5551740b9f1.png)

3. new reward function은 다음과 같이 정의됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45859165-d0796f80-bd9b-11e8-8de9-8b00ebc3d471.png)

4. new transition function은 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/45859177-e4bd6c80-bd9b-11e8-9257-e217e408aad4.png)

위의 새로운 정의들을 기반으로 다음과 같이 Bellman equation이 정의됩니다.  
![image](https://user-images.githubusercontent.com/40893452/45859210-0880b280-bd9c-11e8-8971-dcc9d3084998.png)

![image](https://user-images.githubusercontent.com/40893452/45859267-54cbf280-bd9c-11e8-9bcd-3b5d32a7807b.png)

위의 Bellman equation을 통해서, partially-observable case에서도 non-stationary term ![image](https://user-images.githubusercontent.com/40893452/45859281-6ca37680-bd9c-11e8-9919-fe146faafa1f.png)에 Bellman equation이 의존한다는 것을 알 수 있습니다.  

## [Multi-Agent Fingerprints]
importance sampling 기법은 true objective를 위해 "unbiased estimate"를 할 수 있게 해줍니다.  
그러나, 빈번하게 크고 (large) 경계 없는 (unbounded) variance를 만들어 냅니다.  
이를 해결하기 위해서, "truncating 혹은 adjust" 기법들이 연구되어 variance를 줄이는 데는 성공했지만 "bias"를 만들어 내는 현상이 있습니다.  

그러므로, 이 논문에서는 이 자체를 고치기 보다, multi-agent 의 non-stationary 문제를 포용할 수 있는 기법으로의 변형을 제시합니다.  

우선, 이 논문의 베이스가 되는 IQL 에 대해서 다시 살펴보겟습니다.  
Independent Q-Learning 기법의 약점은 other agent들을 환경 (environment)의 일부로 보는 것입니다.  
이로 인해서, 시간에 따라 변화 하는 other agent들의 policy들이 환경으로 여겨지기 때문에  
Q-function의 "non-stationary" 문제를 발생시키게 됩니다.  

즉, 다른 agent에게 environment로써 영향을 준다면 Q-function이 "stationary" 해야한다는 것을 의미합니다.  
> hyper-Q-Learning (Tesauro, 2003) 에서 방법을 제시하기도 했습니다.  

각각의 agent들의 state space는 "Bayesian inference"를 통해서 계산되어지고 추측되어 agent의 state space로 들어갑니다.  
직관적으로, 이 방법을 통해서 각각의 agent들의 학습을 standard single agent 문제로 간단하게 변화시키지만  
훨씬 더 큰 space 공간의 계산을 필요로 하게 됩니다.  

그러므로, hyper-Q-learning의 실질적인 어려움은 Q-function의 차원을 증가시킨다는 점입니다.  
이로인해 학습시에 하드웨어의 제약에 따라 어려움이 발생합니다.  
이 문제는 "deep learning" 기법을 통해서 가속화되고 있으며, 다른 agent들의 정책들도 high dimensional deep neural entwork들로 구성 됩니다.  
> 즉 다른 agent의 Q-function을 근사하는 neural network를 가지고 있다는 것입니다.  

그러나, 이때 새롭게 발생하는 문제는 other agent들이 자신의 Q-function을 neural network로써 근사시키고 있다면, 이를 추측하기 위한 neural network도 커져야 한다는 것입니다.   

그래서, "experinece replay" 방법을 안정화시키기 위해 각 agent들은 가능한 θ-a에서 조건을 지정할 필요가 없지만 "replay memory"에서 실제로 발생하는 θ-a의 값만 조정할 수 있다는 것이 중요한 방법 요소입니다.
"replay buffer"에서 데이터를 생성 한 정책 순서는 high dimensional policy space를 통해 단일 차원 trajectory를 따르는 것으로 생각할 수 있습니다.
"experinece replay" 을 안정화시키기 위해서는 각 agent들의 observation이 trajectory를 따라 현재 training 을 위한 sample이 어디서 발생했는지를 충분히 구별가능하게 해야합니다.  

이러한 정보를 포함할 수 있는 "low-dimensional fingerprint"로써 "iteration number e"가 좋은 대상입니다.  
그 뿐만 아니라, exploration rate epsilon을 포함하여 observation function을 다음과 같이 변환 시키는 것으로 "experience replay"를 효과적으로 수행 할 수 있게 해줍니다.  

![image](https://user-images.githubusercontent.com/40893452/45862620-81d4d100-bdad-11e8-8a26-9148c6f04d31.png)
![image](https://user-images.githubusercontent.com/40893452/45862611-77b2d280-bdad-11e8-8520-0a5129c4711a.png)

















