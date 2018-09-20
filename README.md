# -Review-Stabilising-Experience-Replay-for-Deep-Multi-Agent-Reinforcement-Learning
[Review]
> Jakob Foerster, Nantas Nardelli,  Gregory Farquhar, Triantafyllos Afouras, Philip. H. S. Torr, Pushmeet Kohli, Shimon Whiteson

### [Motivation]
(1) 대부분의 real-world 문제는 multi-agent reinforcement learning으로 모델링 되어야 합니다.  
그러나, 기존에 존재하는 multi-agent RL 방법들은 문제에 따라서 확장성이 좋지 못한 상태입니다.  
그러므로, 현재의 성공적인 single-agent RL 기법을 multi-agent RL로 성공적인 확장을 이루는 것이 중요한 연구 분야로써 고려되어지고 있습니다.  

(2) 1993년 independent Q-learning (IQL) 의 기법이 발표되면서 multi-agent-RL에 대한 연구의 기반이 되었습니다.  
그러나, 이 논문에서는 각자 학습을 독립적으로 수행하는 agent들에 의해서 학습 과정에서의 environment가 non-stationary해지는 문제로 인해  
학습이 convergence 되지 못하는 문제에 대해서 거론하였습니다. non-stationary 문제로 인해 replay buffer를 사용할 수 없게 되었으며, 이는 DQN 과 같은 최근 사용되는 효과적인 기법을 적용하지 못하게 됨을 의미합니다.  

### [Methodology]
(1) importance sampling을 사용하여 오래된 data를 replay buffer에서 자연스럽게 사라지도록 합니다.  
(2) agent들의 value function을 기반으로 replay buffer로 부터 샘플링 된 데이터의 나이 (age) 를 모호하게 하는 fingerprint를 만듭니다.  

### [Related Work]
(1) Leibo et al., 2017  
이 연구에서는, non-stationary로 인해 발생하는 문제를 해결하기 위해서, replay buffer의 사용 기한을 짧게 만듭니다.  
(2) Foerster et al., 2016  
replay buffer 내의 데이터에서 오래된 데이터를 disable 합니다.  

