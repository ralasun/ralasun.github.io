---
layout : post
title: Model-Free Policy Control, Monte Carlo와 Temporal Difference에 대하여
category: Reinforcement Learning
tags: cs234 reinforcement-learning david-silver sutton
---

이번 포스팅은 지난 포스팅 [Model-Free Policy Evaluation](https://ralasun.github.io/reinforcement%20learning/2020/07/28/mc-td-eval/)에 이어 Model-Free Policy Control에 대해 다루도록 하겠습니다. CS234 4강, Deep Mind의 David Silver 강화학습 강의 5강, Richard S. Sutton 교재 Reinforcement Learning: An Introduction의 Chapter 5, 6 기반으로 작성하였습니다.

***

지난 포스팅에서는 일정 정책 $\pi$ 아래 환경 모델을 모를 때 가치함수를 추정하는 방법인 Monte-Carlo(MC) policy evaluation과 Temporal Difference(TD) policy evaluation에 대해 다뤘습니다. 그러나 sequential decision prcoess 문제의 최종 목표는 최적화된 정책을 갖는 것(Control)입니다. 환경 모델을 알 때 Dynamic Programming(DP)는 policy iteration과 value iteration을 통해 최적 정책을 구할 수 있습니다. 환경 모델을 모를 때 최적 정책을 찾는 방법 Model-Free Control에 대해 자세히 다루기 전에 먼저, Generalized Policy Iteration에 대해 알아보겠습니다.

<h2>Generalized Policy Iteration</h2>
DP에서의 policy iteration을 다시 자세히 살펴봅시다. 정책 발전(policy improvement)를 greedy하게 하였으며, policy evaluation과 policy improvement를 번갈아 반복하는 policy iteration을 통해 최적 가치함수와 최적 정책을 구했습니다. 

$$\pi_0 \overset E\to v_{\pi_0} \overset I\to \pi_1 \overset E\to v_{\pi_1} \overset I\to \pi_2 \overset E\to \cdots \overset I\to \pi_\ast \overset E\to v_\ast$$

<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89000330-0ee5fb00-d332-11ea-82a4-d700a773ada1.png'>
<figcaption align='center'>그림 1. Policy Iteration</figcaption></p>

<그림 1.>은 policy iteration을 그림으로 표현한 것입니다. 두 선은 각각 수렴된 가치함수와 정책들을 의미하고, 화살표는 policy evaluation과 policy improvement를 나타냅니다. 이 과정은 모두 결국 최적 정책과 최적 가치함수를 찾기 위한 것이기 때문에 두 선은 한 점에서 만납니다. 그런데, policy evaluation은 수렴할 때까지 시간이 오래 소요됩니다. 따라서, 위 가치함수 라인에 다다를 때까지 policy evaluation을 수행할 필요가 있을까요 ?

<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89000708-fd512300-d332-11ea-99f1-ebe65c2e5fc7.jpg'><figcaption align='center'>그림 2. Value Iteration</figcaption></p>

policy evaluation은 수렴할 때까지 시간이 오래 소요되기 때문에, 수렴할 때까지 기다리는 것이 아니라, 좀 더 효율적으로 접근하는 방법이 value iteration입니다. 가치함수를 한 스텝에 대해서만 업데이트를 하고, greedy policy improvement를 수행하는 value iteration을 통해 최적 가치함수와 정책을 찾았습니다.

위 두 방법 모두 결국 policy evaluation과정과 policy improvement과정의 상호작용으로 이뤄집니다. 두 과정 모두 안정화될 때, 즉 더 이상의 변화나 발전이 이뤄지지 않을 때, 그 때의 가치함수와 정책은 최적입니다. 따라서, 상호작용되는 과정이 조금씩은 차이가 있을 수 있지만 결국 둘의 상호작용으로 최적점에 다다르게 되는 것입니다. 이것이 바로 Generalized Policy Iteration(GPI)입니다.  
<p align='center'>
<img height='400' src='https://user-images.githubusercontent.com/37501153/89001031-d34c3080-d333-11ea-85cd-0893f0f52fa7.png'><figcaption align='center'>그림 3. Generalized Policy Iteration</figcaption></p>

Model-free control도 마찬가지로 GPI를 통해 최적 가치 함수와 최적 정책을 구합니다. Model-free control에 대해 알아보도록 하겠습니다. Model-free policy evaluation하는 방법으로 Monte-Carlo(MC)와 Temporal Difference(TD)가 있습니다. 마찬가지로, model-free control 하는 방법으로도 Monte-Carlo control와 Temporal-Difference control이 있습니다. 먼저, Monte-Carlo control부터 알아보겠습니다.

<h2>Monte-Carlo Control</h2>
지난 포스팅에서 알아본 monte-carlo estimation이 이제 control에 어떻게 사용되는지 생각해봅시다. 
 
<h3>Monte Carlo Estimation of Action Values</h3>
Monte-Carlo control도 Monte-Carlo estimation과 함께 GPI를 통해 최적정책을 찾아나갑니다. 그러나, DP에서 다른 점이 있습니다. DP는 현재 상태 $s$ 에서 행동 $a$ 를 취했을 때, 받을 수 있는 보상과 다음 상태가 어떻게 될지 알 수 있습니다. 따라서 다음 상태로 올 수 있는 모든 후보들과 보상을 고려하여 최대 가치를 반환하는 다음 상태를 찾은 후 그 상태로 가게 되는 행동을 취합니다. 즉, 상태 가치 함수 정보만으로 충분합니다. 

그러나 model-free 환경의 문제점은 직접 경험하지 않는 이상 다음 상태와 보상이 어떻게 될지 알 수 없습니다. 따라서 상태 가치 함수만으로 행동을 선택할 때 충분한 정보를 제공하지 못합니다. 이러한 이유로 model-free control에서는 상태 가치 함수 $v(s)$ 에 대한 evaluation이 아니라, <b>상태-행동 가치 함수 $q(s,a)$ 에 대한 evaluation</b>을 수행합니다. 상태 s에 대해 모든 행동 a에 대해 $q(s,a)$ 를 비교하여 가장 가치가 높은 행동을 선택하는 것이 상태 s에 대한 정책이 되는 것입니다.
<p align='center'>
<img width="171" alt="model-free-gpi" src="https://user-images.githubusercontent.com/37501153/89005037-76a24300-d33e-11ea-88ad-2f2cdb180929.png"><figcaption align='center'>그림 4. GPI with Q value</figcaption></p>

<h3>Importance of Exploration</h3>
GPI는 '좋은' 정책 $\pi$ 을 계속 찾아나가면 언젠간 최적 정책 $\pi_*$ 에 수렴합니다. 그러면 '좋은' 정책 $\pi$ 는 '좋은' $Q_\pi$ 추정치를 찾아야 합니다. 그래야지만 policy improvement를 통해 최적 정책을 찾아나갈 수 있기 때문입니다.

$$q_{\pi}(s, \pi'(s)) \geq v_\pi(s)$$

'좋은' $Q_\pi$ 추정치는 어떻게 찾을까요? 가능한 한 나올 수 있는 모든 $(s,a)$ 시퀀스를 경험하면 됩니다. MC policy evaluation에서 true expected value에 수렴하기 위해서 에피소드 샘플링을 많이 해야 하는 것과 같습니다. 
<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89006289-5922a880-d341-11ea-9bfc-1fdf7d99a074.jpg'><figcaption align='center'>그림 5. Greedy policy improvement in MDP and Model-Free</figcaption></p>

그러나 model-free 환경에서 모든 $(s,a)$ 쌍으로 구성된 모든 시퀀스를 경험하기는 어렵습니다. 경우의 수가 너무 많기 때문이고, 환경을 모르기 때문에 예측도 어렵습니다. <그림 5.>를 보면 DP같은 경우는 환경을 알기 때문에 V(s)를 추정하기 위해 다음에 나올 trasition model $P^{a}_{ss'}$ 와 함께 모든 상태 s를 고려할 수 있습니다. 따라서, '좋은' 추정치를 계산할 수 있습니다. 즉, 이렇게 찾아진 가치 함수 추정치 기반으로 greedy하게 행동을 선택해도 policy improvement가 일어납니다( [DP포스팅 policy improvement](https://ralasun.github.io/reinforcement%20learning/2020/07/13/dp/) 참고 ). 

반면에, model-free 같은 경우, MC와 TD모두 샘플링을 통해 (s,a)를 경험해 나갑니다. 그렇기 때문에 많은 (s,a)쌍을 방문하지 못하는 문제가 발생합니다. 이로 인해 어떤 (s,a)에 대해서는 좋은 추정치를 얻지 못합니다. 따라서 부정확한 추정치 기반으로 greedy하게 행동을 선택하는 건 심각한 문제를 일으킵니다. Q(s,a)를 추정하는 이유는 상태 s에 있을 때, 여러 행동 a들을 비교하기 위해서입니다. 그러나 어떤 행동 a에 대해서 Q(s,a)가 나쁜 값을 가지게 된다면 공정한 비교가 되지 않습니다. 즉, 학습이 제대로 이뤄지지 않게 되는 것입니다. 이 문제가 바로 <span style='color:red'>'exploration'</span> 문제입니다. 따라서 정책을 평가하기 위한 좋은 Q(s,a)를 구하기 위해선 충분하고 지속적인 탐험(continual exploration)이 보장되어야 합니다. 

충분하고 지속적인 탐험을 가장 심플하게 구현한 건 모든 행동들에 대해 선택할 가능성을 열어두는 것입니다. 이러한 방법 중 하나가 $\epsilon-greedy$ 입니다. 

$$ \begin{align*} \pi(a \mid s)&=m \underset a ax{\mathbb E[R_{t+1} = \gamma v_k(S_{t+1})|S_t=s, A_t=a]}\\&=m \underset a ax{\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]} \end{align*} $$
<blockquote>The simplest idea for ensuring continual exploration is that all actions are tried with non-zero probability.</blockquote>

$\epsilon-greedy$ 는 $\epsilon$ 의 확률로  행동을 랜덤하게 선택하고, $1-\epsilon$ 의 확률로 greedy한 행동을 선택합니다. $\frac{\epsilon}{m} + 1-\epsilon + \frac{\epsilon}{m}\times(m-1) = 1$ 이 되므로 $\epsilon-greedy$ 식을 아래와 같이 구축할 수 있습니다. 여전히 $\epsilon$ 의 확률로 탐험할 가능성을 두는 것입니다. 

$$\pi(a \mid s) = 
	\begin{cases}
		\frac{\epsilon}{m} + 1-\epsilon & \quad \text{if} \quad a^{*}=arg \underset m maxQ(s,a) \\
		\frac{\epsilon}{m} & \quad \text{otherwise}
		\end{cases}$$

<h3> on-policy Monte-Carlo Control </h3>
<i>이 단락에서 다루는 MC Control은 on-policy 기반입니다. on-policy와 off-policy의 차이는 off-policy MC Control에서 설명하도록 하겠습니다.</i>
<p align='center'><img width="835" alt="mcgpi" src="https://user-images.githubusercontent.com/37501153/89014690-d9043f00-d350-11ea-98b7-544f99f73981.png"><figcaption align='center'>그림 6. Monte-Carlo Policy Iteration</figcaption></p> 

진짜로 이제 MC기반의 Control에 대해 알아보겠습니다. 위에서 model-free인 경우도 Generalized Policy Iteration을 통해 최적 가치함수와 최적 정책을 찾는다고 하였습니다. MC기반 policy evaluation은 상태-행동 가치 함수인 $Q_\pi$ 를 찾는 것이고, MC기반 policy improvement는 $\epsilon-greedy$ 를 따릅니다. 그런데, <그림 6.>의 왼쪽 그림처럼, policy evaluation을 $Q_\pi$ 를 수많은 에피소드를 샘플링해서 수렴할 때 반복하는 건 너무 번거롭습니다. 따라서 DP의 value iteration처럼 에피소드 하나가 끝날 때까지만 상태-행동 가치 함수 Q를 업데이트하고, policy improvement를 수행합니다.

이런 방식의 MC 기반 GPI가 과연 최적 정책을 찾게 해주는지에 대해선 아직 해결해야 할 문제가 남았습니다. $\epsilon-greedy$ 방식이 진짜로 정책을 발전 시키는지에 대한 문제와 하나의 최정정책, 가치함수로의 수렴하는지에 대한 문제입니다. 먼저 첫번째 문제부터 살펴보겠습니다.

<b>$\epsilon-greedy$ Improvement</b><br>
$\epsilon-greedy$ 방식으로 정책을 발전시키려면 $V_{\pi_{i+1}}(s) \geq V_{\pi_{i}}(s)$ 를 만족해야 합니다. 아래는 이와 관련된 증명입니다.
<p align='center'>
<img src='https://user-images.githubusercontent.com/37501153/89119377-16033980-d4e9-11ea-8e30-c556dbf7263f.jpg'><figcaption align='center'>그림 7. $\epsilon-greedy$ policy improvement</figcaption></p>

$Q^{\pi_i}(s,\pi_{i+1}(s)) \geq V_\pi(s)$ 이므로, $V_{\pi_{i+1}}(s) \geq V_{\pi_{i}}(s)$ 가 성립합니다. 이렇게 되는 자세한 과정은 지난 [DP 포스팅 policy improvement](https://ralasun.github.io/reinforcement%20learning/2020/07/13/dp/)쪽을 참고 바랍니다. 따라서, $\epsilon-greedy$ 에 의한 정책 발전이 일어납니다.

<b>Greedy in the Limit of Infinite Exploration</b><br>
정책 발전이 일어나는 것과 동시에, 매 스텝마다 정책을 발전시키려면 결국 greedy한 정책으로 수렴해야 합니다. $\epsilon-greedy$ 방식은 모든 행동이 선택될 확률이 non-zero probability라 가정을 하지만, 결국 iteration을 반복해 나가면서 하나의 행동에 대해 $\pi(a \mid s) = 1$ 의 확률을 가져야 되는 것입니다. 이것에 관한 내용을 Greedy in the Limit of Infinite Exploration(GLIE) 라 합니다. 따라서, GLIE를 만족해야 수렴된 정책을 가질 수 있습니다.
<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89119773-c5411000-d4eb-11ea-9982-1904dd2dcfc3.jpg'><figcaption align='center'>그림 8. Greedy in the Limit with Infinite Exploration(GLIE)</figcaption></p>

$e-greedy$ 가 GLIE를 만족하게 하는 가장 심플한 방법은 $\epsilon = \frac{1}{k}$ 로 하여 매 스텝마다 $\epsilon$ 을 감소시켜 0에 수렴하게 하는 것입니다. 따라서, GLIE Monte-Carlo Control 알고리즘을 정리하면 아래와 같습니다.
<p align='center'><img width='500' src='https://user-images.githubusercontent.com/37501153/89120084-4699a200-d4ee-11ea-8bfe-9c2cfe05a53a.jpg'><figcaption align='center'>그림 9. On Policy Monte-Carlo Control</figcaption></p>

<h3>off-policy Monte-Carlo Control</h3>

<h4>Importance Sampling</h4>

<h2>Temporal-Difference Control </h2>



