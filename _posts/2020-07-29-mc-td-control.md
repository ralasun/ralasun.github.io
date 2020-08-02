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
<img width='500' src='https://user-images.githubusercontent.com/37501153/89119377-16033980-d4e9-11ea-8e30-c556dbf7263f.jpg'><figcaption align='center'>그림 7. $\epsilon-greedy$ policy improvement</figcaption></p>

$Q^{\pi_i}(s,\pi_{i+1}(s)) \geq V_\pi(s)$ 이므로, $V_{\pi_{i+1}}(s) \geq V_{\pi_{i}}(s)$ 가 성립합니다. 이렇게 되는 자세한 과정은 지난 [DP 포스팅 policy improvement](https://ralasun.github.io/reinforcement%20learning/2020/07/13/dp/)쪽을 참고 바랍니다. 따라서, $\epsilon-greedy$ 에 의한 정책 발전이 일어납니다.

<b>Greedy in the Limit of Infinite Exploration</b><br>
정책 발전이 일어나는 것과 동시에, 매 스텝마다 정책을 발전시키려면 결국 greedy한 정책으로 수렴해야 합니다. $\epsilon-greedy$ 방식은 모든 행동이 선택될 확률이 non-zero probability라 가정을 하지만, 결국 iteration을 반복해 나가면서 하나의 행동에 대해 $\pi(a \mid s) = 1$ 의 확률을 가져야 되는 것입니다. 이것에 관한 내용을 Greedy in the Limit of Infinite Exploration(GLIE) 라 합니다. 따라서, GLIE를 만족해야 수렴된 정책을 가질 수 있습니다.
<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89119773-c5411000-d4eb-11ea-9982-1904dd2dcfc3.jpg'><figcaption align='center'>그림 8. Greedy in the Limit with Infinite Exploration(GLIE)</figcaption></p>

$e-greedy$ 가 GLIE를 만족하게 하는 가장 심플한 방법은 $\epsilon = \frac{1}{k}$ 로 하여 매 스텝마다 $\epsilon$ 을 감소시켜 0에 수렴하게 하는 것입니다. 따라서, GLIE Monte-Carlo Control 알고리즘을 정리하면 아래와 같습니다.
<p align='center'><img width='500' src='https://user-images.githubusercontent.com/37501153/89120084-4699a200-d4ee-11ea-8bfe-9c2cfe05a53a.jpg'><figcaption align='center'>그림 9. On Policy Monte-Carlo Control</figcaption></p>

<h3>off-policy Monte-Carlo</h3>
Off-policy MC에 대해 설명하기 전에 on-policy learning과 off-policy learning에 대해 알아보겠습니다.

<b>on-policy vs. off-policy</b><br>
이제까지 설명한 MC control 방법은 on-policy control입니다. On-policy란 탐험할 때 따르는 정책과 찾고자 하는 최적 정책이 같은 경우입니다. $\epsilon-greedy$ MC control이 왜 on-policy인지 살펴보면 다음과 같습니다. 한 에피소드 내에서 매 상태 s마다 행동 a를 샘플링합니다. 이 때, $\epsilon$ 의 확률로 정책에 따른 행동 $a=arg \underset m maxQ(s,a)$ 을 샘플링합니다. 그리고 한 에피소드가 다 끝나고 정책을 업데이트 할 때도 이렇게 정책에 따른 행동들을 기반으로 가치함수를 업데이트하여 $\pi_k = \epsilon-greedy(Q)$ 로 정책을 발전시킵니다. 즉, <span style='color:red'>기존 행동 샘플링할 때 사용된 정책 기반으로 정책을 발전시키는 것</span>입니다. 이것이 바로 <span style='color:red'>on-policy learning</span>입니다.

<blockquote>On-policy learning is<br>- learn on the job<br>- learn about policy $\pi$ from experience sampled from $\pi$</blockquote>

그러나 이미 정책을 발전시키는 과정이 greedy한 정책을 한번 찾은 후, 그 정책 위에서 $\epsilon-greedy$ 같은 방법으로 탐험을 하는 것입니다. 그렇기 때문에 탐험을 하는 (s,a)공간이 매우 협소합니다. 이미 발전시킨 정책 위에서 탐험을 하기 때문에, (s,a) 공간위에서 보면 이미 발전시킨 정책 $\pi$ 에 해당되는 공간 근처에서만 탐험이 이뤄지는 것입니다. 마치 그림으로 표현하면 아래와 같을 수 있습니다. 
<p align='center'>
<img width='500' src='https://user-images.githubusercontent.com/37501153/89122084-78ffcb00-d4ff-11ea-8a2a-89d288f13a3a.jpg'><figcaption align='center'>그림 10. On-policy exploration</figcaption></p>
그렇다면 이를 해결할 수 있는 방법은 어떤 것이 있을까요? 바로, <span style='color:red'>탐험하는 정책과 최적 정책을 찾기 위해 학습하는 정책을 분리하는 것</span>입니다. 이것이 바로 <span style='color:red'>off-policy learning</span>입니다. 마치 분류 모델을 위한 지도학습을 진행 할 때, 모든 라벨에 해당되는 데이터가 존재하고, 분포도 고루 존재하면 학습이 더 잘되는 것과 비슷하다고 생각하면 됩니다. 좀 더 다양한 경험을 한 시퀀스 데이터가 많으면 best 답안에 가까운 정책을 찾을 수 있습니다. 그렇기 위해선 탐험의 범위가 넓어야 합니다. 
<blockquote>Off-policy learning is<br>- look over someone's shoulder<br>- learn about policy $\pi$ from experience sampled from $\mu$</blockquote>
일반적으로, 학습하고자 하는 정책을 target policy라 하고, 학습을 위한 데이터를 생성하기 위한 정책을 behavior정책이라 합니다.
학습을 위한 데이터를 학습하고 하는 정책에서 '벗어나서' 수집하기 때문에, 'off-policy'라 합니다.

<b>Importance Sampling</b><br>
대부분의 off-policy 방식은 importance sampling을 이용합니다. Importance sampling이란 기댓값을 계산하고자 하는 확률 분포 $p(x)$ 의 확률 밀도 함수(probability density function, PDF)를 모르거나 안다고 해도 $p$ 에서 샘플을 생성하기 어려울 때, 비교적 샘플을 생성하기 쉬운 $q(x)$ 에서 샘플을 생성하여 $p$ 의 기댓값을 계산하는 것입니다.

$$ \begin{align*} E_{x \sim p}[f(x)]&=\int p(x)f(x)dx\\&=\int \frac{p(x)}{q(x)}q(x)f(x)dx\\&=E_{x \sim q}[\frac{p(x)}{q(x)}f(x)] \end{align*} $$

<b>Importance Sampling for Off-Policy Monte-Carlo</b><br>
그렇다면 importance sampling을 off-policy MC에서 어떻게 이용하는지 알아보도록 하겠습니다. 위에서 설명한 importance sampling 개념대로 target policy와 behavior policy를 보면, 기댓값을 계산하고자 하는 확률 분포 $p(x)$ 에 해당하는 건 target policy $\mu$ 이고, 실제 샘플하는 분포 $q(x)$ 는 behavior policy $\pi$ 입니다. 우리가 계산하고자 하는 기댓값은 $V(s) = E[G_t \mid S_t=s]$ 이므로, 즉 두 분포의 비율 $\frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}$ 을 $G_t$ 에 곱하여 기댓값을 계산해야 합니다. 그러나, 단일 샘플링이 아니라 전체 에피소드에 대한 샘플링이기 때문에 importance sampling한 $G_t^{\pi/\mu}$ 는 아래와 같습니다.

$$G_t^{\pi/\mu} = \frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)}\frac{\pi(A_{t+1} \mid S_{t+1})}{\mu(A_{t+1} \mid S_{t+1})} \dots \frac{\pi(A_{T} \mid S_{T})}{\mu(A_{T} \mid S_{T})}G_t$$

따라서 MC policy evaluation에서 $G_t$ 가 아닌 $G_t^{\pi/\mu}$ 로 가치함수를 업데이트해주면 됩니다.

$$V(S_t) \leftarrow V(S_t) + \alpha(G_t^{\pi/\mu} - V(S_t))$$

그러나 importance sampling 같은 경우 infinite variance를 가지는 단점이 있습니다. 이러한 이유로 수렴하기가 매우 어렵습니다. 따라서 현실적으로 importance sampling을 통한 off-policy Monte-Carlo방식은 사용되지 않습니다. 

<h2>Temporal-Difference Control </h2>
다음은 Temporal-Difference(TD) Control에 대해 알아보겠습니다. On-policy TD control을 Sarsa이고, off-policy TD control을 Q-Learning이라 합니다. SARSA에 대해서 먼저 알아보겠습니다.

<h3>Sarsa : On-policy TD Control</h3>
TD control도 MC control과 마찬가지로, Generalized policy iteration(GPI) 을 따릅니다. Policy evaluation만 TD update을 이용하고 그 외 다른 건 모두 MC control와 같습니다.

$$Q(S,A) \leftarrows Q(S,A) + \alpha(R+\gamma Q(S',A') - Q(S,A))$$

위 update 식에서 샘플링 단위가 (S, A, R, S', A')이기 때문에 Sarsa 라는 이름이 붙여졌습니다. Srasa 알고리즘 전체는 아래와 같습니다.

<p align='center'><img width="7500" alt="sarsa" src="https://user-images.githubusercontent.com/37501153/89125122-19adb500-d517-11ea-80e7-ed18b5449e7a.png"><figcaption align='center'>그림 11. Sarsa</figcaption></p>

Sarsa도 on-policy MC에서 살펴본 것처럼 정책 발전 문제와 수렴 문제를 살펴보겠습니다. 정책 발전 문제는 on-policy MC와 동일하게 $\epsilon-greedy$ 를 사용하기 때문에 정책 발전은 일어납니다. 

반면에 수렴문제는 GLIE를 만족시키는 것 이외에 업데이트 스텝 크기인 $\alpha$ 에 대한 조건이 더 필요합니다. 그 이유는 MC와 다르게 TD는 스텝마다 업데이트가 일어나는 on-line 방식이기 때문에 스텝사이즈 크기에 따라 수렴이 되지 않고 발산이 될 수 있습니다. 스텝크기 $\alpha$ 는 Q 가치함수가 변화가 일어나야 하므로 충분히 크며 동시에 Q 가치함수가 수렴해야 하므로 충분히 작아야 합니다.

$$\sum_{t=1}^{\infty}\alpha_t=\infty$$

$$\sum_{t=1}^{\infty}\alpha^2_t<\infty$$

그러나 실제 문제를 풀 때 $\alpha$ 를 결정하는 건 위의 이론을 이용하진 않고 domain-specific하게 또는 실험적으로 정한다고 합니다. 

sarsa 문제 예를 보겠습니다. 아래는 S에서 시작해서 G로 가야하는 문제입니다. 행동은 위, 아래, 좌, 우이며, 화살표가 있는 곳에서 아래에서 위로 바람이 불고 있습니다. 따라서 이 곳을 지날 때 실제 위로 가는 행동을 해도 실제 움직임은 대각선 우상향으로 가게 됩니다. 매 스텝마다 보상은 -1이며 discount factor 1입니다. 
<p align='center'>
<img width="490" alt="sarsa-example" src="https://user-images.githubusercontent.com/37501153/89125501-ed476800-d519-11ea-89ef-24c2ee66c8dc.png"><figcaption align='center'>그림 12. Sarsa on the Windy Gridworld</figcaption></p>
<그림 12.> 그래프는 Sarsa 학습 결과입니다. 1에피소드가 끝날 때 까지 2000 스텝을 밟지만 그 다음부터 학습속도가 빨라지는 것을 볼 수 있습니다. 

<h3>Q-Learning : Off-policy TD Control</h3>
다음은 off-policy TD 방식인 Q-Learning에 대해서 알아봅시다. Off-policy MC와 다르게 importance sampling이 필요 없습니다. Sarsa 에서 $Q(S_t, A_t)$ 를 업데이트 하기 위해, 정책 $\pi$ 에 따라 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ 을 샘플링 한 후, $Q(S_{t+1}, A_{t+1})$ 기반으로 현재 상태 $(S_t, A_t)$ 를 수정했습니다. 즉, 샘플링된 정책과 학습하는 정책이 일치합니다. 그러나 Q-Learning은 off-policy로 샘플링되는 정책(behavior policy)과 학습하는 정책(target policy)이 다릅니다. 

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1}+\gamma Q(S_{t+1},A')-Q(S_t, A_t))$$ 

- Next action is chosen using behavior policy $A_{t+1} \sim \mu(\cdot \mid S_t)$
- But we consider alternative successor action $A' \sim \pi(\cdot \mid S_t)$ 

다음 상태에 대한 행동을 behavior policy에 따라 선택하지만 실제 $Q(S_t, A_t)$ 를 업데이트하기 위해서 다음 상태에 대한 행동은 behavior policy와 다른 target policy에 의해 선택합니다. 즉, $A'$ 에 대해서 $Q(S_t, A_t)$ 를 업데이트하고 다음 업데이트 할 (s,a) 쌍은 $(S_{t+1}, A')$ 이 아니라 behavior policy 따른 $(S_{t+1}, A_{t+1})$ 인 것입니다. 이 때, $A'=A$ 일수도 있고, $A' \ne A$ 일수도 있습니다.

Q-learning은 taret policy와 behavior policy를 같이 발전시켜 나갑니다. 이때, target policy $\mu$ 는 $Q(s,a)$ 에 관한 greedy policy이고 behavior policy $\pi$ 는 $Q(s,a)$ 에 관한 $\epsilon-greedy$ 입니다. 

- Target policy : $\pi(S_{t+1}) = arg \underset {a'} maxQ(S_{t+1},a')$
- Behavior policy : <br>
$$\mu(a \mid s) = 
	\begin{cases}
		\frac{\epsilon}{m} + 1-\epsilon & \quad \text{if} \quad a^{*}=arg \underset m maxQ(s,a) \\
		\frac{\epsilon}{m} & \quad \text{otherwise}
		\end{cases}$$

위의 behavior policy와 target policy를 가지고 Q-Learning 식을 다시 쓰면 아래와 같습니다.

$$ \begin{align*} Q(S_t,A_t) &\leftarrow Q(S_t, A_t) + \gamma Q(S_{t+1},A')\\&\leftarrow Q(S_t, A_t) + \gamma Q(S_{t+1},arg \underset {a'} maxQ(S_{t+1},a')\\&\leftarrow Q(S_t, A_t) + \gamma m \underset {a'} ax Q(S_{t+1}, a') \end{align*} $$

따라서, Q-learning 알고리즘은 아래와 같습니다. 
<p align='center'><img width="500" alt="qlearning-alg" src="https://user-images.githubusercontent.com/37501153/89126483-ecfe9b00-d520-11ea-8264-347a78f9b7d5.png"><figcaption align='center'>그림 13. Q-Learning</figcaption></p>

<b>Sarsa vs. Q-Learning</b><br>
그렇다면 Sarsa와 Q-Learning은 실제 학습 시 어떤 차이를 보일까요 ? 아래 cliff Waling 예제를 통해 확인해 보겠습니다. 
<p align='center'><img width="500" alt="q-vs-sarsa" src="https://user-images.githubusercontent.com/37501153/89126761-410a7f00-d523-11ea-8613-7dac00ed99d1.png"><figcaption align='center'>그림 14. Cliff-walking 예</figcaption></p>

S에서 시작해서 G로 가는 문제입니다. The Cliff에 도달하면 R=-100을 받고 다시 S로 돌아갑니다. 다른 곳에 밟으면 R=-1을 받습니다. 행동은 위, 아래, 좌, 우이며, $\epsilon=0.1$ 입니다. 학습이 완료됐을 때 최적 정책 결과는 <그림 14.>에서 위에 있는 그림입니다. Sarsa는 safe path로 학습되지만 Q-learning은 optimal path로 학습됩니다. 

Sarsa는 학습되는 방향이 현재 따르는 정책에서 선택된 행동이 고려되기 때문에 path의 길이는 길지만 좀 더 안전한 길을 선택하게 됩니다. 반면에, Q-Learning 같은 경우 학습되는 방향이 현재 따르는 정책과 무관하기 때문에 현재 상태에서 가장 최적의 선택이 될 수 있는 길로 학습이 됩니다. 그렇기 때문에 Sarsa 같은 경우 보상의 합이 Q-Learning보다 크게 나타납니다. 

***

이상으로 이번 포스팅을 마치겠습니다. 읽어주셔서 감사합니다.





***

1. [Importance Sampling, https://untitledtblog.tistory.com/135](https://untitledtblog.tistory.com/135)



