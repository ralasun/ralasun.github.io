---
layout : post
title: General Approach to Time Series Analysis - Time Series Data, Stationarity 등에 대하여
category: Time Series Analysis
tags: time-series-analysis
---

이번 포스팅을 시작으로, 시계열 분석에 대해서 다루도록 하겠습니다. 메인 교재는 Brockwell와 Richard A. Davis의 \< Introduction to Time Series and Forecasting \> 와 패스트캠퍼스의 \<파이썬을 활용한 시계열 분석 A\-Z\> 를 듣고 정리하였습니다. 

---

<h2>1.1. What is Time Series?</h2>

시계열이란, 일정 시간 간격으로 배치된 데이터들의 수열입니다. 그 중에서 discrete한 시계열이란 관측이 발생한 시각 t의 집합 $T_0$ 가 discrete한 경우이며, 관측치가 시간 구간 안에서 연속적으로 발생한다면 continuous한 시계열입니다.

시계열 시퀀스는 일반적으로 자기 상관성이 있는 수열입니다. 즉, 과거의 데이터가 현재를 넘어서 미래까지 영향을 미치는 것을 뜻합니다.

$$Cov(X_i, X_j) \neq 0$$

따라서, 시계열 데이터로 모델링을 하기 위해선 먼저 데이터를 최대한 분해해서 살펴봐야 합니다. 확률 모델링을 하기 위해선 i.i.d 여야 하기 때문입니다. 일반적으로 시계열 데이터는 trend, seasonality, noise 항으로 구성되어 있습니다. 여기서 시계열 데이터가 자기 상관성을 가지게 되는 요인은 trend와 seasonality 요소 때문이고, noise는 i.i.d한 독립변수로 구성된 에러항입니다.

<h2>1.2. Objectives of Time Series Analysis</h2>
시계열 분석의 목적은 주로 시계열 데이터를 보고 앞으로 일어날 일들을 예측하는 것입니다. 그러기 위에선 기존에 있는 시계열 데이터를 가지고 추론을 해야합니다. 따라서, 이러한 추론을 하기 위해선 가정에 맞는 적절한 확률 모델을 선택하여 모델링을 진행해야 합니다. 

그러나, 시계열 데이터는 자기 상관성이 존재하는 데이터입니다. 따라서 확률적 모델링을 통해 이 시계열 데이터를 서로 독립인 데이터로 변환해야 하는데 이 과정이 seasonal adjustment 또는 trend and seasonal decomposition입니다. 그 밖에 log transformation, differencing 같은 과정도 존재합니다. 

어쨌든, 시계열 분석의 궁극적인 목표는 독립적인 변수로 최대한 변환한 뒤, 이를 기반으로 확률적 통계 모델링을 해서, inference를 하는 것입니다. Inference 결과는 다시 우리가 얻고자 하는 예측값으로 바꾸기 위한 reverting 과정을 거쳐야 합니다. 왜냐하면, seasonal adjustment나 Decomposition을 통해 상관성을 제거했기 때문에 원하는 예측값을 얻기 위해선 다시 원래대로 이 과정을 뒤집어서 돌아가야 하는 것입니다. 


<h2>1.3. Some Simple Time Series Models</h2>
위에서 말씀 드린 것과 같이 시계열 데이터를 보고 적절한 확률 모델을 선택하는 것은 매우 중요합니다. 따라서, 몇가지 간단한 time series model을 소개하겠습니다.

<h3>1.3.1. Definition of a time series model</h3>
관측된 $\{x_t\}$ 에 대한 time-series 모델이란, 랜덤 변수 $\{X_t\}$ 시퀀스들의 joint distribution 을 의미합니다. 

> A time series model for the observed data $\{x_t\}$ is a specification of the joint distributions(or possibly only the means and covariances) of a sequence of random variables $\{X_t\}$ of which $\{x_t\}$ is postulated to be a realization.

즉, 랜덤 변수들의 시퀀스 $\{X_1, X_2, \dots \}$ 로 구성된 time-series 확률 모델은 랜덤 벡터 $(X_1, \dots, x_n)' ,\,\, n=1,2,\dots,$ 의 결합 분포입니다. 아래 그림은 랜던 변수들의 시퀀스 $\{S_t, t=1, \dots, 200\}$ 로 나올 수 있는 가능성 중 한가지가 '실현' 된 것입니다. 

<p align='center'><img src='https://imgur.com/izJrvZl.png'><figcaption align='center'>그림 1. Time-series 예시</figcaption></p>

<h3>1.3.2. Some Simple Time Series Model</h3>

<ol><li>iid Noise<br>
가장 기본적인 time series 모델은 noise항으로만 이뤄진 경우입니다.(거의 현실세계에선 없다고 생각하시면 됩니다.)</li>
<li>Binary Process<br>
i.i.d Noise의 종류로, binary 분포를 따르는 noise인 경우입니다. 랜덤 변수들의 시퀀스 $\{X_t,\,\,t=1,2,\dots,\}$ 가 $P[X_t = 1]=p$ , $P[X_t = -1] = 1-p$ 를 따릅니다.</li>
<li>Models with only Trend<br>
trend요소와 noise항만 있는 경우입니다. 여기서 trend요소란 패턴이 선형관계를 가지고 있을 때입니다. 자세히 말하면, 시계열이 시간에 따라 증가, 감소, 또는 일정 수준을 유지하는 경우입니다.

$$X_t = m_t + Y_t$$

<p align='center'><img src='https://imgur.com/AjlrE80.png'><figcaption align='center'>그림 2. Time series with only trend component</figcaption></p></li>

<li>Models with only Seaonality<br>
seasonal요소와 noise항만 있는 경우입니다. 여기서 seasonal요소란 일정한 빈도로 주기적으로 반복되는 패턴을 말합니다. 반면에, 일정하지 않은 빈도로 발생하는 패턴은 Cycle이라 합니다.(여기서는 seasonal 기준으로 설명하겠습니다.)

$$X_t = S_t + Y_t$$</li>

<p align='center'><img src='https://imgur.com/6NZcDiO.png'><figcaption align='center'>그림 3. Times series with only seasonality(period=12month)</figcaption></p>
</ol>

<h3>1.3.3 A General Approach to Time Series Modeling</h3>
시계열 분석에 대해 깊게 들어가기 전에, 시계열 데이터 모델링하는 방법에 대해 대략적으로 알아봅시다.

1) 그래프로 그린 후, 그래프 상에서 아래와 같은 요소가 있는지 체크한다.(Plot the series and examine the main features of the graph)<br> 
<ul><li>trend</li><li>a seasonal component</li><li>any apparent sharp changes in behavior</li><li>any outlying observations</li></ul>

2) 정상상태의 잔차를 얻기 위해, trend와 seasonality 요소를 제거한다. (Remove the trend and seasonal components to get stationary residuals)<br>
&nbsp;&nbsp;&nbsp;&nbsp; trend와 seasonality 요소를 제거하기 전에, 전처리를 해야하는 경우가 있습니다. 예를 들어, 아래와 같이 지수적으로 증가하는 경우에, 로그를 취해서 variance가 일정하도록 만든 후 모델링을 하면 정확도를 높일 수 있습니다.

<p align='center'><img src='https://imgur.com/V85l07h.png'><figcaption align='center'>그림 1. 로그 취하기 전</figcaption></p>
<p align='center'><img src='https://imgur.com/e0GKRKU.png'><figcaption align='center'>그림 2. 로그 취한 후</figcaption></p>

이외에도 여러 방법이 있습니다. 추후에 설명하도록 하겠습니다. 어쨌든, 이 모든 방법들의 핵심은 <b>정상상태의 잔차</b>를 만드는 것입니다.

3) auto-correlation 함수, 여러 다양한 통계량을 이용하여 잔차를 핏팅할 모델을 선택한다. (Choose a model to fit the residuals, making use of various sample statistics including the sample autocorrelation function)

4) 핏팅된 모델로 예측한다.<br> 
&nbsp;&nbsp;&nbsp;&nbsp; 여기서 잔차를 예측하는 것이고, 예측된 잔차를 원래 예측해야 할 값으로 변환한다.

<h2>1.4. Stationary Models and the Autocorrelation Function</h2>

시계열 데이터가 정상상태(stationarity)를 가지기 위해서, 시계열이 확률적인 특징이 시간이 지남에 따라 변하지 않는다는 가정을 충족시켜야 합니다. 그러나 시계열 데이터는 trend와 seasonality요소로 인해, 평균과 분산이 변할 수 있습니다.

> a time series ${\{X_t, t=0, \pm1, ...\}}$ is said to be stationary if it has statistical properties similar to those of the "time-shifted" series ${\{X_{t+h}, t=0, \pm1, ...\}}$ for each integer h.

> Trends can result in a varying mean over time, whereas seasonality can result in a changing variance over time, both which define a time series as being non-stationary. Stationary datasets are those that have a stable mean and variance, and are in turn much easier to model.

시계열에 대한 평균과 공분산은 아래와 같이 정의됩니다.
<p align='center'><img src='https://imgur.com/65biJ1q.png'><figcaption align='center'>그림 3. 시계열의 평균과 공분산</figcaption></p>

<h4>Strict Stationarity vs. Weak Stationarity</h4>
엄격한 정상상태가 되려면,  $(X_1,\dots , X_n)$ 의 결합분포와 $(X_{1+h}, \dots, X_{n+h})$ 의 결합분포가 시간간격 h에 상관없이 동일해야 합니다. 그러나 이를 이론적으로 증명하기 어렵기 때문에, 약한 정상상태(weak stationarity)만을 만족하면 정상상태에 있다고 생각하고 시계열 문제를 풉니다. 약한 정상상태는 아래 조건을 만족합니다. 즉, 결합분포가 동일해야 한다는 강력한 조건이 사라졌기 때문에 약한 정상상태라고 하는 것입니다. 

$$(1) \,\, E(X_t) = u$$
$$(2) \,\, Cov(X_{t+h}, X_{t}) = \gamma_h, for\; all\; h$$
$$(3)\,\, Var(X_t) = Var(X_{t+h})$$

(2)식은 공분산은 t에 독립임을 의미합니다. 정상상태 시계열의 공분산은 아래와 같이 하나의 변수 h에 대해 나타낼 수 있습니다.

$$\gamma_X(h) = \gamma_X(h,0) = \gamma_X(t+h, t)$$

이때 함수 $\gamma_X(\cdot)$ 을 lag h에 대한 auto-covariance 함수(ACVF)라 합니다. auto-correlation 함수(ACF)는 ACVF를 이용해 아래와 같이 정의됩니다.

$$\rho_X(h)=\frac{\gamma_X(h)}{\gamma_X(0)}=Cor(X_{t+h}, X_t)$$

<h4>White Noise</h4>
시계열 ${\{X_t}\}$ 가 독립적인 랜덤 변수의 시퀀스이고, 평균이 0이고, 분산이 $\sigma^2$ 이면, White Noise라 합니다. 아래는 White Noise의 조건입니다.

- $$E(X_t)=0$$
- $$V(X_t)=V(X_{t+h})=\sigma^2$$
- $$\gamma_X(t+h, t)=0\;(h\neq0)$$

<h3>1.4.1 The Sample Autocorrelation Function</h3>

관측 데이터 가지고 자기 상관의 정도를 볼때, sample auto-correlation 함수(sample ACF)를 사용합니다. Sample ACF는 ACF의 추정으로, 계산은 아래와 같습니다.
<p align='center'><img src='https://imgur.com/jjzBk3z.png'><figcaption align='center'>그림 4. Sample ACF</figcaption></p>

White Noise인 경우, 시계열 그래프와 ACF 그래프는 아래와 같습니다. lag가 1이상인 경우, 거의 ACF값이 0에 가까운 것을 볼 수 있고, 95% 신뢰구간 안에 들어와 있습니다. 

<p align='center'><img src='https://imgur.com/RaoTZJj.png'><figcaption align='center'>그림 5. White Noise ACF</figcaption></p>

아래는 그림 1. 그래프에 플롯된 데이터를 가지고 그린 ACF입니다. 보시면, ACF가 lag가 커짐에 따라 서서히 감소하는 형태를 띄는데 이는 trend가 있는 데이터에서 나타납니다. 
<p align='center'><img src='https://imgur.com/N6vk5oN.png'><figcaption align='center'>그림 6. Sequence with trend ACF</figcaption></p>

<h2>1.5. Estimation and Elimination of Trend and Seasonal Components</h2>


trend와 seasonality가 존재하는 시계열의 모델링인 경우, 아래와 같이 additive 형태를 띌 수 있습니다.

$$X_t = m_t + s_t + Y_t$$

시계열 모델링의 최종 목표는 잔차항 $Y_t$ 가 정상상태에 놓이게 하는 것입니다. 따라서 잔차항을 분석하기 위해서 trend 요소 $m_t$ 와 seasonal 요소 $s_t$ 를 제거해야 합니다.

<h3>1.5.1. Estimation and Elimination of Trend in the Absence of Seasonality</h3>
seasonal 요소가 없고, trend요소만 있는 모델링은 아래와 같이 진행할 수 있습니다. 

$$X_t = m_t + Y_t, \quad t=1, \dots ,n, \; where \; EY_t = 0$$

<h4>method1. Trend Estimation</h4>

trend 요소를 추정하는 방법은 Moving Average와 Smoothing을 이용하는 방법 2가지가 있습니다.

<h5>a) Smoothing with a finite moving average filter</h5>

과거 n개의 시점을 평균을 구해 다음 시점을 예측하는 방식입니다. 

$$W_t = (2q+1)^{-1}\sum_{j=-q}^{q}X_{t-j}$$

이때, $X_t = m_t + Y_t$ 이므로, 아래와 같은 식으로 유도됩니다.

$$W_t = (2q+1)^{-1}\sum_{j=-q}^{q}X_{t-j} = (2q+1)^{-1}\sum_{j=-q}^{q}m_{t-j} + (2q+1)^{-1}\sum_{j=-q}^{q}Y_{t-j}$$

만약에 $m_t$ 가 대략 선형관계를 띄고 있다면 잔차항의 평균은 0에 가까울 것입니다. 즉, 트렌드가 선형관계를 띄고 있을 때, moving average filter를 씌어주면 trend요소만 추출할 수 있는 것을 의미합니다.

$$W_t = (2q+1)^{-1}\sum_{j=-q}^{q}X_{t-j} = (2q+1)^{-1}\sum_{j=-q}^{q}m_{t-j} + (2q+1)^{-1}\sum_{j=-q}^{q}Y_{t-j} \approx m_t$$	 

<p align='center'><img src='https://imgur.com/rEHZBt2.png'><figcaption align='center'>그림 7. Moving average filter 취하기 전</figcaption></p>

<p align='center'><img src='https://imgur.com/QPByqUu.png'><figcaption align='center'>그림 8. Moving average filter 취한 후</figcaption></p>

<p align='center'><img src='https://imgur.com/dPTzLn3.png'><figcaption align='center'>그림 9. Trend 제거 후 잔차항</figcaption>

위에 그림 7,8,9 를 살펴 봅시다. 그림 8은 그림 7에서 과거시점 5개를 이용하여 moving average 필터를 씌운 후입니다. 뚜렷한 트렌드가 있지 않음을 보실 수 있습니다. ~~잔차항에 대한 분석은 다시 한번 살펴봐야 할 것 같습니다.~~

Simple Moving Average Filter는 trend가 linear하고, Noise가 White Noise일 때, 시계열 데이터에서 trend요소를 잘 추출할 수 있습니다. 그러나 Non-linear한 trend라면, Noise가 White Noise라 하더라도, trend 추정이 올바르지 않습니다. 그럴 땐, 적절한 가중치를 부여하여 Moving Average Filter를 씌워야 합니다.

$$\sum_{j=-7}^{j=7}a_jX_{t-j} = \sum_{j=-7}^{j=7}a_j m_{t-j}+\sum_{j=-7}^{j=7}a_jY_{t-j} \approx \sum_{j=-7}^{j=7}a_j m_{t-j} = m_t$$

<h5>b) Exponential smoothing</h5>
Moving averages는 과거 n개의 시점에 동일한 가중치를 부여하는 방법입니다. 그러나, 현재시점과 가까울수록 좀 더 현재시점에 영향을 많이 미치는 경우가 일반적으로 생각하기엔 자연스러울 수 있습니다. 예로 주식을 생각하면 될 것 같습니다. 따라서, Exponential smoothing 방법은 현재 시점에 가까울수록 더 큰 가중치를 주는 방법입니다. 

<p align='center'><img src='https://imgur.com/ciknR6Y.png'><figcaption align='center'>그림 10. Exponential Smoothing</figcaption></p>

Exponential Smoothing 수식은 아래와 같습니다.

$$\hat{m}_t = \alpha X_t + (1-\alpha)\hat{m}_{t-1},\,\,t=2, \dots, n,$$
$$\hat{m}_1=X_1$$

아래 그림은 그림 7을 exponential smoothing을 취한 trend 추정 그래프입니다.
<p align='center'><img src='https://imgur.com/hKOWuWu.png'><figcaption align='center'>그림 11. Exponential Smoothing 취한 후</figcaption></p>

<h5>c) Smoothing by elimination of high-frequency component</h5>
trend를 추출하는 방법 중 하나로, 여러 frequency의 합으로 trend를 표현해서 이를 제거하는 것입니다(이 부분은 추후에 4장에 가서 다시 설명하도록 하겠습니다).

<p align='center'><img src='https://imgur.com/hn90Hgr.png'><figcaption align='center'>그림 12. frequency합으로 smoothing을 취한 후( $\alpha=0.4$ )</figcaption></p>

<h5>d) Polynomial fitting</h5>
$m_t = a_0 + a_1t + a_2t^2 + \dots + a_nt^n$ 으로 모델링하여, $\sum_{t=1}^n(x_t-m_t)^2$ 을 최소화하는 방식으로 파라미터 $a_k,\,(k=0, \dots, k=n$ 을 구하는 방식으로 trend를 추정할 수 있습니다. 

<del>$X_t - Y_t = m_t$ 에서, $Y_t$ 는 stationary state을 가정하고 있기 때문에, polynomial model을 구축할 수 있는 것입니다.</del>

<h4>method2. Trend Elimination by Differencing</h4>
method1 방법은 trend를 추정한 뒤, 시계열 $\{X_t\}$ 에서 빼주는 방식으로 trend를 제거하였습니다. 이번엔 difference(차분)를 통해서 trend요소를 제거하는 방법을 알아보도록 하겠습니다. Lag-1 difference operator $\bigtriangledown$ 는 아래와 같습니다.

$$\bigtriangledown X_t = X_t-X_{t-1} = (1-B)X_t$$

B는 backward-shift operator로 $BX_t = X{t-1}$ 입니다. j lag difference는 $\bigtriangledown (X_t) = \bigtriangledown (\bigtriangledown^{j-1} (X_t))$ 입니다. 예를 들어, 2-lag difference는 아래와 같습니다.

$$ \begin{align*} \bigtriangledown^2 X_t&=\bigtriangledown (\bigtriangledown (X_t))=\bigtriangledown ((\bigtriangledown (X_t))\\&=(1-B)(1-B)X_t=(1-2B+B^2)X_t = X_t - 2X_{t-1} + X_{t-2}\end{align*} $$

<h5>Why difference helps eliminating trend components? (Maybe or seasonal components)</h5>
여기서, 제가 공부하면서 궁금했던 포인트는 왜 difference가 trend 제거에 도움이 되는가? 였습니다. 제가 생각한 답은 아래와 같습니다. trend와 seasonal 요소를 제거하려는 이유는 '고정된 평균과 분산을 가지는 분포'를 가지기 위해서입니다. 그래야지 통계적 모델링이 가능하기 때문입니다. 즉 반대로 말하면, trend와 seasonal 요소는 시간에 따라 평균과 분산이 변함을 의미합니다. 즉 그 변하는 요소를 제거하기 위해서 difference를 하는 것입니다. 

difference를 통해서 변동성을 제거하는 건 고등학교 수학 때 배웠던 미분을 통해 이해할 수 있습니다. 예를 들어, 일차함수 $y=a+bx$ 는 x값에 따라 y값이 변합니다. 그러나 일차미분을 통해 구한 기울기 b값은 고정이 됩니다. 반면에 이차함수 $y=ax^2 + bx + c$ 는 이차미분을 통해 2a라는 고정값을 갖게 됩니다. 여기서 미분 과정을 difference라 생각하시면 됩니다.

> 영어로도 미분이 differentiation 임을 생각하면 와닿습니다.

일차함수 y는 변하는 특성 + 고정된 특성을 둘다 가지고 있는데 일차 미분을 통해 a라는 고정된 특성만을 추출하는 것입니다. 

만약에 trend가 일차함수와 같은 관계를 가지고 있다면 1-lag difference 만으로도 변동성을 잡을 수 있게 되는 것이지요. 마찬가지로 2-lag difference는 trend가 이차함수와 같은 관계를 가지고 있다면 적용되는 것입니다. 

그러나, 과도한 difference는 시계열을 과하게 변동성을 제거해 버려서, over-correction이 될 수도 있기 때문에 조심해야 합니다.

<p align='center'><img src='https://imgur.com/dPdnSMm.png'><figcaption align='center'>그림 13. Difference 적용 전</figcaption></p>

<p align='center'><img src='https://imgur.com/RPMUFSJ.png'><figcaption align='center'>그림 14. Difference 적용 후</figcaption></p>

<h3>1.5.2. Estimation and Elimination of Both Trend and Seasonality</h3>

trend와 seasonal 요소가 다 있는 경우 아래와 같이 표현될 수 있습니다(additive model인 경우).<del>multiplicative model인 케이스도 있습니다.</del>

$$X_t = m_t + s_t + Y_t, \,\, t=1, \dots, n,$$
$$where,\,\,EY_t = 0, s_{t+d}=s_t,\,\,and\,\,\sum_{j=1}^{d}s_j=0$$

두 가지 방법을 소개하겠습니다. 먼저, 첫번째 방법입니다.

<h4>method 1. Estimation of Trend and Seasonal components</h4>
아래와 같은 데이터가 있을 때, trend와 seasonal 요소를 제거해 봅시다. 아래 시계열 같은 경우, 주기가 d=12로, 1년 단위로 싸이클이 반복되는 것을 확인할 수 있습니다.

<p align='center'><img src='https://imgur.com/hCcOOp9.png'><figcaption align='center'>그림 15. Accidental Deaths, U.S.A., 1973-1978</figcaption></p>

<ol><li>먼저, trend 요소를 제외합니다. trend 요소를 제외하는 방법으로 moving average filter를 이용할 수 있습니다.<br><br>
예를 들어, 시계열 시퀀스 $\{x_1, x_2, \dots, x_n\}$ 이고, 주기 period $d=2q$ 라 한다면, 아래와 같은 moving average filter 식을 세울 수 있습니다.

$$\hat{m_t} = (0.5x_{t-q} + x_{t-q+1} + \dots + x_{t+q-1} + 0.5x_{t+q})/d,\,\, q<t\leq n-q$$ 

> 양 끝에 0.5씩 붙는 이유는 분자의 갯수는 홀수개 즉 $2q+1$ 이지만, 분모는 짝수 $d=2q$ 이기 때문에, 양 끝에 항의 가중치를 0.5씩만 해주는 것입니다.

반면에, 주기가 $d=2q+1$ 이라면, 아래와 같은 식을 세울 수 있습니다.

$$\hat{m_t} = (2q+1)^{-1}\sum_{j=-q}^{q}X_{t-j}$$
</li>
<li>그 다음은 seasonality 요소를 구하는 차례입니다. 먼저, 위에서 구한 trend요소를 원 시계열 데이터에서 $x_{k+jd} - \hat{m_{k+jd}}$ 와 같이 제거해야 합니다. 그런 다음, 동일한 주기에 해당하는 $x-\hat{m}$ 요소들을 가지고 평균 $w_k, \,\,(k=1, \dots, d)$ 를 구해줍니다. </li>
<li>이 때, $w_k$ 들의 평균은 0이 아닐 수 있습니다. 따라서, seasonal 요소들의 평균이 0이 되도록 다시 한번 평균을 빼줍니다.<del>다시 한번 정규화가 되도록 해주는 것입니다.</del>) 

$$\hat{s_k} = w_k - \frac{1}{d}\sum_{i=1}^{d}w_i,\,\,k=1, \dots, d$$

$$and, \,\, \hat{s_k}=\hat{s_{k-d}},\,k>d$$

따라서, deseaonalized된 데이터는 $d_t = x_t - \hat{s_t},\,\, t=1,\dots ,n$ 이며, detrended된 데이터는 $d_t = x_t - \hat{m_t},\,\, t=1, \dots, n$ 입니다.
</li>
<li>마지막으로, noise 추정값은 trend와 seasonal 요소를 모두 제거한 항입니다.</li>
<li>또한 trend 모델링을 위해, 다시 한번 parametric form으로 다시 한번 재추정하는 과정을 거칩니다. Parametric form으로 다시 한번 trend 요소를 재추정 하는 목적은 prediction과 simulation을 하기 위해서 입니다.</li>
</ol>

<p align='center'><img src='https://imgur.com/srdCVkU.png'><figcaption align='center'>그림 16. Trend and seasonal decomposition 예시</figcaption></p>
<p align='center'><img src='https://imgur.com/oYkGLqN.png'><figcaption align='center'>그림 17. Trend and seasonal decomposition 예시</figcaption></p>

<h4>method 2. Elimination of Trend and Seasonal components by Differencing</h4>
Trend 요소를 Differencing 방법을 통해 제거한 것과 동일하게 진행됩니다. Differencing operator $\bigtriangledown_d$ 을 $X_t = m_t + s_t + Y_t$ 식 양변에 취해주면 아래와 같습니다.

$$\bigtriangledown_dX_t = m_t - m_{t-d} + Y_t - Y_{t-d}$$


<h2>1.6. Testing the Estimated Noise Sequence</h2>

1.5까지 과정을 거치면 우린 Noise 항을 갖게 됩니다. 그러나 이 Noise 항이 White Noise 항인지는 확인이 필요합니다. 만약에 white noise 항이 맞다면, noise sequence를 모델링 하는 것입니다. 만약에 noise 항이 white noise가 아니라 여전히 depedency가 보인다면 다른 방법을 적용해야 합니다.

이번 챕터에서는 white noise인지를 확인하는 방법에 대해 살펴봅니다.

<h3>(a) The sample autocorrelation function</h3>

Sample acf를 그려서, 95%신뢰구간 안에 대부분 들어와 있는지 확인합니다. 만약 2,3개 이상이 신뢰구간 밖에 있거나 1개가 유난히 구간 안에 멀리 벗어 났다면, 우린 white noise라고 세웠던 가설을 기각해야 합니다.

<h3>(b) The portmanteau test(Ljung-Box test)</h3>

Portmanteau 검정 통계량은 <b>일정 기간 동안 일련의 관측치가 랜덤이고 독립적인지 여부를 검사하는데 사용합니다</b>. 통계량은 아래와 같습니다(Box-pierece 검정이라고도 합니다.). 

$$Q = n\sum_{j=1}^{h}\hat{\rho(j)^2}$$

$\hat{\rho(j)}$ 가 0에 가깝다면, $\hat{\rho(j)^2}$ 은 더욱 0에 가까울 것이지만, 몇몇 $\hat{\rho(j)}$ 의 절대값이 크다면, 그 항들에 영향을 받아 전체적인 Q값도 커지게 될 것입니다. 

> h는 lag입니다. h를 무리하게 크게 잡는다면, Q값은 커질 위험이 있습니다. 적당한 h를 잡는 것이 중요합니다.

귀무가설은 시차 h에 대한 자기 상관이 0이라는 귀무가설을 검정합니다. 통계량이 지정된 임계값보다 크면 하나 이상의 시차에 대한 자기 상관이 0과 유의하게 다르며, 일정 기간 랜덤 및 독립적이지 않음을 뜻합니다.

아래는 좀 더 refined된 통계량으로 Ljung-Box 라 합니다.

$$Q = n(n+2)\sum_{k=1}^{h}\hat{\rho(k)}/(n-k)$$

그 밖에, turning point test, difference-sign test, rank test, fitting an autoregressive model, checking for normality등이 있습니다. 

<hr>
이상으로, <Introduction to Time Series and forecasting 리뷰) 1. Introduction to Time Series> 포스팅을 마치겠습니다. 

<hr>

<ol><li>
<a href url='https://blog.naver.com/sw4r/221024668866'>Strict Stationarity vs. Weak Stationarity, https://blog.naver.com/sw4r/221024668866</a></li>
<li>고려대학교 김성범 교수님 <예측모델> 수업자료</li>
<li><a href url = 'https://otexts.com/fppkr/residuals.html'>portmanteau 검정 : https://otexts.com/fppkr/residuals.html</li></ol>
