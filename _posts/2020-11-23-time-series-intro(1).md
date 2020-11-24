---
layout : post
title: Introduction to Time Series and forecasting 리뷰) 1-1. Introduction to Time Series
category: Time Series Analysis
tags: brockwell, richard-davis, statistics, time-series, time-series-analysis, arima, arma 
---
이번 포스팅을 시작으로, 시계열 분석에 대해서 다루도록 하겠습니다. 메인 교재는 Brockwell와 Richard A. Davis의 \< Introduction to Time Series and Forecasting \> 와 패스트캠퍼스의 \<파이썬을 활용한 시계열 분석 A\-Z\> 를 듣고 정리하였습니다. 

---

<h2>1.1. Examples of Time Series</h2>

<h2>1.2. Objectives of Time Series Analysis</h2>

<h2>1.3. Some Simple Time Series Models</h2>

<h3>1.3.3 A General Approach to Time Series Modeling</h3>
시계열 분석에 대해 깊게 들어가기 전에, 시계열 데이터 모델링하는 방법에 대해 대략적으로 알아봅시다.

1) 그래프로 그린 후, 그래프 상에서 아래와 같은 요소가 있는지 체크한다.(Plot the series and examine the main features of the graph)<br> 
<ul><li>trend</li><li>a seasonal component</li><li>any apparent sharp changes in behavior</li><li>any outlying observations</li></ul>

2) 정상상태의 잔차를 얻기 위해, trend와 seasonality 요소를 제거한다. (Remove the trend and seasonal components to get stationary residuals)<br>
&nbsp;&nbsp;&nbsp;&nbsp; trend와 seasonality 요소를 제거하기 전에, 전처리를 해야하는 경우가 있습니다. 예를 들어, 아래와 같이 지수적으로 증가하는 경우에, 로그를 취해서 variance가 일정하도록 만든 후 모델링을 하면 정확도를 높일 수 있습니다.

<p align='center'><img src='https://imgur.com/V85l07h.png'><figcaption align='center'>그림 1. 로그 취하기 전</figcaption></p>
<p align='center'><img src='https://imgur.com/e0GKRKU.png'><figcaption align='center'>그림 1. 로그 취한 후</figcaption></p>

이외에도 여러 방법이 있습니다. 추후에 설명하도록 하겠습니다. 어쨌든, 이 모든 방법들의 핵심은 <b>정상상태의 잔차</b>를 만드는 것입니다.

3) auto-correlation 함수, 여러 다양한 통계량을 이용하여 잔차를 핏팅할 모델을 선택한다. (Choose a model to fit the residuals, making use of various sample statistics including the sample autocorrelation function)

4) 핏팅된 모델로 예측한다.<br> 
&nbsp;&nbsp;&nbsp;&nbsp; 여기서 잔차를 예측하는 것이고, 예측된 잔차를 원래 예측해야 할 값으로 변환한다.

<h2>1.4. Stationary Models and the Autocorrelation Function</h2>

시계열 데이터가 정상상태(stationarity)를 가지기 위해서, 시계열이 확률적인 특징이 시간이 지남에 따라 변하지 않는다는 가정을 충족시켜야 합니다. 

> a time series ${\{X_t, t=0, \pm1, ...\}}$ is said to be stationary if it has statistical properties similar to those of the "time-shifted" series ${\{X_{t+h}, t=0, \pm1, ...\}}$ for each integer h.

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
<p align='center'><img src='https://imgur.com/jjzBk3z.png'></figcaption align='center'>그림 4. Sample ACF</figcaption></p>

White Noise인 경우, 시계열 그래프와 ACF 그래프는 아래와 같습니다. lag가 1이상인 경우, 거의 ACF값이 0에 가까운 것을 볼 수 있고, 95% 신뢰구간 안에 들어와 있습니다. 
<p align='center'><img src='https://imgur.com/RaoTZJj.png'><figcaption align='center'>그림 5. White Noise ACF</figcaption></p>

아래는 그림 1. 그래프에 플롯된 데이터를 가지고 그린 ACF입니다. 보시면, ACF가 lag가 커짐에 따라 서서히 감소하는 형태를 띄는데 이는 trend가 있는 데이터에서 나타납니다. 
<p align='center'><img src='https://imgur.com/N6vk5oN.png'><figcaption align='center'>그림 6. Sequence with trend ACF</figcaption></p>

<h2>Estimation and Elimination of Trend and Seasonal Components</h2>





***
1. [Strict Stationarity vs. Weak Stationarity, https://blog.naver.com/sw4r/221024668866](https://blog.naver.com/sw4r/221024668866)
2. [Strict Stationarity vs. Weak Stationarity, https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221029452892&proxyReferer=https:%2F%2Fwww.google.com%2F](https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221029452892&proxyReferer=https:%2F%2Fwww.google.com%2F)


