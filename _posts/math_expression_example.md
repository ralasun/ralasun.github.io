$$[^{i-1}T_i] = \left( \begin{array}{r}
    \cos(θ_i) & -\sin(θ_i) \cos(⍺_{i,i+1}) & \sin(θ_i) \sin(⍺_{i,i+1}) & a_{i,i+1} \cos(θ_i) \\
    \sin(θ_i) & \cos(θ_i) \cos(⍺_{i,i+1}) & -\sin(θ_i) \sin(⍺_{i,i+1}) & a_{i,i+1} \sin(θ_i) \\
    0 & \sin(⍺_{i,i+1}) & \cos(⍺_{i,i+1}) & d_i \\
    0 & 0 & 0 & 1
\end{array} \right)$$

% thin space
\,
\thinspace
 
% negative thin space
\!
 
% medium space
\:
 
% large space
\;
 
% 0.5em space
\enspace
 
% 1em space
\quad
 
% 2em space
\qquad
 
% custom space
\hspace{3em}
 
% fill empty space
\hfill

<p style="text-align:center">
$$p(\text{x}_t = k|\alpha,\text{z}) = \frac{m_{\alpha, \,k}exp(f_{tk})}{\sum_{j=1}^Km_{\alpha, \,k}exp(f_{tj})}$$
<figcaption align="center">수식1. masked distribution at timestep t</figcaption>
</p>