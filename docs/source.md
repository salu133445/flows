class: center, middle

# Flow-based Deep Generative Models

Jiarui Xu and Hao-Wen Dong

---

# Outlines

- __Deep generative models__
  - Different generative models
  - GAN vs VAE vs Flow-based models
- __Linear algebra basics__
  - Jacobian matrix and determinant
  - Change of variable theorem
- __Normalizing Flows__
- __Models__
  - RealNVP
  - NICE
  - Glow

---

class: center, middle

# Deep Generative Models

---

# Different generative models

.footer[Ian Goodfellow, "Generative Adversarial Networks," _NeurIPS tutorial_, 2016.]

.center[![taxonomy](images/taxonomy.png)]

---

# GANs vs VAEs vs Flow based models

.footer[Lilian Weng, "[Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)," _blog post_, 2018.]

.center[![taxonomy](images/gan-vae-flow.png)]

---

class: center, middle

# Linear Algebra Basics

---

# Jacobian matrix and determinant

---

# Change of variable theorem

Given some random variable $z \sim \pi(z)$ and a invertible mapping $x = f(z)$ (i.e., $z = f^{-1}(x) = g(x)$). Then, the distribution of $x$ is

$$p(x) = \pi(z) \left|\frac{dz}{dx}\right| = \pi(g(x)) \left|\frac{dg}{dx}\right|$$

--

The multivariate version takes the following form:

$$p(\mathbf{x}) = \pi(\mathbf{z}) \left|\det\frac{d\mathbf{z}}{d\mathbf{x}}\right| = \pi(g(\mathbf{x})) \left|\det\frac{dg}{d\mathbf{x}}\right|$$

where $\det\frac{dg}{d\mathbf{x}}$ is the _Jacobian determinant_ of $g$.

---

class: center, middle

# Normalizing Flow Models

---

# Normalizing flow models

.footer[Lilian Weng, "[Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)," _blog post_, 2018.]

__Key__: Transform a simple distribution into a complex one by applying a sequence of _invertible transformations_.

.center[![normalizing-flow](images/normalizing-flow.png)]

--

- In each step, substitute the variable with the new one by change of variables theorem.
- Eventually,  obtain a distribution close enough to the target distribution.

---

# Normalizing flow models

For each step, we have `$\mathbf{z}_i \sim p_i(\mathbf{z}_i)$`, `$\mathbf{z}_i = f_i(\mathbf{z}_{i-1})$` and `$\mathbf{z}_{i-1} = g_i(\mathbf{z}_i)$`. Now,

`$$\begin{align}
  p_i(\mathbf{z}_i)
  &= p_{i-1}(g_i(\mathbf{z}_i)) \left|\det\frac{dg_i(\mathbf{z}_i)}{d\mathbf{z}_i}\right| &&\text{(by change of variables theorem)}\\
  &=p_{i-1}(\mathbf{z}_{i-1})\left|\det\frac{d\mathbf{z}_{i-1}}{df_i(\mathbf{z}_{i-1})}\right| &&\text{(by definition)}\\
  &=p_{i-1}(\mathbf{z}_{i-1})\left|\det\left(\frac{df_i(\mathbf{z}_{i-1})}{d\mathbf{z}_{i-1}}\right)^{-1}\right| &&\text{(by inverse function theorem)}\\
  &=p_{i-1}(\mathbf{z}_{i-1})\left|\det\frac{df_i}{d\mathbf{z}_{i-1}}\right|^{-1} &&\text{(by $\det M \det (M^{-1}) = \det I = 1$)}
\end{align}$$`

Thus, we have `$\log p_i(\mathbf{z}_i) = \log p_{i-1}(\mathbf{z}_{i-1}) - \log \left|\det\frac{df_i}{d\mathbf{z}_{i-1}}\right|$`.

---

# Normalizing flow models

Now, we obtain `$\log p_i(\mathbf{z}_i) = \log p_{i-1}(\mathbf{z}_{i-1}) - \log \left|\det\frac{df_i}{d\mathbf{z}_{i-1}}\right|$`

Recall that `$\mathbf{x} = \mathbf{z}_K = f_K \circ f_{K-1} \dots f_1 (\mathbf{z_0})$`.

Thus, we have

`$$\begin{align}
  \log p(\mathbf{x})
  &= \log p_K(\mathbf{z}_K)\\
  &= \log p_{K-1}(\mathbf{z}_{K-1}) - \log\left|\det\frac{df_K}{d\mathbf{z}_{K-1}}\right|\\
  &=\dots\\
  &= \log p_0(\mathbf{z}_0) - \sum_{i=1}^K \log\left|\det\frac{df_i}{d\mathbf{z}_{i-1}}\right|
\end{align}$$`

---

# Normalizing flow models

In normalizing flows, the exact log-likelihood $\log p(\mathbf{x})$ of input data $x$ is

`$$\log p(\mathbf{x}) = \log p_0(\mathbf{z}_0) - \sum_{i=1}^K \log\left|\det\frac{df_i}{d\mathbf{z}_{i-1}}\right|$$`

--

To make the computation tractable, it requires

- $f_i$ is easily invertible
- The Jacobian determinant of $f_i$ is easy to compute

--

Then, we can train the model by maximizing the log-likelihood over some training dataset $\mathcal{D}$

$$LL(\mathcal{D}) = \sum_{\mathbf{x}\in\mathcal{D}} \log p(\mathbf{x})$$

---

class: center, middle

# Models

---

# NICE

.footer[Laurent Dinh, David Krueger, and Yoshua Bengio, "NICE: Non-linear Independent Components Estimation," _ICLR_, 2015.]

The core idea behind NICE (Non-linear Independent Components Estimation) is to

- split $\mathbf{x} \in \mathbb{R}^D$ into two blocks $\mathbf{x}_1 \in \mathbb{R}^d$ and $\mathbf{x}_2 \in \mathbb{R}^{D-d}$

--

- apply the following transformation from $(\mathbf{x}_1, \mathbf{x}_2)$ to $(\mathbf{y}_1, \mathbf{y}_2)$

  `$$\begin{cases}
    \mathbf{y}_1 &= \mathbf{x}_1\\
    \mathbf{y}_2 &= \mathbf{x}_2 + m(\mathbf{x}_1)
  \end{cases}$$`

  where $m(\cdot)$ is an arbitrarily function (e.g., deep neural network).

---

# NICE

.footer[Laurent Dinh, David Krueger, and Yoshua Bengio, "NICE: Non-linear Independent Components Estimation," _ICLR_, 2015.]

The transformation

`$$\begin{cases}
  \mathbf{y}_1 &= \mathbf{x}_1\\
  \mathbf{y}_2 &= \mathbf{x}_2 + m(\mathbf{x}_1)
\end{cases}$$`

- is trivially invertible.

  `$$\begin{cases}
    \mathbf{x}_1 &= \mathbf{y}_1\\
    \mathbf{x}_2 &= \mathbf{y}_2 - m(\mathbf{y}_1)
  \end{cases}$$`

---

# NICE

.footer[Laurent Dinh, David Krueger, and Yoshua Bengio, "NICE: Non-linear Independent Components Estimation," _ICLR_, 2015.]

The transformation

`$$\begin{cases}
  \mathbf{y}_1 &= \mathbf{x}_1\\
  \mathbf{y}_2 &= \mathbf{x}_2 + m(\mathbf{x}_1)
\end{cases}$$`

- has a unit Jacobian determinant.

  `$$\mathbf{J} = \begin{bmatrix}\mathbf{I}_d&\mathbf{0}_{d\times(D-d)}\\\frac{\partial m(\mathbf{x}_1)}{\partial\mathbf{x}_1}&\mathbf{I}_{D-d}\end{bmatrix}$$`

  `$$\det(\mathbf{J}) = \mathbf{I}$$`

---

# RealNVP

.footer[Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio, "Density Estimation using Real NVP," _ICLR_, 2017.]

The core idea behind RealNVP (Real-valued Non-Volume Preserving) is to

- split $\mathbf{x} \in \mathbb{R}^D$ into two blocks $\mathbf{x}_1 \in \mathbb{R}^d$ and $\mathbf{x}_2 \in \mathbb{R}^{D-d}$

--

- apply the following transformation from $(\mathbf{x}_1, \mathbf{x}_2)$ to $(\mathbf{y}_1, \mathbf{y}_2)$

  `$$\begin{cases}
    \mathbf{y}_{1:d} &= \mathbf{x}_{1:d}\\
    \mathbf{y}_{d+1:D} &= \mathbf{x}_{d+1:D} \odot e^{s(\mathbf{x}_{1:d})} + t(\mathbf{x}_{1:d})
  \end{cases}$$`

  where $s(\cdot)$ and $t(\cdot)$ are _scale_ and _translation_ functions that map $\mathbb{R}^d$ to $\mathbb{R}^{D-d}$, and $\odot$ denotes the element-wise product.

--

  (Note that NICE does not have the scaling term.)

---

# RealNVP

.footer[Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, "Density Estimation using Real NVP," _ICLR_, 2017.]

The transformation

`$$\begin{cases}
  \mathbf{y}_{1:d} &= \mathbf{x}_{1:d}\\
  \mathbf{y}_{d+1:D} &= \mathbf{x}_{d+1:D} \odot e^{s(\mathbf{x}_{1:d})} + t(\mathbf{x}_{1:d})
\end{cases}$$`

- is easily invertible.

  `$$\begin{cases}
    \mathbf{x}_{1:d} &= \mathbf{y}_{1:d}\\
    \mathbf{x}_{d+1:D} &= (\mathbf{y}_{d+1:D} - t(\mathbf{x}_{1:d})) \odot e^{-s(\mathbf{x}_{1:d})}
  \end{cases}$$`

  (Note that it does not involve computing $s^{-1}$ and $t^{-1}$.)

---

# RealNVP

.footer[Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, "Density Estimation using Real NVP," _ICLR_, 2017.]

The transformation

`$$\begin{cases}
  \mathbf{y}_{1:d} &= \mathbf{x}_{1:d}\\
  \mathbf{y}_{d+1:D} &= \mathbf{x}_{d+1:D} \odot e^{s(\mathbf{x}_{1:d})} + t(\mathbf{x}_{1:d})
\end{cases}$$`

- has a Jacobian determinant that is easy to compute.

  `$$\mathbf{J} = \begin{bmatrix}\mathbf{I}_d&\mathbf{0}_{d\times(D-d)}\\\frac{\partial\mathbf{y}_{d+1:D}}{\partial\mathbf{x}_{1:d}}&\text{diag}\left(e^{s(\mathbf{x}_{1:d})}\right)\end{bmatrix}$$`

  `$$\det(\mathbf{J}) = \prod_{j=1}^{D-d} e^{s(\mathbf{x}_{1:d})_j} = \exp\left(\sum_{j=1}^{D-d}s(\mathbf{x}_{1:d})_j\right)$$`

  (Note that it does not involve computing the Jacobian of $s$ and $t$.)

---

# RealNVP

- We can parameterize $s$ and $t$ with complex models (e.g., deep neural networks) as we don't need to the inverse functions and the Jacobian of $s$ and $t$.
- Some dimensions remain unchanged in an affine coupling layer. Thus, we need to alternate the dimensions being modified.

---

# GLOW

---

class: center, middle

# Summary

---

# Summary

---

class: center, middle

# Thank you!
