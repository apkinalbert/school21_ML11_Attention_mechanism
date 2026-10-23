## 1. The RNN Encoder–Decoder bottleneck

The main drawback of the basic model without attention is that all source information available to the decoder must pass through a fixed-size final encoder state. This creates an information bottleneck, especially for long or complex sequences.

### 1.1 The basic model

Let the source sequence contain $T$ tokens:

$$
x_1,x_2,\ldots,x_T.
$$

A token can be a word, a part of a word, or a character.

The encoder processes the tokens one at a time:

$$
h_t=f_{\mathrm{enc}}\bigl(E_{\mathrm{src}}(x_t),h_{t-1}\bigr).
$$

Here:

- $E_{\mathrm{src}}(x_t)$ is the embedding of source token $x_t$;
- $h_t$ is the encoder hidden state after reading this token;
- $f_{\mathrm{enc}}$ is the recurrent update, such as a GRU;
- $h_0$ is the initial state, a zero vector.

Each state combines the current token with information kept from earlier tokens. After the last token, the final state $h_T$ is used to initialize the decoder:

$$
s_0=g(h_T).
$$

The function $g$ can simply copy the state if the dimensions match. Otherwise, it can be a learned transformation. For an LSTM, both the hidden state and the cell state must be handled.

At output step $j$, the decoder receives the previous target token:

$$
s_j=f_{\mathrm{dec}}\bigl(E_{\mathrm{tgt}}(y_{j-1}),s_{j-1}\bigr).
$$

It then predicts a distribution over the target vocabulary:

$$
P(y_j\mid y_{<j},x_{1:T})
=\operatorname{softmax}(W_{\mathrm{out}}s_j+b_{\mathrm{out}}).
$$

Here, $y_{<j}$ means all target tokens before position $j$. The first decoder input is a special start token, $\mathrm{SOS}$. The matrix $W_{\mathrm{out}}$ and bias $b_{\mathrm{out}}$ transform the state into one score for each possible output token.

During training, the previous correct target token is commonly supplied. This is called **teacher forcing**. During generation, the decoder receives its own previous prediction and continues until it produces $\mathrm{EOS}$, the end token.

### 1.2 The final state becomes a bottleneck

Suppose that $h_T$ contains 128 numbers. Its size remains 128 whether the source contains 5 characters or 50 characters.

The encoder therefore has to preserve all useful details in the same amount of state space. These details can include spelling, order, repeated characters, and information needed much later in the output.

In the basic model, the decoder cannot directly return to the state created for an earlier source token. It only receives the final summary and carries its information forward through its own states.

For example, when producing the beginning of a translated name, the decoder may need information about the beginning of the English name. That information has already passed through all later encoder steps before reaching the decoder.

This is a practical learning and memory limitation. It does not mean that every long sequence must fail or that short sequences are always translated correctly.

### 1.3 Information loss and vanishing gradients are related but different

The **information bottleneck** concerns what the final state preserves.

The **vanishing-gradient problem** concerns how well training signals reach earlier computations.

GRU and LSTM gates help control information flow and improve gradient propagation. However, they do not remove the final-state bottleneck: the decoder still depends on a fixed-size summary if attention is absent.

### 1.4 Another limitation: sequential computation

An RNN needs $h_{t-1}$ before it can compute $h_t$. Its recurrent steps therefore cannot all be evaluated independently.

This limits parallel computation across positions. It is a separate issue from the information bottleneck. Adding attention to an RNN does not remove the RNN's sequential dependence.

---

## 2. How attention addresses the bottleneck

**Attention allows the decoder to retrieve information from all encoder output states at each decoding step.**

Instead of making only $h_T$ available, the encoder provides:

$$
h_1,h_2,\ldots,h_T.
$$

For each output step, attention produces a new context vector. The decoder can therefore use different source information when predicting different target tokens.

### 2.1 Queries, keys, and values

For the following equations, vectors are written as columns. One possible recurrent attention design uses:

$$
q_j=W_Qs_{j-1},
\qquad
k_t=W_Kh_t,
\qquad
v_t=W_Vh_t.
$$

- The **query** $q_j$ represents the decoder's current information need.
- The **key** $k_t$ represents source position $t$ for comparison with the query.
- The **value** $v_t$ contains the information that can be retrieved from that position.

The query is computed from information already available to the decoder. It is not computed from the unknown next target token.

Keys and values come from the same encoder states, but their roles differ. Keys determine the attention weights; values are combined using those weights. Separate learned projections allow these roles to use different features. Simpler models can use $k_t=v_t=h_t$.

### 2.2 From similarity scores to a context vector

Scaled dot-product attention first calculates a score:

$$
e_{jt}=\frac{q_j^\top k_t}{\sqrt{d_k}},
$$

where $d_k$ is the number of components in a query or key.

Softmax turns the scores into nonnegative weights that sum to one:

$$
\alpha_{jt}
=\frac{\exp(e_{jt})}{\sum_{u=1}^{T}\exp(e_{ju})},
\qquad
\sum_{t=1}^{T}\alpha_{jt}=1.
$$

These are weights over source positions, not probabilities over the output vocabulary.

The context vector is the weighted sum:

$$
c_j=\sum_{t=1}^{T}\alpha_{jt}v_t.
$$

One possible decoder update is:

$$
s_j=f_{\mathrm{dec}}
\left(
\left[E_{\mathrm{tgt}}(y_{j-1});c_j\right],
s_{j-1}
\right),
$$

where the semicolon means vector concatenation. Other architectures combine the decoder state and attention output at a different point.

### 2.3 The scores are scaled

The dot product adds $d_k$ component products. As a simplified statistical model, suppose these products are independent, with mean zero and variance one. Their sum then has variance $d_k$ and standard deviation $\sqrt{d_k}$.

Dividing by $\sqrt{d_k}$ keeps the typical score size more stable as the dimension grows. Without scaling, large score differences can make softmax almost one-hot, leaving very small gradients for many entries.

### 2.4 What improves, and what remains limited

Attention provides shorter routes from relevant encoder states to output predictions. Early source information no longer has to survive only inside the final encoder state.

However:

- useful information must still exist in the encoder outputs;
- attention weights must be learned successfully;
- an RNN encoder or decoder remains sequential;
- comparing $S$ target queries with $T$ source keys creates $S\times T$ scores.

---

## 3. Positional encoding and why it is needed

Positional encoding supplies information about where tokens occur in a sequence. It allows a model to use order and position in addition to token identity.

### 3.1 Token identity does not describe token position

An embedding table assigns a vector to each token. Before any contextual processing, the same token receives the same embedding wherever it appears.

For example, the two occurrences of the letter a in Anna would have the same embedding if letter case were normalized. The embedding alone does not indicate that one is at the beginning and the other is at the end.

Order matters because rearranging the same tokens can change the meaning or spelling.

### 3.2 What ordinary self-attention knows

Consider self-attention with shared projections, without positional information or a position-dependent mask.

For a fixed query, rearranging the source key–value pairs simply rearranges the scores, weights, and values together. The weighted sum remains the same.

### 3.3 Adding a position vector

A common method is:

$$
z_p=e_p+r_p,
$$

where:

- $p$ is a position index, starting from zero;
- $e_p$ is the token embedding at that position;
- $r_p$ is a positional vector of the same size;
- $z_p$ is the representation supplied to the model.

Addition keeps the vector size unchanged. The model is trained to work with this combined representation. It is not generally possible to uniquely recover two arbitrary vectors from their sum; this method does not require such a reconstruction.

### 3.4 The RNN case is different

An RNN already processes tokens in order. Its state depends on the previous state, so rearranging tokens generally changes the states.

Attention applied to RNN encoder outputs can therefore receive order information through those outputs.

---

## 4. Sinusoidal positional encoding

This method assigns a fixed vector to each position using sine and cosine functions. The vector is calculated, rather than learned.

### 4.1 Building the formula step by step

The position index $p$ can be treated like discrete time. A sinusoid:

$$
\sin(\omega p)
$$

changes with position, stays between $-1$ and $1$, and has a rate of change controlled by the angular frequency $\omega$.

However, one sine value does not fully describe a phase. For example, $\sin(0)=\sin(\pi)=0$.

Adding cosine gives a pair:

$$
u_\omega(p)=
\begin{bmatrix}
\sin(\omega p)\\
\cos(\omega p)
\end{bmatrix}.
$$

At phase zero, the pair is $[0,1]^\top$. At phase $\pi$, it is $[0,-1]^\top$. The two phases can now be distinguished.

The pair still repeats after a full period. Multiple frequencies provide a richer position pattern, with some pairs changing quickly and others slowly.

### 4.2 Why the denominator contains a power

Let the embedding size $d$ be even. There are $m=d/2$ sine–cosine pairs, numbered $i=0,\ldots,m-1$.

One convenient design spaces the frequencies by a constant ratio:

$$
\omega_i=B^{-i/m}=B^{-2i/d}.
$$

The ratio between neighboring frequencies is:

$$
\frac{\omega_{i+1}}{\omega_i}=B^{-1/m}.
$$

This gives a geometric sequence of frequencies, covering different scales without assigning a separate rule to every pair.

With the conventional choice $B=10000$, the formula becomes:

$$
\mathrm{PE}(p,2i)
=\sin\left(\frac{p}{10000^{2i/d}}\right),
$$

$$
\mathrm{PE}(p,2i+1)
=\cos\left(\frac{p}{10000^{2i/d}}\right).
$$

The component indices $2i$ and $2i+1$ place each sine next to its matching cosine.

**The number 10000 is a design choice, not a mathematical requirement.** The formula is one useful construction, not the only possible way to represent positions.

For $d=8$, the frequencies are:

$$
1,\quad 0.1,\quad 0.01,\quad 0.001.
$$

The vector therefore contains:

$$
\mathrm{PE}(p)=
\begin{bmatrix}
\sin p\\
\cos p\\
\sin(p/10)\\
\cos(p/10)\\
\sin(p/100)\\
\cos(p/100)\\
\sin(p/1000)\\
\cos(p/1000)
\end{bmatrix}.
$$

---

## 5. Learned positional encoding

Instead of choosing a formula, the model can learn a table:

$$
P\in\mathbb{R}^{L_{\max}\times d},
$$

where $L_{\max}$ is the allocated number of positions and $d$ is the embedding size.

Position $p$ selects row $P_p$:

$$
r_p=P_p,
\qquad
z_p=e_p+P_p.
$$

This resembles a token embedding table, but its index is a position rather than a token ID.

The rows are model parameters. In a simple gradient-descent update:

$$
P_p\leftarrow P_p-\eta\frac{\partial\mathcal{L}}{\partial P_p},
$$

where $\eta$ is the learning rate and $\mathcal{L}$ is the training loss.

---

## 6. Comparing the two methods

| Property                             | Sinusoidal encoding                       | Learned encoding                         |
| ------------------------------------ | ----------------------------------------- | ---------------------------------------- |
| Position vectors                     | Calculated from a fixed formula           | Selected from a learned table            |
| Extra trainable parameters           | None                                      | $L_{\max}d$                            |
| Relations between positions          | Fixed structure from the frequencies      | Learned from the task                    |
| Adaptation to the dataset            | The model learns how to use fixed vectors | The position vectors adapt themselves    |
| Positions beyond training lengths    | Vectors can still be calculated           | Unseen rows have no task-trained meaning |
| Positions beyond the allocated table | Formula can be evaluated                  | Table must be extended or replaced       |
| Performance on longer sequences      | Not guaranteed                            | Not guaranteed                           |
