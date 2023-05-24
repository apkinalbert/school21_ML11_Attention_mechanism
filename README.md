# Attention

Summary: This project is an introduction to attention mechanism

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Machine Translation before attention](#machine-translation-before-attention) \
    2.2. [How attention works](#how-attention-works) \
    2.3. [Positional encoding](#positional-encoding) \
    2.4. [Attention for sequences with temporal dependency](#attention-for-sequences-with-temporal-dependency)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

In the last project, we got acquainted with recurrent neural networks - the architecture that allows process sequential data. And earlier every time when you need to solve any text processing task, you meet this architecture as one of the approaches. But in 2017 Google had published an article titled “Attention is all you need”, where the authors proposed a new architecture for working with sequences and a particularly new model for Machine Translation task called Transformer. But the main change was the Attention mechanism inside the model, and it was this that allowed the approach to overtake [the existing State of the Art](https://paperswithcode.com/sota/machine-translation-on-wmt2014-english-german) (State of the Art or SOTA is the leading approach for a specific task). The community quickly picked up and tested this idea in various tasks. And now many well-known models use this architecture internally (BERT, GPT, Dall-e, AlphaFold, etc.). Let's try to understand how this model works and why it works so well.

## Chapter II. Introduction

### Machine Translation before attention

The first task for which the attention mechanism was applied was the task of machine translation - i.e. a sentence in one language was translated into another. And before we touch on the topic of Attention, I would like to analyze how this problem was solved using neural networks before the article was published. The majority of approaches were based on the Encoder-Decoder architecture, where both Encoder and Decoder were separate RNN models. Nevertheless, these models were trained together, and the last hidden state of the Encoder was fed to the input of the Decoder as the initial state of the hidden state.

![Machine Translation before attention](misc/images/machine_translation_before_attention.png)

But this approach has a drawback. Understanding exactly what the disadvantage is - part of your task.

### How attention works

When we translate a phrase within one sentence into another language, in most cases we don't need to see the entire sentence. The translation of a particular phrase will usually only be affected by a few words, as shown in the figure below:

![How attention works](misc/images/how_attention_works.png)

Hence the question - can we train the neural network so that it will form such connections itself. Or in other words, in such a way, that net will define which word is strongly associated with the translation and which is not. If we take the already known RNN, then there is no direct answer to these questions inside. To achieve this in Transformer model, the authors suggested following next steps (below, some layers with transformations are omitted for simplicity):

1. for each word in the original sentence and in the translation, we form a vector of fixed size and call them "key" and "request" respectively

    ![translation](misc/images/translation.png)

2. between each word in the original sentence and in the translation, we consider the similarity by taking the scalar product of their “key” and “query” representations (the greater scalar product the more similar the words)

    ![translation 2](misc/images/translation_2.png)

3. in parallel, for each word from the original sentence, we also form the “value” vector

    ![Value vector](misc/images/value_vector.png)

4. and finally, to describe the hidden state of the translated word, we take the weighted sum of the “value” vectors of the original sentence, where the weight is the similarity of the words

    ![Hidden state](misc/images/hidden_state.png)

Thus, the neural network directly considers the relationship between words (multiplying their Q and K representations), and based on these values, forms a hidden state vector for each word of the translated sentence. By following these steps for each word in the translation, we will get exactly the same output from the network as in RNN:

![RNN](misc/images/rnn.png)

However, we noted above that we omitted some layers in the description. In the original article, the attention mechanism is depicted as follows:

![Scaled Dot-Product Attention](misc/images/scaled_dot_product_attention.png)

```math
\text {Attention}(Q, K, V) = \text {softmax}(\frac {QK^T} {\sqrt{d_k}})V
```

where:

- scale layer - simple division of scalar product by coefficient that depends on the size of embedding. This trick lets the softmax layer be more numerically stable: “We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by √(d_k )”
- mask layer we will consider further
- softmax layer is required to transform scaled dot product into probabilities, that will sum up to 1

But the authors did not limit themselves on adding one Attention layer and decided to use several such layers in parallel. They called it Multi-head attention:

![Multi-head attention](misc/images/multi_head_attention.png)

```math
\begin {align*}
\text {MultiHead}(Q, K, V) &= \text {Concat}(\text{head}_1, \dots, \text {head}_n)W^O \\
\text {where } \text{head}_i &= \text {Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end {align*}
```

And this approach works better than using a single Attention layer (even if the models have the same number of parameters). This is due to the fact that each head can look at the same data from different angles. For example, the first will determine the combination of words’ gender, the second - the combination of tense, and so on.

It is also important to note that attention can be used not only for 2 different sequences, but also when we are dealing with one sequence. Then the model will learn how the elements of the same sequence are interconnected with each other. This case is called self-attention:

![transformer-novel-neural-network](misc/images/transformer-novel-neural-network.png)
![transformer-novel-neural-network](misc/images/transformer-novel-neural-network_2.png)

### Positional encoding

Having become acquainted with how the attention mechanism works, we can conclude that such construction allows solving the main problem of RNN - the loss of connection between distant words. But there is also a drawback, currently we do not take into account the order of words. If we want to translate the text (and we don’t have any RNN that pre-processed it) and feed the model two different phrases: “Growth has slowed down” and “down Growth has slowed”, then the decoder that works on the Attention mechanism will not see the difference. The set of words remains the same, their embeddings are the same, {Q, V, K} will be the same, and therefore attention weights and hidden state will not change. Fortunately, this problem is very easily solved by changing the embeddings that we pass as input to calculate {Q, V, K}. In this project, you will get acquainted with 2 possible solutions:

1. use sine and cosine functions of different frequencies
2. use trainable position weights \
    How they are arranged you have to study on your own.

### Attention for sequences with temporal dependency

Let's return to the Mask layer that we observed at the very beginning. Imagine a situation where we use a model with a self-attention mechanism inside to predict from a user's clickstream when they will leave the site. And let the clickstream contain the following events:

*visit page A, add to cart item, visit page B, visit cart, visit payment page, afk for 2 min, visit account page, exit*

If we will use self-attention as described above, then when calculating the hidden state for the “visit cart” event, we will use all subsequent events. But then the model will use information from the future to describe the current event. This situation is fraught with the fact that the model might learn patterns that it cannot observe on real data, since in this case it will not be able to look into the future. This is called data leakage. It usually leads to overly optimistic outcomes during the model building phase, followed by the unpleasant surprise of poor results after the prediction model is implemented and tested on new data. To avoid it in general case always use validation and test data and split them similar to the case if the model will work in production.

To avoid this problem for the attention mechanism - we will just assign -□inf  for all weights “from the future”, then after the softmax layer they will turn into 0:

<table>
  <tr>
    <th>key <br> query</th>
    <th>visit page A</th>
    <th>add to cart item</th>
    <th>visit page B</th>
    <th>visit cart</th>
    <th>visit payment page</th>
  </tr>
  <tr>
    <th>visit page A</th>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>add to cart item</th>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit page B</th>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit cart</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit payment page</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
  </tr>
</table>

## Chapter III. Goal

The goal of this task is to get a deep understanding of how to build models with Attention mechanism.

## Chapter IV. Instructions

- This project will only be evaluated by humans. You are free to organize and name your files as you desire.
- Here and further we use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try Google Colab. It offers kernels (Runtime) with GPU for free that is faster than CPU for such tasks. The norm is not applied to this project. Nevertheless, you are asked to be clear and structured in the conception of your source code.
- Store the datasets in the subfolder data

## Chapter V. Task

In this project, you will work with a dataset of names and their Russian translations. We will build various machine translation models using different architectures.

You can use any framework that you find convenient: PyTorch, TensorFlow, Keras,
etc.

### Dataset

You will work with the dataset that was generated from an open dataset of baby names - [https://data.world/alexandra/baby-names](https://data.world/alexandra/baby-names). The original dataset consists of 6782 records. Every record contains the baby name and gender. The extended version also contains a Russian translation of all names. The translation was produced by Google Translate and then manually corrected for some cases.

### Task

1. Provide the answer to the question from introduction \
    a. TODO
2. Data preparation \
    a. Download dataset from the course files \
    b. Create 2 dictionaries for English letters and Russian ones. Do not forget special tokens \
    c. Encode names \
    d. Split dataset to train, valid and test part (take 80-10-10%). Be sure to use random_state
3. RNN Encoder \
    a. Train 1-layer RNN (GRU or LSTM) model to encode English name. Use built-in class of neural network library \
    b. Add early stopping by validation metric to train procedure \
    c. Generate 10 names using trained model
4. RNN Machine Translation
    a. As encoder take fitted RNN model from the 3rd point \
    b. Implement RNN decoder to produce translation. Use built-in class of neural network library \
    c. Train the network using loss for next letter prediction. Use early stopping on validation metric. Do not train encoder parameters, train only decoder. \
    d. Implement method to translate name. Method should not be stochastic - choose the next letter using argmax \
    e. Evaluation \
        i. plot loss convergence on train and valid data \
        ii. print translation for 5 names from train and 5 names from valid part of dataset every n epochs. Are they realistic? \
        iii. compute perplexity on test set
5. Attention Machine Translation \
    a. As encoder take fitted RNN model from the 3rd point \
    b. Implement decoder with attention mechanism to produce translation. Do not use the built-in class of neural network library for Attention. Compute attention between encoder output and decoder input. \
    c. Train the network using loss for next letter prediction. Use early stopping on validation metric. Do not train encoder parameters, train only decoder. \
    d. repeat 4.e for this model \
    e. compare results. What can you note from these results
6. Positional Encoding \
    a. Implement positional encoding with sine and cosine \
        i. train separate RNN Encoder with this type of positional encoding \
        ii. train RNN Decoder for Machine Translation with this type of positional encoding \
        iii. train Attention Decoder for Machine Translation with this type of positional encoding \
        iv. evaluate models and compare results. Provide local conclusions about results \
    b. Implement positional encoding with trainable weights \
        i. repeats steps 6.a.i - 6.a.iii for this positional encoding \
        ii. visualize learned positional encoding \
    c. What pros and cons of using sine and cosine positional encoding instead of positional encoding with trainable weights. Provide the answer
7. Multi-head Attention Machine Translation \
    a. Implement Multi-head attention. Use n_heads = 3 \
    b. Select the name and add visualization of the attention matrix inside the training process. Display the plot every n epochs. \
    c. Train Machine Translation with this block \
    d. Evaluate model (repeat 4.e) \
    e. One of the heads should be almost diagonal with a shift on 1 block. \ Explain why this shift exists. Example:

![Example](misc/images/example.png)

### Submission

Save your model from Multi-head Attention Machine Translation in the pickle format. Your peer will load it and use it to make predictions for the test dataset once again (don’t forget to use the same random state). The predictions should be stored in a file with the name test_predictions.csv.

Your repository should contain one or several notebooks with your solutions.

## Chapter VI. Bonus part

- Repeat [Transformer architecture](https://arxiv.org/pdf/1706.03762.pdf) for names translation task
- Transform names and translation into n-gram vectors. Implement and measure BLEU metric

💡 [Tap here](https://forms.yandex.ru/cloud/646b47cd02848f2ef1031b24/) **to leave your feedback on the project**. Pedago Team really tries to make your educational experience better.
