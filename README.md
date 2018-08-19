# Coursera_Deep_Learning

<img width="1175" alt="default" src="https://user-images.githubusercontent.com/33269462/44305049-55581000-a33c-11e8-8eec-b2d07e06583c.png">

## Completed Assignments

### Logistic Regression with a Neural Network mindset
Building a logistic regression classifier to recognize cats
* Build the general architecture of a learning algorithm, including:
* Initializing parameters
* Calculating the cost function and its gradient
* Using an optimization algorithm (gradient descent)
* Gather all three functions above into a main model function, in the right order.

### Planar data classification with one hidden layer
* Implement a 2-class classification neural network with a single hidden layer
* Use units with a non-linear activation function, such as tanh
* Compute the cross entropy loss
* Implement forward and backward propagation

### Building your Deep Neural Network: Step by Step
* Use non-linear units like ReLU to improve my model
* Build a deeper neural network (with more than 1 hidden layer)
* Implement an easy-to-use neural network class

### Deep Neural Network for Image Classification: Application
I used the functions I have implemented in the previous assignment to build a deep network, and apply it to cat vs non-cat classification. Hopefully, there is an improvement in accuracy relative to my previous logistic regression implementation.
* Build and apply a deep neural network to supervised learning.

### Initialization
It shows how a well chosen initialization can:
* Speed up the convergence of gradient descent
* Increase the odds of gradient descent converging to a lower training (and generalization) error

### Regularization
Deep Learning models have so much flexibility and capacity that overfitting can be a serious problem, if the training dataset is not big enough. Sure it does well on the training set, but the learned network doesn't generalize to new examples that it has never seen!This assignment is about how to use regularization in my deep learning models.

### Gradient Checking
It uses mathematic techniques to check the implementation of backpropagation is correct. Since backpropagation is quite challenging to implement, and sometimes has bugs

### Optimization Methods
I learnt more advanced optimization methods that can speed up learning and perhaps even get me to a better final value for the cost function. Having a good optimization algorithm can be the difference between waiting days vs. just a few hours to get a good result.

### TensorFlow Tutorial
How we can use some existing frameworks that allow me to build neural networks more easily. Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and many others can speed up my machine learning development significantly. All of these frameworks also have a lot of documentation, which we should feel free to read. In this assignment, I learnt the following in TensorFlow:
* Initialize variables
* Start my own session
* Train algorithms
* Implement a Neural Network

### Convolutional Neural Networks: Step by Step
Implementing convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation.

### Convolutional Neural Networks: Application
* Implement helper functions that will use when implementing a TensorFlow model
* Implement a fully functioning ConvNet using TensorFlow
* Build and train a ConvNet in TensorFlow for a classification problem

### Keras tutorial - the Happy House
* Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK.
* See how to build a deep learning algorithm in a couple of hours.

### Residual Networks
How to build very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by He et al., allow we to train much deeper networks than were previously practically feasible.
* Implement the basic building blocks of ResNets.
* Put together these building blocks to implement and train a state-of-the-art neural network for image classification.

### Autonomous driving - Car detection
Learning about object detection using the very powerful YOLO model.
* Use object detection on a car detection dataset
* Deal with bounding boxes

### Deep Learning & Art: Neural Style Transfer
* Implement the neural style transfer algorithm
* Generate novel artistic images using my algorithm
Notes that most of the algorithms we have studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, we optimize a cost function to get pixel values!

### Face Recognition for the Happy House
Face recognition problems commonly fall into two categories:
* Face Verification - "is this the claimed person?". For example, at some airports, I can pass through customs by letting a system scan my passport and then verifying that I (the person carrying the passport) am the correct person. A mobile phone that unlocks using my face is also using face verification. This is a 1:1 matching problem.
* Face Recognition - "who is this person?". For example, the video lecture showed a face recognition video of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.
* Implement the triplet loss function
* Use a pretrained model to map face images into 128-dimensional encodings
* Use these encodings to perform face verification and face recognition

### Building your Recurrent Neural Network - Step by Step
Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory". They can read inputs, such as words, one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next. This allows a uni-directional RNN to take information from the past to process later inputs. A bidirection RNN can take context from both the past and the future.

### Character level language model - Dinosaurus land
It covers
* How to store text data for processing using an RNN
* How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
* How to build a character-level text generation recurrent neural network
* Why clipping the gradients is important

### Improvise a Jazz Solo with an LSTM Network
* Apply an LSTM to music generation.
* Generate jazz music with deep learning.

### Operations on word vectors
Because word embeddings are very computionally expensive to train, most ML practitioners will load a pre-trained set of embeddings.
* Load pre-trained word vectors, and measure similarity using cosine similarity
* Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
* Modify word embeddings to reduce their gender bias
