# Fundamentals-of-ML-and-Deep-Learning
## Studied from DeepLizard Youtube channel

To understand Deep Learning,first we have to understand what machine learning is.So,in machine learning we train our model by giving some training datasets so that the model can learn from it and whenever some new testdata is tested ,it can give the correct results.

**Example**-Suppose we have to identify whether the given news is positive or negative.Then first of all lets find the difference between how the traditional algorithm works and how ML algorithm works.So,according to Traditional code,we would ask our model to check whether the words like can't,unfortunate,bad,sorry,unhappy,etc. or 'happy,pleased,amazing,fortunate,blessed,etc. occur such that our model can compare between final no. of positive or negative words and decide whether the news is positive or negative.But in ML algo,we would show some datasets to our model so that it is 'trained' ,rather it memorizes or learns from these dataset and hence can identify the completely new testdata when shown with great accuracy.

## Deep Learning-
Deep Learning is a subfield of ML that uses algorithm inspired by structure and function of Brain's neural network.
As a subfield of ML,DL also algorithms to analyze data,learn from the data and then make the prediction or determination of the new data.The basic difference between ML and DL is that DL uses structure and function oriented programming of brain's neural network.So ,this learning can be of Superwised type or Unsuperwised type.
**Suerwised Model** basically learns from the data and make inferences from the data that has been already been labelled(i.e.we know the desired output).Whereas in **Unsuperwised Data** ,model is unaware of the labelled data.
Example-Suppose we want to identify whether the given image is of dog or cat.In superwised learning model,the model already knows that if the image contains features like A,B,C,etc.then the image s of cat,or X,Y,Z then it is of dog.However in Unsuperwised model,it has to make predictions by itself only and divide the predictions into some categories depending on the features observed.

So,these models in which it learns or is getting trained is called **Artificial Neural Network** or Neural nets or models,etc.

## Artificial Neural networks-
ANN is basically a computational system comprising of various layers(input layer,hidden layer,output layer) which takes some input ,processes the signal and gives the desired output.These layers are made up of nodes or neurons which transmits and receive the signal.These layers are in form of stack.

## Layers in ANN-
There are many layers in ANN like Dense,Convolutional layer,pooling layer,recurrent layer,normalization layer,etc. according to the processing they perform.For ex.Convulational layer deals with the data associated with image,recurrent layer deals with time series,etc.
Basically the model contains input layer,connections,hidden layer1,connections,...,connections,output layer.Here the output from the previous layer is served as input to next layer via the connections.Each connection has the **weight** associated with it which depicts the strength of connection.The weighted sum of inputs is calculated and then the activation function is applied on it.The output of this result is served as output.

## Activation Function-
Activation function basically follows the layer.In ANN,activation function of a neuron defines the output of a neuron ,given the set of inputs.when a neuron is activated significantly,its value is 1 and when it is not,its value is 0.There are many activation functions like Sigmoid,relu,etc.
In **Sigmoid function** when the neuron is near to one,it will transform to a no. less than 1,but near to it,when the no.is negative it will transform it to a no. close to zero and when it is near to zero,it will transform into some no. between 0 and 1.
In **RELU function** (Rectified Linear Unit) when the no.is negative,it transforms it to zero and when it is positive ,the output is the positive no.itself.

## Training a Neural Network-
The Neutral Network has to predict the output most accurately via the weights assigned to connections and activation function.So the network needs to optimize its algorithm so that it gives its output most accurately.One of the optimizer or the method to reduce the loss function is **Stochastic Gradient Decend**.SGD is just a method to direct the neurons such that our output is most likely.Here ,the loss function refers to the difference between the original output and the output gained by just passing some of the input data.Basically the outputs are just probability that how much the model thinks of the desired output.
Example-The prediction of Cats and dogs.Let the output be cat.Suppose by giving some input,we got outut as Cat-0.75 and Dog-0.25.So there is loss and we have to minimize it.This can be done by giving more no. of input datasets so that the model learns from it and its accuracy increases.The method used in this is SGD.

## Learning of a Neural Network-
Firstly the weights are assigned randomly to the connections and output is spitted,so the initial loss is calculated.Then for a particular weight,we calculate d(loss)/d(weight) and multiply it by a small no.(usually between 0.001 to 0.1) called as **learning rate**.For each such weight this value is calculated and the value of weight is updated by ditchng its previous value by new_weight=old_weight-calculated value.For each such epoch,our model is more close to the desired result.

## Loss in Neural Network-
There are many types of loss functions,we will be mainly dealing with 'mean squared loss function'.At the end of every epoch,the loss function is calculated which determines the accuracy of the model.Basically the loss function intakes all the errors/loss and perform squaring and taking its mean.The main objective of SGD is minimizing its value.

## Learning Rate-
How fast is the model being close to the correct output is determined by the learning rate.Learning rate is a kind of typical parameter which is tested and tuned according to the model.Small value of LR tells that we require more footsteps to reach the output whereas Large value of LR tells that we require less footsteps but with larger gap between them.

## Datasets-
There are three types-Training Set,Testing set and Validation Set.Training data and validation data have inputs and desired labels(output).Testing data contains inputs but not hte labels.Validation Set is there to ensure that whether the model is not overfitting.We will see overfitting later.

## Predicting in Neural Networks-
The neural network intakes testdata(unlabeled one) and processes it to give the output.The output is basically what the model thinks according to whatever it has learned during its training.

## Overfitting-
The model is said to be overfit when the model is extremely trained and even slight deviation in test data can lead to wrong output.That is the model is unable to generalize the algorithm.Hence,validation data is used to overcome issue of overfitting(a most common issue in many neural networks).One way to do this is to increase the no. of data sets so that our model is trained diversly which can be done by just adding new data or by **data augmentation**(topic to be discussed later) .Also,in order to reduce complexity in model,we can reduce layers in the model or simply the no. of neurons.Another method to reduce overfitting is **Dropout** i.e.it ignores some nodes/neurons randomly in our model ,hence our model improves from this.

## Underfitting-
Underfitting is opposite to Overfitting.The model is not even able to classify the desired output from training data sets.This caused by lower accuracy or higher losses.One way to overcome this is to complicate our model.This can be done by increasing layers,neurons or even the new data sets.Or else just reduce the dropout in the model.

## Superwised,Unsuperwised and Semi- Learning-
When our training set is labelled ,it is Superwised.

When our training set is not labelled,it is USL.Accuracy is not measured in USL.In USL,the model analyzes the features and groups predicted data into some categories(clusters) according to their similar characteristics.Clustering Algorithm basically starts learning data eventhough the model is unaware of labels.The next thing is that **Autoencoders** use the USL network to recontruct the output data from input data.Basically,autoencoders just transfrom the input data into some new transformed data(ex-can be used in preprocessing and denoising of images). 

When we have huge amount of data,and practical labelling to each and every input is tedious job,then semi-superwised learning method is used.In this,some input samples have the labels,whereas rest dont.labelled data set is passed and then unlabelled dataset is passed which makes predictions of output on basis of previously trained data.And then the labelles are named,the method used is **pseudo labelling**.Thus,we trained huge dataset using semi-superwised model.

## Data Augmentation-
It is the method to make new dataset from existing dataset by just transforming its features(for ex.flipping,rotating,cropping img).Data augmentation is used when we have less no. of input datasets(usually for overfitting).

## One-Hot Encoding-
Suppose our Neural-Network is an image classifier of cats,dogs,etc.So,in which format will we able to get the desired output? So,each output is assigned a one dimensional vector of 0's and 1's.The element at each position of the vector will represent its output.So,for three outputs as cats,dogs,lizards,[1,0,0] simply means that our output is a cat.Just make sure the mapping of outputs to corresponding labels in the vector,i.e.whether it is cat,dog,lizard or dog,cat,lizard.(can be studied by keras library)

## _Convolutional Neural Network_
A famous Artificial Neural Network mostly used in Image Analysis.This neural network is expert in finding the patterns of an image and then predicting the outputs.There are basically hidden layers called as **convolutional layers** which are determining patterns/images.The neurons perform the convolutional operation in thes convolutional layers.

First,we will discuss the overview of it.With each convolutional layer,it has some filters associated with it which detect the patterns.Here,pattern in an image can be anything like edges,circles,birds,eyes,stones,triangles,etc.Likewise these filters are differentiated as edge-detector filter,circle-detector filter,etc.The deeper our network goes,more sophisticated this filter become(i.e.they start to recognize minute details also).

Consider the example of digit recognition of MNIST dataset.The filters associated with each convolutional network is decided by us.A filter is just basically a matrix which convolve(sliding by performing certain operations) throughout the image in specific order to give transformed matrix.Here,the convolutional operation is basically the dot product of kernal matrix with original image pixel values.So,basically filters are pattern detectors.More the complicated patterns,deeper the network and more complicated filters in the convolutional layer.

## Zero Padding in CNN-
When we convolve a 3x3 matrix over original image of 28x28 ,we get 26x26 sized image.This is because our convolutional matrix is restricted to slide only to some positions.Here,the output's dimension is less than that of original image,i.e.we can say that we have lost some data from the image.Suppose our neural network contains more such layers,then eventually our resultant image would have lost a huge amount of valuable data,which is a problem.Also,in case of image having small size,it would be meaningless after applying matrix over it.The problem is that the edges here are not being convolved as the interior portion is being done.Hence,Zero padding is used to overcome this problem.Basically,zero padding is method to preserve the dimensions of input.An extra layer of zero pixels is added to all four sides of original image so that the output image has same dimension as that of the input implying the valuable data is not lost.Sometimes,more than one layer is also added to the input image.
There are two types of Padding-**valid padding** and **same padding**.When no padding is required in a convolutional layer,it is called valid and when padding is required to retain the dimensions,it is called 'same' padding.

## Max Pooling in CNN-
Max pooling is a method which is applied to image after it has undergone the convolutional operation,i.e.original image->convoluted image->max-pooled image.So,now we have convoluted image.Let's define the size and stride of the empty matrix to slide over the image,say 2x2 is the size and stride is 2.Stride is nothing but the pixels to be moved by the empty matrix after each max-pooled operation.Now,we have 26x26 dimensioned image.We slide the empty matrix(also called as pool) and pick out the max.pixel value from those 4 pixel values in the pool(4 since it is 2x2).Likewise carry out this same max.value operation over the intervals of 2 pixels and so we get the final output image of 13x13.Hence,called 'max-pooling'.

The main aim of max-pooling is to reduce the computational load of the neural network.As the resolution is reduced by this method,the network can actually look at the larger areas of image at a timegoing forward,which reduces the parameters in the network and reduces the computational load.Also,max-pooling is useful for the problem of overfitting.

## _Backpropagation_-
Recall the concepts of Stochastic Gradient Descend,Loss function,etc.

**Intution-**
First we **pass the data** as input to the neural network via **forward propagation**.In this,we repeatedly pass on the data by calculating the weighted sum of previous layer activation output with the corresponding weights and passing this sum to next layer's activation function.We do this until we reach the output layer.And at this point we **calculate the loss**.That is the **Stochastic Gradient descend** minimizes the loss.SGD does this _by calculating the gradient of loss function and updating the weights_.To do the actual calculation of the gradient,SGD uses the method called as **Backpropagation**.There is Math behind all these scenes,Backprop uses calculus for all this.

For ex-we have 4 output ranges and for a certain input we have highest activation for 2nd output.So,what the neural network does is minimization of loss,i.e.maximize the activation for 2nd and minimize the activation for 1st,3rd and 4th.So,to do this the network has to update the weights for just previous connections.But it can't,since the values of weights are dependent upon the activation of just previous neurons of layer.Similarly,these values are dependent on their previous connections and node values.
Hence,we understood that how the SGD seeks the help of n-1,n-2,n-3,...3,2,1 layers and its in-between connections to update the values of weights in order to reduce the loss function.Hence,we say that the process is taking place in backward direction.Hence,said to be backpropagation.

**Mathematical Notations-**












