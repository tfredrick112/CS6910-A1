
# CS6910 Assignment 1
## Authors: Dhruvjyoti Bagadthey EE17B156, D Tony Fredrick EE17B154


This Github repository presents the codes for assignment 1 of CS6910. For ease of uploading and wandb integration, we have uploaded different versions of the code according to the tasks performed. 

**1. Assignment_1_functions.ipynb** is the code which contains all the necessary functions for training the Neural Network model used in questions 4-10. The only difference of the other codes with this version is that this version provides the flexibility to choose different number of neurons in each layer (e.g [64,56,128]), instead of having the same number of neurons in every layer. This feature was omitted in other versions of the code for simplicity in using wandb and because the tasks specified did not call for such a flexibility as asked in question 2. It contains the optimiser routines sgd, momentum based GD, nesterov accelerated GD, RMSProp, Adam, NAdam as asked in question 3. This code tests on the MNIST data.

**2. Assignment_1_Deep_Learning_wandb.ipynb** is the code which contains 120 wandb sweeps and simulations for the fashion MNIST dataset with the categorical crossentropy loss function asked in question 4. This code also generates the validation accuracy v created plot in question 5, the confusion matrix in question 7 as well as the parallel coordinates plot of question 6.

**3. Assignment_1_MSE_wandb.ipynb** is the code which contains 91 wandb sweeps and simulations for the Fashion MNIST datsaset with the Mean squared error loos function as specified by question 8.

**4. Assignment_1_MNIST.ipynb** is the code which trains on the 3 recommended hyperparameter configurations for the MNIST dataset as specified in question 10.




The NN training framework:

Our codes are based on a procedural framework and make no use of classes for NN models like keras does for the simplicity of understanding as well as the code. Our code works only for classification tasks and by default assumes that the activation function for the last layer is softmax. This was done for simplicity as because the tasks involved in the assignment did not call for a different output layer activation. For hyperparameter search and training one needs to use only 3 functions provided in the code. NN_fit, NN_evaluate and NN_predict. 

**1. NN_fit()**

The NN_fit() function takes the training data, the validation data and the hyperparameters and Trains a NN specified by num_neurons and num_hidden. 
Our code provides flexibility in choosing the following hyperparameters:


* **learning_rate**: the learning rate 


* **activation_f**: activation functions for all the layers except the last layer which is softmax  $\epsilon$ (sigmoid, ReLU, tanh)                          


* **init_mode**: initialization mode $\epsilon$ (random_uniform, random_normal, xavier)


* **optimizer**: optimization routine $\epsilon$ (sgd, momentum, nesterov, RMSprop, Adam, nadam)


* **bach_size**: minibatch size


* **loss**: loss function $\epsilon$ (MSE, Categorical Crossentropy)


* **epochs**: number of epochs to be used


* **L2_lamb**: lambda for L2 regularisation of weights


* **num_neurons**: number of neurons in every hidden layer


* **num_hidden**: number of hidden layers


In addition, the function in Assignment_1_functions.ipynb code provide the additional flexibility to have different number neurons in each layers. This was not utilised in the wandb plots for simplicity as well as for the instructions given in question 6 of the assignment. 

The function returns 

* params: a dictionary containing all weights and biases. for e.g params["Wi"] is the Weight matrix from i-1 th layer to the ith layer.

* epoch costs: a list containing Cost function values vs epochs

The function contains 2 loops, one epoch loop and one batch loop. Note the optimizers are not implemented with those loops instead they are called as parameter updates hence the nomenclature update_params_sgd etc. Hence to include a new optimiser routine, we simply need to include a function that can update the parameters in each epoch in each batch. The necessary spaces and instructions regarding nesterov is provided in the source code with comments.

A summary of the update stages is given below:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vThojdzpoROd58z6yJqOMQFrnlhuPmhhuZYZvgQoQmGLcL9G3F12Or1bWxzBT_HCA2DSfSvhz9h4Uqv/pub?w=960&amp;h=720">

**2. NN_predict()**

The function NN_predict() takes the parameters, the activation function and the data for which predictions are to be made, performs one pass by forward propagation and returns the output labels.

**3. NN_evaluate()**

The function NN_evaluate() takes the parameters, the activation function, train and test data along with their respective labels and calculates and prints the train accuracy, test accuracy and the classification report.



```python

```
