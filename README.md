# machine-learning
machine-learning codes

Neural Network is an important part of machine learning. If trained properly, it can be used to predict unseen data with a high precision and accuracy. However, for a beginner it can be a little tough to implement them in practical situations. So here is a small step by step implementation for a Neural Network in python. Please feel free to ask any questions and suggest changes if required.

Note: the most important part of building a Neural network is to use your own visualization. Try to visualize and have an rough idea of how many layers you want, how the weights will be joining layers, how much neurons you want in each layer. Which activation function (sigmoid, tanh, ReLU, leaky ReLu, maxout norm) you want to use and what function you will use to evaluate loss. Whether it will be a softmax, logistic loss, cross entropy loss or any other loss. Once you visualize it, it becomes easy. Go with the forward pass and then perform backward pass for calculating gradient. This gradient will be used to adjust weights and improve accuracy of prediction.

Step 1: Import all the libraries you might need. Here we basically need NumPy (numerical python ) library. You can either import whole library or some specific functions of it. If you want to learn NumPy from scratch, Here is the book I wrote for beginners, you can download and read it by going at this link: 'https://www.amazon.com/dp/B0716J5769'. This will help you better understand the NumPy library. 

Step 2: feed input  training data columns and corresponding labels. 

Step 3: Separate input data and labels if they are in same matrix. Here we have seperated input data in matrix x (3,3) and vector y of labels.  

Step 4: Set hyperparameters for the NN. you can always change and experiment with the hyperparameters
a)	learning_rate=0.01 
b)	hidden_units=4
c)	weight matrix
d)	bias unit
e)	Regularization Parameter
f)	Optimization
We have not used Regularization and optimization in this code, to keep it simple and short. 

Step 5: Perform Forward pass:
Forward pass is used to predict the output. It has a linear function which is followed by a non linear function. Here we have used a linear function:
∑wixi= W1X1+W2X2+W3X3
and sigmoid function as non-linearity: sigmoid(x)

Step 5: calculate loss or error. 

Loss= Expected output-predicted output

Step 6: Perform backward pass

Calculate gradient for sigmoid function and multiply it by the error to find delta. We will use this delta to update weight and predict the output again with better accuracy.

Step 7: update weights and biases

Step 8: supply Test data to predict label.

