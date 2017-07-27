#CODE for a 3 layer neural network in python

from numpy import exp,array,random,dot,zeros,sum,int8

#create a Neural Network class and initialise all variables

class Neural_Network():
    def __init__(self):
        #setting hyperparameters for the network like bias, weights, hidden_unit neurons and learning rate. 
        #we can also add regularlarization term for the weight.
        
        self.hidden_units=4                               #define no. of neurons in each hidden unit
        self.learning_rate=0.99                            #define learning rate, try with different options (like 0.001,0.1,1)
        self.bias_in=zeros((1,4),dtype=int8)              # bias for input  hidden layer
        self.bias_hidden=zeros((1,4),dtype=int8)          # bias for first hidden layer
        self.bias_out=zeros((1,1),dtype=int8)             # bias for second hidden layer
        
        
        #set weights to an intial random value and normalize it to mean 1
        self.synaptic_weights0 = 2 * random.random((3, self.hidden_units)) - 1                    #weights for first layer
        self.synaptic_weights1 = 2 * random.random((self.hidden_units, self.hidden_units)) - 1    #weights for second layer
        self.synaptic_weights2 = 2 * random.random((self.hidden_units, 1)) - 1                    #weights for third layer
        
    #define the activation function, there is a list of activation functions you can use(tanh,ReLu,leaky Relu etc.)
    #here we have used sigmoid function
    def sigmoid(self, x):
        return (1 / (1 + exp(-x)))
    
    #define the derivative of your sigmoid function
    def derivative_sigmoid(self, x):
        derivative=(self.sigmoid(x) * (1 - self.sigmoid(x)))
        return derivative
    
    #train the Neural Network
    def train(self, training_inputs, training_labels, epochs):
        random.seed(5)
        for i in range(epochs):
            
            #forward propogation:
            
            # we are trying to non-linearity over a linear function. so hidden layer output is sigmoid function applied over z 
            #where z is z=weight*x +bias
            hidden_layer1_output = self.sigmoid(dot(training_inputs, self.synaptic_weights0)+self.bias_in)
            hidden_layer2_output = self.sigmoid(dot(hidden_layer1_output, self.synaptic_weights1)+self.bias_hidden)
            predicted_output= self.sigmoid(dot(hidden_layer2_output, self.synaptic_weights2)+self.bias_out)
            #hidden_layer1_output, hidden_layer2_output, predicted_output= self.test(training_inputs)
            
            #calculate error or loss
            error = training_labels - predicted_output
            
            #Backward propogation to find gradients
            delta_output=error*self.learning_rate*self.derivative_sigmoid(predicted_output)
            hidden2_error=delta_output.dot(self.synaptic_weights2.T)
            delta_hidden2=hidden2_error*self.learning_rate*self.derivative_sigmoid(hidden_layer2_output)
            hidden1_error=delta_hidden2.dot(self.synaptic_weights1.T)
            delta_hidden1=hidden1_error*self.learning_rate*self.derivative_sigmoid(hidden_layer1_output)
            
            #update weights and bias at each layer  according to gradients calculated by backward prop
            self.updated_weight0=self.synaptic_weights0 + training_inputs.T.dot(delta_hidden1)*self.learning_rate
            self.updated_weight1=self.synaptic_weights1 + hidden_layer1_output.T.dot(delta_hidden2)*self.learning_rate
            self.updated_weight2=self.synaptic_weights2 + hidden_layer2_output.T.dot(delta_output)*self.learning_rate
            self.updated_biasin=self.bias_in+sum(delta_hidden1, axis=0)*self.learning_rate
            self.updated_biashidden=self.bias_hidden+sum(delta_hidden2, axis=0)*self.learning_rate
            self.updated_biasout=self.bias_out+sum(delta_output, axis=0)*self.learning_rate
            
    #create test function to predict the value of label after training with updated weights and biases
            
    def test(self, inputs):
        hidden_layer1_output = self.sigmoid(dot(inputs, self.updated_weight0)+self.updated_biasin)
        hidden_layer2_output = self.sigmoid(dot(hidden_layer1_output, self.updated_weight1)+self.updated_biashidden)
        predicted_output= self.sigmoid(dot(hidden_layer2_output, self.updated_weight2)+self.updated_biasout)
        return hidden_layer1_output,hidden_layer2_output, predicted_output
    
if __name__ == "__main__":
    #Intialise the neural network.
    nn = Neural_Network()
    
    # supply the training set data. We have 4 examples, each consisting of 3 input values and 1 output value.
    data=array([[1,0,1,0],[1,1,0,1],[0,1,1,1],[1,0,0,0]])
    training_inputs=data[:,:-1] 
    training_labels=data[:,-1:] 
    
    # Train the neural network using a training set data and no. of epochs.
    nn.train(training_inputs, training_labels, 60000)
       
    # Test the neural network with a new situation.
    print "predict label for input [1,0,1]"
    hidden_layer1_output,hidden_layer2_output, predicted_output = nn.test(array([1,0,1]))
    print "predicted_output=" 
    print predicted_output