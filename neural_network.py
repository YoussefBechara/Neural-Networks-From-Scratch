import math
import pandas as pd
import random 

class Loss_functions:
    def mean_squared_error(self, actual_y, y_predicted):
        squared_error = 0
        for y, y_hat in zip(actual_y, y_predicted):
            squared_error+= (y-y_hat)**2
        return squared_error / len(actual_y)
    
class Optimizers:
    def __init__(self) -> None:
        self.learning_rate = 0.01
    
    def Adam(self):
        self.learning_rate = 0.001

class Activation_Functions:
    def perform_activation_function(self,activation,x):
        self.x = x
        if activation=='sigmoid':
            return self.sigmoid()
        elif activation=='relu':
            return self.relu()
        elif activation=='leaky_relu':
            return self.leaky_relu()
        elif activation=='tanh':
            return self.tanh()
        
    def sigmoid(self):
        return 1/(1+math.exp(-self.x))
    
    def relu(self):
        return max(0, self.x)
    
    def leaky_relu(self, small_constant=0.01):
        if self.x < 0:
            return small_constant*self.x
        else:
            return self.x

    def tanh(self):
        return (math.exp(self.x)-math.exp(-self.x))/(math.exp(self.x)+math.exp(-self.x))
    #there are def more activation funcs but these are the most popular ones
    
class NeuralNetwork(Activation_Functions, Loss_functions, Optimizers):
    def __init__(self) -> None:
        self.output = 0
    def fit(self, X_train, y_train): #only supports pandas dataframes
        self.X_train, self.y_train = X_train , y_train
        
        try:
            self.input_layer_node_count = len(X_train.columns)
        except AttributeError:
            self.input_layer_node_count = 1
        self.create_neural_network()
        
    def back_propagation(self, epochs=5): #takes curr neural network and outputs it with the optimal weights
        def forward_pass(self):
            list_of_y_predicted = []
            for i in range(len(self.X_train[X_train.columns[0]])):#this is wrong cuz inp is flattened
                #inp layer
                inp_node_value = list(self.X_train)  #this is also wrong for the same reason
                for r in range(self.hidden_layer_node_count):
                    weighted_sum = 0
                    for c in range(len(self.input_layer)):
                        weighted_sum += inp_node_value[c]*self.input_layer[c]['weights'][r]
                    weighted_sum += self.hidden_layers[0][r]['bias'][0]
                    weighted_sum_after_activation_function = self.perform_activation_function(activation=self.hidden_layers[0][r]['activation function'][0],x=weighted_sum)
                    self.hidden_layers[0][r]['node_value'].append(weighted_sum_after_activation_function)
                #hidden layers
                for r in range(self.hidden_layer_count):
                    for c in range(self.hidden_layer_node_count):
                        for z in range(self.hidden_layer_node_count):
                            self.hidden_layers[r][c]                        
                #output layer
        def compute_loss(self):
            pass
        def backward_pass(self):
            pass
        def update_weights_bias(self):
            pass
        for i in range(epochs):
            forward_pass()
            compute_loss()
            backward_pass()
            update_weights_bias()
    
    def configure(self,hidden_layer_count, hidden_layer_node_count, optimizer, loss, activation, output_layer_activation):
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_node_count = hidden_layer_node_count
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.output_layer_activation = output_layer_activation
        
    def create_neural_network(self):
        self.input_layer = [{'weights':[random.uniform(-0.05, 0.05) for i in range(self.hidden_layer_node_count)]} for i in range(len(self.X_train.columns))]
        self.hidden_layers = []
        for r in range(self.hidden_layer_count):
            self.hidden_layers.append([])
            for c in range(self.hidden_layer_node_count):
                if r!= self.hidden_layer_node_count-1:
                    self.hidden_layers[r].append({'activation function':self.activation,'weights':[random.uniform(-0.05, 0.05) for i in range(self.hidden_layer_node_count)],'bias': 0, 'node_value': 0})
                else:
                    self.hidden_layers[r].append({'activation function':self.activation,'weights':[random.uniform(-0.05, 0.05) for i in range(self.num_of_classes)],'bias': 0, 'node_value': 0})
        if self.is_regression_or_classification(self.y_train) == 'regression' or self.is_regression_or_classification(self.y_train) == 'classification':
            self.output_layer = [{'activation': self.output_layer_activation, 'output':self.output}]
        elif self.is_regression_or_classification(self.y_train) == 'multi-classification':
            self.output_layer = [{'activation': self.output_layer_activation, 'output':self.output} for i in range(self.num_of_classes)]
        
        self.neural_network = [self.input_layer, self.hidden_layers, self.output_layer]
        print(self.neural_network)
        
    def is_regression_or_classification(self, target):
        if list(target.unique()) == [0, 1] or list(target.unique()) == [1, 0]:
            return 'classification'
        elif target.dtype == str and len(list(target.unique()))>2:
            self.num_of_classes = len(list(target.unique()))
            return 'multi-classification'
        else:
            return 'regression'
    
    def evaluate(self):
        pass
    
    def predict(self):
        pass
       
    def show_output_function():#only if the data is 2d
        pass
    
if __name__ == '__main__':
    df = pd.read_csv('test_scores.csv')
    X_train, y_train = df[['math','cs']], df.passed
    model = NeuralNetwork()
    model.configure(hidden_layer_count=2, hidden_layer_node_count=5, optimizer='SGD', loss='MSE', activation='RELU', output_layer_activation='sigmoid')
    model.fit(X_train, y_train)
