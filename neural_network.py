import math
import pandas as pd
import random 
import numpy as np

class Loss_functions:
    def perform_loss_function(self,function,y_true, y_pred):
        if function=='MSE' or function == 'mse':
            return self.mean_squared_error(y_true, y_pred)
        elif function=='MAE' or function == 'mae':
            return self.mean_absolute_error(y_true, y_pred)
        elif function=='binary cross-entropy':
            y_true, y_pred = np.array(y_true) ,  np.array(y_pred)
            return self.binary_cross_entropy(y_true, y_pred)
        elif function=='cross-entropy':
            y_true, y_pred = np.array(y_true) ,  np.array(y_pred)
            return self.cross_entropy(y_true, y_pred)
        elif function == 'categorical cross-entropy':
            y_true, y_pred = np.array(y_true) ,  np.array(y_pred)
            return self.categorical_cross_entropy(y_true, y_pred)

    def mean_squared_error(self, actual_y, y_predicted):
        squared_error = 0
        for y, y_hat in zip(actual_y, y_predicted):
            squared_error+= (y-y_hat[0])**2
        return squared_error / len(actual_y)
    
    def mean_absolute_error(self, actual_y, y_predicted):
        absolute_error = 0
        for y, y_hat in zip(actual_y, y_predicted):
            absolute_error+= (y-y_hat)
        return absolute_error / len(actual_y)

    def categorical_cross_entropy(self, actual_y, y_predicted, epsilon=1e-15):
        y_pred = np.clip(y_predicted, epsilon, 1. - epsilon)  # Prevent log(0)
        return -np.sum(actual_y * np.log(y_pred)) / actual_y.shape[0]
    
    def cross_entropy(self, actual_y, y_predicted, epsilon=1e-15):
        y_pred = np.clip(y_predicted, epsilon, 1. - epsilon)  # Prevent log(0)
        return -np.mean(np.log(y_pred[np.arange(len(actual_y)), actual_y]))

    def binary_cross_entropy(self, actual_y, y_predicted, epsilon=1e-15):
        y_pred = np.clip(y_predicted, epsilon, 1. - epsilon)  # Prevent log(0)
        return -np.mean(actual_y * np.log(y_pred) + (1 - actual_y) * np.log(1 - y_pred))

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
        
    def back_propagation(self, epochs=1): #takes curr neural network and outputs it with the optimal weights
        def forward_pass():
            list_of_y_predicted = []
            for i in range(len(self.X_train[self.X_train.columns[0]])):#this is wrong cuz inp is flattened
                #inp layer
                
                inp_node_value = tuple(self.X_train[self.X_train.columns[d]].iloc[i] for d in range(len(self.X_train.columns)))   #this is also wrong for the same reason
                for r in range(self.hidden_layer_node_count):
                    weighted_sum = 0
                    for c in range(len(self.input_layer)):
                        weighted_sum += inp_node_value[c]*self.input_layer[c]['weights'][r]
                    weighted_sum += self.hidden_layers[0][r]['bias']
                    #print(self.hidden_layers[0][r]['activation'][0],weighted_sum)
                    weighted_sum_after_activation_function = self.perform_activation_function(activation=self.hidden_layers[0][r]['activation'],x=weighted_sum)
                    self.hidden_layers[0][r]['node_value'] = weighted_sum_after_activation_function
                
                #hidden layers
                for r in range(1,self.hidden_layer_count):
                    for c in range(self.hidden_layer_node_count):
                        weighted_sum = 0
                        for z in range(self.hidden_layer_node_count):
                            weighted_sum += self.hidden_layers[r-1][z]['node_value']*self.hidden_layers[r-1][z]['weights'][c]
                        weighted_sum += self.hidden_layers[r][c]['bias']
                        weighted_sum_after_activation_function = self.perform_activation_function(activation=self.hidden_layers[r][c]['activation'],x=weighted_sum)
                        self.hidden_layers[r][c]['node_value'] = weighted_sum_after_activation_function
                
                #output layer
                list_of_y_predicted.append([])
                for r in range(len(self.output_layer)):
                    for c in range(self.hidden_layer_node_count):
                        weighted_sum = 0
                        weighted_sum += self.hidden_layers[-1][c]['node_value']*self.hidden_layers[-1][c]['weights'][r]
                        weighted_sum += self.output_layer[r]['bias']
                    weighted_sum_after_activation_function = self.perform_activation_function(activation=self.output_layer[r]['activation'],x=weighted_sum)
                    self.output_layer[r]['node_value'] = weighted_sum_after_activation_function
                    list_of_y_predicted[i].append(weighted_sum_after_activation_function)
        
                if len(list_of_y_predicted[i]) > 1: #means that the case is a classification
                    #then do softmax
                    sum_of_exponentials = 0
                    for x in range(len(list_of_y_predicted[i])):
                        sum_of_exponentials += math.exp(list_of_y_predicted[i][x])
                    for x in range(len(list_of_y_predicted[i])):    
                        softmaxed_out = math.exp(list_of_y_predicted[i][x]) / sum_of_exponentials
                        del list_of_y_predicted[i][x]
                        list_of_y_predicted[i].insert(x, softmaxed_out)
            self.y_pred = list_of_y_predicted    
            
        def compute_loss():
            self.loss = self.perform_loss_function(self.loss_function, list(self.y_train) ,self.y_pred)

        def backward_pass():#outputs gradient for each weight and bias
            def derivative(x, y):
                return 
        def update_weights_bias():#optimizers
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
        self.loss_function = loss
        self.activation = activation
        self.output_layer_activation = output_layer_activation
        
    def create_neural_network(self):
        def flatten_inp(x):
            pass
        self.input_layer = [{'weights':[random.uniform(-0.05, 0.05) for i in range(self.hidden_layer_node_count)]} for i in range(len(self.X_train.columns))]
        self.hidden_layers = []
        for r in range(self.hidden_layer_count):
            self.hidden_layers.append([])
            for c in range(self.hidden_layer_node_count):
                if r!= self.hidden_layer_node_count-1:
                    self.hidden_layers[r].append({'activation':self.activation,'weights':[random.uniform(-0.05, 0.05) for i in range(self.hidden_layer_node_count)],'bias': 0, 'node_value': 0})
                else:
                    self.hidden_layers[r].append({'activation':self.activation,'weights':[random.uniform(-0.05, 0.05) for i in range(self.num_of_classes)],'bias': 0, 'node_value': 0})
        if self.is_regression_or_classification(self.y_train) == 'regression' or self.is_regression_or_classification(self.y_train) == 'classification':
            self.output_layer = [{'activation': self.output_layer_activation, 'output':self.output, 'bias': 0}]
        elif self.is_regression_or_classification(self.y_train) == 'multi-classification':
            self.output_layer = [{'activation': self.output_layer_activation,'bias': 0, 'output':self.output} for i in range(self.num_of_classes)]
        
        self.neural_network = [self.input_layer, self.hidden_layers, self.output_layer]
        self.back_propagation()
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
    model.configure(hidden_layer_count=2, hidden_layer_node_count=5, optimizer='SGD', loss='MSE', activation='tanh', output_layer_activation='sigmoid')
    model.fit(X_train, y_train)
