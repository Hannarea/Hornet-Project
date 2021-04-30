import numpy as np




def dot(inputs, weights):
    return np.dot(inputs, weights)

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)


# We may need to replace this with cross entropy??
def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    
    '''
    A neural network with:
      - 3 input 
      - an output layer with 1 neuron (o1)
    '''
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w = np.array([self.w1, self.w2, self.w3])
        
        
        # Bias
        self.b = np.random.normal()
  

  

    def feedforward(self, x):
        # x is a numpy array with 3 elements.
        o1 = sigmoid(dot(x, self.w) + self.b)
        return o1

  


  


    def train_bce(self, data, y_true):
        learn_rate = 0.2
        epochs = 1000
        epsilon = 1e-5
 
        for epoch in range(epochs): 
        # Calculate the partials with respect to the loss function 
         
            dLdw1 = 0
            for i in range(len(y_true)):
                z = np.dot(self.w, data[i, :])
                y_pred = sigmoid(z)
                dLdw1 -= (y_true[i]/(y_pred+epsilon) - (1-y_true[i])/(1-y_pred+epsilon))*y_pred*(1-y_pred)*data[i,0]
                 
              
            dLdw2 = 0
            for i in range(len(y_true)):
                z = np.dot(self.w, data[i, :])
                y_pred = sigmoid(z)
                dLdw2 -= (y_true[i]/(y_pred+epsilon) - (1-y_true[i])/(1-y_pred+epsilon))*y_pred*(1-y_pred)*data[i,1]
                  
             
            dLdw3 = 0
            for i in range(len(y_true)):
                z = np.dot(self.w, data[i, :])
                y_pred = sigmoid(z)
                dLdw3 -= (y_true[i]/(y_pred+epsilon) - (1-y_true[i])/(1-y_pred+epsilon))*y_pred*(1-y_pred)*data[i,2]
                   
             
            dLdb = 0
            for i in range(len(y_true)):
                z = np.dot(self.w, data[i, :])
                y_pred = sigmoid(z)
                dLdb -= (y_true[i]/(y_pred+epsilon) - (1-y_true[i])/(1-y_pred+epsilon))*y_pred*(1-y_pred)
                  
             
            # Preform gradient descent
            self.w1 -= learn_rate*dLdw1
            self.w2 -= learn_rate*dLdw2
            self.w3 -= learn_rate*dLdw3
            self.w = np.array([self.w1, self.w2, self.w3])
            self.b -= learn_rate*dLdb




    def train_gradient_descent(self, data, all_y_trues):
 
        '''
        Trains the model via gradient descent
        '''
        
        learn_rate = 0.2
        epochs = 500
        n = len(all_y_trues) 
        
        for epoch in range(epochs):
            # calculate the partial derivatives with the current weights and biased
            q1 = 0
            for i in range(n):
                z = np.dot(self.w, data[i, :]) +self.b
                a = sigmoid(z)
                q1 += (all_y_trues[i] - a)*a*(1-a)* data[i, 0]
                dCdw1 = -2/n * q1 
            
            q2 = 0
            for i in range(n):
                z = np.dot(self.w, data[i, :]) +self.b
                a = sigmoid(z)
                q2 += (all_y_trues[i] - a)*a*(1-a)* data[i, 1]
                dCdw2 = -2/n * q2 
              
            q3 = 0
            for i in range(n):
                z = np.dot(self.w, data[i, :]) +self.b
                a = sigmoid(z)
                q3 += (all_y_trues[i] - a)*a*(1-a)*data[i,2]
                dCdw3 = -2/n * q3 
            
            qb = 0
            for i in range(n):
                z = np.dot(self.w, data[i, :]) +self.b
                a = sigmoid(z)
                qb += (all_y_trues[i] - a)*a*(1-a)
                dCdb = -2/n * qb
            
            
            # update the weights and biased
            self.w1 -= learn_rate* dCdw1
            self.w2 -= learn_rate* dCdw2
            self.w3 -= learn_rate* dCdw3
            self.w = np.array([self.w1, self.w2, self.w3])
            
            self.b -= learn_rate* dCdb
 
          
          

    def train(self, data, all_y_trues):
        '''
        trains data based on the method discussed in: 
        https://victorzhou.com/blog/intro-to-neural-networks/   
        '''
        learn_rate = 0.2
        epochs = 500 # number of times to loop through the entire dataset
        n = len(data[:,0])
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                o1 = sigmoid(dot(x, self.w) + self.b)
                y_pred = o1
                n = len(all_y_trues)
                
                # Calculate the partial derivatives
                # dC / dwi 
                dcdw1 = -2/n * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[0]
                dcdw2 = -2/n * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[1]
                dcdw3 = -2/n * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[2]
                
                # dC / db
                dCdb = -2/n * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)
                
                # Updating wi's
                self.w1 = self.w1 - learn_rate*dcdw1
                self.w2 = self.w2 - learn_rate*dcdw2
                self.w3 = self.w3 - learn_rate*dcdw3
                
                self.w = np.array([self.w1, self.w2, self.w3])
                
                # Updating b
                self.b -= learn_rate*dCdb
        

        


# # EXAMPLE OF USE
# #Define dataset
# data = np.array([
#     [0,0,0],
#     [0, 0, 0],
#     [1,0,0],
#     [0,1,1],
#     [1,1,1]
#     ])

# all_y_trues = np.array([0, 0, 0, 1, 1])


# # Train our neural network!
# network = OurNeuralNetwork()
# network.train_bce(data, all_y_trues)


# # # Make some predictions
# x1 = np.array([0, 0, 0]) 
# x2 = np.array([1, 1, 1])
# print('should be near 0') 
# print("\t x1: %.3f" % network.feedforward(x1)) 
# print('should be near 1')
# print("\t x2: %.3f" % network.feedforward(x2)) 