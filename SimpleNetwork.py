import numpy as np

# calculates the 
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

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 3) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset
    # n = len(data[:,0])

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        o1 = sigmoid(dot(x, self.w) + self.b)
        y_pred = o1
        
        
        # Calculate the partial derivatives
        # dC / dwi 
        dcdw1 = -2 * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[0]
        dcdw2 = -2 * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[1]
        dcdw3 = -2 * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)*x[2]
        
        # dC / db
        dCdb = -2 * np.sum(y_true - y_pred)*deriv_sigmoid(dot(x, self.w) + self.b)
        
        # Updating wi's
        self.w1 = self.w1 - learn_rate*dcdw1
        self.w2 = self.w2 - learn_rate*dcdw2
        self.w3 = self.w3 - learn_rate*dcdw3
        
        self.w = np.array([self.w1, self.w2, self.w3])
        
        # Updating b
        self.b -= learn_rate*dCdb
        

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        # print('used this data:', )
        loss = mse_loss(all_y_trues, y_preds)


# EXAMPLE OF USE
# #Define dataset
# data = np.array([
#     [0,0,0],
#     [1,0,0],
#     [0,1,0],
#     [1,1,1]
#     ])

# all_y_trues = np.array([0, 0, 1, 1])


# # Train our neural network!
# network = OurNeuralNetwork()
# network.train(data, all_y_trues)


# # Make some predictions
# x1 = np.array([0, 0, 0]) # 128 pounds, 63 inches
# x2 = np.array([1, 1, 1])  # 155 pounds, 68 inches
# print("x1: %.3f" % network.feedforward(x1)) # 0.951 - F
# print("x2: %.3f" % network.feedforward(x2)) # 0.039 - M