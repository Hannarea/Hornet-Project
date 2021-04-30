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
    
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    
    self.ww1 = np.array([self.w1, self.w2, self.w3])
    self.ww2 = np.array([self.w4, self.w5, self.w6])
    self.ww3 = np.array([self.w7, self.w8])
    

    # Bias
    self.b = np.random.normal()
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    

  def feedforward(self, x):
    # x is a numpy array with 3 elements.
    h1 = sigmoid(dot(x, self.ww1) + self.b)
    h2 = sigmoid(dot(x, self.ww2) + self.b1)
    h = np.array([h1, h2])

    o1 = sigmoid(dot(h, self.ww3) + self.b2)
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
        
        h1 = sigmoid(dot(x, self.ww1) + self.b)
        h2 = sigmoid(dot(x, self.ww2) + self.b1)
        h = np.array([h1, h2])

        o1 = sigmoid(dot(h, self.ww3) + self.b2)
        y_pred = o1
        
                
        # Calculate the partial derivatives
        # Neuron o1
        d_ypred_d_w7 = h1 * deriv_sigmoid(dot(h, self.ww3) + self.b2)
        d_ypred_d_w8 = h2 * deriv_sigmoid(dot(h, self.ww3) + self.b2)
        
        d_ypred_d_b3 = deriv_sigmoid(dot(h, self.ww3) + self.b2)

        d_ypred_d_h1 = self.w7 * deriv_sigmoid(dot(h, self.ww3) + self.b2)
        d_ypred_d_h2 = self.w8 * deriv_sigmoid(dot(h, self.ww3) + self.b2)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(dot(x, self.ww1) + self.b)
        d_h1_d_w2 = x[1] * deriv_sigmoid(dot(x, self.ww1) + self.b)
        d_h1_d_w3 = x[2] * deriv_sigmoid(dot(x, self.ww1) + self.b)
        d_h1_d_b1 = deriv_sigmoid(dot(x, self.ww1) + self.b)

        # Neuron h2
        d_h2_d_w4 = x[0] * deriv_sigmoid(dot(x, self.ww2) + self.b1)
        d_h2_d_w5 = x[1] * deriv_sigmoid(dot(x, self.ww2) + self.b1)
        d_h2_d_w6 = x[2] * deriv_sigmoid(dot(x, self.ww2) + self.b1)
        d_h2_d_b2 = deriv_sigmoid(dot(x, self.ww2) + self.b1)
        
    # Updating wi's
        # Neuron h1
        self.w1 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h1 * d_h1_d_w2
        self.w3 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h1 * d_h1_d_w3
        self.b -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w4 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h2 * d_h2_d_w4
        self.w5 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h2 * d_h2_d_w5
        self.w6 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h2 * d_h2_d_w6
        self.b1 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w7 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_w7
        self.w8 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_w8
        self.b2 -= -2 * np.sum(y_true - y_pred)*learn_rate * d_ypred_d_b3
        
        self.ww1 = np.array([self.w1, self.w2, self.w3])
        self.ww2 = np.array([self.w4, self.w5, self.w6])
        self.ww3 = np.array([self.w7, self.w8])
        
        #self.w3 = self.w3 - learn_rate*dcdw3
        
        self.w = np.array([self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8])
        

      # --- Calculate total loss at the end of each epoch
# =============================================================================
#       if epoch % 10 == 0:
#         y_preds = np.apply_along_axis(self.feedforward, 1, data)
#         loss = mse_loss(all_y_trues, y_preds)
#         print("Epoch %d loss: %.3f" % (epoch, loss))
# =============================================================================


# =============================================================================
# 
# # Define dataset
# data = np.array([
#   [3, -2, -1],  # Alice
#   [5, 25, 6],   # Bob
#   [6, 17, 4],   # Charlie
#   [-2, -15, -6], # Diana
# ])
# all_y_trues = np.array([
#   1, # Alice
#   0, # Bob
#   0, # Charlie
#   1, # Diana
# ])
# 
# # Train our neural network!
# network = OurNeuralNetwork()
# network.train(data, all_y_trues)
# 
# # Make some predictions
# emily = np.array([2, -7, -3]) # 128 pounds, 63 inches
# frank = np.array([10, 20, 2])  # 155 pounds, 68 inches
# print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
# print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
# =============================================================================


