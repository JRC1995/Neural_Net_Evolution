
# coding: utf-8

# In[1]:


import numpy as np
import math
np.random.seed(0)

XOR_inputs = [[0,0],
              [0,1],
              [1,0],
              [1,1]]

XOR_outputs = [0,
               1,
               1,
               0]

XOR_outputs = np.asarray(XOR_outputs,np.float32)
XOR_inputs = np.asarray(XOR_inputs,np.float32)

def Neural_Network(w1,w2,b1,b2,inputs,solutions):
    
    layer1 = np.maximum(np.dot(inputs,w1) + b1,0)
    layer2 = np.maximum(np.dot(layer1,w2) + b2,0)
    neural_net_out = layer2.reshape(-1)
    
    reward = -np.sum(np.square(neural_net_out-solutions))

    return reward,neural_net_out

# hyperparameters
population_size = 100 # population size
sigma = 0.1 # noise standard deviation
learning_rate = 0.001 
hidden_size = 3

# Network parameters random initialization
w1 = np.random.randn(2,hidden_size) 
w2 = np.random.randn(hidden_size,1)
b1 = np.random.randn(hidden_size)
b2 = np.random.randn(1)

MAX_ITERATIONS = 100000
display_step = 20
convergence_threshold = 0.00001

for i in range(MAX_ITERATIONS):
    
    reward,neural_net_out =  Neural_Network(w1,w2,b1,b2,XOR_inputs,XOR_outputs)
  
    if i % display_step == 0:
        
        print('iter %d:\nsolution: %s, prediction: %s,\nfitness: %f\n' % 
              (i, str(XOR_outputs), str(neural_net_out), reward))
        
    if math.fabs(reward) <= convergence_threshold:
        print('AFTER CONVERGENCE:\nsolution: %s, prediction: %s,\nfitness: %f\n' % 
              (str(XOR_outputs), str(neural_net_out), reward))
        break

   # initialize memory for a population of w's, and their rewards
    Nw1 = np.random.randn(population_size, 2, hidden_size) # samples from a normal distribution N(0,1)
    Nw2 = np.random.randn(population_size, hidden_size, 1) # samples from a normal distribution N(0,1)
    Nb1 = np.random.randn(population_size, hidden_size) # samples from a normal distribution N(0,1)
    Nb2 = np.random.randn(population_size, 1) # samples from a normal distribution N(0,1)
    Rewards = np.zeros(population_size)
    
    for j in range(population_size):
        
        mutated_w1 = w1 + sigma*Nw1[j] #adding jitter
        mutated_w2 = w2 + sigma*Nw2[j] 
        mutated_b1 = b1 + sigma*Nb1[j] 
        mutated_b2 = b2 + sigma*Nb2[j] 
        
        Rewards[j],_ = Neural_Network(mutated_w1,mutated_w2,
                                      mutated_b1,mutated_b2,
                                      XOR_inputs,XOR_outputs) 
    very_small_number = 0e-20
    #Standardize rewards
    fitness_scores = (Rewards - np.mean(Rewards))/max(np.std(Rewards),very_small_number)
    
    #Next_generation
    w1 = w1 + (learning_rate/(sigma*population_size))* np.matmul(np.transpose(Nw1,(1,2,0)),fitness_scores)
    w2 = w2 + (learning_rate/(sigma*population_size))* np.matmul(np.transpose(Nw2,(1,2,0)),fitness_scores)
    b1 = b1 + (learning_rate/(sigma*population_size))* np.dot(Nb1.T,fitness_scores)
    b2 = b2 + (learning_rate/(sigma*population_size))* np.dot(Nb2.T,fitness_scores)
    
   
    

