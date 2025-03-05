from bayesNet import constructEmptyBayesNet, Factor
from inference_group import approximateInferenceByRejectionSampling, approximateInferenceByLikelihoodWeighting
import matplotlib.pyplot as plt

# Define variables, edges, and variable domains
variableList = ['P', 'B', 'C'] # P = Pacman, B = Blinky (Ghost), C = Clyde (Ghost)
edgeTuplesList = [('P', 'B'), ('P', 'C')]
variableDomainsDict = {
    'P': ['normal', 'energized'],
    'B': ['chasing', 'scared'],
    'C': ['chasing', 'scared']
}

# Construct an empty BayesNet
bayesNet = constructEmptyBayesNet(variableList, edgeTuplesList, variableDomainsDict)

# Define CPTs as Factors and populate them
pacmanCPT = Factor(['P'], [], variableDomainsDict)
pacmanCPT.setProbability({'P': 'energized'}, 0.2)
pacmanCPT.setProbability({'P': 'normal'}, 0.8)

blinkyCPT = Factor(['B'], ['P'], variableDomainsDict)
blinkyCPT.setProbability({'B': 'chasing', 'P' : 'normal'}, 0.9)
blinkyCPT.setProbability({'B': 'scared', 'P' : 'normal'}, 0.1)
blinkyCPT.setProbability({'B': 'chasing', 'P' : 'energized'}, 0.3)
blinkyCPT.setProbability({'B': 'scared', 'P' : 'energized'}, 0.7)

clydeCPT = Factor(['C'], ['P'], variableDomainsDict)
clydeCPT.setProbability({'C': 'chasing', 'P': 'normal'}, 0.75)
clydeCPT.setProbability({'C': 'scared', 'P': 'normal'}, 0.25) 
clydeCPT.setProbability({'C': 'chasing', 'P': 'energized'}, 0.1)
clydeCPT.setProbability({'C': 'scared', 'P': 'energized'}, 0.9)

# Set the CPTs for the BayesNet
bayesNet.setCPT('P', pacmanCPT)
bayesNet.setCPT('B', blinkyCPT)
bayesNet.setCPT('C', clydeCPT)

# Print the constructed BayesNet
print(bayesNet)

variableList2 = ['P', 'G', 'E'] # P = Pacman state, G = Ghost distance from Pacman, E = Ghost Eaten by Pacman
edgeTuplesList2 = [('P', 'E'), ('G', 'E')]
variableDomainsDict2 = {
    'E': ['yes', 'no'],
    'P': ['energized', 'normal'],
    'G': ['far', 'close']  
}

bayesNet2 = constructEmptyBayesNet(variableList2, edgeTuplesList2, variableDomainsDict2)

eatenCPT = Factor(['E'], ['G', 'P'], variableDomainsDict2)
eatenCPT.setProbability({'E' : 'no', 'G' : 'far', 'P' : 'normal'}, 0.9)
eatenCPT.setProbability({'E' : 'yes', 'G' : 'far', 'P' : 'normal'}, 0.1)
eatenCPT.setProbability({'E' : 'no', 'G' : 'close', 'P' : 'normal'}, 0.8)
eatenCPT.setProbability({'E' : 'yes', 'G' : 'close', 'P' : 'normal'}, 0.2)
eatenCPT.setProbability({'E' : 'no', 'G' : 'far', 'P' : 'energized'}, 0.5)
eatenCPT.setProbability({'E' : 'yes', 'G' : 'far', 'P' : 'energized'}, 0.5)
eatenCPT.setProbability({'E' : 'no', 'G' : 'close', 'P' : 'energized'}, 0.3)
eatenCPT.setProbability({'E' : 'yes', 'G' : 'close', 'P' : 'energized'}, 0.7)

pacmanCPT = Factor(['P'], [], variableDomainsDict2)
pacmanCPT.setProbability({'P': 'energized'}, 0.2)
pacmanCPT.setProbability({'P': 'normal'}, 0.8)

ghostCPT = Factor(['G'], [], variableDomainsDict2)
ghostCPT.setProbability({'G': 'close'}, 0.75)
ghostCPT.setProbability({'G': 'far'}, 0.25)

bayesNet2.setCPT('E', eatenCPT)
bayesNet2.setCPT('P', pacmanCPT)
bayesNet2.setCPT('G', ghostCPT)


exact_prob1 = 0.6923077
exact_prob2 = .8

# Check if the are within .1 of what the exact inference returns
approx_prob = approximateInferenceByRejectionSampling(bayesNet, ["P", "C"], {"B": "chasing"}, 1000).getProbability({"B" : "chasing", "P" : "normal", "C" : "chasing"})
assert(abs(approx_prob - exact_prob1) <= .1)

approx_prob = approximateInferenceByRejectionSampling(bayesNet2, ["P"], {}, 1000).getProbability({"P" : "normal"})
assert(abs(approx_prob - exact_prob2) <= .1)

print("All Rejection Sampling Tests Passed!")


approx_prob = approximateInferenceByLikelihoodWeighting(bayesNet, ["P", "C"], {"B": "chasing"}, 1000).getProbability({"B" : "chasing", "P" : "normal", "C" : "chasing"})
assert(abs(approx_prob - exact_prob1) <= .1)

approx_prob = approximateInferenceByLikelihoodWeighting(bayesNet2, ["P"], {}, 1000).getProbability({"P" : "normal"})
assert(abs(approx_prob - exact_prob2) <= .1)

print("All Importance Sampling Inference Tests Passed!")


exact_prob =  0.00967741935483871

rs_error = []
lw_error = []

for num_samples in range(20, 1001, 20):

    rs_prob = approximateInferenceByRejectionSampling(bayesNet, ["P", "B"], {"C": "chasing"}, num_samples).getProbability( {"B" : "chasing", "P" : "energized", "C" : "chasing"})
    lw_prob = approximateInferenceByLikelihoodWeighting(bayesNet, ["P", "B"], {"C": "chasing"}, num_samples).getProbability( {"B" : "chasing", "P" : "energized", "C" : "chasing"})
        
    # Compute error 
    rs_error.append(abs(exact_prob - rs_prob))
    lw_error.append(abs(exact_prob - lw_prob))
   
# Plot the errors
plt.figure(figsize=(8, 5))
plt.plot(range(20, 1001, 20), rs_error, linestyle='--', color='green', label='Rejection sampling')
plt.plot(range(20, 1001, 20), lw_error, linestyle='-', color='brown', label='Likelihood weighting')

plt.xlabel("Number of samples")
plt.ylabel("Error")
plt.legend()
plt.show()

