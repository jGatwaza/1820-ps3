from bayesNet import constructEmptyBayesNet, Factor
from inference_ind import inferenceByEnumeration, inferenceByVariableElimination

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

assert(inferenceByEnumeration(bayesNet, ["P", "C"], {"B": "chasing"}) == inferenceByVariableElimination(bayesNet, ["P", "C"], {"B": "chasing"}))
assert(inferenceByEnumeration(bayesNet2, ["P"], {}) == inferenceByVariableElimination(bayesNet2, ["P"], {}))
print("All Variable Elimination Tests Passed!")