'''
Store the following in the HopfieldNetwork
	- States Si of the neurons in an 1x100 array
		1. each index filled randomly with -1 or 1, (represents spin down, spin up like Ising Model)
	- Biases Bi of the neurons in an 1x100 array
		1. each index filled with random number(-1, 1)
	- Weights Wij in an 100 x 100 matrix 
		1. filled with random number(-1, 1), 
		2. diag = 0
		3. Wij = Wji

1. Need a function that updates the states according to the energy rule(choose a random state and update it)
	a) use that equation in the instructions for E = 1/2.......
	b) if E < 0; flip the spin -1 to 1, 1 to -1

2. Some function that determines if the network has converged(meaning wont change anymore)
	a) Network has reached a local min
	b) iterate each state in states, find E, check to see all E >= 0, if this is the case then network is converged

3. Function that computes the energy
4. Functions that set and return the state
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

class HopfieldNetDiagram:
	def __init__(self, state1, state2):
		self.size = len(state1)
		self.state1 = state1
		self.state2 = state2
		self.state = None
		self.biases = np.array([0]*(self.size))
		self.weights = np.zeros((self.size, self.size))
	
	def initializeBiases(self):
		for i in range(len(self.biases)):
			self.biases[i] = np.random.uniform(-1, 1)
	
	def initializeWeights(self):
		self.weights = (np.outer(self.state1, self.state1) + np.outer(self.state2, self.state2)) / 2
		np.fill_diagonal(self.weights, 0)
	
	def changeInputState(self):
		for i in range(len(self.state1)):
			if self.state1[i] == 0:
				self.state1[i] = -1
			if self.state2[i] == 0:
				self.state2[i] = -1

	def stateToBinary(self, state):
		state_list = list(state)
		state_list = [0 if x == -1 else 1 for x in state_list]
		state_str = ''.join(map(str, state_list))
		binary_number = int(state_str, 2)
		return binary_number
	
	def updateState(self, state, i):
		flipped_state = list(state)
		flipped_state[i] = -flipped_state[i]
		flipped_state_array = np.array(flipped_state)
		delta_E = 2 * flipped_state_array[i] * (np.dot(self.weights[i], flipped_state_array) + self.biases[i])
		flippedStateBinary = self.stateToBinary(flipped_state_array)
		return delta_E, flippedStateBinary

	def buildGraph(self, neurons):
		states = list(itertools.product([-1, 1], repeat=neurons))
		energy_landscape = defaultdict(list)

		for state in states:
			stateBinary = self.stateToBinary(state)
			for i in range(self.size):
				deltaE, flippedStateBinary = self.updateState(state, i)
				if deltaE < 0:
					energy_landscape[stateBinary].append(flippedStateBinary)
		return energy_landscape