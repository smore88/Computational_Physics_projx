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

class HopfieldNetwork:
	def __init__(self, size, name):
		self.name = name
		self.size = size * size
		self.state = np.array([0]*(self.size))
		self.biases = np.array([0]*(self.size))
		self.weights = np.zeros((self.size, self.size))
	
	def initializeState(self):
		for i in range(len(self.state)):
			self.state[i] = np.random.choice([-1, 1])
	
	def initializeBiases(self):
		for i in range(len(self.biases)):
			self.biases[i] = np.random.uniform(-1, 1)
	
	def initializeWeights(self):
		for row in range(self.weights.shape[0]):
			for col in range(self.weights.shape[1]):
				if row == col:
					self.weights[row][col] = 0
				else:
					weight = np.random.uniform(-1, 1)
					self.weights[row][col] = weight
					self.weights[col][row] = weight
	
	def getEnergy(self):
		term1 = -0.5 * np.sum(self.weights * np.outer(self.state, self.state))
		term2 = np.sum(self.biases * self.state)
		totalEnergy = term1 + term2
		return totalEnergy
	
	def updateState(self):
		i = np.random.randint(0, self.size)
		delta_E = 2 * self.state[i] * (np.dot(self.weights[i], self.state) + self.biases[i])
		if delta_E < 0:
			self.state[i] = -self.state[i]
		return delta_E, i
	
	def checkConvergence(self):
		for i in range(len(self.state)):
			delta_E = 2 * self.state[i] * (np.dot(self.weights[i], self.state) + self.biases[i])
			if delta_E < 0:
				return False
		return True
	
	def run_simulation(self):
		energies = []
		current_energy = self.getEnergy()
		energies.append(current_energy)
		MAX_ITERATIONS = 2500
		iterationCounter = 0
		while iterationCounter < MAX_ITERATIONS:
			delta_E, i = self.updateState()
			if delta_E < 0:
				current_energy += delta_E				
			if iterationCounter % 100 == 0:
				energies.append(current_energy)
				if self.checkConvergence():
					break
			iterationCounter += 1
		
		return energies

	


				