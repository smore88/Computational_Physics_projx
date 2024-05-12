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


class TrainHopfieldNet:
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
		return delta_E
	
	def checkConvergence(self):
		for i in range(len(self.state)):
			delta_E = 2 * self.state[i] * (np.dot(self.weights[i], self.state) + self.biases[i])
			if delta_E < 0:
				return False
		return True
	
	def visualize_state(self):
		image = self.state.reshape((10, 10))
		plt.matshow(image, cmap='viridis')
		plt.colorbar() 
		plt.title("Hopfield Network State")
		plt.show()
	
	def run_simulation(self, corrupted_state, max_iterations=50000):
		self.state = corrupted_state
		for i in range(max_iterations):
			self.updateState()			
			if i % 100 == 0:
				print(f"Step: {i}")
				self.visualize_state()
				if self.checkConvergence():
					break
		return self.state
	


				