import numpy as np

class RBM_MNIST:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.zeros((self.num_visible, self.num_hidden))
        self.visible_biases = np.zeros(self.num_visible)
        self.hidden_biases = np.zeros(self.num_hidden)
    
    def initializeBiases(self):
        self.visible_biases = np.random.uniform(0, 0, size=self.num_visible)
        self.hidden_biases = np.random.uniform(0, 0, size=self.num_hidden)
    
    def initializeWeights(self):
        self.weights = np.random.uniform(-0.01, 0.01, size=(self.num_visible, self.num_hidden))
    
    def prob_v_given_h(self, h):
        eff_mag = np.dot(self.weights, h) + self.visible_biases
        prob_v = 1 / (1 + np.exp(eff_mag)) 
        return prob_v

    def get_newV(self, h):
        prob_v = self.prob_v_given_h(h)
        newV = np.where(np.random.rand(self.num_visible) < prob_v, 1, 0) 
        return newV
    
    def prob_h_given_v(self, v):
        eff_mag = np.dot(v, self.weights) + self.hidden_biases
        prob_h = 1 / (1 + np.exp(eff_mag)) 
        return prob_h

    def get_newH(self, v):
        prob_h = self.prob_h_given_v(v)
        newH = np.where(np.random.rand(self.num_hidden) < prob_h, 1, 0) 
        return newH
    
    def gibbs_sampling(self, k, num_samples):
        vf = []
        hf = []
        for _ in range(num_samples):
            v = np.random.choice([0, 1], self.num_visible)
            h = np.random.choice([0, 1], self.num_hidden)
            for _ in range(k):
                h = self.get_newH(v)
                v = self.get_newV(h)
            vf.append(v)
            hf.append(h)
        return vf, hf
    
    def compute_energy(self, v, h):
        term1 = -np.sum(v @ self.weights @ h)
        term2 = -np.sum(self.visible_biases * v)
        term3 = -np.sum(self.hidden_biases * h)
        return term1 + term2 + term3
    
    def gibbs_sampling2(self, k, h):
        for _ in range(k):
            v = self.get_newV(h) 
            h = self.get_newH(v)  
        return v, h

    def train_rbm(self, data, learning_rate, mini_batch_size, k, num_epochs):
        for _ in range(num_epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), mini_batch_size):
                mini_batch = data[i:i + mini_batch_size]
                dW = np.zeros_like(self.weights)
                da = np.zeros_like(self.visible_biases)
                db = np.zeros_like(self.hidden_biases)
                for v in mini_batch:
                    h = self.get_newH(v)
                    dW -= np.outer(v, h)
                    da -= v
                    db -= h
                    vk, hk = self.gibbs_sampling2(k, h=h)
                    dW += np.outer(vk, hk)
                    da += vk
                    db += hk
                self.weights -= learning_rate * (dW/mini_batch_size)
                self.visible_biases -= learning_rate * (da/mini_batch_size)
                self.hidden_biases -= learning_rate * (db/mini_batch_size)
    
    def freeEnergy1(self, v):
        vbias_term = -np.dot(v, self.visible_biases)
        term1 = -np.dot(v, self.weights) - self.hidden_biases
        term2 = np.dot(v, self.weights) + self.hidden_biases
        hidden_term_sum = -np.sum(np.logaddexp(term1, term2))
        return vbias_term + hidden_term_sum
    
    def freeEnergy2(self, v):
        vbias_term = -np.dot(v, self.visible_biases)
        term1 = np.dot(v, self.weights) + self.hidden_biases
        hidden_term_sum = -np.sum(np.log(1 + np.exp(term1)))
        return vbias_term + hidden_term_sum
    
    def average_free_energy(self, data):
        total_free_energy = 0
        for v in data:
            total_free_energy += self.freeEnergy1(v)
        return total_free_energy / len(data)
    
    def train_rbm2(self, data, learning_rate, mini_batch_size, k, num_epochs):
        free_energies = []
        initial_free_energy = self.average_free_energy(data)
        # free_energies.append(nrg)
        for _ in range(num_epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), mini_batch_size):
                mini_batch = data[i:i + mini_batch_size]
                dW = np.zeros_like(self.weights)
                da = np.zeros_like(self.visible_biases)
                db = np.zeros_like(self.hidden_biases)
                for v in mini_batch:
                    h = self.get_newH(v)
                    dW -= np.outer(v, h)
                    da -= v
                    db -= h
                    vk, hk = self.gibbs_sampling2(k, h=h)
                    dW += np.outer(vk, hk)
                    da += vk
                    db += hk
                self.weights -= learning_rate * (dW/mini_batch_size)
                self.visible_biases -= learning_rate * (da/mini_batch_size)
                self.hidden_biases -= learning_rate * (db/mini_batch_size)
            
            # free_energies.append(self.average_free_energy(data))
            final_free_energy = self.average_free_energy(data)
            delta = final_free_energy - initial_free_energy
            free_energies.append(delta)
            initial_free_energy = final_free_energy

        return free_energies
        
            

