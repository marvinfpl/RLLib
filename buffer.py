class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []

    def __len__(self):
        return len(self.states)

    def append(self, state, action, reward, done, value, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logprobs.clear()

class ReplayBuffer:
    def __init__(self, alpha, beta, eps):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.priorities = []
    
    def append(self):
        pass

    def sample(self):
        pass

    def update(self):
        pass