import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(
            torch.as_tensor(state, dtype=torch.float32).clone().detach()
        )
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(float(torch.as_tensor(log_prob).item()))
        self.values.append(float(torch.as_tensor(value).item()))
        self.dones.append(bool(done))
        

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.states)