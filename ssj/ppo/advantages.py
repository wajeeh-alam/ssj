""" Compute Generalized Advantage Estimation (GAE) for the rollout buffer to 
calculate advantages for policy gradient updates (train NN)."""

def compute_gae(buffer, last_val, gamma=0.99, lam=0.95):
    advantages = [0.0] * len(buffer)
    gae = 0.0

    # Go through the rollout in reverse (A(t) requires A(t+1))
    for step in reversed(range(len(buffer))):
        # determine next val state
        if step == len(buffer) - 1:
            next_val = last_val 
        else:
            next_val = buffer.values[step + 1]

        # Non-terminal state check: if done, don't consider value of next state
        non_terminal = 1.0 - float(buffer.dones[step])
        delta = buffer.rewards[step] + gamma * next_val * non_terminal - buffer.values[step]
        
        # Compute GAE recursively and store in advantages list
        gae = delta + gamma * lam * non_terminal * gae
        advantages[step] = gae

    returns = [
        advantages[step] + buffer.values[step]
        for step in range(len(buffer))
    ]

    return advantages, returns