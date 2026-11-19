import torch
from torch.distributions import Categorical

def compute_policy_loss(
        model, 
        states,
        actions,
        old_log_probs, 
        advantages, 
        clip_epsilon=0.2):

    logits, values = model(states)
    dist = Categorical(logits=logits)

    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped_loss = ratio * advantages
    clipped_objective = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_objective = -torch.min(unclipped_loss, clipped_objective).mean()

    return policy_objective


def compute_value_loss(model, states, returns):
    _, values = model(states)

    # Makes both tensors the same shape for MSE computation
    values = values.squeeze(-1)

    value_loss = torch.nn.functional.mse_loss(values, returns)
    return value_loss

def ppo_update(model, optimizer, buffer, advantages, returns,
               epochs=4, minibatch_size=64):
    # Buffers -> Tensors
    states = torch.stack(buffer.states)
    actions = torch.tensor(buffer.actions, dtype=torch.long)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize advantages to train stabley
    advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    for epoch in range(epochs):
        # Shuffle the indices for minibatch sampling
        indices = torch.randperm(len(states)) 

        for start in range(0, len(buffer), minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            actor_loss = compute_policy_loss(
                model,
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_advantages
            )

            critic_loss = compute_value_loss(
                model,
                batch_states,
                batch_returns
            )

            logits, _ = model(batch_states)
            dist = Categorical(logits=logits)
            mean_entropy = dist.entropy().mean()
        
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * mean_entropy
        
            # Gradient descent step
            optimizer.zero_grad()
        
            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    return total_loss.item()