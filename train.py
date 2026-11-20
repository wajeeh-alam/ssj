import torch

from ssj.env import CombatEnv
from ssj.models.actor_critic import ActorCritic
from ssj.ppo.buffer import RolloutBuffer
from ssj.ppo.rollout import collect_rollout
from ssj.ppo.advantages import compute_gae
from ssj.ppo.trainer import ppo_update
from ssj.evaluate import evaluate

def train():
    env = CombatEnv()

    model = ActorCritic(state_dim=5, action_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = RolloutBuffer()

    num_updates = 5
    rollout_steps = 256

    print("Before training evaluation: ", evaluate(model, num_episodes=100))

    for update in range(num_updates):
        buffer.clear()

        last_value = collect_rollout(
            env, model, buffer, num_steps=rollout_steps)

        advantages, returns = compute_gae(buffer, last_value)
        loss = ppo_update(model, optimizer, buffer, advantages, returns)

        print(f"Update {update + 1}/{num_updates} | Loss: {loss:.4f}")

    print("After training evaluation: ", evaluate(model, num_episodes=100))

if __name__ == "__main__":
    train()