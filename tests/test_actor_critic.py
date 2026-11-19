from ssj.models.actor_critic import ActorCritic
import torch

model = ActorCritic(state_dim=4, action_dim=4)

state = torch.tensor(
    [80, 40, 50, 20],
    dtype=torch.float32
)

logits, value = model(state)

print("Logits:", logits)
print("Value:", value)
print("Logits shape:", logits.shape)
print("Value shape:", value.shape)