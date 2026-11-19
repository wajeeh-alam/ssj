
import torch

from ssj.models.actor_critic import ActorCritic
from ssj.ppo.trainer import compute_policy_loss


def test_policy_loss():
    model = ActorCritic(state_dim=5, action_dim=4)

    states = torch.rand(10, 5)
    actions = torch.randint(0, 4, (10,))

    with torch.no_grad():
        logits, _ = model(states)
        dist = torch.distributions.Categorical(logits=logits)
        old_log_probs = dist.log_prob(actions)

    advantages = torch.randn(10)

    loss = compute_policy_loss(
        model,
        states,
        actions,
        old_log_probs,
        advantages
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert any(
        p.grad is not None
        for p in model.actor.parameters()
    )


import torch

from ssj.env import CombatEnv
from ssj.models.actor_critic import ActorCritic
from ssj.ppo.buffer import RolloutBuffer
from ssj.ppo.rollout import collect_rollout
from ssj.ppo.advantages import compute_gae
from ssj.ppo.trainer import ppo_update


def test_ppo_updates_weights():
    env = CombatEnv()
    model = ActorCritic(state_dim=5, action_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    buffer = RolloutBuffer()

    last_value = collect_rollout(
        env, model, buffer, num_steps=100
    )

    advantages, returns = compute_gae(
        buffer, last_value
    )

    # Save weights before training
    old_weights = [
        param.detach().clone()
        for param in model.parameters()
    ]

    loss = ppo_update(
        model, optimizer, buffer, advantages, returns
    )

    # Check if at least one parameter changed
    weights_changed = any(
        not torch.equal(old, new)
        for old, new in zip(old_weights, model.parameters())
    )

    assert weights_changed
    assert torch.isfinite(torch.tensor(loss))

def test_ppo_runs_all_epochs():
    model = ActorCritic(state_dim=5, action_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = RolloutBuffer()

    # Expects:
    # ceil(100 / 64) = 2 minibatches per epoch
    # 4 epochs * 2 minibatches = 8 optimizer steps
    
    for _ in range(100):
        state = torch.rand(5)

        with torch.no_grad():
            action, log_prob, value = model.act(state)

        buffer.add(
            state=state,
            action=action,
            reward=torch.rand(1).item(),
            log_prob=log_prob,
            value=value,
            done=False
        )

    advantages = torch.randn(100).tolist()
    returns = torch.randn(100).tolist()

    ppo_update(
        model,
        optimizer,
        buffer,
        advantages,
        returns,
        epochs=4,
        minibatch_size=64
    )

    # See how many times Adam updated the parameters
    first_param = next(model.parameters())
    step_count = optimizer.state[first_param]["step"].item()

    assert step_count == 8