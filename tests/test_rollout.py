from ssj.env import CombatEnv
from ssj.models.actor_critic import ActorCritic
from ssj.ppo.buffer import RolloutBuffer
from ssj.ppo.rollout import collect_rollout


def test_collect_rollout():
    env = CombatEnv()

    model = ActorCritic(
        state_dim=5,
        action_dim=4
    )

    buffer = RolloutBuffer()

    last_value = collect_rollout(
        env=env,
        model=model,
        buffer=buffer,
        num_steps=100
    )

    assert len(buffer) == 100
    assert len(buffer.states) == 100
    assert len(buffer.actions) == 100
    assert len(buffer.rewards) == 100
    assert len(buffer.log_probs) == 100
    assert len(buffer.values) == 100
    assert len(buffer.dones) == 100

    print("Collected transitions:", len(buffer))
    print("First action:", buffer.actions[0])
    print("First reward:", buffer.rewards[0])
    print("Last value:", last_value)