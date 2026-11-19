from ssj.ppo.buffer import RolloutBuffer
import torch


def test_buffer():
    buffer = RolloutBuffer()

    buffer.add(
        state=[80, 40, 50, 20],
        action=0,
        reward=5,
        log_prob=-1.2,
        value=10,
        done=False
    )

    assert len(buffer) == 1
    assert len(buffer.states) == 1
    assert len(buffer.rewards) == 1
    assert buffer.actions[0] == 0

    buffer.clear()

    assert len(buffer) == 0
    assert len(buffer.dones) == 0

    buffer.add(
        state=torch.tensor([80, 40, 50, 20], dtype=torch.float32),
        action=0,
        reward=5,
        log_prob=torch.tensor(-1.2),
        value=torch.tensor([4.56]),
        done=False
    )

    assert len(buffer) == 1
    assert len(buffer.states) == 1
    assert len(buffer.dones) == 1

    buffer.clear()

    assert len(buffer) == 0