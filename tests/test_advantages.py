import pytest
from ssj.ppo.advantages import compute_gae


def test_gae_terminal():
    class FakeBuffer:
        rewards = [1.0, 2.0]
        values = [0.5, 1.0]
        dones = [False, True]

        def __len__(self):
            return len(self.rewards)

    advantages, returns = compute_gae(
        FakeBuffer(),
        last_val=0.0,
        gamma=1.0,
        lam=1.0
    )

    assert advantages == pytest.approx([2.5, 1.0])
    assert returns == pytest.approx([3.0, 2.0])