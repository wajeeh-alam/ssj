import random
import torch


def get_opponent_action():
    return random.randint(0, 3)


def collect_rollout(env, model, buffer, num_steps):
    state = env.reset()

    for _ in range(num_steps):

        # 1. Convert observation to tensor array
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32
        )

        # 2. AI chooses action
        action, log_prob, value = model.act(state_tensor)

        player_action = get_opponent_action()

        # 4. Execute combat turn
        next_state, reward, done, info = env.step(
            player_action,
            action
        )

        # 5. Store experience in the rollout buffer for GAE later
        buffer.add(
            state=state_tensor,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            done=done
        )

        if done:
            state = env.reset()
        else:
            state = next_state

    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32
        )
        _, last_value = model(state_tensor)

    return last_value.item()