import random
import torch
from ssj.env import CombatEnv


def evaluate(model, num_episodes=100):
    action_counts = {
        "attack": 0,
        "blast": 0,
        "dodge": 0,
        "block": 0
    }
        
    env = CombatEnv()

    wins = 0
    losses = 0
    draws = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        info = {"winner": None}
        
        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)

            with torch.no_grad():
                logits, _ = model(state_tensor)
                ai_action = torch.argmax(logits).item()

                action_names = ["attack", "blast", "dodge", "block"]
                action_counts[action_names[int(ai_action)]] += 1

            player_action = random.randint(0, 3)

            state, reward, done, info = env.step(player_action, ai_action)

        if info["winner"] == "ai":
            wins += 1
        elif info["winner"] == "player":
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_episodes": num_episodes,
        "win_rate": wins / num_episodes,
        "loss_rate": losses / num_episodes,
        "draw_rate": draws / num_episodes,
        "action_counts": action_counts
    }