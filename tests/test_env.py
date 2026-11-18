from ssj.env import CombatEnv
import pytest

def test_reset():
    env = CombatEnv()
    obs = env.reset()

    assert obs == (1.0, 1.0, 1.0, 1.0, 0.0)
    assert env.player_health == 100
    assert env.ai_health == 100
    assert env.player_ki == 100
    assert env.ai_ki == 100
    assert env.turns == 0

def test_attack_vs_attack():
    env = CombatEnv()
    env.reset()

    obs, reward, done, info = env.step(0, 0) # player and ai attack
    assert obs == (0.9, 0.9, 1.0, 1.0, 0.01) 
    assert reward == 0  # No damage dealt
    assert not done
    assert info["winner"] == None

    def test_blast_vs_dodge():
        env = CombatEnv()
        env.reset()

        obs, reward, done, info = env.step(1, 2)  # player blasts, AI dodges
        assert obs == (1.0, 1.0, 0.9, 1.0, 0.01)
        assert reward == 0.0
        assert not done
        assert info["winner"] == None

    def test_invalid_ki():
        env = CombatEnv()
        env.ai_ki = 0

        obs, reward, done, info = env.step(1, 1)

        assert obs == (1.0, 0.8, 0.9, 0.1, 0.01)
        assert reward == -2.0
        assert not done
        assert info["winner"] is None

        assert info["player_action_valid"] is True
        assert info["ai_action_valid"] is False

    def test_simultaneous_knockout():
        env = CombatEnv()
        env.player_health = 10
        env.ai_health = 10

        obs, reward, done, info = env.step(0, 0)

        assert env.player_health == 0
        assert env.ai_health == 0
        assert done is True
        assert info["winner"] == "draw"
        assert reward == 0


    def test_ai_wins():
        env = CombatEnv()
        env.player_health = 10

        obs, reward, done, info = env.step(3, 0)

        assert done is True
        assert info["winner"] == "ai"
        assert reward == 6.0  # 1 damage reward + 5 win bonus


    def test_player_wins():
        env = CombatEnv()
        env.ai_health = 10

        obs, reward, done, info = env.step(0, 2)

        assert done is True
        assert info["winner"] == "player"
        assert reward == -6.0


    def test_turn_limit():
        env = CombatEnv()
        env.turns = 99
        env.player_health = 80
        env.ai_health = 60

        obs, reward, done, info = env.step(2, 2)

        assert done is True
        assert info["winner"] == "player"
        assert reward == -5.0


    def test_step_after_done():
        env = CombatEnv()
        env.player_health = 10

        env.step(3, 0)

        with pytest.raises(RuntimeError):
            env.step(0, 0)


    def test_invalid_action():
        env = CombatEnv()

        with pytest.raises(ValueError):
            env.step(5, 0)

        with pytest.raises(ValueError):
            env.step(0, -1)