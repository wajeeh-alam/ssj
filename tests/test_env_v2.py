
from ssj.env_v2 import CombatEnvV2


def test_reset():
    env = CombatEnvV2()
    obs = env.reset(player_action=1)

    assert env.player_health == 100
    assert env.ai_health == 100
    assert env.player_stamina == 100
    assert env.ai_stamina == 100
    assert env.distance == 1

    assert len(obs) == 16


def test_blast_telegraph():
    env = CombatEnvV2()
    obs = env.reset(player_action=1)

    telegraph = obs[-6:]

    assert telegraph == (0, 1, 0, 0, 0, 0)


def test_observation_normalized():
    env = CombatEnvV2()
    obs = env.reset(player_action=0)

    assert all(0.0 <= value <= 1.0 for value in obs)

def test_spend_stamina():
    env = CombatEnvV2()
    env._spend_resources("ai", 0)

    assert env.ai_stamina == 85


def test_charge_has_no_cost():
    env = CombatEnvV2()
    env._spend_resources("ai", 4)

    assert env.ai_ki == 100
    assert env.ai_stamina == 100


def test_attack_blocked():
    env = CombatEnvV2()
    env.distance = 0

    damage = env._calculate_damage(
        attacker_action=0,
        defender_action=3,
        attacker_valid=True,
        defender_valid=True
    )

    assert damage == 0


def test_attack_close_range():
    env = CombatEnvV2()
    env.distance = 0

    damage = env._calculate_damage(0, 4, True, True)

    assert damage == 12


def test_attack_out_of_range():
    env = CombatEnvV2()
    env.distance = 2

    damage = env._calculate_damage(0, 4, True, True)

    assert damage == 0


def test_blast_dodged():
    env = CombatEnvV2()

    damage = env._calculate_damage(1, 2, True, True)

    assert damage == 0


def test_invalid_dodge_does_not_counter_blast():
    env = CombatEnvV2()

    damage = env._calculate_damage(1, 2, True, False)

    assert damage == 20

def test_blast_close_range():
    env = CombatEnvV2()
    env.distance = 0

    damage = env._calculate_damage(1, 4, True, True)

    assert damage == 10


def test_blast_long_range():
    env = CombatEnvV2()
    env.distance = 2

    damage = env._calculate_damage(1, 4, True, True)

    assert damage == 20

def test_move_cooldown():
    env = CombatEnvV2()

    assert env._can_perform_action("ai", 5)

    env.ai_move_cooldown = True

    assert not env._can_perform_action("ai", 5)
    assert env._can_perform_action("ai", 0)

def test_attack_charge():
    env = CombatEnvV2()
    env.reset(player_action=0)
    env.distance = 0
    env.ai_ki = 20

    # Player attacks, AI charges
    env.step(player_action=0, ai_action=4)

    assert env.ai_health == 88
    assert env.ai_ki == 20

def test_charge_not_interrupted_at_range():
    env = CombatEnvV2()
    env.reset(player_action=0)
    env.distance = 2
    env.ai_ki = 20

    env.step(player_action=0, ai_action=4)

    assert env.ai_health == 100
    assert env.ai_ki == 45

def test_player_block_recovers_ki():
    env = CombatEnvV2()
    env.reset(player_action=3)
    env.player_ki = 20
    env.ai_ki = 40

    env.step(player_action=3, ai_action=4)

    assert env.player_ki == 30
    assert env.ai_ki == 65

def test_move_closer():
    env = CombatEnvV2()
    env.reset(player_action=4)
    env.distance = 2

    env.step(player_action=4, ai_action=5)

    assert env.distance == 1
    assert env.ai_move_cooldown is True


def test_dodge_farther():
    env = CombatEnvV2()
    env.reset(player_action=4)
    env.distance = 1

    env.step(player_action=4, ai_action=2)

    assert env.distance == 2


def test_move_and_dodge_cancel():
    env = CombatEnvV2()
    env.reset(player_action=5)
    env.distance = 1

    env.step(player_action=5, ai_action=2)

    assert env.distance == 1

def test_stamina_regeneration():
    env = CombatEnvV2()
    env.reset(player_action=4)
    env.ai_stamina = 50

    env.step(player_action=4, ai_action=4)

    assert env.ai_stamina == 60


def test_reward_for_damage():
    env = CombatEnvV2()
    env.reset(player_action=4)
    env.distance = 0

    obs, reward, done, info = env.step(
        player_action=4,
        ai_action=0
    )

    assert env.player_health == 88
    assert reward == 12 / 20
    assert done is False


def test_max_turns():
    env = CombatEnvV2()
    env.reset(player_action=4)
    env.turns = 99

    obs, reward, done, info = env.step(
        player_action=4,
        ai_action=4
    )

    assert done is True
    assert info["winner"] == "draw"

def test_prepare_turn():
    env = CombatEnvV2()
    env.reset()

    obs = env.prepare_turn(player_action=1)

    assert env.telegraphed_action == 1
    assert obs[-6:] == (0, 1, 0, 0, 0, 0)


def test_telegraph_cleared_after_step():
    env = CombatEnvV2()
    env.reset()
    env.prepare_turn(player_action=1)

    obs, reward, done, info = env.step(
        player_action=1,
        ai_action=2
    )

    assert env.telegraphed_action is None
    assert obs[-6:] == (0, 0, 0, 0, 0, 0)


def test_winner_logic():
    env = CombatEnvV2()
    env.reset()
    env.prepare_turn(player_action=4)

    env.player_health = 10
    env.ai_health = 50
    env.turns = 99

    obs, reward, done, info = env.step(
        player_action=4,
        ai_action=4
    )

    assert done is True
    assert info["winner"] == "ai"
    assert reward == 10