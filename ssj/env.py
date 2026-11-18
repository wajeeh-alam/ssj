class CombatEnv:
    def __init__(self):
        self.max_health = 100
        self.max_turns = 100
        self.max_ki = 100

        self.reset()

    def reset(self):
        self.player_health = self.max_health
        self.ai_health = self.max_health
        self.player_ki = self.max_ki
        self.ai_ki = self.max_ki
        self.turns = 0

        return self._get_observation()

    def _get_observation(self):
        return self.player_health / 100, self.ai_health / 100, self.player_ki / 100, self.ai_ki / 100, self.turns / self.max_turns


    def step(self, player_action, ai_action):
        if (
            self.player_health <= 0
            or self.ai_health <= 0
            or self.turns >= self.max_turns
        ):
            raise RuntimeError("Cannot step: match has already ended")

        # Update health/ki from actions, process player then ai
        # 0: Attack, 1: Blast, 2: Dodge, 3: Block

        valid_actions = (0, 1, 2, 3)
        winner = None
        
        player_action_valid = True
        ai_action_valid = True

        if player_action not in valid_actions:
            raise ValueError(f"Invalid player action: {player_action}")
        if ai_action not in valid_actions:
            raise ValueError(f"Invalid AI action: {ai_action}")

        if player_action == 1 and self.player_ki < 20:
            player_action_valid = False

        if ai_action == 1 and self.ai_ki < 20:
            ai_action_valid = False

        damage_to_player = 0
        damage_to_ai = 0

        # Player attacks AI
        if player_action_valid:
            if player_action == 0 and ai_action != 3:
                damage_to_ai = min(10, self.ai_health)

            elif player_action == 1 and ai_action != 2:
                damage_to_ai = min(20, self.ai_health)

        # AI attacks player
        if ai_action_valid:
            if ai_action == 0 and player_action != 3:
                damage_to_player = min(10, self.player_health)

            elif ai_action == 1 and player_action != 2:
                damage_to_player = min(20, self.player_health)

        # Update health based on damage

        if damage_to_player > 0:
            self.player_health -= damage_to_player
            self.player_health = max(self.player_health, 0)

        if damage_to_ai > 0:
            self.ai_health -= damage_to_ai
            self.ai_health = max(self.ai_health, 0)

        # Update ki after actions

        if player_action_valid and player_action == 1:
            self.player_ki -= 20
            

        if ai_action_valid and ai_action == 1:
            self.ai_ki -= 20

        self.turns += 1
        self.player_ki = min(self.player_ki+10, self.max_ki)
        self.ai_ki = min(self.ai_ki+10, self.max_ki)
        done = self.player_health <= 0 or self.ai_health <= 0 or self.turns >= self.max_turns
        reward = (damage_to_player - damage_to_ai) / 10

        if done:
            if self.ai_health > self.player_health:
                winner = "ai"
                reward += 5
            elif self.player_health > self.ai_health:
                winner = "player"
                reward -= 5
            else:
                winner = "draw"

        return (
            self._get_observation(),
            reward,
            done,
            {
                "winner": winner,
                "damage_to_player": damage_to_player,
                "damage_to_ai": damage_to_ai,
                "player_action_valid": player_action_valid,
                "ai_action_valid": ai_action_valid,
            }
        )

env = CombatEnv()
env.reset()