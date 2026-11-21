
class CombatEnvV2:
    ACTIONS = ("attack", "blast", "dodge", "block", "charge", "move")

    ACTION_COSTS = {
        0: ("stamina", 15),
        1: ("ki", 25),
        2: ("stamina", 20),
        3: ("stamina", 5),
        4: (None, 0),
        5: ("stamina", 10),
    }

    def __init__(self):
        self.max_health = 100
        self.max_ki = 100
        self.max_stamina = 100
        self.max_distance = 2
        self.max_turns = 100

        self.reset()

    def reset(self, player_action=None):
        self.player_health = self.max_health
        self.ai_health = self.max_health

        self.player_ki = self.max_ki
        self.ai_ki = self.max_ki

        self.player_stamina = self.max_stamina
        self.ai_stamina = self.max_stamina

        self.distance = 1
        self.turns = 0
        self.player_move_cooldown = False
        self.ai_move_cooldown = False

        # Telegraph for the next combat turn
        self.telegraphed_action = player_action

        return self._get_observation()

    def _get_observation(self):
        telegraph = [0.0] * 6

        if self.telegraphed_action is not None:
            telegraph[self.telegraphed_action] = 1.0

        return (
            self.player_health / self.max_health,
            self.ai_health / self.max_health,
            self.player_ki / self.max_ki,
            self.ai_ki / self.max_ki,
            self.player_stamina / self.max_stamina,
            self.ai_stamina / self.max_stamina,
            self.distance / self.max_distance,
            self.turns / self.max_turns,
            float(self.player_move_cooldown),
            float(self.ai_move_cooldown),
            *telegraph
        )

    def _can_perform_action(self, fighter, action):
        resource, cost = self.ACTION_COSTS[action]

        if action == 5:
            if getattr(self, f"{fighter}_move_cooldown"):
                return False

        if resource is None:
            return True

        available = getattr(self, f"{fighter}_{resource}")

        return available >= cost
    
    def _spend_resources(self, fighter, action):
        resource, cost = self.ACTION_COSTS[action]

        if resource is None:
            return
        
        attr = f"{fighter}_{resource}"
        available = getattr(self, attr)
        setattr(self, attr, available-cost)

    def _calculate_damage(
        self,
        attacker_action,
        defender_action,
        attacker_valid,
        defender_valid
    ):
        if not attacker_valid:
            return 0

        # Attack: 12 damage at close range
        if attacker_action == 0:
            if self.distance != 0:
                return 0

            if defender_action == 3 and defender_valid:
                return 0

            return 12

        # Blast: 20 damage at any distance
        if attacker_action == 1:
            if defender_action == 2 and defender_valid:
                return 0

            return 10 if self.distance == 0 else 20

        return 0

    def _charge_interrupted(self, opponent_action, opponent_valid):
        return (
            self.distance == 0
            and opponent_action == 0
            and opponent_valid
        )

    def prepare_turn(self, player_action):
        if player_action not in self.ACTION_COSTS:
            raise ValueError("Invalid player action")

        if (
            self.player_health <= 0
            or self.ai_health <= 0
            or self.turns >= self.max_turns
        ):
            raise RuntimeError("Match already ended")

        self.telegraphed_action = player_action

        return self._get_observation()

    def _update_distance(
        self,
        player_action,
        ai_action,
        player_valid,
        ai_valid
    ):
        change = 0

        if player_valid:
            if player_action == 5:
                change -= 1
            elif player_action == 2:
                change += 1

        if ai_valid:
            if ai_action == 5:
                change -= 1
            elif ai_action == 2:
                change += 1

        self.distance = max(
            0,
            min(self.max_distance, self.distance + change)
        )

    def step(self, player_action, ai_action):
        # Check whether the match has ended
        if (
            self.player_health <= 0
            or self.ai_health <= 0
            or self.turns >= self.max_turns
        ):
            raise RuntimeError("Match already ended")

        # Validate action IDs before using them
        if player_action not in self.ACTION_COSTS:
            raise ValueError("Invalid player action")

        if ai_action not in self.ACTION_COSTS:
            raise ValueError("Invalid AI action")

        # Player must perform the telegraphed action
        if player_action != self.telegraphed_action:
            raise ValueError("Action does not match telegraph")

        # Determine which actions are affordable
        player_valid = self._can_perform_action(
            "player", player_action
        )

        ai_valid = self._can_perform_action(
            "ai", ai_action
        )

        next_player_cooldown = player_action == 5 and player_valid
        next_ai_cooldown = ai_action == 5 and ai_valid
        
        # Spend resources only for valid actions
        if player_valid:
            self._spend_resources("player", player_action)

        if ai_valid:
            self._spend_resources("ai", ai_action)

        # Calculate both fighters' damage before changing health
        damage_to_player = self._calculate_damage(
            ai_action,
            player_action,
            ai_valid,
            player_valid
        )

        damage_to_ai = self._calculate_damage(
            player_action,
            ai_action,
            player_valid,
            ai_valid
        )

        self.player_move_cooldown = next_player_cooldown
        self.ai_move_cooldown = next_ai_cooldown

        self.ai_health = max(0, self.ai_health - damage_to_ai)
        self.player_health = max(0, self.player_health - damage_to_player)

        if ai_valid:
            if ai_action == 4:
                if not self._charge_interrupted(player_action, player_valid):
                    self.ai_ki = min(self.max_ki, self.ai_ki + 25)

            elif ai_action == 3:
                self.ai_ki = min(self.max_ki, self.ai_ki + 10)

        if player_valid:
            if player_action == 4:
                if not self._charge_interrupted(ai_action, ai_valid):
                    self.player_ki = min(self.max_ki, self.player_ki + 25)

            elif player_action == 3:
                self.player_ki = min(self.max_ki, self.player_ki + 10)

        self._update_distance(
            player_action,
            ai_action,
            player_valid,
            ai_valid
        )

        self.player_stamina = min(
            self.max_stamina,
            self.player_stamina + 10
        )
    
        self.ai_stamina = min(
            self.max_stamina,
            self.ai_stamina + 10
        )

        self.turns += 1
        done = (
            self.player_health <= 0
            or self.ai_health <= 0
            or self.turns >= self.max_turns
        )

        self.telegraphed_action = None

        winner = None

        if done:
            if self.ai_health > self.player_health:
                winner = "ai"
            elif self.ai_health < self.player_health:
                winner = "player"
            else:
                winner = "draw"

        reward = (damage_to_player - damage_to_ai) / 20

        if done:
            if winner == "ai":
                reward += 10
            elif winner == "player":
                reward -= 10

        info = {
            "winner": winner,
            "player_action_valid": player_valid,
            "ai_action_valid": ai_valid,
            "damage_to_player": damage_to_player,
            "damage_to_ai": damage_to_ai,
        }

        return self._get_observation(), reward, done, info