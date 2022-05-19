import asyncio

from gym.spaces import Box
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate
from poke_env.server_configuration import ServerConfiguration
from poke_env.player.env_player import Gen8EnvSinglePlayer
import numpy as np

from threading import Thread

# class SimpleRLPlayer(Gen8EnvSinglePlayer):
#     def calc_reward(self, last_battle, current_battle) -> float:
#         return self.reward_computing_helper(
#             current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
#         )

#     def embed_battle(self, battle):
#         # -1 indicates that the move does not have a base power
#         # or is not available
#         moves_base_power = -np.ones(4)
#         moves_dmg_multiplier = np.ones(4)
#         for i, move in enumerate(battle.available_moves):
#             moves_base_power[i] = (
#                 move.base_power / 100
#             )  # Simple rescaling to facilitate learning
#             if move.type:
#                 moves_dmg_multiplier[i] = move.type.damage_multiplier(
#                     battle.opponent_active_pokemon.type_1,
#                     battle.opponent_active_pokemon.type_2,
#                 )

#         # We count how many pokemons have fainted in each team
#         fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
#         fainted_mon_opponent = (
#             len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
#         )

#         # Final vector with 10 components
#         final_vector = np.concatenate(
#             [
#                 moves_base_power,
#                 moves_dmg_multiplier,
#                 [fainted_mon_team, fainted_mon_opponent],
#             ]
#         )
#         return np.float32(final_vector)

#     def describe_embedding(self):
#         low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
#         high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
#         return Box(
#             np.array(low, dtype=np.float32),
#             np.array(high, dtype=np.float32),
#             dtype=np.float32,
#         )

server_config = ServerConfiguration(
    "ps:8000",
    "authentication-endpoint.com/action.php?"
)

POKE_LOOP = asyncio.new_event_loop()


class RandomGen8EnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        return np.array([0])

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )


def env_algorithm(player, n_battles):
    for _ in range(n_battles):
        done = False
        player.reset()
        while not done:
            _, _, done, _ = player.step(
                np.random.choice(player.action_space))


def to_id_str(name):
    return "".join(char for char in name if char.isalnum()).lower()


async def launch_battles(player, opponent):
    battles_coroutine = asyncio.gather(
        player.send_challenges(
            opponent=to_id_str(opponent.username),
            n_challenges=1,
            to_wait=opponent.logged_in,
        ),
        opponent.accept_challenges(opponent=to_id_str(
            player.username), n_challenges=1),
    )
    await battles_coroutine


def env_algorithm_wrapper(player, kwargs):
    env_algorithm(player, **kwargs)

    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break


p1 = RandomGen8EnvPlayer(log_level=25, server_configuration=server_config)
p2 = RandomGen8EnvPlayer(log_level=25, server_configuration=server_config)

p1._start_new_battle = True
p2._start_new_battle = True

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 5}

t1 = Thread(target=lambda: env_algorithm_wrapper(p1, env_algorithm_kwargs))
t1.start()

t2 = Thread(target=lambda: env_algorithm_wrapper(p2, env_algorithm_kwargs))
t2.start()

while p1._start_new_battle:
    loop.run_until_complete(launch_battles(p1, p2))
t1.join()
t2.join()
