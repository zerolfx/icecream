import functools
import logging
import math
from typing import Callable, Dict, List, Tuple, Union, NamedTuple

import numpy as np
from collections import defaultdict

TOTAL_UNITS = 24
# test weighted = 0.01, 0.1, 0.5, 0.8, 1 and find the performance of 0.5 is the best
# although I don't know why, but I guess we can test it for several rounds later
WEIGHT = 0.5


class Choice(NamedTuple):
    flavors: List[int]
    max_depth: int
    index: Tuple[int, int]


def generate_choices(top_layer: np.ndarray, curr_level: np.ndarray) -> List[Choice]:
    res = list()
    for i in range(top_layer.shape[0] - 1):
        for j in range(top_layer.shape[1] - 1):
            cur = list()
            max_depth = -1
            for x in range(2):
                for y in range(2):
                    max_depth = max(max_depth, curr_level[i + x][j + y])
            if max_depth == -1:
                continue
            for x in range(2):
                for y in range(2):
                    if max_depth == curr_level[i + x][j + y]:
                        cur.append(top_layer[i + x][j + y])
            res.append(Choice(cur, max_depth, (i, j)))
    return res


def activate(x: float) -> float:
    return x


def estimate_preference(history: Dict[int, int]) -> Dict[int, float]:
    s = sum(map(activate, history.values()))
    res = dict()
    for k, v in history.items():
        res[k] = activate(v) / s
    return res


def score(choice: Choice, preferences: Dict[int, float]) -> float:
    res = 0
    for v in choice.flavors:
        res += preferences[v]
    return res


def score_all(preferences: Dict[int, float], choices: List[Choice]) -> float:
    scores = sorted(list(map(functools.partial(score, preferences=preferences), choices)), reverse=True)
    logging.info(scores)
    return sum(scores[:48])


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = [0]

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
              get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]
              ) -> Dict[str, Union[Tuple[int, int], int]]:
        remain = 24 - self.state[-1]
        all_choices = generate_choices(top_layer, curr_level)
        choices = list(filter(lambda x: len(x.flavors) <= remain, all_choices))
        # self.logger.info(choices)
        if not choices:
            turns = get_turns_received()

            next_players = list()
            if len(set(turns)) == 1:  # last turn for current round
                next_players = [i for i in range(len(turns)) if i != player_idx]
            else:
                for idx, turn in enumerate(turns):
                    if turn == min(turns) and idx != player_idx:
                        next_players.append(idx)

            if len(self.state) <= 4:
                self.state.append(0)
                return dict(action="pass", values=self.rng.choice(next_players))

            estimated_preferences = list(map(estimate_preference, get_served()))
            self.logger.debug(estimated_preferences)
            next_player = max(next_players, key=lambda i: score_all(estimated_preferences[i], all_choices))

            self.state.append(0)
            return dict(action="pass", values=next_player)

        def f(choice: Choice) -> float:
            res = 0
            for flavor in choice.flavors:
                res -= self.flavor_preference.index(flavor)
            res /= len(choice.flavors)
            res += choice.max_depth * 0.2
            res -= 0.01 * len(choice.flavors)
            return res

        choice = max(choices, key=f)
        self.state[-1] += len(choice.flavors)
        return dict(action='scoop', values=choice.index)
