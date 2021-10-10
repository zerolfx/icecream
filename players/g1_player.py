import logging
import math
from typing import Callable, Dict, List, Tuple, Union, NamedTuple

import numpy as np
from collections import defaultdict


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


def choose_next_player(now_turn: int, possible_players: List[int], served_situation: List[Dict[int, int]],
                       top_layer: np.ndarray,
                       curr_level: np.ndarray, rng: np.random.Generator, group_id: int) -> int:
    if len(possible_players) == 1:
        return possible_players[0]

    # if now_turn = 0, it means for the first turn, we just randomly choose a person to pass to
    if now_turn == 0 and len(possible_players) != 0:
        return rng.choice(possible_players)

    # if there are no candidates, which means our team is the last one to serve
    if len(possible_players) == 0:
        possible_players = [i for i in range(10) if i != group_id]

    choices = generate_choices(top_layer, curr_level)
    # if all ice cream has been taken
    if len(choices) == 0:
        return rng.choice(list(possible_players))

    greedy_flavor_preference = defaultdict(list)
    for player_index in possible_players:
        player_served_situation = sorted(served_situation[player_index].items(), key=lambda item: -item[1])
        for flavor, _ in player_served_situation:
            greedy_flavor_preference[player_index].append(flavor)

    max_score = -math.inf
    next_player_list = set()
    for player_index, flavor_preference in greedy_flavor_preference.items():
        score_list = []
        for choice in choices:
            score_list.append((score(choice, flavor_preference), len(choice.flavors)))

        score_list.sort(key=lambda x: -x[0])
        remain = 24
        player_max_score = 0
        for score_num, count in score_list:
            if remain < 0:
                break
            remain -= count
            player_max_score += score_num

        if max_score < player_max_score:
            max_score = player_max_score
            next_player_list = set()
            next_player_list.add(player_index)
        elif max_score == player_max_score:
            next_player_list.add(player_index)

    # if len(next_player_list) == 1, just return,
    # if the length is larger than 1, we may choose randomly choose a from the set
    # or use a weighted function(still needs to think)
    if len(next_player_list) == 1:
        return list(next_player_list)[0]
    return rng.choice(list(next_player_list))


# copy from the below f function
def score(choice: Choice, flavor_preference) -> float:
    res = 0
    for flavor in choice.flavors:
        res -= flavor_preference.index(flavor)
    res /= len(choice.flavors)
    res += choice.max_depth * 0.2
    res += 0.01 * len(choice.flavors)
    return res


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = [0]
        self.group_id = 1 - 1

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
              get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]
              ) -> Dict[str, Union[Tuple[int, int], int]]:
        remain = 24 - self.state[-1]
        choices = generate_choices(top_layer, curr_level)
        choices = list(filter(lambda x: len(x.flavors) <= remain, choices))
        self.logger.info(choices)
        if not choices:
            turns = get_turns_received()
            players = list()
            for idx, turn in enumerate(turns):
                if turn == min(turns) and idx != self.group_id:
                    players.append(idx)
            self.state.append(0)
            # if we choose the person with the highest score
            next_player = choose_next_player(min(turns), players, get_served(), top_layer, curr_level, self.rng,
                                             self.group_id)
            return dict(action="pass", values=next_player)

            # if we just randomly choose one person
            # if len(players) == 0:
            #     players = [i for i in range(10) if i != self.group_id]
            # return dict(action="pass", values=self.rng.choice(players))

        def f(choice: Choice) -> float:
            res = 0
            for flavor in choice.flavors:
                res -= self.flavor_preference.index(flavor)
            res /= len(choice.flavors)
            res += choice.max_depth * 0.2
            res += 0.01 * len(choice.flavors)
            return res

        choice = max(choices, key=f)
        self.state[-1] += len(choice.flavors)
        return dict(action='scoop', values=choice.index)
