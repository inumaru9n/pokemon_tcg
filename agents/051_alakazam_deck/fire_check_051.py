"""051の新カード（Tool Scrapper 1137 / Battle Cage 1264）が実際にプレイされるかの発火確認。
in-processで run_match の game loop を簡略再現し、051側の選択を検査する。"""
import random
import sys
import time

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from arena.run_match import load_agent_module, read_deck, _validate_selection
from cg.api import OptionType, AreaType, to_observation_class
from cg.game import battle_start, battle_select, battle_finish

CARD = int(sys.argv[1])          # 1137 or 1264
OPP = sys.argv[2]                # opponent agent dir
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

A_DIR = "/Users/akira/kaggle/pokemon_tcg/agents/051_alakazam_deck"
agent_a = load_agent_module(A_DIR, "A")
agent_b = load_agent_module(OPP, "B")
deck_a = read_deck(A_DIR)
deck_b = read_deck(OPP)

plays = 0
games_with_play = 0
tool_card_prompts = 0
wins = 0
for g in range(N):
    random.seed(1000 + g)
    a0 = g % 2 == 0
    obs_dict, start = battle_start(deck_a if a0 else deck_b, deck_b if a0 else deck_a)
    assert obs_dict is not None
    played_this_game = 0
    steps = 0
    try:
        while True:
            res = obs_dict["current"]["result"]
            if res != -1:
                if res == (0 if a0 else 1):
                    wins += 1
                break
            if steps > 4000:
                break
            player = obs_dict["current"]["yourIndex"]
            is_a = (player == 0) == a0
            sel = (agent_a if is_a else agent_b)(obs_dict)
            err = _validate_selection(sel, obs_dict["select"])
            assert not err, err
            if is_a:
                obs = to_observation_class(obs_dict)
                for i in sel:
                    o = obs.select.option[i]
                    if o.type == OptionType.PLAY:
                        me = obs.current.players[obs.current.yourIndex]
                        try:
                            if me.hand[o.index].id == CARD:
                                played_this_game += 1
                        except (IndexError, TypeError):
                            pass
                    elif o.type == OptionType.TOOL_CARD:
                        tool_card_prompts += 1
            obs_dict = battle_select(sel)
            steps += 1
    finally:
        battle_finish()
    plays += played_this_game
    if played_this_game:
        games_with_play += 1

print(f"card={CARD} opp={OPP} games={N} wins_051={wins} "
      f"plays={plays} games_with_play={games_with_play} tool_card_selections={tool_card_prompts}")
