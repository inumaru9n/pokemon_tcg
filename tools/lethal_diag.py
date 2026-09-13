"""方策が「そのターンで勝てる詰み」を見逃している頻度を測る。

**なぜ先に測るか**: EXP-053 で探索ファミリーは5連敗し、生き残ったのは lethal-only
（確定検証×高頻度×効果確定）だけだった。そして EXP-099 で打点計算を直した結果、
**KO判定の真値一致が100%**になった（102は74.9%）。つまり「今KOできるか」は
探索なしで確実に分かる。

見逃しが多ければ lethal層は安価で確実な改善になる。少なければ、価値誘導探索のような
重い手段に進む理由が強まる。**headroom を測ってから道具を選ぶ。**

判定: 決定時点で
  - 自分のアクティブが相手アクティブをKOできる（ko_opp_active）
  - そのKOで取るサイド枚数 >= 自分の残りサイド（lethal_for_me）
のとき、方策が ATTACK を選んだか。

usage:
  uv run python tools/lethal_diag.py --agent agents/104_alakazam_feat4 --games 40
"""

from __future__ import annotations

import argparse
import collections
import importlib.util as ilu
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="agents/104_alakazam_feat4")
    ap.add_argument("--games", type=int, default=40)
    a = ap.parse_args()

    import cg.game as cgg
    import run_match as rm
    from cg.api import OptionType, to_observation_class

    ag = os.path.join(REPO, a.agent)
    if ag not in sys.path:
        sys.path.insert(0, ag)
    fn = next(f for f in sorted(os.listdir(ag))
              if f.startswith("feat_") and f.endswith(".py"))
    sp = ilu.spec_from_file_location(fn[:-3], os.path.join(ag, fn))
    ft = ilu.module_from_spec(sp)
    sys.modules[fn[:-3]] = ft
    sp.loader.exec_module(ft)

    OPPS = [("101_marnie_luca", 4242), ("098_spidops_gen", 777),
            ("085_kangaskhan_e57e", 99), ("041_garchomp_replica", 5150)]
    st = collections.Counter()
    per = collections.defaultdict(lambda: [0, 0])

    for opp, seed in OPPS:
        pa = rm.load_agent_module(ag, "A")
        pb = rm.load_agent_module(os.path.join(REPO, "agents", opp), "B")
        da = rm.read_deck(ag)
        db = rm.read_deck(os.path.join(REPO, "agents", opp))
        for g in range(a.games):
            random.seed(g)
            obs, _ = cgg.battle_start_seeded(da, db, seed + g)
            while obs is not None:
                if obs["current"]["result"] != -1:
                    break
                me = obs["current"]["yourIndex"]
                if me == 0:
                    o = to_observation_class(obs)
                    sel = o.select
                    if sel is not None and int(sel.context) == 0:
                        v, K = ft.feat_vector(o.current, me)
                        gv = dict(zip(K, v))
                        atk = [i for i, x in enumerate(sel.option)
                               if x.type == OptionType.ATTACK]
                        if atk and gv.get("ko_opp_active", 0) > 0:
                            lethal = gv.get("lethal_for_me", 0) > 0
                            chosen = pa(obs)
                            took = bool(chosen) and chosen[0] in atk
                            k = "詰み" if lethal else "KOのみ"
                            st[(k, took)] += 1
                            per[opp.split("_")[1]][0] += int(lethal and took)
                            per[opp.split("_")[1]][1] += int(lethal)
                obs = cgg.battle_select((pa if me == 0 else pb)(obs))
            cgg.battle_finish()

    print(f"エージェント: {a.agent}\n")
    for k in ("詰み", "KOのみ"):
        t, f = st[(k, True)], st[(k, False)]
        n = t + f
        if n:
            lbl = ("**そのターンで勝てる詰み**" if k == "詰み" else "KOできる（勝ちには足りない）")
            print(f"{lbl}: {n} 回")
            print(f"    攻撃を選んだ {t} ({t/n:.1%}) / **見逃した {f} ({f/n:.1%})**")
    print(f"\n{'対面':<14}{'詰み局面':>10}{'取った':>8}{'取得率':>9}")
    for k, (t, n) in sorted(per.items()):
        if n:
            print(f"  {k:<12}{n:>10}{t:>8}{t/n:>8.1%}")


if __name__ == "__main__":
    main()
