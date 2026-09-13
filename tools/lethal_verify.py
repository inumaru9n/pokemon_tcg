"""「詰みを見逃した」と判定した局面を、**エンジンで1件ごとに検証**する（EXP-126）。

**なぜ要るか**: lethal_diag.py は特徴器のフラグ（ko_opp_active / lethal_for_me）で
見逃しを数えるが、フラグが偽陽性なら「見逃し26.9%」は幻になる。
**実装の前に、発火した局面だけを取り出して1件ごとに正誤を確定させる**
（EXP-049の教訓: 発火率が低い改善はA/Bでは検出できないので、1件ごとの検証が主証拠になる）。

やること: 方策が攻撃を選ばなかった詰み局面で、`search_begin` → `search_step(攻撃)` と
**強制的に攻撃させ、その場で勝ちになるか**を見る。ローカルなので隠れ情報は真値が使える
（EXP-125が殺したのは「相手の未来を読む」ロールアウトであって、
**自分の攻撃がKOするかの確定判定**はここでも本番でも隠れ情報に依存しない）。

出すもの:
  - 詰み判定の**真陽性率**（強制攻撃で即勝ちになった割合）
  - 見逃した試合の**最終結果**（見逃しても勝てたのか、それが敗着だったのか）

usage:
  uv run python tools/lethal_verify.py --agent agents/186_alakazam_v8 --games 60
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agents/186_alakazam_v8")
    ap.add_argument("--games", type=int, default=60)
    a = ap.parse_args()

    import cg.game as cgg
    import run_match as rm
    from cg.api import OptionType, to_observation_class, search_begin, search_step, search_end
    from selfplay.state_encoder import _hidden

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
    # 検証の結果は「その局面で攻撃したら即勝ちだったか」の3値
    ver = collections.Counter()
    per = collections.defaultdict(lambda: collections.Counter())
    missed_games = []          # (相手, 見逃し件数, 最終結果)

    for opp, seed in OPPS:
        pa = rm.load_agent_module(ag, "A")
        pb = rm.load_agent_module(os.path.join(REPO, "agents", opp), "B")
        da = rm.read_deck(ag)
        db = rm.read_deck(os.path.join(REPO, "agents", opp))
        for g in range(a.games):
            random.seed(g)
            obs, _ = cgg.battle_start_seeded(da, db, seed + g)
            ptr = cgg.Battle.battle_ptr
            nmiss = 0
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
                        if atk and gv.get("ko_opp_active", 0) > 0 \
                                and gv.get("lethal_for_me", 0) > 0:
                            chosen = pa(obs)
                            if not (chosen and chosen[0] in atk):
                                # **見逃し候補**。強制的に攻撃させて結果を確かめる
                                nmiss += 1
                                key = "検証不能"
                                try:
                                    h0, h1 = _hidden(ptr, 0), _hidden(ptr, 1)
                                    rt = search_begin(
                                        o, list(h0.get("deck") or []),
                                        list(h0.get("prize") or []),
                                        list(h1.get("deck") or []),
                                        list(h1.get("prize") or []),
                                        list(h1.get("hand") or []), [])
                                    ch = search_step(rt.searchId, [atk[0]])
                                    # 攻撃後、こちらの入力を要さずに決着するはず。
                                    # 決着しなければ「詰みではなかった」＝偽陽性。
                                    for _ in range(40):
                                        s2 = ch.observation.current
                                        if s2.result != -1:
                                            break
                                        s = ch.observation.select
                                        if s is None or not s.option:
                                            break
                                        ch = search_step(ch.searchId, [0])
                                    r = ch.observation.current.result
                                    key = ("**即勝ち**" if r == me else
                                           "引分" if r == 2 else
                                           "**偽陽性(負け)**" if r != -1 else "**偽陽性(未決着)**")
                                    search_end()
                                except Exception as _e:      # noqa: BLE001
                                    try:
                                        search_end()
                                    except Exception:        # noqa: BLE001
                                        pass
                                    key = f"検証不能({type(_e).__name__})"
                                ver[key] += 1
                                per[opp.split("_")[1]][key] += 1
                obs = cgg.battle_select((pa if me == 0 else pb)(obs))
            res = obs["current"]["result"] if obs is not None else -1
            if nmiss:
                missed_games.append((opp.split("_")[1], nmiss, res))
            cgg.battle_finish()

    tot = sum(ver.values())
    print(f"■ {a.agent} / {len(OPPS)}相手 × {a.games}試合\n")
    print(f"「詰みを見逃した」と判定した {tot} 件を、強制攻撃させて検証:")
    for k, n in ver.most_common():
        print(f"   {k:22} {n:>4}  ({n/max(1,tot):.1%})")
    tp = ver["**即勝ち**"]
    print(f"\n   **真陽性率 {tp}/{tot} = {tp/max(1,tot):.1%}**"
          "（この割合だけが本物の見逃し＝lethal層で拾える）")

    print(f"\n{'対面':<14}{'見逃し':>8}{'即勝ち':>8}{'真陽性率':>10}")
    for k, c in sorted(per.items()):
        n = sum(c.values())
        print(f"  {k:<12}{n:>8}{c['**即勝ち**']:>8}{c['**即勝ち**']/max(1,n):>9.1%}")

    if missed_games:
        lost = [x for x in missed_games if x[2] == 1]
        print(f"\n■ 見逃しがあった {len(missed_games)}試合 の最終結果: "
              f"勝ち {sum(1 for x in missed_games if x[2]==0)} / "
              f"**負け {len(lost)}** / 引分 {sum(1 for x in missed_games if x[2]==2)}")
        print(f"   → **見逃しが直接の敗着になったのは最大 {len(lost)}試合**"
              f"（{len(lost)/(len(OPPS)*a.games):.1%} の試合）")


if __name__ == "__main__":
    main()
