"""特徴の**正しさ**をエンジンの真値と突き合わせる（被覆監査とは別の観点）。

**なぜ必要か**: 被覆（feat_audit.py）は「使っていないデータ」を見つけるが、
「**使っているが計算が間違っている**」特徴は見つけられない。実際この案件では2度踏んだ:

  EXP-097  Alakazam の Powerful Hand の打点が常に0（効果文にしか打点が無い技を落としていた）
  EXP-099  弱点×2 を打点計算に入れていない（教師の試合の54.5%で脅威特徴が過小評価）

どちらも**例外を出さない**ので、テストでも被覆監査でも捕まらない。唯一の確実な検証は
**エンジンに実際に攻撃させて、その結果と我々の予測を突き合わせること**。

やり方: MAIN決定で ATTACK 選択肢があるとき、`search_begin`/`search_step` でその攻撃を
実行し、相手アクティブのHPの減少量を測る。それを `_atk_scan` の予測打点と比較する。

usage:
  uv run python tools/feat_verify_damage.py --games 12
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
    ap.add_argument("--agent", default="agents/103_alakazam_feat3")
    ap.add_argument("--games", type=int, default=12)
    a = ap.parse_args()

    import cg.game as cgg
    import run_match as rm
    from cg.api import (AreaType, OptionType, search_begin, search_end,
                        search_step, to_observation_class)

    ag = os.path.join(REPO, a.agent)
    if ag not in sys.path:
        sys.path.insert(0, ag)
    fname = next(f for f in sorted(os.listdir(ag))
                 if f.startswith("feat_") and f.endswith(".py"))
    sp = ilu.spec_from_file_location(fname[:-3], os.path.join(ag, fname))
    ft = ilu.module_from_spec(sp)
    sys.modules[fname[:-3]] = ft
    sp.loader.exec_module(ft)

    OPPS = [("101_marnie_luca", 4242), ("098_spidops_gen", 777),
            ("041_garchomp_replica", 99), ("085_kangaskhan_e57e", 5150)]
    rows = []
    ko_rows = []
    per_opp = collections.defaultdict(list)

    def determinize(obs):
        """相手の非公開領域を適当なカードで埋める（打点計算には影響しない）。"""
        st = obs.current
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        deck_ids = sorted(ft.DECK_COUNTS)
        pad = deck_ids[0]
        need = me.deckCount + len(me.prize)
        mine = [pad] * need
        return (mine[:me.deckCount], mine[me.deckCount:],
                [pad] * op.deckCount, [pad] * len(op.prize),
                [pad] * op.handCount,
                [pad] if (op.active and op.active[0] is None) else [])

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
                me_i = obs["current"]["yourIndex"]
                if me_i == 0:
                    o = to_observation_class(obs)
                    st, sel = o.current, o.select
                    atk = [i for i, x in enumerate(sel.option)
                           if x.type == OptionType.ATTACK] if sel else []
                    op_p = st.players[1].active[0] if st.players[1].active else None
                    if atk and op_p is not None and int(sel.context) == 0:
                        act = st.players[0].active[0]
                        # **特徴器と同じ引数で呼ぶ**。def_field を渡さないと
                        # 無効化特性の判定が働かず、検証にならない。
                        opf = ([op_p] + [x for x in st.players[1].bench if x])
                        _dfn = getattr(ft, "_cd", None)
                        try:      # 新しい特徴器（弱点・無効化・型付きコストあり）
                            pred = ft._atk_scan(
                                act, len(act.energyCards or []),
                                ft._atk_ctx(st.players[0], act),
                                defender=_dfn(op_p), etypes=ft._etypes(act),
                                def_field=opf, defender_obj=op_p)[0]
                        except TypeError:   # 旧世代（102など）: 打点のみ
                            pred = ft._atk_scan(
                                act, len(act.energyCards or []),
                                ft._atk_ctx(st.players[0], act))[0]
                        hp0 = op_p.hp
                        ser0 = getattr(op_p, "serial", None)
                        # **エンジンは攻撃・ターン終了・ポケモンチェックを1ステップで
                        # 解決する**ので、測定値にはチェックアップの確定ダメージが
                        # 含まれる。分離できないので予測側に足して比較する
                        # （Froslass「Freezing Shroud」= 特性持ちに毎ターン10）。
                        chk = 0
                        _CK = getattr(ft, "CHECKUP_DMG", {})
                        _cdf = getattr(ft, "_cd", lambda x: None)
                        if _CK and getattr(_cdf(op_p), "skills", None):
                            for _s in (st.players[0], st.players[1]):
                                for _q in ((([_s.active[0]] if _s.active
                                             and _s.active[0] else []))
                                           + [x for x in _s.bench if x]):
                                    if _q.id in _CK and _q.id != op_p.id:
                                        chk += _CK[_q.id]
                        best = None
                        try:
                            root = search_begin(o, *determinize(o))
                            for idx in atk:
                                ch = search_step(root.searchId, [int(idx)])
                                # **攻撃は1ステップでは解決しない**。対象選択・エネ
                                # トラッシュ・コイン等の子selectが挟まるので、
                                # 相手の手番になるまで進めてから盤面を読む。
                                # **ポケモンチェックを含めない**。相手の手番まで進めると
                                # Froslass「Freezing Shroud」等のチェックアップ処理で
                                # 10ダメージ乗り、攻撃の打点として測ってしまう
                                # （Marnie戦の不一致が全て正確に +10 だった原因）。
                                cs = ch.observation.current
                                for _ in range(24):
                                    ob2 = ch.observation
                                    cs = ob2.current
                                    t0 = (cs.players[1].active[0]
                                          if cs.players[1].active else None)
                                    if t0 is None or getattr(t0, "serial",
                                                             None) != ser0:
                                        break            # KO・入替が起きた
                                    if t0.hp != hp0:
                                        break            # ダメージが入った
                                    s2 = ob2.select
                                    if s2 is None or not s2.option:
                                        break
                                    if ob2.current.yourIndex != 0:
                                        break
                                    k = max(1, min(s2.maxCount or s2.minCount or 1,
                                                   len(s2.option)))
                                    ch = search_step(ch.searchId, list(range(k)))
                                    cs = ch.observation.current
                                tgt = (cs.players[1].active[0]
                                       if cs.players[1].active else None)
                                if tgt is not None and getattr(tgt, "serial",
                                                               None) == ser0:
                                    d, ko = hp0 - tgt.hp, False
                                else:
                                    d, ko = hp0, True     # KO（最低でも hp0 通った）
                                if best is None or d > best[0]:
                                    best = (d, ko)
                        except Exception:        # noqa: BLE001
                            best = None
                        finally:
                            try:
                                search_end()
                            except Exception:    # noqa: BLE001
                                pass
                        if best is not None:
                            # **KO判定の一致**も記録する。方策にとっては生の打点より
                            # 「これでKOできるか」の二値のほうが決定に直結する。
                            ko_pred = (pred + chk) >= hp0
                            ko_real = best[1] or (best[0] >= hp0)
                            rows.append((pred + chk, best[0], best[1]))
                            per_opp[opp.split("_")[1]].append(
                                (pred + chk, best[0], best[1]))
                            ko_rows.append((ko_pred, ko_real, hp0, pred + chk,
                                            best[0]))
                obs = cgg.battle_select((pa if me_i == 0 else pb)(obs))
            cgg.battle_finish()

    if not rows:
        raise SystemExit("攻撃の実行を1件も観測できなかった（search が使えない可能性）")
    def judge(p, r, ko):
        # KO は「hp0 以上通った」しか分からないので、予測が hp0 以上なら正解扱い
        return (p >= r) if ko else (p == r)

    ok = sum(1 for p, r, k in rows if judge(p, r, k))
    nko = sum(1 for _, _, k in rows if k)
    print(f"攻撃を実行して打点を照合: {len(rows)} 件（うちKO {nko} 件）")
    print(f"  予測が真値と一致: {ok}/{len(rows)} = {ok / len(rows):.1%}")
    print(f"\n{'対面':<14}{'件数':>6}{'一致率':>9}{'平均誤差(予測-真値)':>20}")
    for k, v in sorted(per_opp.items()):
        m = sum(1 for p, r, kk in v if judge(p, r, kk))
        nv = [(p, r) for p, r, kk in v if not kk]
        e = (sum(p - r for p, r in nv) / len(nv)) if nv else 0.0
        print(f"  {k:<12}{len(v):>6}{m / len(v):>8.1%}{e:>19.1f}")
    kok = sum(1 for a, b, *_ in ko_rows if a == b)
    fp = [(h, p, r) for a, b, h, p, r in ko_rows if a and not b]
    fn = [(h, p, r) for a, b, h, p, r in ko_rows if b and not a]
    print(f"\n■ KO判定（ko_opp_active に相当）の一致: {kok}/{len(ko_rows)} = "
          f"{kok / max(1, len(ko_rows)):.2%}")
    print(f"    「KOできる」と誤認: {len(fp)}件 / 「KOできない」と誤認: {len(fn)}件")
    if fp[:3] or fn[:3]:
        print(f"    誤認の例(残HP, 予測打点, 実打点): 過大={fp[:3]} 過小={fn[:3]}")
    diff = [(p, r) for p, r, kk in rows if not judge(p, r, kk)]
    if diff:
        c = collections.Counter(diff)
        print("\n不一致の内訳（予測 → 真値）上位12:")
        for (p, r), n in c.most_common(12):
            print(f"    {p:>5} → {r:<5}  {n}件")


if __name__ == "__main__":
    main()
