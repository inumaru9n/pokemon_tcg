"""deck.csv から全選択NN方策用の特徴器 feat_*.py を自動生成する。

**なぜ作るか**: プールの各枠にSLレプリカ（EXP-094形式の純NN方策）を作るには、枠ごとに
盤面特徴と行動クラス分類が要る。手作りすると1枠あたり094相当の作り込み（v1→v6）が必要。
しかし手作り2例（094=Alakazam / 092=Marnie）を突き合わせると、**デッキ依存部分は
「定数」と「行動クラスの名前集合」だけ**で、しかも手作り述語は同じ型の機械展開だった:

  094 ready_zam(Alakazam×エネ≥1)   ≡ 092 ready_grim(Grimmsnarl×エネ≥2)  → 種族×エネ枚数
  094 loaded_pre(進化前×エネ≥1)     ≡ 092 charged_pre(進化前×エネ≥2)      → 進化前×エネ枚数
  094 dudun3(Dudunsparce×エネ≥3)   ≡ 092 munki_online(Munkidori×エネ≥1)  → 同上
  （092のみ）grimline_field / snow_field                                  → 進化ライン数

固定閾値を焼き込むのでなく **枚数そのもの（合計 e_*、最大 emax_*）** を出す。閾値は
MLP側が学べるので情報量は手作り以上になる。

汎用コアは `tools/featgen_core.py`（094からの忠実移植）で、このスクリプトは
**定数ヘッダを生成してコアを連結するだけ**。忠実度の保証をコア1本に集約するのが狙い。

**検証**: 手作りの正解が2つある。同じデッキに生成版を当てて
  (a) 識別不能率（特徴が完全一致する選択肢を教師が選び分けた率）
  (b) val一致率
が手作り版と同等なら、残りのプール枠は抽出+学習だけで量産できる。

usage:
  uv run python tools/gen_featurizer.py --deck agents/094_yushin_nn/deck.csv \\
      --out /tmp/feat_gen094.py
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HEADER = '''"""自動生成された特徴器（tools/gen_featurizer.py）。手で編集しないこと。

デッキ: {deck}
ポケモン {n_sp}種 / 進化ライン {n_line}本 / エネ {n_e}種 / スタジアム {n_st}種
行動クラス {n_cls}個

汎用コア（後半）は tools/featgen_core.py と同一。デッキ依存はこのヘッダの定数のみ。
"""

from __future__ import annotations

import os

from cg.api import CardType, all_card_data

# ---- 攻撃テーブル（EN_Card_Data.csv由来の自動生成モジュール。全1056種を収録） ----
try:
    from attack_table_056 import ATTACKS_056
except ImportError:
    import importlib.util as _ilu

    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = ("/kaggle_simulations/agent"
                if os.path.exists("/kaggle_simulations/agent/main.py")
                else os.getcwd())
    _p = os.path.join(_dir, "attack_table_056.py")
    _sp = _ilu.spec_from_file_location("attack_table_056", _p)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    ATTACKS_056 = _m.ATTACKS_056

card_table = {{c.cardId: c for c in all_card_data()}}

# ---- 効果文から復元した打点（tools/gen_atk_dmg.py 生成。デッキ非依存） ----
# attack_table_056 の Damage 列は「n/a」の技を全て打点不明にしており、その中に
# Alakazam の Powerful Hand（手札1枚につき20）のような主砲が含まれる。
{ATK_TABLES}

# ---- 相手のスタジアム/ツールはメタ上位のカード単位で持つ ----
# スタジアムとツールには「属性」が無く効果がカード固有なので、相手側でも
# カード単位が要る。ただし無限ではなくメタ上位14種/11種で99.6%/100%を覆う。
# **メタが動いたら tools/meta_zones.py を再生成すること。**
{META_ZONES}

# ==== 直近メタの看板カード・構築（tools/gen_meta_cards.py で再生成） ====
{META_CARDS_BLOCK}

# 有効な特徴グループ（None なら全部。tools/featgen_core.py の _GRULES 参照）
FEAT_GROUPS = {FEAT_GROUPS}

# **このデッキの教師データで算出した削除列**（完全重複の余剰 + 定数列）。
# tools/calc_drop_cols.py が出力する JSON を --drop で渡して焼き込む。
DROP_COLS = {DROP_COLS}

# ---- 相手側の判定に使う固定ID（自デッキと無関係。094から継承） ----
BOSS_ORDERS = 1182
TR_ARTICUNO = 414
MIST_IDS = (11, 20)          # Mist Energy / Rock Fighting Energy（PH無効化）

# ---- 自デッキから導出した定数 ----
SPECIES = {species!r}
SPECIES_NAME = {sname!r}
LINE_OF = {line_of!r}        # cardId -> 進化ライン名（最下段の種の名前）
LINE_NAMES = {line_names!r}
PRE_EVO = {pre_evo!r}        # 同デッキ内に進化先を持つ種
ENERGY_NAME = {ename!r}      # 全エネ（基本+特殊）の cardId -> 短縮名
BASIC_ENERGY_NAME = {bname!r}
TOOL_NAME = {toolname!r}     # ポケモンに付けるツール（ATTACHで来る）
STADIUM_NAME = {stname!r}
ATTACH_BUCKET = {bucket!r}   # エネ貼り先の種 -> クラス分類のバケット
CLS_PLAY = {cls_play!r}
CLS_EVO = {cls_evo!r}
CLS_AB = {cls_ab!r}        # ポケモンの特性 + 効果を持つスタジアム
CLS_ATK = {cls_atk!r}      # attackId -> クラス名（技の打ち分け）
CLASSES = {classes!r}


def _load_deck_counts() -> dict[int, int]:
    try:
        _d = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _d = os.getcwd()
    for path in (os.path.join(_d, "deck.csv"),
                 "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            counts: dict[int, int] = {{}}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.split(",")[0].isdigit():
                        continue
                    cid = int(line.split(",")[0])
                    counts[cid] = counts.get(cid, 0) + 1
            return counts
    raise FileNotFoundError("deck.csv not found")


DECK_COUNTS = _load_deck_counts()
DECK_IDS = sorted(DECK_COUNTS)  # 特徴キーの固定順


'''


def _short(name: str, cid: int, maxlen: int = 12) -> str:
    nm = (name or f"c{cid}").lower()
    # **所有者接頭辞を落とす**（"Team Rocket's Spidops" → "spidops"）。
    # 落とさないと Team Rocket 系デッキで全カードが "team_rocket_" に切り詰められ、
    # _uniq のx付けで意味のない名前が並ぶ（衝突はしないが可読性が消える）。
    for sep in ("'s ", "’s ", "‘s "):
        if sep in nm:
            nm = nm.split(sep, 1)[1]
            break
    for junk in (" ex", "-", "'", "’", ".", " ", "‘"):
        nm = nm.replace(junk, "_")
    nm = "".join(ch for ch in nm if ch.isalnum() or ch == "_").strip("_")
    return nm[:maxlen] or f"c{cid}"


def _uniq(names: dict[int, str]) -> dict[int, str]:
    """短縮名の衝突を解消する（別カードが同じ特徴キーに潰れるのを防ぐ）。"""
    used: set[str] = set()
    out: dict[int, str] = {}
    for cid in sorted(names):
        s = names[cid]
        while s in used:
            s += "x"
        used.add(s)
        out[cid] = s
    return out


def build(deck_path: str) -> dict:
    from cg.api import CardType, all_card_data
    tbl = {c.cardId: c for c in all_card_data()}
    with open(deck_path) as fh:
        ids = [int(ln.split(",")[0]) for ln in fh
               if ln.strip() and ln.split(",")[0].strip().isdigit()]
    if not ids:
        raise SystemExit(f"デッキが読めない: {deck_path}")
    counts = collections.Counter(ids)

    def ct(cid):
        return getattr(tbl.get(cid), "cardType", None)

    species = tuple(c for c in sorted(counts) if ct(c) == CardType.POKEMON)
    sname = _uniq({c: _short(getattr(tbl.get(c), "name", ""), c) for c in species})

    # ---- 進化ライン: evolvesFrom を辿って最下段まで遡る（デッキ内に限る） ----
    by_name: dict[str, int] = {}
    for c in species:
        by_name.setdefault(getattr(tbl[c], "name", ""), c)
    parent: dict[int, int] = {}
    for c in species:
        src = getattr(tbl[c], "evolvesFrom", None)
        if src and src in by_name and by_name[src] != c:
            parent[c] = by_name[src]

    def root(c, depth=0):
        # 循環が万一あっても止まるように深さで打ち切る
        return root(parent[c], depth + 1) if (c in parent and depth < 5) else c

    line_of = {c: sname[root(c)] for c in species}
    line_names = tuple(sorted(set(line_of.values())))
    pre_evo = tuple(sorted(set(parent.values())))  # 進化先を同デッキ内に持つ種

    # ---- エネルギー ----
    energies = [c for c in sorted(counts)
                if ct(c) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)]
    ename = _uniq({c: _short(getattr(tbl.get(c), "name", ""), c, 10)
                   for c in energies})
    bname = {c: ename[c] for c in energies if ct(c) == CardType.BASIC_ENERGY}

    stadiums = [c for c in sorted(counts) if ct(c) == CardType.STADIUM]
    stname = _uniq({c: _short(getattr(tbl.get(c), "name", ""), c, 10)
                    for c in stadiums})
    tools = [c for c in sorted(counts) if ct(c) == CardType.TOOL]
    toolname = _uniq({c: _short(getattr(tbl.get(c), "name", ""), c, 10)
                      for c in tools})

    # ---- エネ貼り先バケット: ライン × 最終進化かどうか ----
    # 094の手作りバケット(zam / pre / dun / oth)は「ラインの中で最終進化か手前か」で
    # 切られていた。それを機械化する。
    bucket = {c: f"{line_of[c]}_{'p' if c in pre_evo else 'f'}" for c in species}

    # ---- 行動クラス ----
    # トレーナーも短縮名を _uniq に通す（切り詰めで名前が衝突すると別カードが
    # 同一クラスに潰れ、クラス頭が両者を区別できなくなる）
    trainers = [c for c in sorted(counts)
                if ct(c) in (CardType.SUPPORTER, CardType.ITEM,
                             CardType.TOOL, CardType.STADIUM)]
    tname = _uniq({c: _short(getattr(tbl.get(c), "name", ""), c, 10) for c in trainers})
    cls_play = {c: f"play_{sname[c]}" for c in species}
    cls_play.update({c: f"use_{tname[c]}" for c in trainers})
    # 進化クラスは「たねでない種」全てに与える。進化元が同デッキに無い場合
    # （ふしぎなアメで中間を飛ばす構成）でも EVOLVE 選択肢は出るため、
    # `c in parent` で絞ると本来のクラスが "other" に落ちる。
    cls_evo = {c: f"evo_{sname[c]}" for c in species
               if not getattr(tbl[c], "basic", False)}
    # 特性クラスはポケモンだけでなく**スタジアムにも与える**（スタジアムの効果発動も
    # ABILITY型の選択肢として来る。092のSpikemuth GymはMAIN決定の9.9%）
    cls_ab = {c: f"ab_{sname[c]}" for c in species}
    cls_ab.update({c: f"ab_{stname[c]}" for c in stadiums
                   if getattr(tbl.get(c), "skills", None)})
    # 技クラス: card.attacks は attackId の列。技ごとにクラスを立てる
    cls_atk: dict[int, str] = {}
    for c in species:
        for i, aid in enumerate(getattr(tbl[c], "attacks", None) or []):
            cls_atk[int(aid)] = f"atk_{sname[c]}_{i}"
    attach = {f"attach_{e}_{b}"
              for e in list(ename.values()) + list(toolname.values())
              for b in list(bucket.values()) + ["oth"]}
    classes = tuple(sorted(
        set(cls_play.values()) | set(cls_evo.values()) | set(cls_ab.values())
        | set(cls_atk.values()) | attach
        | {"ab_other", "ab_stadium", "retreat", "atk_other", "end", "other"}))

    # ---- 短縮名の衝突検査 ----
    # 種族名やライン名がコア側の固定キーと衝突すると、別の意味の特徴が同じキーに
    # 潰れて静かに壊れる（例外は出ない）。生成時に落とす。
    reserved = {"hp", "hp_ratio", "dmg", "e", "n_tool", "cant", "fresh", "pre",
                "free", "mine", "op", "basic", "oth"}
    for label, names in (("種族", set(sname.values())),
                         ("ライン", set(line_names)),
                         ("エネ", set(ename.values())),
                         ("スタジアム", set(stname.values()))):
        bad = names & reserved
        if bad:
            raise SystemExit(f"{label}の短縮名がコアの固定キーと衝突: {sorted(bad)}"
                             "（tools/gen_featurizer.py の _short を調整すること）")
    if set(sname.values()) & set(ename.values()):
        raise SystemExit("種族名とエネ名が衝突: "
                         f"{sorted(set(sname.values()) & set(ename.values()))}")
    # 特性クラスは種族名とスタジアム名を同じ ab_ 名前空間に入れるので、そこも見る
    ab_dup = set(sname.values()) & set(stname.values())
    if ab_dup:
        raise SystemExit(f"種族名とスタジアム名が衝突（ab_が潰れる）: {sorted(ab_dup)}")
    # 名前が切り詰めで潰れていないか（_uniq が x を足して逃げた形跡）を警告する。
    # 衝突は起きないが、可読性が落ちた特徴器はデバッグを著しく困難にする。
    for label, names in (("種族", sname), ("エネ", ename), ("スタジアム", stname)):
        pad = [v for v in names.values() if v.endswith("xx")]
        if pad:
            print(f"  WARN: {label}の短縮名が切り詰めで衝突し x で回避: {sorted(pad)}")

    # エネ名とツール名が衝突すると attach_ クラスが潰れる
    et_dup = set(ename.values()) & set(toolname.values())
    if et_dup:
        raise SystemExit(f"エネ名とツール名が衝突（attach_が潰れる）: {sorted(et_dup)}")

    return dict(deck=deck_path, species=species, sname=sname, line_of=line_of,
                line_names=line_names, pre_evo=pre_evo, ename=ename, bname=bname,
                stname=stname, toolname=toolname, bucket=bucket,
                cls_play=cls_play, cls_evo=cls_evo,
                cls_ab=cls_ab, cls_atk=cls_atk, classes=classes)


def emit_agent(agent_dir: str, tag: str, teacher: str) -> None:
    """エージェントディレクトリ一式を書き出す（feat / nn / main / attack_table）。

    deck.csv は呼び出し側が置いておく。**既存ファイルは上書きしない**
    （既存エージェントを壊さない。CLAUDE.mdの比較可能性ルール）。
    """
    deck = os.path.join(agent_dir, "deck.csv")
    if not os.path.exists(deck):
        raise SystemExit(f"{deck} が無い（60枚のdeck.csvを先に置くこと）")
    # **別タグの feat_*.py が既にあるディレクトリには書かない**。既存エージェントに
    # 新しい特徴器を生やすと、そのディレクトリの中身が「どの版か」曖昧になり、
    # CLAUDE.md の比較可能性ルール（既存 agents/NNN_* は変更しない）を破る。
    other = [f for f in os.listdir(agent_dir)
             if f.startswith("feat_") and f.endswith(".py") and f != f"feat_{tag}.py"]
    if other:
        raise SystemExit(
            f"{agent_dir} には既に {', '.join(other)} がある。"
            f"既存エージェントは変更しない規約なので、新しい番号のディレクトリを作ること")
    feat = os.path.join(agent_dir, f"feat_{tag}.py")
    build_and_write(deck, feat)
    # numpy推論（デッキ非依存なのでそのまま複製）
    nn_dst = os.path.join(agent_dir, f"nn_{tag}.py")
    if not os.path.exists(nn_dst):
        with open(os.path.join(REPO, "tools/featgen_nn.py")) as s, \
                open(nn_dst, "w") as t:
            t.write(s.read())
    main_dst = os.path.join(agent_dir, "main.py")
    if os.path.exists(main_dst):
        print(f"  main.py は既存なので触らない: {main_dst}")
        return
    tmpl = open(os.path.join(REPO, "tools", "featgen_main.tmpl")).read()
    for k, v in (("@AGENT@", os.path.basename(agent_dir.rstrip("/"))),
                 ("@TEACHER@", teacher),
                 ("@FEATMOD@", f"feat_{tag}"), ("@NNMOD@", f"nn_{tag}"),
                 ("@MODEL@", f"model_{tag}.npz"),
                 ("@ENVVAR@", f"NN_{tag.upper()}")):
        tmpl = tmpl.replace(k, v)
    with open(main_dst, "w") as t:
        t.write(tmpl)
    print(f"  main.py / nn_{tag}.py を生成")


# 既定は 102(384次元)の構成 + 同カテゴリの拡張のみ。
# etype/typeoh/zones は EXP-099 で希釈が疑われた新カテゴリなので既定で外す。
# EXP-122: optcond は除外（toolIndex と全3,684,180選択肢で値が完全一致）。
# oparch も除外（168 vs 167 で **+0.0pt**。他特徴から F1 0.85-0.96 で予測可能な冗長情報）。
# megaprize は**含める**（171で「冗長」として外したのは誤り。max系2つは非線形要約であり
# 線形結合の1つも実効学習率の倍率として機能する。EXP-122で機序を実測）。
GROUPS = {"base", "slot_attr", "resource", "threat2", "misc",
          "selctx", "oppseen", "megaprize"}

DROP_COLS: list = []   # --drop で差し替える


def _meta_cards(tools_dir: str) -> str:
    """tools/meta_cards.py から定数定義を丸ごと取り込む（docstringは落とす）。

    従来は featgen_core.py に META_CARDS/META_ARCH を直書きしていたため、
    **メタが動いても更新されなかった**（07-22版のままで Mega Lopunny ex と
    Teal Mask Ogerpon ex が欠落）。tools/gen_meta_cards.py で再生成し、ここから注入する。
    """
    src = open(os.path.join(tools_dir, "meta_cards.py")).read()
    q = chr(39) * 3                       # docstring の区切り
    return src.split(q, 2)[-1].strip() if src.lstrip().startswith(q) else src


def _meta_zones(tools_dir: str) -> str:
    """tools/meta_zones.py から定数の代入行だけを抜き出す（docstringは落とす）。"""
    src = open(os.path.join(tools_dir, "meta_zones.py")).read()
    return "\n".join(ln for ln in src.split("\n")
                      if re.match(r"^META_(STADIUM|TOOL) = ", ln))


def build_and_write(deck_path: str, out_path: str) -> dict:
    d = build(deck_path)
    _td = os.path.join(REPO, "tools")
    if _td not in sys.path:
        sys.path.insert(0, _td)
    import gen_atk_dmg
    _fix, _var = gen_atk_dmg.build()
    head = HEADER.format(
        n_sp=len(d["species"]), n_line=len(d["line_names"]), n_e=len(d["ename"]),
        n_st=len(d["stname"]), n_cls=len(d["classes"]),
        ATK_TABLES=gen_atk_dmg.emit(_fix, _var).split("\n", 1)[1],
        META_ZONES=_meta_zones(_td),
        META_CARDS_BLOCK=_meta_cards(_td),
        FEAT_GROUPS=repr(GROUPS), DROP_COLS=repr(sorted(DROP_COLS)), **d)
    core = open(os.path.join(REPO, "tools", "featgen_core.py")).read()
    outdir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(outdir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(head + core)
    # 攻撃テーブルは相手の打点予測（セクション7）に必要。同ディレクトリに無ければ置く
    at = os.path.join(outdir, "attack_table_056.py")
    if not os.path.exists(at):
        src = os.path.join(REPO, "agents/094_yushin_nn/attack_table_056.py")
        with open(src) as s, open(at, "w") as t:
            t.write(s.read())
        print(f"  attack_table_056.py を配置 -> {at}")
    print(f"生成 -> {out_path}")
    print(f"  ポケモン {len(d['species'])}種 / ライン {len(d['line_names'])}本 "
          f"({', '.join(d['line_names'])})")
    print(f"  進化前 {[d['sname'][c] for c in d['pre_evo']]}")
    print(f"  エネ {list(d['ename'].values())} / "
          f"スタジアム {list(d['stname'].values())}")
    print(f"  貼り先バケット {sorted(set(d['bucket'].values()))}")
    print(f"  行動クラス {len(d['classes'])}個")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", help="deck.csv（--agent-dir なしのとき必須）")
    ap.add_argument("--out", help="feat_*.py の出力先（--agent-dir なしのとき必須）")
    ap.add_argument("--agent-dir",
                    help="エージェント一式を生成する（deck.csvを先に置いておく）")
    ap.add_argument("--drop", default=None,
                    help="削除列のJSON（tools/calc_drop_cols.py の出力）。"
                         "**デッキごとに実データで算出したものを渡すこと**")
    ap.add_argument("--groups", default=None,
                    help="有効にする特徴グループをカンマ区切りで（既定: base,slot_attr,resource,threat2,misc / all で全部）")
    ap.add_argument("--tag", help="モジュール名の接尾（例: 097 → feat_097.py）")
    ap.add_argument("--teacher", default="（未記入）",
                    help="模倣元。main.pyのdocstringに書き込む")
    args = ap.parse_args()
    if args.drop:
        import json as _j
        global DROP_COLS
        DROP_COLS = sorted(_j.load(open(args.drop))["drop"])
        print(f"  削除列: {len(DROP_COLS)}個 ({args.drop})")
    if args.groups:
        # **グループを差し替えたら次元が変わる**。生成物のヘッダに焼き込まれ、
        # 学習・推論はそこから読むので、指定は生成時の1回だけでよい。
        global GROUPS
        GROUPS = ({"base", "slot_attr", "resource", "threat2", "etype",
                   "typeoh", "zones", "misc", "selctx", "optcond"}
                  if args.groups == "all"
                  else set(x.strip() for x in args.groups.split(",")) | {"base"})
        print(f"  特徴グループ: {sorted(GROUPS)}")

    if args.agent_dir:
        if not args.tag:
            raise SystemExit("--agent-dir には --tag が必要")
        emit_agent(args.agent_dir, args.tag, args.teacher)
        return
    if not (args.deck and args.out):
        raise SystemExit("--deck と --out、または --agent-dir を指定すること")
    build_and_write(args.deck, args.out)


if __name__ == "__main__":
    main()
