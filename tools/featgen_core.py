# ==== 汎用コア（tools/gen_featurizer.py が生成ファイルの後半へ丸ごと連結する） ====
# ここはデッキに依存しない。デッキ依存の情報は前半の生成ヘッダが定義する定数
#   SPECIES / SPECIES_NAME / LINE_OF / LINE_NAMES / PRE_EVO / ENERGY_NAME /
#   BASIC_ENERGY_NAME / STADIUM_NAME / ATTACH_BUCKET / CLS_PLAY / CLS_EVO /
#   CLS_AB / CLASSES / DECK_COUNTS / DECK_IDS
# だけを参照する。
#
# セクション1/2/3/5/7/8とオプション側特徴は agents/094_yushin_nn/feat_094.py の
# **忠実移植**（EXP-094 v12時点。A-1/A-2・playerIndex・toolIndex/energyIndex・
# NULLオプションの修正込み）。セクション4/6の導出項/9のスタジアムone-hotは、
# 094と092の手作り述語を「種族×エネ量」「進化ライン数」の機械展開に置き換えた版。



# ==== 対面識別・脅威判定に使う定数（メタ知識。全レプリカ共通なのでコア側に置く） ====
# 各アーキタイプを一意に特定できる看板カード（直近メタの個体デッキから弁別力で自動選定）。
# **相手がどのデッキかを表す特徴が1つも無かった**のが従来の最大の穴。教師は対面ごとに
# 全く違う打ち方をするのに、モデルはそれを条件付けられなかった
# （実測: 同一デッキ156952a871でも、対Spidopsの勝率がパイロット間で17.9% vs 51.0%）。
# META_CARDS / META_ARCH / ARCH_DECK / OPP_TRACK / ARCH_MAIN は
# 生成ヘッダ（tools/meta_cards.py 由来）で定義される。
META_ARCH_NAMES = sorted(META_ARCH)
ARCH_SLUG = {a: "".join(ch for ch in a.lower().replace(" ", "_")
                        if ch.isalnum() or ch == "_")[:14]
             for a in META_ARCH}
# 自デッキの各カードから進化できる先（選択肢の「取ると進化がつながるか」判定用）


# 効果無効化を与える特殊エネ。**役割が違うので分ける**（MIST_IDS はまとめていた）。
MIST_NOEFFECT_ANY = 11        # Mist Energy: どのポケモンでも効果を受けない
MIST_NOEFFECT_FIGHTING = 20   # Rock Fighting Energy: 闘ポケモンのみ


import re  # noqa: E402  （グループ判定に使う）


# ==== 特徴グループ（絞り込みを構造化する。EXP-099で768次元は希釈と判明） ====
# **打点の修正（弱点・無効化・チェックアップ）は次元を増やさないので常に有効**。
# ここで切れるのは「102(384次元)から増えた分」だけ。
#   base       102 の 384次元（常に有効）
#   slot_attr  相手/自スロットの属性・関係（102 の os{i}_*/ms{i}_* の素直な拡張）
#   resource   山+サイドの残り枚数 / ACE SPEC（102 の hand_/dis_ と同系）
#   threat2    無効化フラグ / 継続ダメージ / 逃走・攻撃可否 / 状態異常（脅威系と同系）
#   etype      スロット別・アクティブのエネ型 one-hot（154次元。新カテゴリ）
#   typeoh     ポケモンの型 one-hot（66次元。新カテゴリ）
#   zones      ツール種別 / 相手スタジアム種別 / 相手トラッシュ内訳（新カテゴリ）
#   misc       ターン進行
# ==== 削除する列（EXP-122）====
# **デッキごとに実データで算出し、生成ヘッダから注入する**（DROP_COLS）。
# 中身は (a) 完全重複列の各群から1本を除く残り (b) 分散ゼロの定数列。
# **ここにハードコードしてはいけない**。重複も定数もデッキと教師データに依存し、
# 104のリストをMarnieに当てると「そのデッキでは重複でない列」を消してしまう
# （me_dot/me_dot_ability は Alakazam のデータで偶然一致しただけ、ACE SPEC は
#  デッキごとに別カード、等）。算出手順は tools/calc_drop_cols.py。
_DROP = frozenset(globals().get("DROP_COLS", ()))


_GRULES = [
    (re.compile(r"^(ms|os)\d_et\d+$|^(myact|opact)_e_type_\d+$"), "etype"),
    (re.compile(r"^(os\d|opact)_type_\d+$"), "typeoh"),
    (re.compile(r"^os\d_(maxhp|retreat|stage|has_ab|tera|weak_vs_me|resist_vs_me"
                r"|i_am_weak_to_it|noeffect_e)$|^ms\d_(weak_vs_opp|beats_opp|maxhp)$"),
     "slot_attr"),
    (re.compile(r"^left_\d+$|^ace_"), "resource"),
    (re.compile(r"^(me|op)_(asleep|paralyzed|confused|poisoned|burned|dot|dot_ability"
                r"|can_retreat|retreat_cost)$|^(my|op)_can_attack$"
                r"|^(my_dmg_nullified|opp_has_protect|my_has_protect)$"), "threat2"),
    (re.compile(r"^tool_(me|op)_\d+$|^stad_meta_\d+$|^opdis_n_"), "zones"),
    (re.compile(r"^(turn_actions|stadium_played)$"), "misc"),
    # **相手の隠れ情報推論（EXP-113 軸A）**。プールで −0.12/−0.23pt、対Lopunny −3.2pt
    # で**棄却済み**なのにコードが残り、以降に生成した特徴器（156〜164）へ自動混入した。
    # グループを切って `--groups` で外せるようにする。**既定では外すこと**。
    (re.compile(r"^opp_(left|phand)_\d+$|^opp_arch_(conf|margin|known)$"
                r"|^opp_(unseen_n|hand_n|main_left|main_phand)$"), "oppbelief"),
    # **メタ語彙**（相手が見せたメタカード / 相手アーキタイプのone-hot）。
    # 教師データに乏しい相手のカードを足すと**他の特徴の学習を薄める純損失**になる
    # （EXP-120: 104→166 で語彙を+35次元したら対Marnie −4.1pt、加重 −2.29pt）。
    # 除去できるようグループを切る。**既定には含める**（104では働いていることを確認済み）。
    # **生のカード観測**（相手の場・付随カード・**トラッシュ**・スタジアムの累積枚数）。
    # トラッシュを含むので試合が進むほど単調増加し、**ターン進行の代理**としても働く。
    # 教師データに乏しい相手のカードは序盤も終盤もほぼ0のままで、純粋な死に次元になる。
    # **EXP-118で足したが冗長と判明した盤面4個**（EXP-122）。
    # my_act_prize は ms0_prz と同一、my_bench_prize_max / my_prize_exposure は
    # ms1_prz〜ms5_prz の最大/合計、opp_bench_prize_max は os1_prz〜os5_prz の最大。
    # 104に元からある ms*_prz / os*_prz / prz_if_*_ko / prz_me / prz_op / prz_diff で
    # 完全に表現できており、重みノルムも 0.55〜0.87（他の中央値0.952）と低い。
    # **既定から外す**。選択肢側の is_mega_ex / opt_prize は本物の穴を埋めているので残す
    # （is_ex は Mega ex に対して 0.0 になる。Api.h の ex/megaEx は排他）。
    (re.compile(r"^(my_act_prize|my_bench_prize_max|my_prize_exposure"
                r"|opp_bench_prize_max)$"), "megaprize"),
    (re.compile(r"^opp_seen_\d+$"), "oppseen"),
    # **その要約**（op_seen から「看板カードを何割見たか」で系統を推定した one-hot）。
    # oppseen を落として oparch だけ残す構成を試せるように、別グループにする。
    (re.compile(r"^oparch_"), "oparch"),
]


def _group_of(k: str) -> str:
    for rx, g in _GRULES:
        if rx.match(k):
            return g
    return "base"


def _prefix_of(name: str) -> str:
    """所有者接頭辞（"Team Rocket’s Articuno" → "Team Rocket’s "）。無ければ空。"""
    for sep in ("’s ", "'s "):
        i = (name or "").find(sep)
        if i > 0:
            return name[:i + len(sep)]
    return ""


def _p_hold(k: int, n_unseen: int, n_hand: int) -> float:
    """相手の未観測 n_unseen 枚のうち k 枚が対象カードのとき、
    手札 n_hand 枚に**1枚以上含まれる確率**（超幾何分布）。

    P(1枚以上) = 1 - C(n_unseen-k, n_hand) / C(n_unseen, n_hand)
    ループで書くのは、提出物が numpy 以外に依存しないため。
    """
    if k <= 0 or n_hand <= 0 or n_unseen <= 0:
        return 0.0
    if k >= n_unseen or n_hand >= n_unseen:
        return 1.0
    q = 1.0
    for i in range(n_hand):
        q *= (n_unseen - k - i) / (n_unseen - i)
        if q <= 0.0:
            return 1.0
    return 1.0 - q


def _protected(defender, on_bench: bool, def_field, attacker_cd, is_counter: bool):
    """defender が、その側の場にある無効化特性で守られているか。

    PROTECT はカードテキストから自動生成（tools/gen_atk_dmg.py）。**「ダメージ」と
    「効果」は別物**で、ダメカンを置く技（Alakazam の Powerful Hand 等）は効果側。
    Team Rocket's Articuno の Repelling Veil は効果だけを消すので Powerful Hand は
    通らないが、通常のダメージ技は通る。これを入れないと対Spidopsで打点を
    平均249も過大評価する（tools/feat_verify_damage.py で実測）。
    """
    if defender is None:
        return False
    dcd = card_table.get(defender.id)
    # **付いている特殊エネによる効果無効化**（Mist Energy / Rock Fighting Energy）。
    # 「効果を受けない（ダメージは受ける）」なので、ダメカンを置く技だけが消える。
    # Rock Fighting Energy は闘ポケモンにしか効かない（targetEnergyType(Fighting)）。
    if is_counter:
        for e in (getattr(defender, "energyCards", None) or []):
            if e.id == MIST_NOEFFECT_ANY:
                return True
            if (e.id == MIST_NOEFFECT_FIGHTING
                    and getattr(dcd, "energyType", None) == 6):
                return True
    for p in def_field:
        if p is None:
            continue
        t = PROTECT.get(p.id)
        if t is None:
            continue
        kind, scope, cond = t
        if is_counter:
            if kind not in ("effects", "both"):
                continue
        elif kind not in ("damage", "both"):
            continue
        if scope == "self" and p is not defender:
            continue
        if scope == "bench" and not on_bench:
            continue
        if scope == "team_basic":
            pre = _prefix_of(getattr(card_table.get(p.id), "name", "") or "")
            nm = getattr(dcd, "name", "") or ""
            if not (getattr(dcd, "basic", False) and pre and nm.startswith(pre)):
                continue
        if cond == "def_no_rulebox" and (getattr(dcd, "ex", False)
                                         or getattr(dcd, "megaEx", False)):
            continue
        if cond == "atk_ex" and not (getattr(attacker_cd, "ex", False)
                                     or getattr(attacker_cd, "megaEx", False)):
            continue
        if cond == "atk_basic_ex" and not (getattr(attacker_cd, "basic", False)
                                           and getattr(attacker_cd, "ex", False)):
            continue
        if cond == "atk_ability" and not getattr(attacker_cd, "skills", None):
            continue
        return True
    return False


def _resolve_dmg(cid: int, mv: str, dmg, ctx):
    """1つの技の (打点, 解決できたか, ダメカン配置か) を返す。

    **打点解決はここ1か所に集約する**。`attack_table_056.py` は EN_Card_Data.csv の
    Damage 列しか見ておらず、打点が効果文にしか書かれていない技（全体の約26%）を
    すべて None にしている。その中には **Alakazam の唯一の攻撃 Powerful Hand**
    （手札1枚につき20）が含まれる。ATK_FIX/ATK_VAR はその効果文を解析した表
    （tools/gen_atk_dmg.py 生成）。

    ベンチ限定の技（Shaymin の Pinpoint Dive 等）はアクティブに通らないので 0 を返す。
    """
    if dmg is not None:
        return dmg, True, False
    key = (cid, (mv or "").strip())
    t = ATK_FIX.get(key)
    if t is not None:
        return (t[0] if t[1] else 0), True, bool(t[2])
    t = ATK_VAR.get(key)
    if t is not None:
        kind, unit, to_act, is_cnt = t
        return ((unit * max(0, ctx.get(kind, 0)) if to_act else 0), True,
                bool(is_cnt))
    return 0, False, False


def _wr_mult(attacker_type, defender_card):
    """(弱点倍率, 抵抗の減算)。エンジン実装（SetProperty.h CalcDamage）に合わせる:
    弱点は damage *= 2、抵抗は damage -= 30（0未満は0）。

    **弱点は相手ポケモン単体の属性ではなく「自分の型 × 相手の弱点」という関係**。
    これを入れていなかったため、教師の試合の54.5%（対Marnie/Garchomp）で
    `they_ko_me` 等の脅威特徴が2倍ぶん過小評価されていた。
    """
    if defender_card is None or attacker_type is None:
        return 1, 0
    w = 2 if getattr(defender_card, "weakness", None) == attacker_type else 1
    r = 30 if getattr(defender_card, "resistance", None) == attacker_type else 0
    return w, r


def _afford(cid: int, mv: str, cn: int, n_energy: int, etypes, wild: int = 0):
    """その技が撃てるか。etypes（付いているエネが供給する型のlist）が無ければ枚数判定。

    `{P}●●` は超1+任意2であって任意3では撃てない。ATTACKS_056 は個数しか持たない
    ので、型付きコスト表 ATK_COST で判定する。

    wild は「これから貼る、型が未定のエネの個数」。**脅威推定では相手が必要な型を
    貼ってくる前提（＝安全側）で数える**。無色と決めつけると相手の打点を過小評価し、
    「殺されない」と誤認する。
    """
    t = ATK_COST.get((cid, (mv or "").strip()))
    if t is None or etypes is None:
        return cn <= n_energy
    typed, colorless = t
    pool = list(etypes)
    for need_t in typed:
        if need_t in pool:
            pool.remove(need_t)
        elif wild > 0:
            wild -= 1            # これから貼るエネで型要求を満たす
        else:
            return False         # 無色エネは型要求を満たさない
    return len(pool) + wild >= colorless


def _atk_scan(p, n_energy: int, ctx, defender=None, etypes=None, wild: int = 0,
              def_field=(), on_bench: bool = False, defender_obj=None):
    """p が n_energy 個のエネで出せる (最大打点, 未解決可変フラグ, 最小要求エネ)。

    defender を渡すと弱点×2・抵抗−30 を適用した「実際に通る打点」になる。
    etypes を渡すと型付きコストで撃てるかを判定する。wild はこれから貼る任意型のエネ数。
    ctx は打点が状態依存の技のための盤面数値（p の視点で渡す）。
    """
    acd = card_table.get(p.id)
    atype = getattr(acd, "energyType", None)
    wmul, rsub = _wr_mult(atype, defender)
    mx, var, need = 0, 0, 99
    for (cn, dmg, mv) in ATTACKS_056.get(p.id, ()):
        ok_cost = _afford(p.id, mv, cn, n_energy, etypes, wild)
        d, ok, is_cnt = _resolve_dmg(p.id, mv, dmg, ctx)
        if not ok:
            if ok_cost:
                var = 1          # 打点不明だが撃てる技がある
            continue
        if d > 0 and def_field and _protected(defender_obj, on_bench, def_field,
                                              acd, is_cnt):
            d = 0                # 無効化特性で通らない
        if d > 0:
            d = max(0, d * wmul - rsub)
        if d > 0 and cn < need:
            need = cn
        if ok_cost and d > mx:
            mx = d
    return mx, var, (0 if need == 99 else need)


def _atk_ctx(ps, p, extra_energy: int = 0, extra_hand: int = 0):
    """_atk_scan に渡す ctx を、プレイヤー ps・ポケモン p から組み立てる。"""
    return {
        "hand": (ps.handCount if hasattr(ps, "handCount") else len(ps.hand or []))
        + extra_hand,
        "bench": sum(1 for b in ps.bench if b is not None),
        "energy": len(p.energyCards or []) + extra_energy,
        "prize_taken": 6 - len(ps.prize or []),
        "discard_poke": sum(1 for c in ps.discard
                            if getattr(card_table.get(c.id), "cardType", None)
                            == CardType.POKEMON),
    }


def _clock(hp: float, dmg: float) -> float:
    """残HPを打点で割った「あと何回殴られるか」。打点0なら大きな値。"""
    if dmg <= 0:
        return 9.0
    import math
    return min(9.0, math.ceil(hp / dmg))


def _prz_of_id(cid: int) -> int:
    cd = card_table.get(cid)
    if cd is None:
        return 1
    return 3 if cd.megaEx else (2 if cd.ex else 1)


def _cd(p):
    """場のポケモン → CardData（None安全）。弱点/抵抗/型を引くのに使う。"""
    return card_table.get(p.id) if p is not None else None


def _etypes(p):
    """p に付いているエネが供給する型のlist（Pokemon.energies）。無ければ None。"""
    e = getattr(p, "energies", None)
    return list(e) if e is not None else None


def _maxdmg_next(p, opp_player, defender=None, def_field=(),
                 defender_obj=None) -> tuple[int, int]:
    """相手ポケモン p が次のターン（エネ+1・ドロー+1）に出せる (最大打点, 可変フラグ)。

    defender に自分のアクティブを渡すと弱点込みの「実際に食らう打点」になる。
    """
    e_next = len(p.energyCards or []) + 1
    ctx = _atk_ctx(opp_player, p, extra_energy=1, extra_hand=1)
    mx, var, _ = _atk_scan(p, e_next, ctx, defender=defender,
                           etypes=_etypes(p), wild=1,   # 貼る1個は任意型とみなす
                           def_field=def_field, defender_obj=defender_obj)
    return mx, var


EVO_TARGETS = {
    c: tuple(t for t in DECK_IDS
             if getattr(card_table.get(t), "evolvesFrom", None)
             == getattr(card_table.get(c), "name", None))
    for c in DECK_IDS}
EVO_TARGETS = {k: v for k, v in EVO_TARGETS.items() if v}


def _e_count(p, ids) -> int:
    """ポケモンpに付いている、id集合idsのエネ枚数。"""
    return sum(1 for e in (p.energyCards or []) if e.id in ids)


def featurize(state, my_index: int) -> dict[str, float]:
    """State（obs.current）→ 特徴dict。全キー常に出力（0含む）。"""
    me = state.players[my_index]
    op = state.players[1 - my_index]
    f: dict[str, float] = {}

    # ---- 1. ゲーム進行 ----
    f["turn"] = state.turn
    f["first"] = 1.0 if state.firstPlayer == my_index else 0.0
    prz_me, prz_op = len(me.prize), len(op.prize)
    f["prz_me"] = prz_me
    f["prz_op"] = prz_op
    f["prz_diff"] = prz_op - prz_me  # 正=自分リード

    # ---- 2. 枚数カウント ----
    my_hand = me.hand or []
    f["n_hand_me"] = len(my_hand) if me.hand is not None else me.handCount
    f["n_deck_me"] = me.deckCount
    f["n_hand_op"] = op.handCount
    f["n_deck_op"] = op.deckCount
    my_bench = [p for p in me.bench if p is not None]
    op_bench = [p for p in op.bench if p is not None]
    f["n_bench_me"] = len(my_bench)
    f["n_bench_op"] = len(op_bench)
    f["n_dis_me"] = len(me.discard)
    f["n_dis_op"] = len(op.discard)

    # ---- 3. 自分の手札の中身（ID別） ----
    for cid in DECK_IDS:
        f[f"hand_{cid}"] = 0.0
    for c in my_hand:
        k = f"hand_{c.id}"
        if k in f:
            f[k] += 1

    # ---- 4. 自分の場 ----
    # 手作り版の述語（094: ready_zam/loaded_pre/dudun3、092: ready_grim/charged_pre/
    # munki_online/grimline_field/snow_field）は全て「種族またはラインの、場の数 ×
    # エネ枚数の閾値」だった。固定閾値を焼くのでなく **枚数そのもの（合計と最大）** を
    # 出す方が情報量が多く、閾値はMLP側が学べる。
    act = me.active[0] if me.active else None
    for cid in SPECIES:
        f[f"act_{SPECIES_NAME[cid]}"] = 0.0
    f["act_hp"] = 0.0
    f["act_hp_ratio"] = 0.0
    f["act_dmg"] = 0.0
    f["act_e"] = 0.0
    f["act_n_tool"] = 0.0
    for nm in ENERGY_NAME.values():
        f[f"act_e_{nm}"] = 0.0
    f["act_cant"] = 1.0 if (me.asleep or me.paralyzed) else 0.0
    f["act_fresh"] = 0.0
    # **自分側のサイド露出**（EXP-119）。エンジンは倒されたポケモンの種別で
    # 相手が取るサイド枚数を変える（State.h::getPrizeCount）:
    #   通常 1枚 / ex 2枚 / **MegaEx 3枚**（6枚で敗北なので1体で半分）
    # 従来は `opp_act_prize` として**相手側にしか**持たせていなかったため、
    # 「自分のMegaを前に出してよいか」という Megaデッキ最重要の判断材料が
    # モデルに一切入っていなかった。156/157 が忠実度 −36/−33pt で落ちた原因。
    f["my_act_prize"] = _prz_of_id(act.id) if act is not None else 0.0
    f["my_bench_prize_max"] = max([_prz_of_id(p.id) for p in my_bench], default=0.0)
    f["my_prize_exposure"] = float(sum(_prz_of_id(p.id)
                                       for p in ([act] if act else []) + my_bench))
    # 相手側も対称に持たせる（既存の opp_act_prize / opp_bench_prz2 の補完）
    f["opp_bench_prize_max"] = max([_prz_of_id(p.id) for p in op_bench], default=0.0)
    # **サイド系の非対称を解消**（EXP-122）。26特徴を自分側/相手側で全数突合したところ、
    # 欠けていたのはこの2つだけだった。opp_prize_exposure は「相手盤面を総取りしたら
    # 何枚動くか」、my_bench_prz2 は「自分のベンチにex級が何体いるか」。
    f["my_bench_prz2"] = float(sum(1 for p in my_bench if _prz_of_id(p.id) >= 2))
    if act is not None:
        k = SPECIES_NAME.get(act.id)
        if k:
            f[f"act_{k}"] = 1.0
        f["act_hp"] = act.hp
        f["act_hp_ratio"] = act.hp / act.maxHp if act.maxHp else 0.0
        f["act_dmg"] = act.maxHp - act.hp
        f["act_e"] = len(act.energyCards or [])
        f["act_n_tool"] = len(getattr(act, "tools", []) or [])
        f["act_fresh"] = 1.0 if act.appearThisTurn else 0.0
        for e in (act.energyCards or []):
            nm = ENERGY_NAME.get(e.id)
            if nm:
                f[f"act_e_{nm}"] += 1

    field_all = ([act] if act is not None else []) + my_bench
    for cid in SPECIES:
        nm = SPECIES_NAME[cid]
        cps = [p for p in field_all if p.id == cid]
        f[f"n_{nm}"] = len(cps)                    # 場（アクティブ込み）の数
        f[f"e_{nm}"] = sum(len(p.energyCards or []) for p in cps)
        f[f"emax_{nm}"] = max((len(p.energyCards or []) for p in cps), default=0)
        f[f"dmg_{nm}"] = sum(p.maxHp - p.hp for p in cps)
    for ln in LINE_NAMES:                           # 進化ライン単位の場の数
        f[f"line_{ln}"] = sum(1 for p in field_all if LINE_OF.get(p.id) == ln)
    pres = [p for p in field_all if p.id in PRE_EVO]
    f["n_pre"] = len(pres)
    f["e_pre"] = sum(len(p.energyCards or []) for p in pres)
    f["emax_pre"] = max((len(p.energyCards or []) for p in pres), default=0)
    f["bench_free"] = me.benchMax - len(my_bench)
    f["bench_dmg"] = sum(p.maxHp - p.hp for p in my_bench)
    f["field_dmg_me"] = f["bench_dmg"] + f["act_dmg"]

    # ---- 5. 自分のトラッシュ（ID別） ----
    for cid in DECK_IDS:
        f[f"dis_{cid}"] = 0.0
    for c in me.discard:
        k = f"dis_{c.id}"
        if k in f:
            f[k] += 1

    # ---- 6. 山∪サイドのプール（ID別残数） ----
    seen: dict[int, int] = {}

    def _see(cid):
        seen[cid] = seen.get(cid, 0) + 1

    for c in my_hand:
        _see(c.id)
    for c in me.discard:
        _see(c.id)
    for p in field_all:
        _see(p.id)
        for c in (p.energyCards or []):
            _see(c.id)
        for c in (getattr(p, "tools", []) or []):
            _see(c.id)
        for c in (getattr(p, "preEvolution", []) or []):
            _see(c.id)
    for c in state.stadium:
        if c.playerIndex == my_index:
            _see(c.id)
    for cid in DECK_IDS:
        f[f"pool_{cid}"] = max(0, DECK_COUNTS[cid] - seen.get(cid, 0))
    # 導出確率: 各基本エネが「残り全部サイド落ち」の確率（超幾何）。
    # 094の p_psy_all_prized の一般化（基本エネ種ごとに1つ）。
    pool_total = me.deckCount + prz_me
    for cid, nm in BASIC_ENERGY_NAME.items():
        c_left = f.get(f"pool_{cid}", 0.0)
        p_all = 0.0
        if 0 < c_left <= prz_me and pool_total > 0:
            p_all = 1.0
            for i in range(int(c_left)):
                p_all *= max(0.0, (prz_me - i)) / max(1, (pool_total - i))
        f[f"p_{nm}_all_prized"] = p_all

    # ---- 7. 相手の場（属性ベース。デッキ非依存） ----
    op_act = op.active[0] if op.active else None
    op_articuno = any(p.id == TR_ARTICUNO
                      for p in ([op_act] if op_act else []) + op_bench)
    f["opp_act_hp"] = op_act.hp if op_act else 0.0
    f["opp_act_dmg"] = (op_act.maxHp - op_act.hp) if op_act else 0.0
    f["opp_act_prize"] = _prz_of_id(op_act.id) if op_act else 0.0
    f["opp_act_energy"] = len(op_act.energyCards) if op_act else 0.0
    _myf = ([act] if act is not None else []) + my_bench
    mx, var = (_maxdmg_next(op_act, op, defender=_cd(act), def_field=_myf,
                            defender_obj=act) if op_act else (0, 0))
    f["opp_act_maxdmg"] = mx
    f["opp_act_vardmg"] = var
    ph_immune = 0.0
    if op_act is not None:
        mist = _e_count(op_act, MIST_IDS)
        cd = card_table.get(op_act.id)
        veil = (op_articuno and cd is not None and cd.basic
                and "Team Rocket" in cd.name)
        ph_immune = 1.0 if (mist > 0 or veil) else 0.0
    f["opp_act_ph_immune"] = ph_immune
    f["opp_bench_prz2"] = sum(1 for p in op_bench if _prz_of_id(p.id) >= 2)
    # 相手盤面の総サイド（自分側 my_prize_exposure の対称物。EXP-122）。
    # ここに置くのは op_act がこの位置で初めて確定するため。
    f["opp_prize_exposure"] = float(sum(_prz_of_id(p.id)
                                        for p in ([op_act] if op_act else []) + op_bench))
    f["opp_bench_canatk"] = sum(
        1 for p in op_bench
        if any(cn <= len(p.energyCards) for (cn, _d, _m) in ATTACKS_056.get(p.id, ())))
    f["opp_field_maxdmg"] = max(
        [mx] + [_maxdmg_next(p, op, defender=_cd(act), def_field=_myf,
                             defender_obj=act)[0] for p in op_bench],
        default=0)
    f["opp_bench_maxhp"] = max([p.hp for p in op_bench], default=0)
    f["opp_bench_growing"] = sum(1 for p in op_bench
                                 if len(p.energyCards) == 0 and p.hp == p.maxHp)
    f["opp_cant"] = 1.0 if (op.asleep or op.paralyzed) else 0.0

    # ---- 8. 相手のトラッシュ（資源勘定） ----
    f["opp_used_boss"] = sum(1 for c in op.discard if c.id == BOSS_ORDERS)
    n_e = n_sup = n_poke = 0
    for c in op.discard:
        cd = card_table.get(c.id)
        if cd is None:
            continue
        if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_e += 1
        elif cd.cardType == CardType.SUPPORTER:
            n_sup += 1
        elif cd.cardType == CardType.POKEMON:
            n_poke += 1
    f["opp_e_disc"] = n_e
    f["opp_sup_used"] = n_sup
    f["opp_poke_lost"] = n_poke

    # ---- 9. スタジアム・ターンフラグ ----
    st_mine = st_op = 0.0
    for nm in STADIUM_NAME.values():
        f[f"stadium_{nm}"] = 0.0
    for c in state.stadium:
        if c.playerIndex == my_index:
            st_mine = 1.0
        else:
            st_op = 1.0
        nm = STADIUM_NAME.get(c.id)
        if nm:
            f[f"stadium_{nm}"] = 1.0
    f["stadium_mine"] = st_mine
    f["stadium_op"] = st_op
    # **相手のスタジアムは STADIUM_NAME に席が無い**（自デッキから生成した表のため）。
    # 実戦で Spikemuth Gym / Battle Cage 等が計456回出たが stadium_op の1ビットしか
    # 立たず、効果が全く違うものを区別できなかった。+1=自分が置いた / -1=相手。
    for _sc in META_STADIUM:
        f[f"stad_meta_{_sc}"] = 0.0
    for c in state.stadium:
        if c.id in META_STADIUM:
            f[f"stad_meta_{c.id}"] = 1.0 if c.playerIndex == my_index else -1.0
    f["f_eatt"] = 1.0 if state.energyAttached else 0.0
    f["f_sup"] = 1.0 if state.supporterPlayed else 0.0
    f["f_retreated"] = 1.0 if state.retreated else 0.0

    # ---- 10. 対面識別（相手がどのデッキか） ----
    # **従来ここが完全に欠落していた**。教師は対面ごとに全く違う打ち方をするのに、
    # モデルはそれを条件付けられなかった。相手が公開した領域（場・付随カード・
    # トラッシュ・スタジアム）から、アーキタイプの看板カードの観測数を数える。
    op_field = ([op_act] if op_act is not None else []) + op_bench
    op_seen: dict[int, int] = {}
    for p in op_field:
        op_seen[p.id] = op_seen.get(p.id, 0) + 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in (getattr(p, grp, None) or []):
                op_seen[c.id] = op_seen.get(c.id, 0) + 1
    for c in op.discard:
        op_seen[c.id] = op_seen.get(c.id, 0) + 1
    for c in state.stadium:
        if c.playerIndex != my_index:
            op_seen[c.id] = op_seen.get(c.id, 0) + 1
    # META_CARDS=識別（2系統以下）/ THREAT_CARDS=脅威追跡（3系統以上かつ我々に干渉）。
    # **目的が違うだけで表現は同じ**（観測枚数）。基準は tools/meta_cards.py に明記。
    for cid in list(META_CARDS) + list(globals().get("THREAT_CARDS", ())):
        f[f"opp_seen_{cid}"] = float(op_seen.get(cid, 0))
    # 系統ごとの「看板カードを何割見たか」。1枚でも見えれば強い証拠になる
    _best_a, _best_s, _snd = None, 0.0, 0.0
    for a in META_ARCH_NAMES:
        ids = META_ARCH[a]
        hit = sum(1 for cid in ids if op_seen.get(cid))
        sc = hit / max(1, len(ids))
        f[f"oparch_{ARCH_SLUG[a]}"] = sc
        if sc > _best_s:
            _best_a, _snd, _best_s = a, _best_s, sc
        elif sc > _snd:
            _snd = sc

    # ---- 10a. 相手の隠れ情報の推論（**アーキタイプが分かれば60枚は既知**） ----
    # 従来は opp_seen_*（観測した枚数）しか持たず、**残り枚数も手札にある確率も
    # 計算していなかった**。自分側には left_/pool_ と超幾何確率まであるのに非対称だった。
    # 「相手のBoss's Ordersはもう0枚」「Unfair Stampを持っている確率は35%」は
    # ベンチの置き方・手札の抱え方を直接変える情報で、**モデルには計算できない**
    # （デッキ構成という外部知識が要る）。
    f["opp_arch_conf"] = _best_s
    f["opp_arch_margin"] = _best_s - _snd      # 2位との差＝識別の確からしさ
    _dk = ARCH_DECK.get(_best_a) if _best_s > 0.0 else None
    f["opp_arch_known"] = 1.0 if _dk else 0.0
    # 相手の未観測領域（山＋手札＋サイド）。ここに残枚数が散っている
    _op_hand = op.handCount if hasattr(op, "handCount") else len(op.hand or [])
    _op_unseen = max(0, op.deckCount + _op_hand + len(op.prize or []))
    f["opp_unseen_n"] = float(_op_unseen)
    f["opp_hand_n"] = float(_op_hand)
    for cid in OPP_TRACK:
        left = max(0, _dk.get(cid, 0) - op_seen.get(cid, 0)) if _dk else 0
        f[f"opp_left_{cid}"] = float(left)
        f[f"opp_phand_{cid}"] = _p_hold(left, _op_unseen, _op_hand)
    # 主戦力が山にあと何枚いるか＝この先の脅威の量
    _main = ARCH_MAIN.get(_best_a, 0) if _dk else 0
    f["opp_main_left"] = float(max(0, _dk.get(_main, 0) - op_seen.get(_main, 0))) if _dk else 0.0
    f["opp_main_phand"] = _p_hold(int(f["opp_main_left"]), _op_unseen, _op_hand)

    # ---- 10b. 毎ターン確定で入るダメージ（クロックの前に確定させる） ----
    # 毒10 / 火傷20 に加え、**場に出ている特性由来のチェックアップダメージ**も数える。
    # Froslass「Freezing Shroud」はポケモンチェックのとき特性を持つ全ポケモン（両者）に
    # ダメカン1個を置く。Marnie は教師の試合の44%で Froslass は4枚採用なので、
    # 落とすとクロックが毎ターン10ずれる（tools/feat_verify_damage.py で発見）。
    _chk_src = []
    for _side in (me, op):
        for _p in (([_side.active[0]] if _side.active and _side.active[0] else [])
                   + [x for x in _side.bench if x is not None]):
            if _p.id in CHECKUP_DMG:
                _chk_src.append((_p.id, CHECKUP_DMG[_p.id]))

    def _dot_of(ps):
        a0 = ps.active[0] if ps.active and ps.active[0] else None
        v = (10.0 if getattr(ps, "poisoned", False) else 0.0) \
            + (20.0 if getattr(ps, "burned", False) else 0.0)
        if a0 is not None and getattr(card_table.get(a0.id), "skills", None):
            # 発生源自身は対象外（"except any Froslass"）
            v += float(sum(d for cid2, d in _chk_src if cid2 != a0.id))
        return v

    dot_me, dot_op = _dot_of(me), _dot_of(op)

    # ---- 11. 脅威とクロック（ダメージレースの算術） ----
    # **これらは1つも無かった**。「今KOできるか」「次にKOされるか」は決定に直結する。
    op_field_all = ([op_act] if op_act is not None else []) + op_bench
    my_field_all = ([act] if act is not None else []) + my_bench
    my_dmg, my_var, my_need = (
        _atk_scan(act, len(act.energyCards or []), _atk_ctx(me, act),
                  defender=_cd(op_act), etypes=_etypes(act),
                  def_field=op_field_all, defender_obj=op_act)
        if act is not None else (0, 0, 0))
    op_dmg_now, op_var, _ = (
        _atk_scan(op_act, len(op_act.energyCards or []), _atk_ctx(op, op_act),
                  defender=_cd(act), etypes=_etypes(op_act),
                  def_field=my_field_all, defender_obj=act)
        if op_act is not None else (0, 0, 0))
    op_dmg_next, _ = (_maxdmg_next(op_act, op, defender=_cd(act),
                                   def_field=my_field_all, defender_obj=act)
                      if op_act else (0, 0))
    f["my_best_dmg"] = float(my_dmg)
    f["my_var_atk"] = float(my_var)
    f["my_energy_need"] = float(max(0, my_need - (len(act.energyCards or [])
                                                  if act else 0)))
    # **「技が撃てるか」と「打点が出るか」は別物**。無効化されていても攻撃自体は
    # 合法で、0ダメージで撃つ（＝ターンを渡す）のは現実の選択肢。打点だけ見ていると
    # この区別が消える。
    f["my_can_attack"] = 1.0 if (act is not None and any(
        _afford(act.id, mv, cn, len(act.energyCards or []), _etypes(act))
        for (cn, _d, mv) in ATTACKS_056.get(act.id, ()))) else 0.0
    f["op_can_attack"] = 1.0 if (op_act is not None and any(
        _afford(op_act.id, mv, cn, len(op_act.energyCards or []), _etypes(op_act))
        for (cn, _d, mv) in ATTACKS_056.get(op_act.id, ()))) else 0.0
    f["opp_best_dmg_now"] = float(op_dmg_now)
    f["opp_best_dmg_next"] = float(op_dmg_next)
    opp_hp = float(op_act.hp) if op_act else 0.0
    my_hp = float(act.hp) if act else 0.0
    f["ko_opp_active"] = 1.0 if (op_act is not None and my_dmg >= opp_hp > 0) else 0.0
    f["they_ko_me"] = 1.0 if (act is not None and op_dmg_next >= my_hp > 0) else 0.0
    # **継続ダメージは打点と同じ働きをする**ので、クロックに足す
    f["clock_them"] = _clock(opp_hp, my_dmg + dot_op)
    f["clock_me"] = _clock(my_hp, op_dmg_next + dot_me)
    f["clock_diff"] = f["clock_me"] - f["clock_them"]   # 正=こちらが速い
    f["dmg_margin"] = float(my_dmg) - opp_hp         # 打点の過不足
    # サイドレース: KOで何枚動くか / それで決着するか
    f["prz_if_my_act_ko"] = float(_prz_of_id(act.id)) if act else 0.0
    f["prz_if_opp_act_ko"] = float(_prz_of_id(op_act.id)) if op_act else 0.0
    f["lethal_for_me"] = 1.0 if (f["ko_opp_active"] and
                                 prz_me <= f["prz_if_opp_act_ko"]) else 0.0
    f["lethal_for_opp"] = 1.0 if (f["they_ko_me"] and
                                  prz_op <= f["prz_if_my_act_ko"]) else 0.0
    # **無効化が刺さっているか**を明示する。打点0の理由が「エネ不足」なのか
    # 「効果を消されている」なのかで取るべき行動が正反対になる（後者は殴っても無駄で、
    # 別の攻撃役を用意するかベンチを狙う必要がある）。実測で対Spidopsの打点予測を
    # 平均249も過大評価していた原因（tools/feat_verify_damage.py）。
    f["my_dmg_nullified"] = 1.0 if (
        op_act is not None and _protected(op_act, False, op_field_all,
                                          _cd(act), True)) else 0.0
    f["opp_has_protect"] = float(sum(1 for p6 in op_field_all
                                     if p6 is not None and p6.id in PROTECT))
    f["my_has_protect"] = float(sum(1 for p6 in my_field_all
                                    if p6 is not None and p6.id in PROTECT))

    # ---- 12. 盤面のスロット単位（集約をやめる） ----
    # ベンチを合計値でしか持っていなかったが、「どのスロットの誰が傷んでいるか」は
    # ボスの指令の的・進化先・エネの貼り先の判断に直結する。選択肢側の inPlayIndex と
    # 対応が取れる形で、スロットごとに出す。
    # bench は **詰めずに** 使う。空きスロットを詰めると選択肢側の inPlayIndex と
    # 番号が食い違い、「どのスロットを指しているか」を突き合わせられなくなる。
    my_list = [act] + list(me.bench)
    op_list = [op_act] + list(op.bench)
    for side, lst, tag in ((me, my_list, "ms"), (op, op_list, "os")):
        for i in range(6):
            p = lst[i] if i < len(lst) else None
            pre = f"{tag}{i}_"
            f[pre + "on"] = 1.0 if p is not None else 0.0
            f[pre + "hpr"] = (p.hp / p.maxHp) if (p and p.maxHp) else 0.0
            f[pre + "dmg"] = float(p.maxHp - p.hp) if p else 0.0
            f[pre + "e"] = float(len(p.energyCards or [])) if p else 0.0
            f[pre + "tool"] = float(len(getattr(p, "tools", []) or [])) if p else 0.0
            f[pre + "pre"] = float(len(getattr(p, "preEvolution", []) or [])) if p else 0.0
            f[pre + "prz"] = float(_prz_of_id(p.id)) if p else 0.0
            f[pre + "fresh"] = 1.0 if (p and p.appearThisTurn) else 0.0
            if tag == "ms":
                for cid in SPECIES:      # 自分側は種族まで持つ（相手は自デッキ外なので不可）
                    f[pre + SPECIES_NAME[cid]] = 1.0 if (p and p.id == cid) else 0.0
            else:
                f[pre + "dmgnext"] = (
                    float(_maxdmg_next(p, op, defender=_cd(act),
                                       def_field=my_field_all,
                                       defender_obj=act)[0]) if p else 0.0)

    # ---- 13. 資源とデッキアウト ----
    n_by_type = {"poke": 0, "sup": 0, "item": 0, "tool": 0, "energy": 0, "stad": 0}
    for c in my_hand:
        cd = card_table.get(c.id)
        t = getattr(cd, "cardType", None)
        if t == CardType.POKEMON:
            n_by_type["poke"] += 1
        elif t == CardType.SUPPORTER:
            n_by_type["sup"] += 1
        elif t == CardType.ITEM:
            n_by_type["item"] += 1
        elif t == CardType.TOOL:
            n_by_type["tool"] += 1
        elif t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_by_type["energy"] += 1
        elif t == CardType.STADIUM:
            n_by_type["stad"] += 1
    for k, v in n_by_type.items():
        f[f"hand_n_{k}"] = float(v)
    # ACE SPEC はデッキに1枚しか入れられない。使ったかどうかが計画に効く
    f["ace_in_hand"] = float(sum(
        1 for c in my_hand if getattr(card_table.get(c.id), "aceSpec", False)))
    f["ace_used"] = float(sum(
        1 for c in me.discard if getattr(card_table.get(c.id), "aceSpec", False)))
    f["ace_op_used"] = float(sum(
        1 for c in op.discard if getattr(card_table.get(c.id), "aceSpec", False)))
    f["deckout_me"] = float(me.deckCount)      # 1ターン1ドローなので枚数がそのままターン数
    f["deckout_op"] = float(op.deckCount)

    # ---- 13b. 場に付いているツールの種別（枚数しか見ていなかった） ----
    # ツールは装備先のポケモンの性能を直接変える（相手の Hero’s Cape が283回付いていた）。
    # TOOL_NAME は自デッキのツールしか持たず、しかも option_class でしか使っていなかった。
    for _tag, _lst in (("me", my_list), ("op", op_list)):
        for _tc in META_TOOL:
            f[f"tool_{_tag}_{_tc}"] = 0.0
        for _p6 in _lst:
            if _p6 is None:
                continue
            for _t in (getattr(_p6, "tools", None) or []):
                _k = f"tool_{_tag}_{_t.id}"
                if _k in f:
                    f[_k] += 1.0

    # ---- 13c. 相手トラッシュの種別内訳（総枚数しか見ていなかった） ----
    # トラッシュは**両者とも中身が見える**ので、相手が何を使い切ったかが分かる。
    _od = {"poke": 0, "sup": 0, "item": 0, "tool": 0, "energy": 0, "stad": 0}
    for c in op.discard:
        _t = getattr(card_table.get(c.id), "cardType", None)
        if _t == CardType.POKEMON:
            _od["poke"] += 1
        elif _t == CardType.SUPPORTER:
            _od["sup"] += 1
        elif _t == CardType.ITEM:
            _od["item"] += 1
        elif _t == CardType.TOOL:
            _od["tool"] += 1
        elif _t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            _od["energy"] += 1
        elif _t == CardType.STADIUM:
            _od["stad"] += 1
    for _k2, _v2 in _od.items():
        f[f"opdis_n_{_k2}"] = float(_v2)

    # ---- 14. 状態異常（L1。従来は自分の asleep/paralyzed しか見ていなかった） ----
    # 相手が眠り/マヒなら逃げられず殴られ続ける。毒/火傷は毎ターン確定ダメージで
    # クロックを直接変える。
    for tag, ps, dv in (("me", me, dot_me), ("op", op, dot_op)):
        for cond in ("asleep", "paralyzed", "confused", "poisoned", "burned"):
            f[f"{tag}_{cond}"] = 1.0 if getattr(ps, cond, False) else 0.0
        f[f"{tag}_dot"] = dv               # ポケモンチェックでの確定ダメージ/ターン
        f[f"{tag}_dot_ability"] = dv - (
            (10.0 if getattr(ps, "poisoned", False) else 0.0)
            + (20.0 if getattr(ps, "burned", False) else 0.0))
        # 逃げられるか（逃げコスト × 付いているエネ × 行動不能）
        a = ps.active[0] if ps.active and ps.active[0] else None
        rc = getattr(card_table.get(a.id), "retreatCost", None) if a else None
        f[f"{tag}_retreat_cost"] = float(rc if rc is not None else 0)
        # **1ターン1回の制限を条件に入れる**（state.retreated）。入れないと
        # 「逃げられる」と主張しているのに RETREAT 選択肢が無い偽陽性が出る。
        f[f"{tag}_can_retreat"] = 1.0 if (
            a is not None and rc is not None
            and len(a.energyCards or []) >= rc
            and not getattr(ps, "asleep", False)
            and not getattr(ps, "paralyzed", False)
            and not (tag == "me" and state.retreated)) else 0.0

    # ---- 15. 相手スロットの属性（段3。カードの素性でなく属性なので未知カードでも効く） ----
    # 相手のカード空間は直近1週間で138種あり、上位50種で97.7%だが、
    # **メタが動くと陳腐化する**。属性はルール由来なので陳腐化しない。
    my_type = getattr(_cd(act), "energyType", None)
    for i in range(6):
        p6 = op_list[i] if i < len(op_list) else None
        cd6 = _cd(p6)
        pre = f"os{i}_"
        f[pre + "maxhp"] = float(p6.maxHp) if p6 else 0.0
        rc = getattr(cd6, "retreatCost", None)
        f[pre + "retreat"] = float(rc if rc is not None else 0)
        f[pre + "stage"] = (0.0 if cd6 is None else
                            (3.0 if cd6.megaEx else 2.0 if cd6.stage2
                             else 1.0 if cd6.stage1 else 0.0))
        f[pre + "has_ab"] = 1.0 if (cd6 is not None
                                    and getattr(cd6, "skills", None)) else 0.0
        f[pre + "tera"] = 1.0 if getattr(cd6, "tera", False) else 0.0
        wm, rs = _wr_mult(my_type, cd6)
        f[pre + "weak_vs_me"] = 1.0 if wm > 1 else 0.0      # 自分の型で2倍取れる
        f[pre + "resist_vs_me"] = 1.0 if rs else 0.0
        # そのスロットの型で自分のアクティブが2倍食らうか（ボスの指令の判断材料）
        wm2, _ = _wr_mult(getattr(cd6, "energyType", None), _cd(act))
        f[pre + "i_am_weak_to_it"] = 1.0 if wm2 > 1 else 0.0
    # 相手の各枠に「効果を無効化する特殊エネ」が何個ついているか。
    # Mist Energy / Rock Fighting Energy はダメカン系の技を完全に消す。
    for i in range(6):
        p6 = op_list[i] if i < len(op_list) else None
        f[f"os{i}_noeffect_e"] = float(sum(
            1 for e in (getattr(p6, "energyCards", None) or [])
            if e.id in (MIST_NOEFFECT_ANY, MIST_NOEFFECT_FIGHTING))
        ) if p6 is not None else 0.0

    # 相手アクティブの型 one-hot（系統の識別と弱点判定の材料。11種で有限）
    oa_t = getattr(_cd(op_act), "energyType", None)
    for t in range(11):
        f[f"opact_type_{t}"] = 1.0 if oa_t == t else 0.0

    # ---- 16. 自分スロットの関係（自分側は種族one-hotがあるので属性は冗長。関係だけ出す） ----
    op_t = getattr(_cd(op_act), "energyType", None)
    for i in range(6):
        p6 = my_list[i] if i < len(my_list) else None
        cd6 = _cd(p6)
        wm, _ = _wr_mult(op_t, cd6)
        f[f"ms{i}_weak_vs_opp"] = 1.0 if wm > 1 else 0.0
        wm2, _ = _wr_mult(getattr(cd6, "energyType", None), _cd(op_act))
        f[f"ms{i}_beats_opp"] = 1.0 if wm2 > 1 else 0.0

    # ---- 17. 付いているエネの「型」（枚数だけでは技が撃てるか決まらない） ----
    # **アクティブだけでなく全スロットに出す**。ベンチのエネ型は「前に出せば技が
    # 撃てるか」の判断に要り、相手ベンチのそれはボスの指令の対象選びに効く。
    for tag, lst in (("ms", my_list), ("os", op_list)):
        for i in range(6):
            p6 = lst[i] if i < len(lst) else None
            et = (_etypes(p6) or []) if p6 is not None else []
            for t in range(11):
                f[f"{tag}{i}_et{t}"] = float(et.count(t))
    for tag, p6 in (("myact", act), ("opact", op_act)):
        et = _etypes(p6) or []
        for t in range(11):
            f[f"{tag}_e_type_{t}"] = float(et.count(t))
    # 自分スロットの maxHp。種族one-hotでは分からない（ツール/特性で増減するため。
    # 実測で Spidops 130→230 のような変化が531件あった）
    for i in range(6):
        p6 = my_list[i] if i < len(my_list) else None
        f[f"ms{i}_maxhp"] = float(p6.maxHp) if p6 is not None else 0.0
    # 相手ベンチの型 one-hot（アクティブにしか出していなかった）
    for i in range(1, 6):
        p6 = op_list[i] if i < len(op_list) else None
        tt = getattr(_cd(p6), "energyType", None)
        for t in range(11):
            f[f"os{i}_type_{t}"] = 1.0 if tt == t else 0.0

    # ---- 18. 山+サイドに残っている枚数（カード別。L1の穴だった） ----
    # 選択肢側には left_after があったが、盤面側に無かった。「Alakazamが残り何枚か」は
    # 行動選択の前段の計画に効く。
    seen: dict[int, int] = {}
    for c in my_hand:
        seen[c.id] = seen.get(c.id, 0) + 1
    for c in me.discard:
        seen[c.id] = seen.get(c.id, 0) + 1
    for p6 in field_all:
        seen[p6.id] = seen.get(p6.id, 0) + 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in (getattr(p6, grp, None) or []):
                seen[c.id] = seen.get(c.id, 0) + 1
    # **場に出したスタジアムを数え落とさない**。スタジアムは trash でも field でも
    # なく state.stadium へ移るので、除外すると「まだ山にある」と誤答する
    # （実測591件の不一致）。
    for c in state.stadium:
        if c.playerIndex == my_index:
            seen[c.id] = seen.get(c.id, 0) + 1
    for cid, n in DECK_COUNTS.items():
        f[f"left_{cid}"] = float(max(0, n - seen.get(cid, 0)))

    # ---- 19. ターンの進行（L1の未使用フィールド） ----
    f["turn_actions"] = float(getattr(state, "turnActionCount", 0) or 0)
    f["stadium_played"] = 1.0 if getattr(state, "stadiumPlayed", False) else 0.0

    # **有効なグループのキーだけ残す**（dict内包は順序を保つのでキー順は不変）
    if FEAT_GROUPS is not None:
        f = {k: v for k, v in f.items() if _group_of(k) in FEAT_GROUPS}
    if _DROP:
        f = {k: v for k, v in f.items() if k not in _DROP}
    return f


CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def option_class(state, select, opt_i: int, my_index: int) -> str:
    """MAIN選択肢→行動クラス名。単一コードパス（収集・推論共用）。"""
    from cg.api import AreaType, OptionType
    o = select.option[opt_i]
    me = state.players[my_index]
    if o.type == OptionType.ATTACK:
        # **技ごとに分ける**（092で実装。どの技を撃つかはMAIN決定の中核なので、
        # 「atk」1クラスに潰すと打ち分けの情報がクラス頭に載らない）
        return CLS_ATK.get(o.attackId, "atk_other")
    if o.type == OptionType.END:
        return "end"
    if o.type == OptionType.RETREAT:
        return "retreat"
    if o.type == OptionType.ABILITY:
        try:
            # **STADIUMを必ず分岐に入れる**。スタジアムの効果発動もABILITY型で来る
            # （092のSpikemuth GymはMAIN決定の9.9%）。ACTIVE以外をbenchと決めつけると
            # 無関係なベンチのポケモンのIDを読み、例外も出ずに誤分類される。
            if o.area == AreaType.STADIUM:
                # **相手のスタジアムも来る**（Spikemuth Gym等は両者が効果を使える。
                # 094の教師データではMAIN決定の1.5%）。自デッキに無いカードは
                # CLS_ABに席が無いので、雑多な ab_other でなく専用クラスに寄せる。
                return CLS_AB.get(state.stadium[o.index].id, "ab_stadium")
            elif o.area == AreaType.ACTIVE:
                cid = me.active[o.index].id
            elif o.area == AreaType.BENCH:
                cid = me.bench[o.index].id
            else:
                return "ab_other"
        except (IndexError, TypeError, AttributeError):
            return "ab_other"
        return CLS_AB.get(cid, "ab_other")
    try:
        cid = me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return "other"
    if o.type == OptionType.PLAY:
        return CLS_PLAY.get(cid, "other")
    if o.type == OptionType.EVOLVE:
        return CLS_EVO.get(cid, "other")
    if o.type == OptionType.ATTACH:
        # **エネだけでなくツールも ATTACH で来る**（Hero's Cape / Brave Bangle 等）。
        # エネしか見ないとツール装着が全て "other" に落ちる
        # （Spidopsの教師データではMAIN決定の1.9%）。貼り先の選択も伴う行動なので
        # エネと同じ「対象バケット付き」のクラスにする。
        et = ENERGY_NAME.get(cid) or TOOL_NAME.get(cid)
        if et is None:
            return "other"
        try:
            src = me.active if o.inPlayArea == AreaType.ACTIVE else me.bench
            tid = src[o.inPlayIndex].id
        except (IndexError, TypeError, AttributeError):
            return f"attach_{et}_oth"
        return f"attach_{et}_{ATTACH_BUCKET.get(tid, 'oth')}"
    return "other"


FEAT_KEYS = None  # 初回featurize時に確定


def feat_vector(state, my_index: int):
    """特徴dictを固定順のリストにして返す（学習・推論共用）。"""
    global FEAT_KEYS
    f = featurize(state, my_index)
    if FEAT_KEYS is None:
        FEAT_KEYS = sorted(f)
    return [f[k] for k in FEAT_KEYS], FEAT_KEYS


# ==== 全選択NN方策（LSTM）用の追加特徴（feat_094.py から忠実移植・デッキ非依存） ====

N_LOGTYPE = 24          # cg.api.LogType の種類数
# その決定自体の要件（下記 logs_vector 参照）。グループ "selctx" で on/off する。
SELECT_DIM = 5 if "selctx" in FEAT_GROUPS else 0
LOGS_DIM = N_LOGTYPE * 2 + SELECT_DIM
# option_vector の次元（24 + 可能化4 + 状態異常種別1）。最後の1つは "optcond" で切れる。
# +2 は is_mega_ex / opt_prize（EXP-119。Mega ex はサイド3枚＝ex の1.5倍の重さ）。
OPT_DIM = 30 + (1 if "optcond" in FEAT_GROUPS else 0)
N_CTX = 49              # cg.api.SelectContext の種類数（0..48）


def _num094(x, default=-1.0):
    """None/enum混在の安全な数値化（END/ATTACK等は area/index が None）。"""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def logs_vector(obs, my_index: int) -> list[float]:
    """前回決定以降の logs 要約（LogType別カウント）+ **その決定自体の要件**。

    後半5次元は SelectData の未使用フィールド。「あと何個ダメカンを置くか」
    「あと何個ぶんのエネコストが残っているか」「何枚選ぶ決定か」は、その決定の
    文脈そのものなのに盤面からは分からない（featurize は state しか見ないため）。
    """
    v = [0.0] * LOGS_DIM
    for lg in (getattr(obs, "logs", None) or []):
        t = _num094(getattr(lg, "type", None), -1.0)
        ti = int(t)
        if not (0 <= ti < N_LOGTYPE):
            continue
        pi = getattr(lg, "playerIndex", my_index)
        off = 0 if pi == my_index else N_LOGTYPE
        v[off + ti] += 1.0
    sel = getattr(obs, "select", None)
    if sel is not None and "selctx" in FEAT_GROUPS:
        b = N_LOGTYPE * 2
        v[b + 0] = _num094(getattr(sel, "minCount", None), 0.0)
        v[b + 1] = _num094(getattr(sel, "maxCount", None), 0.0)
        v[b + 2] = _num094(getattr(sel, "remainDamageCounter", None), 0.0)
        v[b + 3] = _num094(getattr(sel, "remainEnergyCost", None), 0.0)
        v[b + 4] = 1.0 if getattr(sel, "contextCard", None) is not None else 0.0
    return v


def resolve_option(obs, opt, my_index: int):
    """option → (card_id, 属性dict)。解決不能なら card_id=-1（裏向きサイド等）。

    **opt.playerIndex を尊重する**（ボスの指令のガスト先など、相手の場を対象にする
    選択が実在する。ctx3 SWITCHの約6割がこれ）。
    """
    from cg.api import AreaType, OptionType
    st = obs.current
    owner = opt.playerIndex if opt.playerIndex is not None else my_index
    ps = st.players[owner]
    card = None
    try:
        if opt.area is None:
            # **PLAY は area=None で index が手札index**（cardIdも空）。
            if opt.type == OptionType.PLAY and opt.index is not None:
                card = (ps.hand or [])[opt.index]
        elif opt.area == AreaType.DECK:
            card = (obs.select.deck or [])[opt.index]
        elif opt.area == AreaType.HAND:
            card = (ps.hand or [])[opt.index]
        elif opt.area == AreaType.DISCARD:
            card = ps.discard[opt.index]
        elif opt.area == AreaType.ACTIVE:
            card = ps.active[opt.index]
        elif opt.area == AreaType.BENCH:
            card = ps.bench[opt.index]
        elif opt.area == AreaType.STADIUM:
            card = st.stadium[opt.index]
        elif opt.area == AreaType.LOOKING:
            card = st.looking[opt.index]
    except (IndexError, TypeError, AttributeError):
        card = None
    at = {"is_opp": 1.0 if owner != my_index else 0.0}
    if card is None:
        return -1, at
    cid = getattr(card, "id", -1)
    # **サイド枚数はカードIDだけで決まる**ので、場のポケモンに限らず常に付ける。
    # エンジンAPIは ex と megaEx を別フラグで返し（Api.h: megaEx）、
    # Mega Lucario ex は ex=False / megaEx=True。従来の is_ex だけだと
    # **デッキの主役が「ただのポケモン」として 0.0** になっていた（EXP-119で実測）。
    # とくに「手札からMegaへ進化する」選択はリスク判断が最も要る場面なので、
    # 場のポケモン限定にすると肝心なところで効かない（実測 20回中18回が0.0だった）。
    _mcd = card_table.get(cid)
    at["is_mega_ex"] = 1.0 if (_mcd is not None
                               and getattr(_mcd, "megaEx", False)) else 0.0
    at["opt_prize"] = float(_prz_of_id(cid)) if cid is not None and cid >= 0 else 0.0
    if hasattr(card, "energyCards"):     # ポケモン
        cd = card_table.get(cid)
        mx = float(getattr(card, "maxHp", 0) or 0)
        hp = float(getattr(card, "hp", 0) or 0)   # hp=残HP、maxHp=最大HP
        at["is_pokemon"] = 1.0
        at["hp"] = hp / 300.0
        at["max_hp"] = mx / 300.0
        at["dmg"] = (mx - hp) / 300.0
        at["hurt_ratio"] = (1.0 - hp / mx) if mx > 0 else 0.0
        at["n_energy"] = float(len(card.energyCards or []))
        at["n_tool"] = float(len(getattr(card, "tools", []) or []))
        at["n_pre"] = float(len(getattr(card, "preEvolution", []) or []))
        at["is_ex"] = 1.0 if (cd is not None and getattr(cd, "ex", False)) else 0.0
    else:
        at["is_pokemon"] = 0.0
    return cid, at


def pool_counts(state, my_index: int):
    """(手札内の同カード枚数, 山∪サイド残枚数) の2 dict。"""
    me = state.players[my_index]
    hand = me.hand or []
    hand_counts: dict[int, int] = {}
    for c in hand:
        hand_counts[c.id] = hand_counts.get(c.id, 0) + 1
    seen = dict(hand_counts)
    for c in me.discard:
        seen[c.id] = seen.get(c.id, 0) + 1
    fld = ([me.active[0]] if me.active and me.active[0] else []) \
        + [b for b in me.bench if b is not None]
    for p in fld:
        seen[p.id] = seen.get(p.id, 0) + 1
        for c in (p.energyCards or []):
            seen[c.id] = seen.get(c.id, 0) + 1
        for c in (getattr(p, "tools", None) or []):
            seen[c.id] = seen.get(c.id, 0) + 1
        for c in (getattr(p, "preEvolution", []) or []):
            seen[c.id] = seen.get(c.id, 0) + 1
    # 盤面側 left_{cid} と同じ理由でスタジアムを数える
    for c in state.stadium:
        if c.playerIndex == my_index:
            seen[c.id] = seen.get(c.id, 0) + 1
    left = {cid: max(0, n - seen.get(cid, 0)) for cid, n in DECK_COUNTS.items()}
    return hand_counts, left


def option_vector(obs, opt, my_index: int, hand_counts, pool_left):
    """オプション1つ → (card_id, 数値特徴list[OPT_DIM])。"""
    cid, at = resolve_option(obs, opt, my_index)
    _idx = opt.index if opt.index is not None else -1
    return cid, [
        _num094(opt.type),
        _num094(opt.area),
        float(min(_idx, 20)),
        at.get("is_pokemon", 0.0),
        at.get("hp", 0.0),
        at.get("dmg", 0.0),
        at.get("n_energy", 0.0),
        at.get("n_tool", 0.0),
        at.get("n_pre", 0.0),
        at.get("is_ex", 0.0),
        at.get("is_mega_ex", 0.0),
        at.get("opt_prize", 0.0),
        float(hand_counts.get(cid, 0)),
        float(pool_left.get(cid, 0)),
        _num094(getattr(opt, "number", None), 0.0),
        _num094(getattr(opt, "attackId", None), 0.0),
        at.get("is_opp", 0.0),        # 相手所有の選択肢か（ガスト先など）
        at.get("max_hp", 0.0),
        at.get("hurt_ratio", 0.0),
        # 同一ポケモンに付いた複数のツール/エネを区別する
        _num094(getattr(opt, "toolIndex", None), -1.0),
        _num094(getattr(opt, "energyIndex", None), -1.0),
        _num094(getattr(opt, "count", None), 0.0),
        # **貼り先/進化先**（inPlayArea/inPlayIndex）。無いとMAIN決定の約15%が
        # 原理的に判別不能になる（094 v5までの穴 A-2）。
        _num094(getattr(opt, "inPlayArea", None), -1.0),
        _num094(getattr(opt, "inPlayIndex", None), -1.0),
        *((_num094(getattr(opt, "specialConditionType", None), -1.0),)
          if "optcond" in FEAT_GROUPS else ()),
        *_inplay_target_feats(obs, opt, my_index),
        # **「そのカードが何を可能にするか」**（ここまでは「何であるか」しか無かった）。
        # サーチ先の選択(ctx7)は全決定の15%を占め、一致率が0.62と最も低い区間だった。
        # カードIDの埋め込みだけから「今このカードが要るか」を学ばせるのは無理がある。
        *_enable_feats(obs, opt, my_index, cid, pool_left),
    ]


def _enable_feats(obs, opt, my_index: int, cid: int, pool_left):
    """選択肢のカードが何を可能にするか（4次元）。

      enables_evo   : そのカードの進化元が自分の場にいるか（＝取れば進化できる）
      is_pre_needed : そのカードの進化先を自分が持っている（場or手札）か
      completes_atk : エネなら、アクティブの最小要求エネを満たすか
      left_after    : 取った後に山∪サイドに残る枚数
    """
    try:
        st = obs.current
        ps = st.players[my_index]
        cd = card_table.get(cid)
        if cd is None:
            return [0.0, 0.0, 0.0, 0.0]
        actv = ps.active[0] if (ps.active and ps.active[0]) else None
        field = ([actv] if actv else []) + [b for b in ps.bench if b is not None]
        src = getattr(cd, "evolvesFrom", None)
        enables = 0.0
        if src:
            enables = 1.0 if any(
                getattr(card_table.get(p.id), "name", None) == src for p in field) else 0.0
        # 自分がこのカードから進化する先を持っているか（EVO_TARGETS は事前計算）
        tgt = EVO_TARGETS.get(cid)
        need_pre = 0.0
        if tgt:
            have = {c.id for c in (ps.hand or [])} | {p.id for p in field}
            need_pre = 1.0 if any(t in have for t in tgt) else 0.0
        completes = 0.0
        if cid in ENERGY_NAME and actv is not None:
            _mx, _v, need = _atk_scan(actv, len(actv.energyCards or []),
                                      _atk_ctx(ps, actv), etypes=_etypes(actv))
            completes = 1.0 if (
                need and len(actv.energyCards or []) + 1 >= need) else 0.0
        return [enables, need_pre, completes,
                float(max(0, pool_left.get(cid, 0) - 1))]
    except (IndexError, TypeError, AttributeError):
        return [0.0, 0.0, 0.0, 0.0]


def _inplay_target_feats(obs, opt, my_index: int):
    """貼り先/進化先ポケモンの (残HP割合, エネ数)。取得できなければ0。"""
    from cg.api import AreaType
    ipa, ipi = getattr(opt, "inPlayArea", None), getattr(opt, "inPlayIndex", None)
    if ipa is None or ipi is None:
        return [0.0, 0.0]
    try:
        ps = obs.current.players[my_index]
        p = ps.active[ipi] if ipa == AreaType.ACTIVE else ps.bench[ipi]
        if p is None:
            return [0.0, 0.0]
        mx = float(getattr(p, "maxHp", 0) or 0)
        hp = float(getattr(p, "hp", 0) or 0)
        return [hp / mx if mx > 0 else 0.0, float(len(p.energyCards or []))]
    except (IndexError, TypeError, AttributeError):
        return [0.0, 0.0]


# **【EXP-139で棄却】この関数はどの特徴器からも呼ばれていない（純追加・次元不変）。**
# 狙い所（正解クラスが同点の決定）で **−0.03pt**（3シード sd 0.59pt, t=−0.10）、
# 全体val +0.013pt。理由は **board の ms{slot}_{種族} one-hot + inPlayIndex から
# 対象カードIDが 100.0000% の精度で復元できる**＝情報は既に入力に在ったこと。
# **配線し直す前に EXP-139.md を読むこと。** 残してあるのは記録のためで、
# 「未着手の改善案」ではない。
#
# 貼り先/進化先ポケモンの**カードID**（EXP-139）。opt_vec は対象について
# inPlayArea/inPlayIndex（位置）と 残HP比/エネ数 しか持たず、**種族が入っていない**。
# MAINでは cls_head の ATTACH_BUCKET が代役をするが Abra(741)/Kadabra(742) が
# 同一クラス 'abra_p' という粗さがあり、ctx37 EVOLVE は非MAINなので cls_head 自体が
# 効かない。これを opt_enc 側（two-tower の選択肢塔）へ一元化するための追加出力。
#
# **数値特徴ではないので opt_vec の次元は変えない**（埋め込みの索引として別に渡す）。
# 既存の特徴器・既存モデルには一切影響しない純追加。
PAD_TCID = -2      # train_nn_kaggle の vocab で行0（padding_idx。寄与は厳密に0）


def option_target_cid(obs, opt, my_index: int) -> int:
    """貼り先/進化先ポケモンのカードID。対象が無ければ PAD_TCID。

    **_inplay_target_feats と同じ対象解決**をする（学習データ側の復元もこの規則）。

    inPlayArea には AreaType に存在しない **0（対象なしの番兵）** が来る。
    教師データ435,150決定での実測: ctx 7/5/2/19 に61,550件あり、ipi は常に0で、
    `_inplay_target_feats` は**全件 [0,0] を返している**（= 対象として解決されていない）。
    `ipa is None` では弾けないので、**ACTIVE/BENCH に明示的に限定する**。
    """
    from cg.api import AreaType
    ipa = getattr(opt, "inPlayArea", None)
    ipi = getattr(opt, "inPlayIndex", None)
    if ipa is None or ipi is None:
        return PAD_TCID
    if ipa != AreaType.ACTIVE and ipa != AreaType.BENCH:
        return PAD_TCID
    try:
        ps = obs.current.players[my_index]
        p = ps.active[ipi] if ipa == AreaType.ACTIVE else ps.bench[ipi]
        return int(p.id) if p is not None else PAD_TCID
    except (IndexError, TypeError, AttributeError):
        return PAD_TCID


# 棄権(NULL)オプションの特徴。minCount==0 の決定で選択肢列の末尾に足し、
# pointer networkのstop actionとして機能させる（EXP-094 v7）。
NULL_CID = -3


def null_option_vector():
    """棄権オプションの数値特徴。実オプションと区別できる値にする。"""
    v = [0.0] * OPT_DIM
    v[0] = -1.0   # type: 実optionには無い値
    v[1] = -1.0   # area
    v[2] = -1.0   # index
    return v
