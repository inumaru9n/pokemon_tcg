"""攻撃の打点テーブルを EN_Card_Data.csv から生成する（Damage列が n/a の技を効果文から救う）。

**なぜ必要か**: `attack_table_056.py` は Damage 列だけを見ており、Damage='n/a' の技を
すべて「打点 None」＝打点不明として落としている。ところが **Alakazam の唯一の攻撃
Powerful Hand は Damage='n/a'**（打点が効果文にしか書かれていない）で、これは我々が
今いちばん強くしたいデッキの主砲そのものである。この穴があると「今KOできるか」
「次に殺されるか」といった脅威特徴が Alakazam デッキで**恒久的にゼロ**になる。

出力は2種類:

  ATK_FIX[(card_id, move)] = (dmg, to_active)   効果文に固定打点が書いてあった技
  ATK_VAR[(card_id, move)] = (kind, unit, to_active)
       状態依存の技。kind は打点の数え方、unit は1個あたりのダメージ。
       kind='hand'    自分の手札枚数（Alakazam の Powerful Hand: 20/枚）
       kind='bench'   自分のベンチ数
       kind='energy'  自分についているエネルギー数
       kind='discard_g' トラッシュの基本草エネ数 など、数え方が特殊なもの

`for each` を含む技は固定値化できないので ATK_FIX には入れず ATK_VAR に回す。
どちらにも解決できなかった技は従来どおり「可変フラグのみ」として扱う。
"""

from __future__ import annotations

import csv
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "pokemon-tcg-ai-battle/data/EN_Card_Data.csv")

# 「〜1つにつき」の数え方。効果文の言い回し → kind
_COUNT_KINDS = [
    (re.compile(r"for each card in your hand", re.I), "hand"),
    (re.compile(r"for each of your Benched Pok", re.I), "bench"),
    (re.compile(r"for each Energy attached to this Pok", re.I), "energy"),
    (re.compile(r"for each Prize card you have taken", re.I), "prize_taken"),
    (re.compile(r"for each Pok.mon in your discard", re.I), "discard_poke"),
]
# 打点の書かれ方（damage counters は1個=10ダメージ）
_RE_DMG = re.compile(r"does (\d+) (?:more )?damage(?: to ([^.]*))?", re.I)
_RE_CNT = re.compile(r"(?:Place|Put) (\d+) damage counters?(?: on ([^.]*))?", re.I)


def _parse_damage(effect: str):
    """効果文から (kind, unit, to_active) を返す。kind=None なら固定打点。

    to_active=False は「ベンチにしか当たらない技」（例: Shaymin の Pinpoint Dive は
    相手ベンチのex/Vにしか60を出せない）。アクティブをKOできるかの判定にそのまま
    使うと嘘になるので区別する。
    """
    if not effect:
        return None, None, True, False
    unit, tgt = None, ""
    m = _RE_CNT.search(effect)
    if m:
        unit = int(m.group(1)) * 10       # ダメカン1個 = 10ダメージ
        tgt = m.group(2) or ""
    else:
        m = _RE_DMG.search(effect)
        if m:
            unit = int(m.group(1))
            tgt = m.group(2) or ""
    if unit is None:
        return None, None, True, False
    # 対象が「ベンチの〜」だけならアクティブには当たらない
    to_active = not (re.search(r"Bench", tgt, re.I) and not re.search(r"Active", tgt, re.I))
    # **ダメカンを置く技は「ダメージ」ではなく「効果」**。カードテキストに明記が
    # ある（Skeledirge: "Damage is not an effect."）。Team Rocket's Articuno の
    # Repelling Veil は効果だけを無効化するので、Alakazam の Powerful Hand は
    # 消えるが通常のダメージ技は通る。この区別が無いと対Spidopsで打点を
    # 平均249も過大評価する（実測）。
    is_counter = bool(_RE_CNT.search(effect))
    for rx, kind in _COUNT_KINDS:
        if rx.search(effect):
            return kind, unit, to_active, is_counter
    if re.search(r"for each", effect, re.I):
        return "unknown", unit, to_active, is_counter   # 数え方が未対応
    return None, unit, to_active, is_counter


# 技コストの記号 → energyType 番号（cg.api の CardData.energyType と同じ体系）
_SYM = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8}


def _parse_cost(txt: str):
    """'{P}●●' → ([5], 2) のように (必要な型のリスト, 無色の個数) を返す。

    **枚数だけでは技が撃てるか判定できない**。`{P}●●` は超エネ1枚+任意2枚であって、
    任意3枚では撃てない。従来の attack_table_056 は個数しか持っていなかった。
    """
    if not txt or txt == "n/a":
        return (), 0
    typed, colorless = [], 0
    i = 0
    while i < len(txt):
        ch = txt[i]
        if ch == "{":
            j = txt.find("}", i)
            if j < 0:
                break
            sym = txt[i + 1:j]
            if sym in _SYM:
                typed.append(_SYM[sym])
            else:
                colorless += 1          # {C} {A} など無色扱い
            i = j + 1
        else:
            if not ch.isspace():
                colorless += 1          # ● は無色
            i += 1
    return tuple(typed), colorless


# ---- ダメージ/効果の無効化特性 ----
# scope = 誰を守るか / cond = 追加条件（攻撃側または防御側にかかる）
_PROT_SCOPE = [
    (re.compile(r"done to your Benched|to your Benched", re.I), "bench"),
    (re.compile(r"done to this Pok|to this Pok", re.I), "self"),
    (re.compile(r"to your Basic .{0,24}Pok", re.I), "team_basic"),
    (re.compile(r"done to your", re.I), "team"),
]
_PROT_COND = [
    (re.compile(r"don.t have a Rule Box", re.I), "def_no_rulebox"),
    (re.compile(r"opponent.s Basic Pok.mon .?\{?ex", re.I), "atk_basic_ex"),
    (re.compile(r"opponent.s Pok.mon .?\{?ex", re.I), "atk_ex"),
    (re.compile(r"that have an Ability", re.I), "atk_ability"),
]
_RE_PROT = re.compile(r"[Pp]revent all (damage from and effects of attacks|"
                      r"effects of attacks|damage)", re.I)


def build_protect():
    """{card_id: (kind, scope, cond)} を返す。kind は damage/effects/both。"""
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    out = {}
    for r in rows:
        mv = (r.get("Move Name") or "").strip()
        cid = (r.get("Card ID") or "").strip()
        if not cid.isdigit() or not mv.startswith("[Ability]"):
            continue
        tx = r.get("Effect Explanation") or ""
        m = _RE_PROT.search(tx)
        if not m:
            continue
        g = m.group(1).lower()
        kind = ("both" if "and effects" in g
                else "effects" if "effects" in g else "damage")
        scope = next((v for rx, v in _PROT_SCOPE if rx.search(tx)), "self")
        cond = next((v for rx, v in _PROT_COND if rx.search(tx)), "")
        out[int(cid)] = (kind, scope, cond)
    return out


_RE_CHECKUP = re.compile(
    r"During Pok.mon Checkup,? .{0,40}?(?:put|place) (\d+) damage counter", re.I)


def build_checkup():
    """{card_id: 1ターンあたりの確定ダメージ} を返す（ポケモンチェックの特性）。

    Froslass「Freezing Shroud」型。**毎ターン確定で入るのでクロックに直結する**が、
    毒/火傷しか数えていないと落ちる（Marnie は教師の試合の44%で Froslass 4枚採用）。
    """
    out = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8-sig")):
        cid = (r.get("Card ID") or "").strip()
        mv = (r.get("Move Name") or "").strip()
        if not cid.isdigit() or not mv.startswith("[Ability]"):
            continue
        m = _RE_CHECKUP.search(r.get("Effect Explanation") or "")
        if m:
            out[int(cid)] = int(m.group(1)) * 10
    return out


def build_costs():
    """全ポケモンの技コストを型付きで返す: {(card_id, move): (typed, colorless)}。"""
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    out = {}
    for r in rows:
        mv = (r.get("Move Name") or "").strip()
        cid = (r.get("Card ID") or "").strip()
        cost = (r.get("Cost") or "n/a").strip()
        if (not mv or mv == "n/a" or not cid.isdigit()
                or mv.startswith("[Ability]") or cost in ("n/a", "")):
            continue
        t, c = _parse_cost(cost)
        if t or c:
            out[(int(cid), mv)] = (t, c)
    return out


def build():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    fix, var = {}, {}
    for r in rows:
        mv = (r.get("Move Name") or "").strip()
        cid = (r.get("Card ID") or "").strip()
        cost = (r.get("Cost") or "n/a").strip()
        # 特性・エネルギーの常時効果・[Tera] などは攻撃ではない（コスト欄が n/a）
        if (not mv or mv == "n/a" or not cid.isdigit()
                or mv.startswith("[Ability]") or cost in ("n/a", "")):
            continue
        if (r.get("Damage") or "n/a").strip() not in ("n/a", "", "-"):
            continue                      # Damage列がある技は既存テーブルで足りる
        kind, unit, to_act, is_cnt = _parse_damage(r.get("Effect Explanation") or "")
        if unit is None or kind == "unknown":
            continue
        key = (int(cid), mv)
        if kind is None:
            fix[key] = (unit, int(to_act), int(is_cnt))
        else:
            var[key] = (kind, unit, int(to_act), int(is_cnt))
    return fix, var


def emit(fix, var) -> str:
    out = ['"""EN_Card_Data.csv の効果文から復元した打点（自動生成: tools/gen_atk_dmg.py）。"""',
           "ATK_FIX = {"]
    for (c, m), d in sorted(fix.items()):
        out.append(f"    ({c}, {m!r}): {d!r},")
    out += ["}", "ATK_VAR = {"]
    for (c, m), t in sorted(var.items()):
        out.append(f"    ({c}, {m!r}): {t!r},")
    out += ["}", "CHECKUP_DMG = {"]
    for c, d in sorted(build_checkup().items()):
        out.append(f"    {c}: {d},")
    out += ["}", "PROTECT = {"]
    for c, t in sorted(build_protect().items()):
        out.append(f"    {c}: {t!r},")
    out += ["}", "ATK_COST = {"]
    for (c, m), (t, n) in sorted(build_costs().items()):
        out.append(f"    ({c}, {m!r}): ({t!r}, {n}),")
    out += ["}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    f, v = build()
    print(f"固定打点を復元: {len(f)} 技 / 状態依存: {len(v)} 技")
    print("\n--- 状態依存（全件） ---")
    for (c, m), (k, u, ta) in sorted(v.items()):
        print(f"  {c:>5} {m:<24} {k:<12} {u}/個 {'active可' if ta else 'ベンチ限定'}")
    print("\n--- 固定打点（先頭25件） ---")
    nb = [x for x in f.items() if not x[1][1]]
    print(f"  （うちベンチ限定 {len(nb)} 技: " +
          ", ".join(m for (_c, m), _ in nb) + "）")
    for (c, m), (d, ta) in sorted(f.items())[:20]:
        print(f"  {c:>5} {m:<24} {d} {'active可' if ta else 'ベンチ限定'}")
