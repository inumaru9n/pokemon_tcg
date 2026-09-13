"""直近メタから `tools/meta_cards.py` を再生成する。

従来は手作りのデータファイルで、**07-22メタのまま固定**されていた
（Mega Lopunny ex / Teal Mask Ogerpon ex が入っていない）。メタが動くたびに再生成する。

生成するもの:
  META_CARDS  相手識別に使う看板カードの和集合
  META_ARCH   アーキタイプ → 看板カードid（そのデッキに入り、他が1デッキ以下のもの）
  **ARCH_DECK   アーキタイプ → 代表構築の60枚（cid→枚数）**  ← 新規
  **OPP_TRACK   相手の残枚数/手札確率を追跡するカード（メタ4デッキ以上で共通）** ← 新規
  **ARCH_MAIN   アーキタイプ → 主戦力のcid（最大HPのポケモン）** ← 新規

ARCH_DECK があると「相手のアーキタイプが分かれば60枚は既知」という事実を特徴にできる。
従来は opp_seen_*（観測した枚数）しか無く、**残り枚数も手札にある確率も計算していなかった**。

usage: uv run python tools/gen_meta_cards.py --since 2026-07-30 --min-share 0.02
"""
from __future__ import annotations
import argparse, collections, csv, glob, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-07-30")
    ap.add_argument("--min-share", type=float, default=0.02)
    ap.add_argument("--track-min-decks", type=int, default=4,
                    help="この数以上のメタデッキに入るカードを追跡対象にする")
    ap.add_argument("--out", default=os.path.join(REPO, "tools", "meta_cards.py"))
    a = ap.parse_args()

    cards = {r["Card ID"]: r for r in csv.DictReader(
        open(os.path.join(REPO, "pokemon-tcg-ai-battle/data/EN_Card_Data.csv"),
             encoding="utf-8-sig"))}
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "data/episodes/*/summary.csv"))):
        d = f.split("/")[-2]
        if d >= a.since:
            for r in csv.DictReader(open(f)):
                rows.append(r)
    assert rows, f"{a.since} 以降のエピソードが無い"
    share = collections.Counter(r["archetype"] for r in rows)
    tot = sum(share.values())
    archs = [x for x, n in share.most_common() if n / tot >= a.min_share]

    deck, main_of = {}, {}
    for x in archs:
        sig = collections.Counter(r["deck_signature"] for r in rows
                                  if r["archetype"] == x).most_common(1)[0][0]
        ids = next(r["deck_ids"].split() for r in rows if r["deck_signature"] == sig)
        deck[x] = dict(collections.Counter(int(c) for c in ids))
        poke = [(c, int(cards[str(c)]["HP"])) for c in deck[x]
                if str(c) in cards and cards[str(c)]["HP"] not in ("n/a", "")]
        main_of[x] = max(poke, key=lambda t: t[1])[0] if poke else 0

    n_decks = collections.Counter()
    for x in archs:
        for c in deck[x]:
            n_decks[c] += 1
    # 看板 = そのデッキに入り、他が1デッキ以下（＝識別力がある）
    arch_sig = {x: sorted(c for c in deck[x] if n_decks[c] <= 2
                          and not cards.get(str(c), {}).get("Card Name", "").startswith("Basic "))
                for x in archs}
    meta_cards = sorted({c for v in arch_sig.values() for c in v})
    track = sorted(c for c, k in n_decks.items()
                   if k >= a.track_min_decks
                   and not cards.get(str(c), {}).get("Card Name", "").startswith("Basic "))
    # 各アーキタイプの主戦力も追跡対象に加える（残り枚数が脅威の量に直結する）
    track = sorted(set(track) | set(main_of.values()) - {0})

    def nm(c):
        return cards.get(str(c), {}).get("Card Name", "?")
    with open(a.out, "w") as f:
        f.write(f"'''直近メタの看板カードと構築（自動生成: tools/gen_meta_cards.py）。\n\n"
                f"期間 {a.since}〜 / {tot}デッキ観測 / シェア{a.min_share*100:.0f}%以上の{len(archs)}系統。\n"
                f"**メタが動いたら必ず再生成すること**（旧版は07-22メタのままで\n"
                f"Mega Lopunny ex と Teal Mask Ogerpon ex が欠落していた）。\n'''\n\n")
        f.write(f"META_CARDS = {meta_cards!r}\n\n")
        f.write("META_ARCH = {\n")
        for x in archs:
            f.write(f"    {x!r}: {arch_sig[x]!r},   # share {share[x]/tot*100:.1f}%\n")
        f.write("}\n\n")
        f.write("# 代表構築の60枚。**相手のアーキタイプが分かれば残り枚数が計算できる**\n")
        f.write("ARCH_DECK = {\n")
        for x in archs:
            f.write(f"    {x!r}: {dict(sorted(deck[x].items()))!r},\n")
        f.write("}\n\n")
        f.write(f"# 追跡カード（{a.track_min_decks}デッキ以上で共通 + 各系統の主戦力）\n")
        f.write("OPP_TRACK = [\n")
        for c in track:
            f.write(f"    {c},   # {nm(c)}\n")
        f.write("]\n\n")
        f.write("ARCH_MAIN = {\n")
        for x in archs:
            f.write(f"    {x!r}: {main_of[x]},   # {nm(main_of[x])}\n")
        f.write("}\n")
    print(f"生成 -> {a.out}")
    print(f"  系統 {len(archs)}: " + ", ".join(f"{x}({share[x]/tot*100:.1f}%)" for x in archs))
    print(f"  META_CARDS {len(meta_cards)}枚 / OPP_TRACK {len(track)}枚")
    print("  OPP_TRACK: " + ", ".join(nm(c) for c in track))


if __name__ == "__main__":
    main()
