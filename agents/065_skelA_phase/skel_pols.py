"""骨格A/B/Cの方策オブジェクト（精読spec S1-S9 反映版）。
skel_common.decide(obs, pol) から呼ばれる。基礎価値を共有し A=phase / B=goal /
C=economy で修飾。精読条件(majkel_closeread.md)を各構造に焼き込む。
"""

from __future__ import annotations

import skel_common as K


def _sig(f):
    return K.spec_signals(f)


# 精読S3/S2/S9をMAINクラス優先度へ反映する共通ブースト
def _spec_main_boost(cls, f):
    C = K._C
    s = _sig(f)
    b = 0
    # S3: 2体目ready Alakazam未確保なら、Zam進化/装填を強く後押し（KO後即交代の保険）
    if not s["backup_ready"] and s["ready_zams"] >= 1:
        if cls in ("evo_zam", "attach_basic_zam", "attach_tel_zam", "candy"):
            b += 18
    # S2: アタッカー未確保時はライン構築最優先（対アグロの速攻負け対策）
    if not f["ready_zam"]:
        if cls in ("evo_kad", "evo_zam", "candy", "put_abra"):
            b += 15
        if cls == "atk":
            b -= 10
    # S5: Alakazam完成後は攻撃を後押し（turn5+）
    if f["ready_zam"] and f["turn"] >= 5 and cls == "atk":
        b += 8
    # S9: 山切れ危険時はドロー能力を抑制（Majkel未管理の弱点を補正=模倣超え狙い）
    if s["deckout_risk"] and cls in ("ab_dud", "evo_dud", "put_dun"):
        b -= 22
    # S6: 相手手札<8ではXerosic抑制、>=8で後押し
    if cls == "xero":
        b += 12 if f["opp_hand"] >= 8 else -25
    # S7: turn5-9のHammer後押し
    if cls == "hammer" and 5 <= f["turn"] <= 9:
        b += 10
    # S8: サイド射程(自プライズ<=3)でBoss後押し、遠いと抑制
    if cls == "boss":
        b += 12 if f["prz_me"] <= 3 else -18
    return b


# ==================== A: フェーズ状態機械 ====================
_A_EARLY = {"put_dun": 92, "put_abra": 91, "evo_kad": 90, "candy": 88, "evo_zam": 86,
            "poffin": 80, "pad": 78, "attach": 74, "evo_dud": 68, "ab_dud": 66,
            "dawn": 60, "hilda": 58, "hammer": 50, "put_fez": 40, "put_shay": 30,
            "atk": 40, "xero": 40, "boss": 30, "retreat": 25, "end": 5}
_A_MID = {"ab_dud": 90, "evo_zam": 88, "evo_dud": 86, "candy": 84, "evo_kad": 82,
          "attach": 80, "hammer": 76, "atk": 74, "put_dun": 72, "xero": 68,
          "poffin": 60, "pad": 58, "boss": 62, "dawn": 50, "hilda": 48,
          "put_abra": 55, "retreat": 30, "end": 5}
_A_LATE = {"atk": 90, "ab_dud": 86, "evo_dud": 84, "put_dun": 82, "attach": 80,
           "boss": 78, "evo_zam": 74, "xero": 58, "hammer": 56, "candy": 60,
           "poffin": 36, "pad": 34, "put_abra": 40, "retreat": 30, "end": 8}


class SkelA:
    """フェーズ(early/mid/late)が全選択を支配（S1-S9反映）。"""

    @staticmethod
    def _phase(f):
        t = f["turn"]
        return "early" if t <= 4 else "mid" if t <= 9 else "late"

    def main_priority(self, cls, f):
        tbl = {"early": _A_EARLY, "mid": _A_MID, "late": _A_LATE}[self._phase(f)]
        return tbl.get(cls, 20) + _spec_main_boost(cls, f)

    def discard_keep(self, cid, f):
        v = K.keep_value(cid, f)
        C = K._C
        ph = self._phase(f)
        if ph == "late" and cid in (C["Hilda"], C["Dawn"], C["Xerosic"], C["Boss_Orders"]):
            v += 12
        if ph == "early" and cid in (C["Xerosic"], C["Boss_Orders"]):
            v -= 12
        return v

    def search_pref(self, cid, f):
        v = K.search_value(cid, f)
        C = K._C
        ph = self._phase(f)
        if ph == "early" and cid in (C["Abra"], C["Kadabra"], C["Dunsparce"], C["Rare_Candy"]):
            v += 12
        if ph == "late" and cid in (C["Boss_Orders"], C["Hilda"], C["Dawn"]):
            v += 12
        # S3: 2体目Alakazam確保
        if not _sig(f)["backup_ready"] and cid == C["Alakazam"]:
            v += 15
        return v

    def switch_pref(self, p, f):
        return K.attack_value(p, f)

    def activate_yes(self, f, aid):
        return not _sig(f)["deckout_risk"]  # S9: 山切れ危険なら能力抑制


# ==================== B: 目標サブルーチン ====================
def _goal(f):
    s = _sig(f)
    if f["prz_me"] <= 2 and f["ready_zam"]:
        return "close"
    if not f["ready_zam"] or not s["backup_ready"]:
        return "build"          # S2/S3: アタッカーと予備の確保を最優先
    if f["opp_hand"] >= 8:
        return "disrupt"
    if not s["deckout_risk"] and (f["hand"] <= 6 or f["deck"] > 20):
        return "engine"
    return "attack"


_B = {
    "build": {"put_abra": 92, "evo_kad": 92, "candy": 90, "evo_zam": 90,
              "attach": 86, "put_dun": 74, "poffin": 72, "pad": 70, "ab_dud": 58},
    "engine": {"ab_dud": 95, "evo_dud": 92, "put_dun": 90, "candy": 55, "poffin": 52,
               "pad": 48, "evo_kad": 45, "attach": 40},
    "disrupt": {"xero": 95, "hammer": 90, "boss": 65, "ab_dud": 50, "attach": 45,
                "atk": 55},
    "attack": {"atk": 95, "boss": 70, "attach": 65, "ab_dud": 55, "evo_dud": 50,
               "hammer": 60},
    "close": {"boss": 95, "atk": 92, "attach": 85, "ab_dud": 40},
}


class SkelB:
    """目標(build/engine/disrupt/attack/close)が全選択を支配（S1-S9反映）。"""

    def main_priority(self, cls, f):
        tbl = _B[_goal(f)]
        base = {"atk": 40, "end": 5, "boss": 30, "retreat": 20}
        return tbl.get(cls, base.get(cls, 15)) + _spec_main_boost(cls, f) * 0.5

    def discard_keep(self, cid, f):
        v = K.keep_value(cid, f)
        g = _goal(f)
        C = K._C
        if g == "disrupt" and cid in (C["Xerosic"], C["Enhanced_Hammer"]):
            v += 20
        if g == "close" and cid == C["Boss_Orders"]:
            v += 20
        if g == "build" and cid in (C["Abra"], C["Kadabra"], C["Alakazam"], C["Rare_Candy"]):
            v += 14
        if g == "engine" and cid in (C["Dunsparce"], C["Dudunsparce"]):
            v += 12
        return v

    def search_pref(self, cid, f):
        v = K.search_value(cid, f)
        g = _goal(f)
        C = K._C
        if g == "build" and cid in (C["Kadabra"], C["Alakazam"], C["Abra"], C["Rare_Candy"]):
            v += 16
        if g == "engine" and cid in (C["Dunsparce"], C["Dudunsparce"]):
            v += 15
        if g == "disrupt" and cid in (C["Xerosic"], C["Enhanced_Hammer"]):
            v += 20
        if g in ("close", "attack") and cid == C["Boss_Orders"]:
            v += 20
        return v

    def switch_pref(self, p, f):
        v = K.attack_value(p, f)
        if _goal(f) == "engine" and p.id == K._C["Dunsparce"]:
            v += 15
        return v

    def activate_yes(self, f, aid):
        return _goal(f) in ("engine", "build") and not _sig(f)["deckout_risk"]


# ==================== C: 経済スコアラー ====================
_C_MAIN = {"ab_dud": 74, "evo_dud": 72, "put_dun": 70, "evo_zam": 76, "evo_kad": 74,
           "candy": 72, "attach": 70, "hammer": 60, "atk": 68, "boss": 58,
           "xero": 55, "poffin": 54, "pad": 52, "dawn": 50, "hilda": 48,
           "put_abra": 60, "lana": 40, "mine": 42, "stretcher": 44, "ash": 40,
           "put_fez": 30, "put_shay": 26, "retreat": 25, "end": 10}


class SkelC:
    """全選択を1つの経済スコアで採点（精読spec S1-S9を経済ボーナスとして統合）。"""

    def main_priority(self, cls, f):
        return _C_MAIN.get(cls, 20) + _spec_main_boost(cls, f)

    def discard_keep(self, cid, f):
        return K.keep_value(cid, f)

    def search_pref(self, cid, f):
        v = K.search_value(cid, f)
        if not _sig(f)["backup_ready"] and cid == K._C["Alakazam"]:
            v += 15
        return v

    def switch_pref(self, p, f):
        return K.attack_value(p, f)

    def activate_yes(self, f, aid):
        return not _sig(f)["deckout_risk"]


POLS = {"A": SkelA(), "B": SkelB(), "C": SkelC()}
