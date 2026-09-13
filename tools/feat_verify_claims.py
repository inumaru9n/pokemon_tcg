"""特徴が主張していることを、エンジンの真値（選択肢リスト・デッキ構成）と突き合わせる。

被覆監査（feat_audit.py）は「使っていないデータ」を見つけるが、
「**使っているが計算が間違っている**」特徴は見つけられない。この道具はそちらを見る。

  can_retreat  「逃げられる」と主張 vs RETREAT選択肢が実在するか
  can_attack   「技が撃てる」と主張 vs ATTACK選択肢が実在するか
  left_{cid}   「山+サイドに何枚残っている」 vs 自デッキ構成からの真値
  非有限値      NaN/inf を含む特徴が無いか

打点そのものは tools/feat_verify_damage.py（エンジンに実際に攻撃させる）で見る。

usage:
  uv run python tools/feat_verify_claims.py
"""
import sys, random, collections, math
REPO="/Users/akira/kaggle/pokemon_tcg"; AG=REPO+"/"+(sys.argv[1] if len(sys.argv)>1 else "agents/103_alakazam_feat3")
sys.path[:0]=[REPO, REPO+"/arena", AG]
import importlib.util as ilu, numpy as np, run_match as rm, cg.game as cgg
from cg.api import to_observation_class, OptionType, all_card_data
tbl={c.cardId:c for c in all_card_data()}
import os as _os
_fn=next(f for f in sorted(_os.listdir(AG)) if f.startswith("feat_") and f.endswith(".py"))
sp=ilu.spec_from_file_location(_fn[:-3], AG+"/"+_fn); ft=ilu.module_from_spec(sp)
sys.modules[_fn[:-3]]=ft; sp.loader.exec_module(ft)
R=collections.Counter(); bad_left=collections.Counter(); nonfinite=collections.Counter()
for opp,seed in [("101_marnie_luca",4242),("098_spidops_gen",777),("085_kangaskhan_e57e",99),("041_garchomp_replica",5150)]:
    a=rm.load_agent_module(AG,"A"); b=rm.load_agent_module(REPO+f"/agents/{opp}","B")
    da,db=rm.read_deck(AG),rm.read_deck(REPO+f"/agents/{opp}")
    for g in range(6):
        random.seed(g); obs,_=cgg.battle_start_seeded(da,db,seed+g)
        while obs is not None:
            if obs["current"]["result"]!=-1: break
            me=obs["current"]["yourIndex"]
            if me==0:
                o=to_observation_class(obs); st=o.current; sel=o.select
                v,K=ft.feat_vector(st,me)
                # (E) 非有限値
                arr=np.asarray(v,np.float64)
                for i in np.where(~np.isfinite(arr))[0]: nonfinite[K[i]]+=1
                if sel and int(sel.context)==0:
                    kv=dict(zip(K,v))
                    types={x.type for x in sel.option}
                    # (1) can_retreat vs RETREAT選択肢の実在
                    R[("can_retreat", bool(kv["me_can_retreat"]), OptionType.RETREAT in types)]+=1
                    # (2) 技が撃てる(my_best_dmg>0 or my_var_atk) vs ATTACK選択肢の実在
                    # 「打点が出るか」ではなく「技が撃てるか」で照合する
                    R[("can_attack", bool(kv["my_can_attack"]), OptionType.ATTACK in types)]+=1
                    # (3) enables_evo(選択肢側) vs EVOLVE選択肢の実在
                    hc,pl=ft.pool_counts(st,me)
                    ev_claim=False
                    for i2,opt in enumerate(sel.option):
                        cid,_=ft.resolve_option(o,opt,me)
                        vec=ft.option_vector(o,opt,me,hc,pl)[1]
                        if opt.type==OptionType.PLAY and vec[25]>0: ev_claim=True
                    # (4) left_{cid} の真値照合（自デッキは既知）
                    for cid,n in ft.DECK_COUNTS.items():
                        seen=sum(1 for c in (st.players[0].hand or []) if c.id==cid)
                        seen+=sum(1 for c in st.players[0].discard if c.id==cid)
                        fld=([st.players[0].active[0]] if st.players[0].active and st.players[0].active[0] else [])+[x for x in st.players[0].bench if x]
                        for p in fld:
                            if p.id==cid: seen+=1
                            for grp in ("energyCards","tools","preEvolution"):
                                seen+=sum(1 for c in (getattr(p,grp,None) or []) if c.id==cid)
                        seen+=sum(1 for c in st.stadium if c.id==cid and c.playerIndex==me)  # ★真値はスタジアムも数える
                        true_left=max(0,n-seen)
                        if abs(kv.get(f"left_{cid}",0)-true_left)>1e-9:
                            bad_left[tbl[cid].name]+=1
            obs=cgg.battle_select((a if me==0 else b)(obs))
        cgg.battle_finish()
print("■ 特徴の主張 vs エンジンの真値（MAIN決定）")
for nm in ("can_retreat","can_attack"):
    tp=R[(nm,True,True)]; fp=R[(nm,True,False)]; fn=R[(nm,False,True)]; tn=R[(nm,False,False)]
    tot=tp+fp+fn+tn
    print(f"  {nm:<14}一致 {(tp+tn)/max(1,tot):6.1%}  （主張True/実際False={fp}件, 主張False/実際True={fn}件, n={tot}）")
print(f"\n■ left_{{cid}}（山+サイドの残り）の真値との不一致")
print(f"  {sum(bad_left.values())}件" + (f"  内訳: {bad_left.most_common(5)}" if bad_left else "  → 一致"))
print(f"\n■ 非有限値(NaN/inf)を含む特徴")
print(f"  {sum(nonfinite.values())}件" + (f"  {nonfinite.most_common(5)}" if nonfinite else "  → なし"))
