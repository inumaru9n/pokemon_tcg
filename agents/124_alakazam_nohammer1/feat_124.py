"""自動生成された特徴器（tools/gen_featurizer.py）。手で編集しないこと。

デッキ: agents/104_alakazam_feat4/deck.csv
ポケモン 7種 / 進化ライン 4本 / エネ 3種 / スタジアム 1種
行動クラス 65個

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

card_table = {c.cardId: c for c in all_card_data()}

# ---- 効果文から復元した打点（tools/gen_atk_dmg.py 生成。デッキ非依存） ----
# attack_table_056 の Damage 列は「n/a」の技を全て打点不明にしており、その中に
# Alakazam の Powerful Hand（手札1枚につき20）のような主砲が含まれる。
ATK_FIX = {
    (40, 'Mirage Barrage'): (120, 1, 0),
    (45, 'Pinpoint Dive'): (60, 0, 0),
    (80, 'Twin Shotels'): (50, 1, 0),
    (94, 'Cursed Drop'): (40, 1, 1),
    (140, 'Cruel Arrow'): (100, 1, 0),
    (144, 'Trifrost'): (110, 1, 0),
    (153, 'Garnet Volley'): (180, 1, 0),
    (189, 'Sonic Peridot'): (100, 1, 0),
    (215, 'Painful Memories'): (20, 1, 1),
    (219, 'Law of the Underworld'): (60, 1, 1),
    (241, 'Severe Squall'): (60, 1, 0),
    (247, 'Sneaky Placement'): (20, 1, 1),
    (266, 'Thump-Thump Boom'): (100, 1, 0),
    (302, 'Wide Blast'): (50, 0, 0),
    (372, 'Dual Bolt'): (50, 1, 0),
    (377, 'Thunder Raid'): (210, 0, 0),
    (438, 'Drag Off'): (30, 1, 0),
    (449, 'Spinning Tail'): (30, 1, 0),
    (508, 'Drag Off'): (20, 1, 0),
    (591, 'Telekinesis'): (70, 1, 0),
    (638, 'Sonic Double'): (50, 1, 0),
    (665, 'Jumping Kick'): (40, 1, 0),
    (730, 'Chilling Wings'): (20, 1, 0),
    (759, 'Dashing Kick'): (50, 0, 0),
    (774, 'Sniping Feathers'): (120, 1, 0),
    (805, 'Targeted Dive'): (70, 0, 0),
    (817, 'Sneaky Placement'): (10, 1, 1),
    (844, 'Dual Tail'): (60, 1, 0),
    (868, 'Split Bomb'): (60, 1, 0),
    (880, 'Phantasmal Barrage'): (120, 1, 1),
    (889, 'Tar Cannon'): (140, 1, 0),
    (928, 'Explosion Y'): (280, 1, 0),
    (946, 'Spit Shot'): (120, 1, 0),
    (984, 'Bone Shot'): (50, 1, 0),
    (1021, 'Feather Shot'): (90, 1, 0),
    (1058, 'Haunt'): (30, 1, 1),
    (1064, 'Sonic Ripper'): (220, 1, 0),
}
ATK_VAR = {
    (171, 'Thunderburst Storm'): ('energy', 30, 1, 0),
    (620, 'Power Whip'): ('energy', 20, 1, 0),
    (743, 'Powerful Hand'): ('hand', 20, 1, 1),
}
CHECKUP_DMG = {
    104: 10,
    442: 20,
}
PROTECT = {
    28: ('both', 'self', ''),
    74: ('both', 'bench', ''),
    83: ('damage', 'self', 'atk_basic_ex'),
    117: ('damage', 'self', 'atk_ability'),
    158: ('damage', 'self', ''),
    203: ('effects', 'self', ''),
    207: ('both', 'self', ''),
    330: ('damage', 'self', 'atk_ex'),
    343: ('damage', 'bench', 'def_no_rulebox'),
    345: ('damage', 'self', 'atk_ex'),
    362: ('both', 'self', ''),
    414: ('effects', 'team_basic', ''),
    504: ('both', 'self', ''),
    835: ('effects', 'self', ''),
    1136: ('effects', 'self', ''),
    1138: ('damage', 'self', ''),
}
ATK_COST = {
    (21, 'High Jump Kick'): ((7,), 2),
    (21, 'Nab ’n’ Dash'): ((), 1),
    (22, 'Push Down'): ((6,), 0),
    (23, 'Ram'): ((6, 6), 0),
    (23, 'Super Sandstorm'): ((6, 6), 1),
    (24, 'Comet Punch'): ((), 2),
    (24, 'Wicked Impact'): ((), 3),
    (25, 'Slow Crunch'): ((1,), 1),
    (25, 'Superpowered Horns'): ((1,), 2),
    (26, 'Leaflet Blessings'): ((), 1),
    (26, 'Solar Beam'): ((1,), 1),
    (27, 'Avenging Edge'): ((1,), 2),
    (27, 'Recovery Net'): ((1,), 0),
    (28, 'Hook'): ((1,), 0),
    (29, 'Matcha Splash'): ((1,), 1),
    (29, 'Re-Brew'): ((), 1),
    (30, 'Ground Burn'): ((2, 2), 1),
    (30, 'Hot Magma'): ((2,), 1),
    (31, 'Allure'): ((), 1),
    (31, 'Ground Melter'): ((2,), 1),
    (32, 'Icicle Missile'): ((3,), 1),
    (32, 'Permeating Chill'): ((3,), 0),
    (33, 'Flock'): ((3,), 0),
    (33, 'Flop'): ((3,), 0),
    (34, 'Numbing Water'): ((3,), 0),
    (35, 'Aurora Gain'): ((3,), 0),
    (35, 'Undulating Slice'): ((3, 3), 1),
    (36, 'Pick and Stick'): ((4,), 0),
    (37, 'Volt Cyclone'): ((4,), 2),
    (38, 'Dual Headbutt'): ((), 1),
    (39, 'Heart Sign'): ((), 1),
    (39, 'Love Resonance'): ((5,), 2),
    (40, 'Mirage Barrage'): ((3,), 2),
    (40, 'Shinobi Blade'): ((3,), 0),
    (41, 'Ground Crasher'): ((6,), 0),
    (41, 'Hammer In'): ((6, 6), 1),
    (42, 'Find a Friend'): ((), 1),
    (42, 'Rolling Tackle'): ((1, 2), 0),
    (43, 'Ascension'): ((), 1),
    (43, 'Quick Attack'): ((), 3),
    (44, 'Blood Moon'): ((), 5),
    (45, 'Pinpoint Dive'): ((1,), 0),
    (45, 'Rear Kick'): ((), 2),
    (46, 'Blaze Blitz'): ((2, 2), 1),
    (46, 'Heat Blast'): ((2,), 1),
    (47, 'Big Bite'): ((3,), 0),
    (48, 'Reverse Thrust'): ((3,), 0),
    (49, 'Giant Wave'): ((3, 3), 0),
    (50, 'Aqua Bomb'): ((3, 3), 0),
    (50, 'Lucky Find'): ((3,), 0),
    (51, 'Double Hit'): ((3,), 2),
    (51, 'Vanguard Punch'): ((3,), 0),
    (52, 'Numbing Hold'): ((3, 3), 0),
    (52, 'Tricolor Pump'): ((3,), 0),
    (53, 'Ball Roll'): ((), 1),
    (53, 'Magical Shot'): ((5,), 2),
    (54, 'Mirror Attack'): ((5,), 0),
    (55, 'Evolution Jammer'): ((5,), 0),
    (55, 'Super Psy Bolt'): ((5,), 2),
    (56, 'Hex Hurl'): ((), 3),
    (57, 'Razor Fin'): ((6,), 1),
    (58, 'Giant Tusk'): ((6, 6), 2),
    (58, 'Land Collapse'): ((), 2),
    (59, 'Mysterious Beam'): ((7,), 0),
    (59, 'Suffocating Gas'): ((7, 7), 0),
    (60, 'Super Poison Breath'): ((7, 7), 0),
    (61, 'Speed Wing'): ((7,), 3),
    (61, 'Vengeance Fletching'): ((7, 7), 0),
    (62, 'Primordial Beatdown'): ((6,), 1),
    (62, 'Shred'): ((2, 6), 1),
    (63, 'Bellowing Thunder'): ((4, 6), 0),
    (63, 'Burst Roar'): ((), 1),
    (64, 'Silent Wing'): ((), 2),
    (65, 'Dig'): ((), 2),
    (65, 'Gnaw'): ((), 1),
    (66, 'Land Crush'): ((), 3),
    (67, 'Ram'): ((), 3),
    (68, 'Hang Down'): ((), 2),
    (68, 'Rigidify'): ((1,), 0),
    (69, 'Comet Slap'): ((), 2),
    (69, 'Corkscrew Punch'): ((1,), 0),
    (70, 'Energy Loop'): ((), 2),
    (70, 'Expelling Tornado'): ((1,), 0),
    (71, 'Flop'): ((), 1),
    (71, 'Leaf Litter Tackle'): ((1,), 2),
    (72, 'Superpowered Horns'): ((1,), 2),
    (73, 'Slight Intrusion'): ((), 1),
    (74, 'Psychic'): ((1,), 0),
    (75, 'Prism Edge'): ((1, 1), 1),
    (76, 'Roasting Heat'): ((2,), 0),
    (77, 'Fake Out'): ((2,), 0),
    (78, 'Bite'): ((2,), 0),
    (78, 'Flare Strike'): ((2,), 2),
    (79, 'Blaze Blast'): ((2,), 4),
    (80, 'Twin Shotels'): ((5,), 2),
    (81, 'Sand Spray'): ((6,), 1),
    (82, 'Mud Shot'): ((), 1),
    (82, 'Wild Tackle'): ((6, 6), 1),
    (83, 'Dirty Beam'): ((5,), 2),
    (84, 'Cross Breaker'): ((8, 8), 0),
    (84, 'Steel Wing'): ((), 2),
    (85, 'Dig Claws'): ((8,), 0),
    (85, 'Iron Tackle'): ((8,), 2),
    (86, 'Beam'): ((8,), 2),
    (87, 'Peak Acceleration'): ((), 1),
    (87, 'Sparking Strike'): ((4, 4, 5), 0),
    (88, 'Coordinated Strike'): ((), 2),
    (88, 'Quick Sign'): ((), 1),
    (89, 'Branch Poke'): ((1, 1), 0),
    (89, 'Smash Kick'): ((1,), 0),
    (90, 'Beat'): ((1, 1), 0),
    (91, 'Drum Beating'): ((1,), 0),
    (91, 'Wood Hammer'): ((1, 1), 0),
    (92, 'Tumbling Attack'): ((1,), 0),
    (93, 'Do the Wave'): ((1,), 0),
    (94, 'Cursed Drop'): ((1,), 0),
    (94, 'Spill the Tea'): ((1,), 0),
    (95, 'Mountain Stroll'): ((), 1),
    (95, 'Ogre Comeback'): ((1,), 1),
    (96, 'Myriad Leaf Shower'): ((1, 1, 1), 0),
    (97, 'Call for Family'): ((2,), 0),
    (97, 'Live Coal'): ((2,), 1),
    (98, 'Mind Ruler'): ((2,), 0),
    (99, 'Dynamic Blaze'): ((2, 2, 2), 0),
    (99, 'Wrathful Hearth'): ((2,), 2),
    (100, 'Whirlpool'): ((), 2),
    (101, 'Icy Snow'): ((3,), 0),
    (101, 'Inviting Kiss'): ((), 1),
    (102, 'Hydro Splash'): ((3,), 2),
    (103, 'Astonish'): ((3,), 1),
    (104, 'Frost Smash'): ((3,), 1),
    (105, 'Aqua Slash'): ((3,), 0),
    (106, 'Wave Splash'): ((3,), 1),
    (107, 'Giga Impact'): ((3,), 0),
    (108, 'Sob'): ((), 1),
    (108, 'Torrential Pump'): ((3,), 2),
    (109, 'Beam'): ((5,), 0),
    (110, 'Slurp Slurp'): ((5,), 1),
    (111, 'Sand Attack'): ((5,), 0),
    (111, 'Spooky Shot'): ((5,), 2),
    (112, 'Mind Bend'): ((5,), 1),
    (113, 'Best Punch'): ((6,), 0),
    (114, 'Knuckle Punch'): ((6,), 0),
    (114, 'Superpower'): ((6,), 2),
    (115, 'Gutsy Swing'): ((6,), 3),
    (115, 'Tantrum'): ((6,), 0),
    (116, 'Good Punch'): ((6, 6), 0),
    (117, 'Demolish'): ((6,), 2),
    (118, 'Steel Burst'): ((8,), 2),
    (119, 'Bite'): ((2, 5), 0),
    (119, 'Petty Grudge'): ((5,), 0),
    (120, 'Dragon Headbutt'): ((2, 5), 0),
    (121, 'Jet Headbutt'): ((), 1),
    (121, 'Phantom Dive'): ((2, 5), 0),
    (122, 'Surf'): ((2, 3), 0),
    (123, 'Mach Cut'): ((), 1),
    (124, 'Boundless Power'): ((), 3),
    (124, 'Lucky Attachment'): ((), 1),
    (125, 'Return'): ((), 3),
    (126, 'Shocking Web'): ((1,), 1),
    (127, 'Add On'): ((), 1),
    (127, 'Leafage'): ((1,), 0),
    (128, 'Cutting Wind'): ((1,), 0),
    (128, 'United Wings'): ((), 1),
    (129, 'Power Shot'): ((1,), 0),
    (129, 'Stock Up on Feathers'): ((), 1),
    (130, 'Accelerator Flash'): ((8,), 0),
    (130, 'Shattering Speed'): ((8, 8, 8), 0),
    (131, 'Come and Get You'): ((5,), 0),
    (131, 'Mumble'): ((5, 5), 0),
    (132, 'Will-O-Wisp'): ((5, 5), 0),
    (133, 'Shadow Bind'): ((5, 5), 1),
    (134, 'Disarming Voice'): ((5,), 2),
    (134, 'Mystical Return'): ((5,), 0),
    (135, 'Mad Bite'): ((6, 6), 1),
    (136, 'Double Scratch'): ((7,), 1),
    (136, 'Stampede'): ((), 1),
    (137, 'Claw Slash'): ((7,), 2),
    (137, 'Illusory Hijacking'): ((), 2),
    (138, 'Chain-Crazed'): ((7, 7), 1),
    (138, 'Poisonous Musculature'): ((), 1),
    (139, 'Dirty Headbutt'): ((7, 7), 1),
    (140, 'Cruel Arrow'): ((), 3),
    (141, 'Irritated Outburst'): ((7, 7), 0),
    (142, 'Magnetic Blast'): ((8,), 2),
    (143, 'Headbutt Bounce'): ((8, 8), 0),
    (143, 'Rigidify'): ((8,), 0),
    (144, 'Trifrost'): ((3, 3, 8, 8), 1),
    (145, 'Colorful Catch'): ((), 1),
    (145, 'Headbutt'): ((), 2),
    (146, 'Bind Down'): ((1,), 0),
    (147, 'Miasma Wind'): ((1,), 0),
    (148, 'Reaping Dash'): ((), 1),
    (149, 'Spray Fluid'): ((1,), 0),
    (150, 'Syrup Storm'): ((), 2),
    (151, 'Quick Attack'): ((), 1),
    (152, 'Combustion'): ((2,), 2),
    (152, 'Low Sweep'): ((2,), 0),
    (153, 'Flare Strike'): ((2,), 2),
    (153, 'Garnet Volley'): ((2, 6, 7), 0),
    (154, 'Larimar Rain'): ((3, 5, 8), 0),
    (154, 'Power Splash'): ((3,), 0),
    (155, 'Tidal Wave'): ((3, 3), 0),
    (156, 'Haymaker'): ((3,), 4),
    (157, 'Headbutt'): ((), 3),
    (158, 'Hard Crunch'): ((), 3),
    (159, 'Sonic Edge'): ((), 4),
    (160, 'Jolting Charge'): ((), 1),
    (161, 'Charged Web'): ((4,), 1),
    (161, 'Fulgurite'): ((1, 4, 6), 0),
    (162, 'Dangle Tail'): ((), 1),
    (162, 'Tackle'): ((5,), 1),
    (163, 'Seek Inspiration'): ((5,), 1),
    (163, 'Super Psy Bolt'): ((5, 5), 1),
    (164, 'Flower Shower'): ((5,), 0),
    (164, 'Play Rough'): ((5,), 0),
    (165, 'Allure'): ((), 1),
    (165, 'Beam'): ((), 2),
    (166, 'Iron Tackle'): ((), 2),
    (167, 'Hyper Ray'): ((), 2),
    (168, 'Knickknack Carrying'): ((8,), 0),
    (168, 'Ram'): ((8,), 1),
    (169, 'Hammer In'): ((8,), 0),
    (169, 'Raging Hammer'): ((8, 8), 1),
    (170, 'Iron Blaster'): ((8, 8), 1),
    (171, 'Dragon Headbutt'): ((4, 6), 1),
    (171, 'Thunderburst Storm'): ((4, 6), 0),
    (172, 'Triple Stab'): ((), 1),
    (173, 'Speed Wing'): ((), 2),
    (174, 'Assault Landing'): ((), 1),
    (175, 'Boundless Power'): ((), 3),
    (176, 'Crown Opal'): ((1, 3, 4), 0),
    (176, 'Unified Beatdown'): ((), 2),
    (177, 'Precocious Evolution'): ((), 1),
    (178, 'Jungle Whip'): ((1, 1), 1),
    (178, 'Leaf Drain'): ((1,), 0),
    (179, 'Black Frost'): ((3, 3), 2),
    (179, 'Ice Age'): ((), 3),
    (180, 'Big Bite'): ((3,), 1),
    (181, 'Aqua Edge'): ((3,), 0),
    (182, 'Hydro Splash'): ((3,), 1),
    (183, 'Delightful Kiss'): ((), 6),
    (184, 'Eon Blade'): ((5, 5), 1),
    (185, 'Splashing Dodge'): ((), 1),
    (186, 'Minor Errand-Running'): ((), 1),
    (186, 'Tackle'): ((), 3),
    (187, 'Bite'): ((6,), 1),
    (187, 'Call for Family'): ((), 1),
    (188, 'Cutting Wind'): ((6,), 1),
    (188, 'Screech'): ((), 1),
    (189, 'Reversing Storm'): ((6,), 0),
    (189, 'Sonic Peridot'): ((3, 6, 8), 0),
    (190, 'Metal Defender'): ((8, 8, 8), 0),
    (191, 'Strike It Rich'): ((8,), 0),
    (191, 'Surf Back'): ((), 3),
    (192, 'Deleting Slash'): ((8,), 1),
    (192, 'Slicing Blade'): ((8,), 2),
    (193, 'Swinging Sphene'): ((1, 3, 6), 0),
    (193, 'Tropical Frenzy'): ((1, 3), 0),
    (194, 'Cotton Wings'): ((3, 8), 0),
    (194, 'Humming Charge'): ((3,), 0),
    (195, 'Buster Tail'): ((5, 8), 1),
    (195, 'Time Manipulation'): ((), 1),
    (196, 'Fully Singe'): ((2,), 0),
    (196, 'Steaming Stomp'): ((6,), 2),
    (197, 'Disarming Voice'): ((), 1),
    (198, 'Vengeful Crush'): ((1,), 2),
    (199, 'Call for Family'): ((), 1),
    (200, 'Tackle'): ((1,), 0),
    (200, 'Wander About'): ((), 1),
    (201, 'Entangling Whip'): ((1, 1), 1),
    (201, 'Hazardous Greed'): ((1,), 1),
    (202, 'Flare'): ((2,), 1),
    (203, 'Torcherto'): ((2,), 1),
    (204, 'Flamethrower'): ((2, 2), 1),
    (204, 'Light Punch'): ((2,), 0),
    (205, 'Spicy Rage'): ((2, 2), 0),
    (206, 'Leap Out'): ((), 1),
    (207, 'Hypno Splash'): ((3,), 2),
    (208, 'Sprinkle Water'): ((), 2),
    (209, 'Icicle Loop'): ((3, 3), 1),
    (210, 'Topaz Bolt'): ((1, 4, 8), 0),
    (211, 'Electric Ball'): ((4,), 1),
    (212, 'Crushing Pulse'): ((4,), 0),
    (212, 'Energy Short'): ((4,), 0),
    (213, 'Prize Count'): ((4, 4), 1),
    (213, 'Summon Lightning'): ((), 1),
    (214, 'Speed Wing'): ((), 3),
    (215, 'Painful Memories'): ((5,), 0),
    (216, 'Full Heart'): ((), 1),
    (216, 'Guardian Burst'): ((5, 5), 0),
    (217, 'Neurokinesis'): ((5,), 1),
    (218, 'Mumble'): ((5,), 0),
    (218, 'Petty Grudge'): ((5,), 1),
    (219, 'Law of the Underworld'): ((5,), 0),
    (219, 'Spooky Shot'): ((5,), 2),
    (220, 'Psyshot'): ((5,), 1),
    (220, 'See Through'): ((), 1),
    (221, 'Psyshot'): ((5,), 2),
    (222, 'Electromagnetic Sonar'): ((), 1),
    (222, 'Gnaw'): ((5,), 0),
    (223, 'Barite Jail'): ((3, 5, 6), 0),
    (223, 'Sand Tomb'): ((), 3),
    (224, 'Destined Fight'): ((6,), 1),
    (224, 'Tantrum'): ((6,), 0),
    (225, 'Mud Shot'): ((6,), 2),
    (226, 'Hammer In'): ((6, 6), 1),
    (226, 'Unrelenting Onslaught'): ((), 2),
    (227, 'Bite'): ((7,), 1),
    (227, 'Stomp Off'): ((7,), 0),
    (228, 'Darkness Fang'): ((7,), 2),
    (228, 'Stomp Off'): ((7,), 0),
    (229, 'Crashing Headbutt'): ((7,), 1),
    (229, 'Obsidian'): ((5, 7, 8), 1),
    (230, 'Poison Chain'): ((7,), 1),
    (231, 'Cinnabar Lure'): ((2, 3, 7), 0),
    (231, 'Surprise Pump'): ((2, 3), 0),
    (232, 'Great Swing'): ((), 2),
    (233, 'Ready to Ram'): ((), 2),
    (233, 'Smashing Headbutt'): ((), 4),
    (234, 'Hard Tackle'): ((), 3),
    (234, 'Prism Charge'): ((), 1),
    (235, 'Itchy Pollen'): ((), 6),
    (236, 'Moss Agate'): ((1, 2, 3), 0),
    (236, 'Verdant Storm'): ((1,), 1),
    (237, 'Triple Spin'): ((1,), 0),
    (238, 'Seed Bomb'): ((1,), 0),
    (239, 'Burning Charge'): ((2,), 1),
    (239, 'Carnelian'): ((2, 3, 4), 0),
    (240, 'Rapid Draw'): ((), 1),
    (241, 'Aquamarine'): ((2, 3, 4), 0),
    (241, 'Severe Squall'): ((3,), 1),
    (242, 'Flail'): ((3,), 0),
    (243, 'Euclase'): ((1, 3, 7), 0),
    (243, 'Frost Bullet'): ((3,), 1),
    (244, 'Dravite'): ((2, 3, 4), 0),
    (244, 'Flashing Spear'): ((4,), 1),
    (245, 'Psychic'): ((5,), 0),
    (245, 'Strange Hacking'): ((5,), 0),
    (246, 'Amazez'): ((1, 5, 7), 0),
    (246, 'Psych Out'): ((5,), 2),
    (247, 'Sneaky Placement'): ((5,), 0),
    (248, 'Moon Mirage'): ((7,), 2),
    (248, 'Onyx'): ((4, 5, 7), 0),
    (249, 'Coruscating Quartz'): ((2, 3, 4), 0),
    (250, 'Tackle'): ((), 2),
    (251, 'Jewel Breaker'): ((), 4),
    (252, 'Bug Bite'): ((1,), 0),
    (253, 'Harden'): ((1,), 0),
    (254, 'Scale Hurricane'): ((1,), 0),
    (255, 'Corner'): ((), 1),
    (256, 'Searing Flame'): ((2, 2), 1),
    (257, 'Flare'): ((2, 2), 1),
    (257, 'Rolling Tackle'): ((), 2),
    (258, 'Back Draft'): ((), 2),
    (258, 'Flamebody Cannon'): ((2, 2), 1),
    (259, 'Scorching Cyclone'): ((2, 2), 1),
    (260, 'Water Gun'): ((3,), 0),
    (261, 'Aqua Slash'): ((3, 3), 0),
    (262, 'Hydro Splash'): ((3, 3), 1),
    (263, 'Surf'): ((), 3),
    (264, 'Hydro Pump'): ((), 4),
    (265, 'Voltaic Chain'): ((), 2),
    (266, 'Electric Ball'): ((4, 4), 1),
    (266, 'Thump-Thump Boom'): ((4, 4), 0),
    (267, 'Zapping Short'): ((), 2),
    (268, 'Tiny Charge'): ((4,), 1),
    (269, 'Thunderous Bolt'): ((4, 4, 4), 1),
    (270, 'Quick Attack'): ((4,), 0),
    (271, 'Mach Bolt'): ((4,), 2),
    (272, 'Full Moon Rondo'): ((5,), 1),
    (273, 'Cursed Words'): ((5,), 0),
    (273, 'Spooky Shot'): ((5,), 1),
    (274, 'Beam'): ((5, 5), 0),
    (274, 'Spinning Attack'): ((5,), 0),
    (275, 'Psypunch'): ((5,), 0),
    (275, 'Zen Headbutt'): ((5, 5), 0),
    (276, 'Conjoined Beams'): ((5, 5), 0),
    (276, 'Wrack Down'): ((5,), 0),
    (277, 'Psychic Sphere'): ((5,), 0),
    (277, 'Victory Symbol'): ((5,), 2),
    (278, 'Hold Still'): ((5,), 0),
    (279, 'Magical Shot'): ((5,), 0),
    (280, 'Fade Out'): ((5,), 0),
    (280, 'Inviting Flowers'): ((), 1),
    (281, 'Call for Family'): ((), 1),
    (281, 'Lunge Out'): ((6,), 0),
    (282, 'Impaling Tusk'): ((6, 6), 0),
    (282, 'Strength'): ((), 1),
    (283, 'Rumbling March'): ((6, 6), 0),
    (284, 'Crunch'): ((), 2),
    (285, 'Take Down'): ((), 2),
    (286, 'Dig It Up'): ((), 1),
    (286, 'Stampede'): ((), 2),
    (287, 'Clamping Fangs'): ((), 3),
    (288, 'Gnaw'): ((6,), 1),
    (288, 'Turf Maker'): ((), 1),
    (289, 'Break Ground'): ((6,), 2),
    (289, 'Rumble'): ((6,), 1),
    (290, 'Cracking Stomp'): ((7,), 1),
    (291, 'Thieving Swipe'): ((7,), 1),
    (292, 'Scratch'): ((7,), 0),
    (293, 'Night Joker'): ((7, 7), 0),
    (294, 'Double Spin'): ((), 1),
    (295, 'Confront'): ((8,), 1),
    (295, 'Spinning Gears'): ((), 1),
    (296, 'Magnetic Blast'): ((), 1),
    (296, 'Triple Smash'): ((8, 8), 1),
    (297, 'Spike Draw'): ((), 1),
    (298, 'Shoot Through'): ((), 1),
    (298, 'Steel Wing'): ((8, 8), 1),
    (299, 'Brave Slash'): ((8, 8, 8), 1),
    (299, 'Insta-Strike'): ((), 1),
    (300, 'Bite'): ((), 1),
    (300, 'Reckless Charge'): ((2, 3), 0),
    (301, 'Guard Press'): ((), 2),
    (301, 'Heavy Impact'): ((2, 3), 1),
    (302, 'Dragon Impact'): ((2, 3), 2),
    (302, 'Wide Blast'): ((2,), 2),
    (303, 'Powerful Rage'): ((2, 4), 0),
    (303, 'Virtuous Flame'): ((2, 2, 4), 1),
    (304, 'Dynamic Press'): ((), 3),
    (305, 'Ram'): ((), 2),
    (305, 'Trading Places'): ((), 1),
    (306, 'Destructive Drill'): ((), 3),
    (306, 'Tenacious Tail'): ((), 1),
    (307, 'Intimidating Stare'): ((), 1),
    (307, 'Peck'): ((), 2),
    (308, 'Razor Wing'): ((), 3),
    (308, 'Speed Dive'): ((), 1),
    (309, 'Smash Kick'): ((), 3),
    (310, 'Headbutt'): ((), 3),
    (311, 'Fickle Spitting'): ((), 1),
    (312, 'Magnetic Burst'): ((6,), 0),
    (312, 'Power Gem'): ((6,), 1),
    (313, 'Cyber Drive'): ((4, 5), 1),
    (313, 'Repulsion Bolt'): ((4, 5), 0),
    (314, 'Rollout'): ((), 1),
    (315, 'Double-Edge'): ((5, 5, 5, 5), 0),
    (316, 'Angelite'): ((3, 4, 5), 0),
    (316, 'Magical Charm'): ((5,), 2),
    (317, 'Reckless Charge'): ((), 2),
    (318, 'Flap'): ((2,), 1),
    (318, 'Shining Blaze'): ((2, 2), 1),
    (319, 'Will-O-Wisp'): ((2,), 0),
    (320, 'Abyssal Flames'): ((2,), 0),
    (320, 'Raging Amethyst'): ((2, 5, 8), 0),
    (321, 'Leaf Step'): ((1,), 1),
    (322, 'Spinning Attack'): ((1,), 2),
    (323, 'Rear Kick'): ((), 1),
    (324, 'Scratch'): ((2,), 1),
    (325, 'Slash'): ((2,), 1),
    (326, 'Smolder-sault'): ((2,), 1),
    (327, 'Tail Whap'): ((3,), 1),
    (327, 'Water Gun'): ((3,), 0),
    (328, 'Tail Whap'): ((), 1),
    (328, 'Thunder'): ((4, 4), 1),
    (329, 'Linked Lightning'): ((4,), 1),
    (330, 'Magical Shot'): ((5,), 2),
    (331, 'Aurora Beam'): ((5,), 1),
    (331, 'Rising Horns'): ((5, 5), 1),
    (332, 'Anchor Smash'): ((5, 5), 2),
    (332, 'Bind Down'): ((5,), 2),
    (333, 'Quick Attack'): ((6,), 0),
    (334, 'Claw Slash'): ((7,), 0),
    (335, 'Hammer In'): ((8,), 1),
    (336, 'Slashing Strike'): ((8, 8), 1),
    (336, 'Steel Armament'): ((), 1),
    (337, 'Hyper Whirlpool'): ((), 3),
    (338, 'Rallying Horn'): ((), 3),
    (338, 'Vise Grip'): ((1,), 0),
    (339, 'Razor Wing'): ((1,), 1),
    (339, 'Whirlwind'): ((), 1),
    (340, 'Jet Cyclone'): ((1, 1, 1), 1),
    (341, 'Spike Sting'): ((), 1),
    (342, 'Leaf Step'): ((1,), 2),
    (343, 'Smash Kick'): ((), 2),
    (344, 'Ascension'): ((), 1),
    (345, 'Superb Scissors'): ((1,), 2),
    (346, 'Mini Drain'): ((1,), 0),
    (347, 'Energy Loop'): ((1,), 0),
    (348, 'Hydra Breath'): ((1,), 0),
    (348, 'Whip Smash'): ((1,), 2),
    (349, 'Grass Kagura'): ((), 1),
    (349, 'Ogre’s Hammer'): ((1, 1), 1),
    (350, 'Double Headbutt'): ((), 1),
    (351, 'Fire Mane'): ((2,), 1),
    (352, 'Ember'): ((2,), 0),
    (353, 'Combustion'): ((2,), 0),
    (354, 'Buddy Blast'): ((2,), 0),
    (354, 'Steam Artillery'): ((2, 2), 1),
    (355, 'Steady Firebreathing'): ((2,), 0),
    (356, 'Lava Burst'): ((2, 2, 2), 0),
    (357, 'Shining Feathers'): ((2, 2, 2, 2), 0),
    (358, 'Fire Kagura'): ((), 1),
    (358, 'Searing Flame'): ((2, 2), 1),
    (359, 'Sprinkle Water'): ((3,), 0),
    (360, 'Bubble Beam'): ((3,), 0),
    (361, 'Abrupt Flash'): ((3,), 0),
    (362, 'Splash'): ((3,), 0),
    (363, 'Splashing Panic'): ((3,), 0),
    (363, 'Waterfall'): ((3, 3), 1),
    (364, 'Surf'): ((3,), 1),
    (364, 'Swim Together'): ((3,), 0),
    (365, 'Undulate'): ((), 1),
    (366, 'Aqua Split'): ((3,), 1),
    (367, 'Tail Whap'): ((), 1),
    (367, 'Wave Splash'): ((3,), 2),
    (368, 'Aqua Slash'): ((3,), 2),
    (368, 'Whirlpool'): ((), 2),
    (369, 'Avenging Billow'): ((3,), 1),
    (369, 'Dynamic Dive'): ((3, 3), 2),
    (370, 'Bubble Drain'): ((3, 3), 1),
    (370, 'Water Kagura'): ((), 1),
    (371, 'Electroslug'): ((4,), 1),
    (372, 'Dual Bolt'): ((4,), 1),
    (372, 'High-Voltage Press'): ((4, 4), 1),
    (373, 'Zapping Draw'): ((), 6),
    (374, 'Rear Kick'): ((), 1),
    (374, 'Tiny Bolt'): ((4,), 1),
    (375, 'Flash Impact'): ((4,), 1),
    (375, 'Zap Kick'): ((), 1),
    (376, 'Astonish'): ((4,), 0),
    (376, 'Gadget Show'): ((), 2),
    (377, 'Scratch'): ((), 1),
    (377, 'Thunder Raid'): ((4, 4, 4), 0),
    (378, 'Impound'): ((6,), 0),
    (378, 'Try to Imitate'): ((), 2),
    (379, 'Rock Hurl'): ((6,), 0),
    (380, 'Dragonslice'): ((6,), 0),
    (381, 'Corkscrew Dive'): ((6,), 0),
    (381, 'Draconic Buster'): ((6, 6), 0),
    (382, 'Running Charge'): ((6,), 1),
    (383, 'Heavy Impact'): ((6, 6), 1),
    (384, 'Slight Intrusion'): ((), 1),
    (385, 'Pull'): ((), 1),
    (385, 'Reckless Charge'): ((), 3),
    (386, 'Mountain Ramming'): ((6, 6), 1),
    (386, 'Rock Kagura'): ((), 1),
    (387, 'Raging Curse'): ((), 1),
    (388, 'Confront'): ((), 3),
    (388, 'Stampede'): ((), 1),
    (389, 'Boss Headbutt'): ((), 3),
    (389, 'Vigorous Tackle'): ((), 1),
    (390, 'Clean Hit'): ((), 2),
    (390, 'Horn Attack'): ((), 1),
    (391, 'Gnaw Through'): ((), 1),
    (392, 'Rolling Tackle'): ((), 2),
    (393, 'Hang Down'): ((), 1),
    (394, 'Rolling Tackle'): ((), 2),
    (395, 'Damage Rush'): ((), 2),
    (395, 'Mega Drain'): ((1,), 2),
    (396, 'Gadget Show'): ((), 2),
    (396, 'Trimming Mower'): ((1,), 0),
    (397, 'Cut Up'): ((1,), 0),
    (398, 'Petal Blade Dance'): ((1,), 0),
    (398, 'Razor Leaf'): ((), 1),
    (399, 'Searching Eyes'): ((), 1),
    (400, 'Take Down'): ((1,), 0),
    (401, 'Rocket Rush'): ((1,), 1),
    (402, 'Ram'): ((), 1),
    (403, 'Nutrients'): ((1,), 0),
    (403, 'Tackle'): ((), 2),
    (404, 'Aroma Shot'): ((), 3),
    (404, 'Oil Salvo'): ((1,), 0),
    (405, 'Dig Claws'): ((), 2),
    (405, 'Live Coal'): ((2,), 0),
    (406, 'Flare'): ((), 2),
    (406, 'Punishing Fang'): ((2,), 2),
    (407, 'Evil Incineration'): ((2,), 3),
    (407, 'Flame Screen'): ((2,), 2),
    (408, 'Steady Firebreathing'): ((2,), 0),
    (409, 'Cruel Coal'): ((2,), 0),
    (409, 'Scorching Fire'): ((2,), 1),
    (410, 'Collect'): ((), 1),
    (410, 'Combustion'): ((2,), 0),
    (411, 'Combustion'): ((), 1),
    (411, 'Double Kick'): ((2,), 1),
    (412, 'Heat Blast'): ((), 2),
    (412, 'Inferno Kick Flurry'): ((2, 2), 1),
    (413, 'Gadget Show'): ((), 2),
    (413, 'Singe'): ((2,), 0),
    (414, 'Dark Frost'): ((3,), 2),
    (415, 'Shell Press'): ((3,), 0),
    (416, 'Wave Splash'): ((3,), 2),
    (417, 'Crescendo Wave'): ((3,), 0),
    (418, 'Icicle'): ((3,), 2),
    (418, 'Light Punch'): ((), 2),
    (419, 'Frozen Wood'): ((3, 3, 3), 1),
    (419, 'Lunge Out'): ((), 3),
    (420, 'Gadget Show'): ((), 2),
    (420, 'Manual Wash'): ((3,), 0),
    (421, 'Reckless Charge'): ((3,), 0),
    (422, 'Dive'): ((3, 3), 0),
    (422, 'Sharp Fin'): ((3,), 0),
    (423, 'Frost Smash'): ((3, 3, 3), 1),
    (423, 'Gentle Slap'): ((3,), 1),
    (424, 'Crushing Press'): ((3, 3, 3), 1),
    (425, 'Jamming Wing'): ((), 2),
    (425, 'Wicked Thunder'): ((4,), 2),
    (426, 'Procurement'): ((), 1),
    (426, 'Tiny Bolt'): ((4,), 0),
    (427, 'Thunder Shock'): ((4,), 1),
    (428, 'Head Bolt'): ((4,), 2),
    (429, 'Hypnotic Ray'): ((5,), 0),
    (430, 'Bench Manipulation'): ((5, 5, 5), 0),
    (430, 'Psyshot'): ((5,), 0),
    (431, 'Erasure Ball'): ((5, 5), 1),
    (432, 'Headbutt Bounce'): ((5,), 2),
    (432, 'Rocket Mirror'): ((5,), 1),
    (433, 'Chiming Commotion'): ((), 6),
    (434, 'Gemstone Mimicry'): ((5,), 1),
    (435, 'Disruptive Radar'): ((), 1),
    (435, 'Super Psy Bolt'): ((5,), 1),
    (436, 'Psychic'): ((5,), 2),
    (437, 'Wild Kick'): ((6,), 0),
    (438, 'Drag Off'): ((6,), 0),
    (439, 'Impact Blow'): ((6, 6), 0),
    (440, 'Mountain Munch'): ((), 1),
    (441, 'Explosive Ascension'): ((), 1),
    (442, 'Demolition Tackle'): ((6,), 3),
    (443, 'Headbutt'): ((6,), 0),
    (443, 'Rock Throw'): ((), 3),
    (444, 'Mountain Drop'): ((), 3),
    (444, 'Power Gem'): ((6,), 0),
    (445, 'Steady Punch'): ((6,), 0),
    (446, 'Harmonious Spirit Palm'): ((6,), 0),
    (447, 'Giant Rock'): ((6,), 3),
    (447, 'Regi Charge'): ((), 1),
    (448, 'Drag Down'): ((), 1),
    (448, 'Gnaw'): ((7,), 0),
    (449, 'Spinning Tail'): ((7, 7, 7), 0),
    (450, 'Surprise Attack'): ((7,), 0),
    (451, 'Dark Awakening'): ((7,), 0),
    (451, 'Scratch'): ((7, 7), 0),
    (452, 'Love Impact'): ((7,), 0),
    (452, 'Mega Kick'): ((7, 7), 0),
    (453, 'Hammer In'): ((7,), 1),
    (453, 'Pierce'): ((7,), 0),
    (454, 'Hammer In'): ((7,), 1),
    (454, 'Horn Rend'): ((7, 7), 1),
    (455, 'Kingly Impact'): ((7, 7, 7), 1),
    (455, 'Tainted Horn'): ((7, 7), 1),
    (456, 'Poison Spray'): ((7,), 0),
    (457, 'Confuse Ray'): ((7,), 0),
    (458, 'Assassin’s Return'): ((7, 7), 0),
    (459, 'Corrosive Sludge'): ((7,), 1),
    (460, 'Gooped Up'): ((7,), 1),
    (460, 'Hazardous Venom'): ((7, 7), 1),
    (461, 'Leaking Gas'): ((7,), 1),
    (462, 'Explode Together Now'): ((7,), 1),
    (463, 'Deceit'): ((), 1),
    (463, 'Torment'): ((7,), 1),
    (464, 'Scratch'): ((7,), 0),
    (464, 'Strike the Sleeper'): ((7, 7), 0),
    (465, 'Hurricane of Needles'): ((8,), 3),
    (465, 'Iron Shake-Up'): ((), 1),
    (466, 'Metal Claw'): ((8,), 1),
    (466, 'Roost'): ((), 1),
    (467, 'Strong Bash'): ((8, 8), 1),
    (468, 'Dangerous Incisors'): ((), 1),
    (469, 'Reckless Abandon'): ((), 1),
    (470, 'Fury Swipes'): ((), 2),
    (470, 'Paw-cket Pilfer'): ((), 1),
    (471, 'Cruel Slash'): ((), 3),
    (471, 'Haughty Order'): ((), 2),
    (472, 'Dizzy Punch'): ((), 3),
    (472, 'Gentle Slap'): ((), 2),
    (473, 'Hacking'): ((), 1),
    (474, 'R Command'): ((), 3),
    (475, 'R Command'): ((), 2),
    (476, 'Wing Attack'): ((), 1),
    (477, 'Add On'): ((), 1),
    (477, 'Speed Wing'): ((), 2),
    (478, 'Push Down'): ((), 1),
    (479, 'Tackle'): ((1,), 0),
    (479, 'Vine Whip'): ((1, 1), 0),
    (480, 'Vine Whip'): ((1,), 2),
    (480, 'Wrap'): ((1,), 0),
    (481, 'Command the Grass'): ((1,), 3),
    (482, 'Collect'): ((), 1),
    (482, 'Scratch'): ((), 2),
    (483, 'Gentle Slap'): ((), 2),
    (484, 'Hide'): ((), 1),
    (484, 'Leafage'): ((1,), 0),
    (485, 'Bemusing Aroma'): ((1,), 0),
    (485, 'Cut'): ((1,), 2),
    (486, 'Lively Needles'): ((1,), 0),
    (486, 'Pierce'): ((1,), 1),
    (487, 'Horn Attack'): ((), 1),
    (488, 'Toxic Spore'): ((), 1),
    (489, 'Dangerous Reaction'): ((), 1),
    (489, 'Seed Bomb'): ((1,), 1),
    (490, 'V-Force'): ((2, 2), 0),
    (491, 'Will-O-Wisp'): ((2,), 1),
    (492, 'Searing Flame'): ((2,), 1),
    (492, 'Smashing Headbutt'): ((2,), 3),
    (493, 'Brighten and Burn'): ((2,), 0),
    (494, 'Fire Blast'): ((2,), 0),
    (495, 'Burn It All Up'): ((2, 2), 0),
    (495, 'Incendiary Pillar'): ((2,), 0),
    (496, 'Peck Off'): ((), 1),
    (497, 'Fire Wing'): ((2,), 2),
    (498, 'Collect'): ((), 1),
    (498, 'Scratch'): ((), 2),
    (499, 'Gentle Slap'): ((), 2),
    (500, 'Round'): ((), 2),
    (501, 'Round'): ((), 2),
    (501, 'Wave Splash'): ((3,), 2),
    (502, 'Hyper Voice'): ((3,), 3),
    (502, 'Round'): ((), 3),
    (503, 'Ancient Seaweed'): ((3,), 0),
    (503, 'Surf'): ((3,), 2),
    (504, 'Big Bite'): ((3,), 2),
    (505, 'Waterfall'): ((3,), 2),
    (506, 'Snotted Up'): ((3,), 0),
    (507, 'Continuous Headbutt'): ((), 1),
    (507, 'Sheer Cold'): ((3, 3, 3), 1),
    (508, 'Drag Off'): ((3,), 0),
    (508, 'Icicle'): ((3,), 0),
    (509, 'Blizzard Burst'): ((3, 3), 1),
    (509, 'Slash'): ((), 2),
    (510, 'Call for Family'): ((), 1),
    (510, 'Static Shock'): ((4,), 0),
    (511, 'Hold Still'): ((), 1),
    (512, 'Electric Ball'): ((4, 4), 1),
    (513, 'Buzz Flip'): ((4, 4, 4), 1),
    (513, 'Thunder Fang'): ((4,), 0),
    (514, 'Charge'): ((), 1),
    (514, 'Disaster Volt'): ((4,), 2),
    (515, 'Slash'): ((), 2),
    (515, 'Voltage Burst'): ((4, 4), 1),
    (516, 'Mumble'): ((5,), 0),
    (516, 'Rest'): ((), 1),
    (517, 'Dream Calling'): ((), 1),
    (517, 'Sleep Pulse'): ((5,), 0),
    (518, 'Rollout'): ((), 1),
    (519, 'Cellular Evolution'): ((), 1),
    (519, 'Spray Fluid'): ((), 1),
    (520, 'Cellular Ascension'): ((), 1),
    (520, 'Evo-Lariat'): ((), 1),
    (521, 'Beam'): ((), 3),
    (521, 'Slight Shift'): ((5,), 0),
    (522, 'Calm Mind'): ((5,), 0),
    (522, 'Psychic'): ((), 3),
    (523, 'Best Punch'): ((), 2),
    (524, 'Double Smash'): ((), 3),
    (524, 'Golurk Hammer'): ((5, 5), 3),
    (525, 'Echoed Voice'): ((5,), 0),
    (526, 'Corkscrew Punch'): ((6, 6), 0),
    (526, 'Mud-Slap'): ((6,), 0),
    (527, 'Piercing Drill'): ((6, 6), 0),
    (527, 'Rock Tumble'): ((6, 6, 6), 0),
    (528, 'Low Kick'): ((6,), 0),
    (528, 'Strength'): ((6,), 2),
    (529, 'Hammer Arm'): ((6,), 2),
    (529, 'Low Kick'): ((6,), 0),
    (530, 'Swing Around'): ((6,), 3),
    (531, 'Shoulder Throw'): ((6,), 1),
    (532, 'Dig Claws'): ((), 2),
    (532, 'Flail'): ((), 1),
    (533, 'Stone Edge'): ((6,), 2),
    (534, 'Abundant Harvest'): ((), 1),
    (534, 'Earthquake'): ((6,), 2),
    (535, 'Poison Spray'): ((), 1),
    (536, 'Venoshock'): ((), 1),
    (537, 'Venoshock'): ((7,), 1),
    (538, 'Tighten Up'): ((7,), 0),
    (539, 'Tighten Up'): ((7,), 1),
    (540, 'Cursed Slug'): ((7, 7), 1),
    (540, 'Tighten Up'): ((7,), 1),
    (541, 'Playful Kick'): ((), 1),
    (542, 'Cutting Wind'): ((), 3),
    (543, 'Wild Lances'): ((8,), 0),
    (544, 'Corner'): ((8,), 0),
    (545, 'Cut Up'): ((8,), 0),
    (545, 'Finishing Blow'): ((8,), 1),
    (546, 'Metal Arms'): ((8, 8), 1),
    (546, 'Righteous Edge'): ((8,), 0),
    (547, 'Protect Charge'): ((8, 8), 1),
    (548, 'Gather Strength'): ((), 1),
    (549, 'Bite'): ((), 1),
    (549, 'Boundless Power'): ((6, 8), 0),
    (550, 'Axe Blast'): ((6, 8), 1),
    (550, 'Cross-Cut'): ((), 2),
    (551, 'Scout'): ((), 1),
    (551, 'Stampede'): ((), 1),
    (552, 'Fly'): ((), 1),
    (553, 'Add On'): ((), 1),
    (553, 'Swift Flight'): ((), 2),
    (554, 'Return'): ((), 1),
    (555, 'Tail Slap'): ((), 1),
    (556, 'Do the Wave'): ((), 1),
    (557, 'Bug Bite'): ((1,), 0),
    (558, 'Bug Buzz'): ((1,), 1),
    (559, 'Healing Wrapping'): ((), 1),
    (559, 'X-Scissor'): ((1,), 1),
    (560, 'Absorb'): ((1,), 0),
    (561, 'Energy Gift'): ((1,), 0),
    (561, 'Wondrous Cotton'): ((1,), 0),
    (562, 'Rear Kick'): ((), 2),
    (563, 'Push Down'): ((), 2),
    (563, 'Solar Beam'): ((1,), 2),
    (564, 'Headbutt Bounce'): ((), 1),
    (565, 'Acid Spray'): ((1,), 0),
    (566, 'Emerald Blade'): ((1, 1), 1),
    (566, 'Giga Drain'): ((1,), 0),
    (567, 'Rollout'): ((2, 2), 0),
    (567, 'Tackle'): ((2,), 0),
    (568, 'Combustion'): ((2,), 0),
    (568, 'Heat Crash'): ((2, 2), 1),
    (569, 'Heat Crash'): ((2, 2, 2), 1),
    (570, 'Collect'): ((), 1),
    (570, 'Scratch'): ((), 2),
    (571, 'Gentle Slap'): ((), 2),
    (572, 'Fire Claws'): ((2, 2), 0),
    (572, 'Licking Catch'): ((2,), 0),
    (573, 'Blazing Burst'): ((2, 2), 1),
    (573, 'Slash'): ((), 2),
    (574, 'Tackle'): ((3,), 0),
    (574, 'Water Gun'): ((3, 3), 0),
    (575, 'Energized Shell'): ((3,), 0),
    (576, 'Energized Slash'): ((3,), 0),
    (577, 'Bared Fangs'): ((3,), 0),
    (577, 'Bite'): ((), 1),
    (578, 'Firefighting'): ((), 1),
    (578, 'Wing Attack'): ((), 2),
    (579, 'Air Slash'): ((), 3),
    (579, 'Flap'): ((), 1),
    (580, 'Beat'): ((), 1),
    (580, 'Ice Edge'): ((3,), 1),
    (581, 'Ice Beam'): ((3,), 2),
    (581, 'Ram'): ((), 2),
    (582, 'Double Freeze'): ((3,), 2),
    (582, 'Ram'): ((), 2),
    (583, 'Gale Thrust'): ((3,), 1),
    (583, 'Sonic Edge'): ((3,), 2),
    (584, 'Smash Kick'): ((), 1),
    (584, 'Zap Kick'): ((4,), 1),
    (585, 'Electrobullet'): ((4, 4), 1),
    (585, 'Smash Kick'): ((), 1),
    (586, 'Surprise Attack'): ((4,), 0),
    (587, 'Discharge'): ((4,), 0),
    (588, 'Flop'): ((4,), 1),
    (588, 'Muddy Bolt'): ((), 1),
    (589, 'Heart Stamp'): ((), 1),
    (590, 'Gust'): ((), 2),
    (590, 'Happy Return'): ((), 1),
    (591, 'Reflect'): ((), 1),
    (591, 'Telekinesis'): ((5,), 2),
    (592, 'Focused Wish'): ((5,), 1),
    (593, 'Extended Damagriiigus'): ((5,), 1),
    (593, 'Perplex'): ((5,), 2),
    (594, 'Super Psy Bolt'): ((5,), 0),
    (595, 'Fortunate Eye'): ((5,), 0),
    (595, 'Psyshot'): ((5,), 1),
    (596, 'Synchro Shot'): ((5,), 1),
    (597, 'Oceanic Gloom'): ((5,), 0),
    (598, 'Power Press'): ((5,), 1),
    (599, 'Harden'): ((6,), 0),
    (599, 'Rolling Rocks'): ((6, 6), 0),
    (600, 'Power Gem'): ((6, 6, 6), 0),
    (600, 'Smack Down'): ((6,), 0),
    (601, 'Heavy Impact'): ((6, 6, 6), 0),
    (601, 'Vengeful Cannon'): ((6,), 0),
    (602, 'Elbow Strike'): ((6,), 0),
    (602, 'Rising Chop'): ((6,), 0),
    (603, 'Acrobatics'): ((6,), 1),
    (604, 'Rock Throw'): ((6,), 1),
    (605, 'Kick'): ((6,), 0),
    (606, 'Low Sweep'): ((6,), 0),
    (606, 'Smash Uppercut'): ((6, 6), 0),
    (607, 'Land Crush'): ((6, 6), 1),
    (607, 'Retaliate'): ((6,), 1),
    (608, 'Invite Evil'): ((7,), 0),
    (609, 'Knock Off'): ((7,), 0),
    (610, 'Headbutt'): ((7,), 0),
    (610, 'Invade'): ((7, 7), 0),
    (611, 'Ruffians Attack'): ((7, 7), 0),
    (612, 'Drool'): ((7,), 0),
    (612, 'Sludge Bomb'): ((7,), 1),
    (613, 'Gunk Shot'): ((7, 7), 1),
    (613, 'Suffocating Gas'): ((7,), 0),
    (614, 'Take Down'): ((7,), 0),
    (615, 'Foul Play'): ((), 3),
    (615, 'Mind Jack'): ((7,), 0),
    (616, 'Body Slam'): ((), 2),
    (616, 'Darkness Fang'): ((7,), 2),
    (617, 'Double Hit'): ((), 2),
    (617, 'Pitch-Black Fangs'): ((7, 7), 2),
    (618, 'Dark Bite'): ((7, 7, 7), 2),
    (619, 'Metal Claw'): ((8,), 2),
    (619, 'Zzzt'): ((8,), 0),
    (620, 'Metal Claw'): ((8,), 3),
    (620, 'Power Whip'): ((8,), 0),
    (621, 'Hard Gears'): ((8,), 0),
    (622, 'Hard Gears'): ((8,), 1),
    (623, 'Hammer In'): ((8,), 2),
    (624, 'Bite Together'): ((), 1),
    (624, 'Vise Grip'): ((8,), 1),
    (625, 'Ambush'): ((2, 3), 1),
    (625, 'Shred'): ((), 1),
    (626, 'Gnaw'): ((), 1),
    (626, 'Procurement'): ((), 1),
    (627, 'Focus Energy'): ((), 1),
    (627, 'Hyper Fang'): ((), 1),
    (628, 'Play Rough'): ((), 1),
    (629, 'Lunge Out'): ((), 3),
    (629, 'Roar'): ((), 1),
    (630, 'Odor Sleuth'): ((), 2),
    (630, 'Special Fang'): ((), 4),
    (631, 'Gold Breaker'): ((), 3),
    (632, 'Glide'): ((), 2),
    (632, 'Peck'): ((), 1),
    (633, 'Aerial Ace'): ((), 1),
    (633, 'Speed Wing'): ((), 4),
    (634, 'Hurricane'): ((), 3),
    (634, 'Wrapped in Wind'): ((), 1),
    (635, 'Psychic Sphere'): ((5,), 0),
    (635, 'Summoning Sign'): ((), 1),
    (636, 'Clay Blast'): ((5, 5), 1),
    (636, 'Eerie Light'): ((5,), 0),
    (637, 'Magical Shot'): ((5,), 2),
    (638, 'Razor Wing'): ((), 2),
    (638, 'Sonic Double'): ((8, 8), 1),
    (639, 'Ram'): ((8,), 1),
    (640, 'Metal Slash'): ((8,), 1),
    (641, 'Metal Stomp'): ((8,), 2),
    (642, 'Pointy Nails'): ((7,), 0),
    (643, 'Pointy Claws'): ((7, 7), 0),
    (644, 'Crunch'): ((7, 7), 1),
    (645, 'Rear Kick'): ((7,), 0),
    (645, 'Wild Tackle'): ((7, 7), 1),
    (646, 'Corkscrew Punch'): ((7,), 0),
    (646, 'Filch'): ((), 1),
    (647, 'Corkscrew Punch'): ((7, 7), 0),
    (648, 'Shadow Bullet'): ((7, 7), 0),
    (649, 'Spiky Wheel'): ((), 3),
    (650, 'Bind Down'): ((1,), 0),
    (651, 'Razor Leaf'): ((1, 1), 0),
    (652, 'Jungle Dump'): ((1, 1, 1, 1), 0),
    (653, 'Jam-Packed'): ((), 1),
    (654, 'Guard Press'): ((1,), 0),
    (654, 'Stomping Wood'): ((), 3),
    (655, 'Solar Cutter'): ((1,), 0),
    (655, 'Traverse Time'): ((1,), 0),
    (656, 'Nap'): ((), 1),
    (656, 'Seed Bomb'): ((1,), 1),
    (657, 'Low Kick'): ((1,), 1),
    (657, 'Pound'): ((1,), 0),
    (658, 'Perplex'): ((1,), 1),
    (658, 'Reversing Gust'): ((1,), 0),
    (659, 'Combustion'): ((2,), 1),
    (659, 'Stampede'): ((), 1),
    (660, 'Combustion'): ((2,), 1),
    (660, 'Supernatural Shapeshifter'): ((), 1),
    (661, 'Call for Family'): ((2,), 0),
    (661, 'Flare'): ((2,), 2),
    (662, 'Roasting Heat'): ((2,), 0),
    (662, 'Volcanic Meteor'): ((2,), 3),
    (663, 'Backfire'): ((2, 2), 1),
    (663, 'Singe'): ((2,), 0),
    (664, 'Wild Kick'): ((), 1),
    (665, 'Jumping Kick'): ((), 1),
    (666, 'Turbo Flare'): ((), 1),
    (667, 'Take Down'): ((), 2),
    (667, 'Vise Grip'): ((), 1),
    (668, 'Slap'): ((), 1),
    (669, 'Dig Claws'): ((), 1),
    (669, 'Mud-Slap'): ((6,), 1),
    (670, 'Mud Shot'): ((6,), 2),
    (670, 'Sand Attack'): ((), 2),
    (671, 'Bind'): ((), 2),
    (671, 'Strength'): ((), 4),
    (672, 'Pow-Pow Punching'): ((), 6),
    (673, 'Confront'): ((6, 6), 0),
    (673, 'Corkscrew Punch'): ((6,), 0),
    (674, 'Wild Press'): ((6, 6, 6), 0),
    (675, 'Power Gem'): ((6, 6), 0),
    (676, 'Cosmic Beam'): ((6,), 0),
    (677, 'Accelerating Stab'): ((6,), 0),
    (678, 'Aura Jab'): ((6,), 0),
    (678, 'Mega Brave'): ((6, 6), 0),
    (679, 'Smack'): ((6,), 0),
    (680, 'Reckless Charge'): ((6,), 0),
    (681, 'Shadowy Side Kick'): ((6, 6), 0),
    (682, 'Boundless Power'): ((6, 6), 1),
    (682, 'Stony Kick'): ((6,), 0),
    (683, 'Ram'): ((), 1),
    (684, 'Rock Hurl'): ((6,), 1),
    (685, 'Hammer In'): ((6, 6), 1),
    (686, 'Cutting Riposte'): ((7, 7), 1),
    (686, 'Vise Grip'): ((), 1),
    (687, 'Claw of Darkness'): ((7, 7), 1),
    (687, 'Terminal Period'): ((7,), 1),
    (688, 'Mountain Breaker'): ((7,), 0),
    (689, 'Clutch'): ((7,), 0),
    (689, 'Dark Feather'): ((7, 7), 1),
    (690, 'Darkness Fang'): ((7,), 0),
    (691, 'Greedy Hunt'): ((), 1),
    (691, 'Pitch-Black Fangs'): ((7,), 1),
    (692, 'Poison Jab'): ((7,), 1),
    (693, 'Miraculous Paint'): ((7,), 1),
    (694, 'Skull Bash'): ((8, 8), 2),
    (694, 'Welcoming Tail'): ((), 2),
    (695, 'Gobble Down'): ((8, 8), 0),
    (695, 'Huge Bite'): ((8, 8), 1),
    (696, 'Beam'): ((8,), 0),
    (696, 'Chrono Burst'): ((8, 8), 1),
    (697, 'Beat'): ((8,), 0),
    (698, 'Light Punch'): ((8,), 0),
    (699, 'Windup Swing'): ((8,), 0),
    (700, 'All-You-Can-Grab'): ((), 1),
    (700, 'Speed Attack'): ((8,), 2),
    (701, 'Pluck'): ((), 1),
    (702, 'Repeating Drill'): ((), 1),
    (703, 'Bellyful of Milk'): ((), 2),
    (703, 'Tackle'): ((), 3),
    (704, 'Collect'): ((), 1),
    (704, 'Gnaw'): ((), 2),
    (705, 'Bite'): ((), 2),
    (706, 'Hook'): ((1,), 2),
    (706, 'Poison Powder'): ((1,), 0),
    (707, 'Absorb'): ((1,), 1),
    (707, 'Pumped-Up Whip'): ((1, 1), 2),
    (708, 'Razor Leaf'): ((1,), 0),
    (709, 'Push Down'): ((1,), 1),
    (710, 'Solar Beam'): ((1, 1), 2),
    (711, 'Rollout'): ((1,), 1),
    (712, 'Scratch'): ((), 1),
    (713, 'U-turn'): ((1,), 1),
    (714, 'Earthen Power'): ((1,), 1),
    (715, 'Flare'): ((2,), 1),
    (716, 'Searing Flame'): ((2,), 2),
    (717, 'Combustion'): ((2, 2), 1),
    (717, 'Ram'): ((2,), 0),
    (718, 'Coiling Crush'): ((), 2),
    (718, 'Heat Crawler'): ((2, 2), 2),
    (719, 'Scorching Earth'): ((2,), 0),
    (720, 'Call for Family'): ((), 1),
    (720, 'Waterfall'): ((3,), 1),
    (721, 'Riptide'): ((3,), 0),
    (721, 'Swirling Waves'): ((3, 3), 1),
    (722, 'Beat'): ((3,), 0),
    (722, 'Icy Snow'): ((3, 3), 0),
    (723, 'Frost Barrier'): ((3, 3, 3), 0),
    (723, 'Hammer-lanche'): ((3, 3), 0),
    (724, 'Wave Splash'): ((3, 3), 0),
    (725, 'Aqua Launcher'): ((3, 3, 3), 0),
    (726, 'Surprise Attack'): ((3,), 0),
    (727, 'Double Stab'): ((3,), 0),
    (728, 'Bring Down'): ((3,), 0),
    (728, 'Water Shot'): ((3,), 0),
    (729, 'Hide'): ((3,), 0),
    (730, 'Chilling Wings'): ((3,), 0),
    (731, 'Freezing Headbutt'): ((), 1),
    (731, 'Tackle'): ((3,), 2),
    (732, 'Beam'): ((4,), 0),
    (733, 'Thunder Shock'): ((4,), 0),
    (734, 'Flashing Bolt'): ((4, 4), 0),
    (734, 'Upper Spark'): ((4,), 0),
    (735, 'Electro Fall'): ((4, 4), 0),
    (736, 'Thunder Jolt'): ((4,), 0),
    (737, 'Flash Ray'): ((4, 4), 0),
    (737, 'Riotous Blasting'): ((4, 4, 4), 0),
    (738, 'Electrified Incisors'): ((4,), 0),
    (739, 'Double Scratch'): ((), 1),
    (740, 'Dazzle Blast'): ((), 1),
    (740, 'Head Bolt'): ((4,), 1),
    (741, 'Teleportation Attack'): ((5,), 0),
    (742, 'Super Psy Bolt'): ((5,), 0),
    (743, 'Powerful Hand'): ((5,), 0),
    (744, 'Psychic'): ((5, 5), 0),
    (745, 'Collect'): ((), 1),
    (745, 'Headbutt'): ((5,), 0),
    (746, 'Call Sign'): ((5,), 0),
    (746, 'Psyshot'): ((5,), 0),
    (747, 'Mega Symphonia'): ((5,), 0),
    (747, 'Overflowing Wishes'): ((5,), 0),
    (748, 'Damage Beat'): ((5,), 0),
    (749, 'Triple Spin'): ((5,), 0),
    (750, 'Psychic Sphere'): ((5,), 2),
    (751, 'Bright Horns'): ((5, 5), 1),
    (751, 'Geo Gate'): ((5,), 0),
    (752, 'Stampede'): ((5,), 0),
    (752, 'Take Down'): ((5,), 1),
    (753, 'Hammer In'): ((5, 5), 1),
    (753, 'Horrifying Bite'): ((5,), 0),
    (754, 'Illusory Impulse'): ((2, 5), 1),
    (754, 'Strafe'): ((), 1),
    (755, 'Dragon Claw'): ((3, 5), 1),
    (756, 'Rapid-Fire Combo'): ((), 3),
    (757, 'Gentle Slap'): ((), 2),
    (757, 'Quick Gift'): ((), 1),
    (758, 'Charm'): ((), 1),
    (758, 'Skip'): ((), 1),
    (759, 'Dashing Kick'): ((), 1),
    (759, 'Spiral Kick'): ((), 2),
    (760, 'Flop'): ((), 2),
    (760, 'Light Punch'): ((), 1),
    (761, 'Hyper Lariat'): ((), 3),
    (761, 'Knuckle Punch'): ((), 2),
    (762, 'Tackle'): ((5,), 1),
    (763, 'Bite'): ((5,), 1),
    (763, 'Finishing Blow'): ((5, 5), 1),
    (764, 'Aurora Beam'): ((5, 5), 1),
    (764, 'Swelling Light'): ((5,), 0),
    (765, 'Magical Shot'): ((5,), 1),
    (765, 'Soothing Melody'): ((5,), 0),
    (766, 'Garland Ray'): ((5, 5), 0),
    (767, 'Call for Family'): ((), 1),
    (767, 'Scratch'): ((5,), 0),
    (768, 'Draining Kiss'): ((5,), 0),
    (769, 'Sweet Circle'): ((5,), 0),
    (770, 'Petty Grudge'): ((7,), 0),
    (771, 'Spooky Shot'): ((7,), 0),
    (772, 'Void Gale'): ((7, 7), 0),
    (773, 'Ambush'): ((7,), 0),
    (774, 'Sniping Feathers'): ((7, 7), 1),
    (774, 'Wind of Darkness'): ((7,), 0),
    (775, 'Cocky Claw'): ((7,), 0),
    (776, 'Allure'): ((), 1),
    (776, 'Dark Cutter'): ((7,), 1),
    (777, 'Power Rush'): ((7, 7, 7), 0),
    (777, 'Shatter'): ((7, 7), 0),
    (778, 'Seed Bomb'): ((1,), 0),
    (779, 'Disperse Drool'): ((1,), 0),
    (780, 'Lively Flower'): ((1,), 0),
    (780, 'Pollen Bomb'): ((1,), 0),
    (781, 'Juggernaut Horn'): ((1, 1), 0),
    (781, 'Mountain Ramming'): ((1, 1, 1), 0),
    (782, 'Headbutt'): ((1,), 1),
    (783, 'Mega Drain'): ((1,), 1),
    (784, 'Lunge Out'): ((1,), 1),
    (785, 'Bug’s Cannon'): ((1,), 0),
    (785, 'Speed Attack'): ((1, 1), 1),
    (786, 'Flail Around'): ((1,), 0),
    (787, 'Jumping Shot'): ((), 3),
    (787, 'Low Kick'): ((1,), 0),
    (788, 'Live Coal'): ((2,), 0),
    (789, 'Steady Firebreathing'): ((2,), 0),
    (790, 'Inferno X'): ((2, 2), 0),
    (791, 'Fighting Wings'): ((2,), 0),
    (792, 'Blaze Ball'): ((), 3),
    (793, 'Blaze Ball'): ((), 4),
    (794, 'Burning Flare'): ((2, 2, 2, 2), 0),
    (794, 'Combustion'): ((2,), 0),
    (795, 'Fire Wing'): ((2, 2), 1),
    (796, 'Chop'): ((2,), 0),
    (796, 'Gather Strength'): ((2,), 0),
    (797, 'Infernal Slash'): ((2,), 0),
    (798, 'Bubble Drain'): ((3,), 1),
    (799, 'Slam'): ((3,), 1),
    (800, 'Icy Snow'): ((3,), 1),
    (800, 'Stampede'): ((), 1),
    (801, 'Frost Smash'): ((3,), 2),
    (801, 'Rising Lunge'): ((), 2),
    (802, 'Blizzard Edge'): ((3,), 3),
    (802, 'Wreck'): ((), 3),
    (803, 'Crystal Fall'): ((3, 3), 0),
    (804, 'Call for Support'): ((), 1),
    (804, 'Tackle'): ((), 2),
    (805, 'Peck'): ((), 1),
    (805, 'Targeted Dive'): ((), 3),
    (806, 'Thunderbolt'): ((4,), 1),
    (807, 'Play Rough'): ((4,), 1),
    (808, 'Electric Run'): ((4,), 1),
    (809, 'Growl'): ((4,), 0),
    (809, 'Tiny Charge'): ((4,), 0),
    (810, 'Electric Punch'): ((4, 4), 0),
    (811, 'Voltaic Fist'): ((4, 4), 0),
    (812, 'Petty Grudge'): ((5,), 0),
    (813, 'Hexa-Magic'): ((5, 5), 0),
    (814, 'Collect'): ((), 1),
    (815, 'Healing Fluff'): ((), 1),
    (815, 'U-turn'): ((5,), 0),
    (816, 'Limit Break'): ((5,), 1),
    (817, 'Sneaky Placement'): ((5,), 0),
    (818, 'Psychic Sphere'): ((5,), 2),
    (819, 'Double-Edge'): ((6, 6), 0),
    (819, 'Raging Charge'): ((6,), 0),
    (820, 'Poison Jab'): ((6,), 0),
    (821, 'Poison Ring'): ((6,), 0),
    (822, 'Double Headbutt'): ((6,), 0),
    (823, 'Super Vibration'): ((6, 6), 0),
    (824, 'Cutting Wind'): ((6, 6), 0),
    (825, 'Dig Claws'): ((7,), 0),
    (825, 'Scratch'): ((7, 7), 0),
    (826, 'Cut'): ((7, 7), 0),
    (826, 'Retaliatory Claw'): ((7, 7), 0),
    (827, 'Reckless Charge'): ((7,), 0),
    (828, 'Greedy Fang'): ((7,), 0),
    (828, 'Hungry Jaws'): ((7, 7), 0),
    (829, 'Pitch-Black Fangs'): ((7, 7, 7), 0),
    (830, 'Ram'): ((7,), 0),
    (830, 'Rear Kick'): ((7,), 1),
    (831, 'Bite'): ((7,), 0),
    (831, 'Confront'): ((7,), 2),
    (832, 'Hammer In'): ((7,), 3),
    (832, 'Vengeful Fang'): ((7,), 0),
    (833, 'Call for Family'): ((7,), 0),
    (833, 'Playful Kick'): ((7,), 1),
    (834, 'Gentle Slap'): ((7, 7), 1),
    (835, 'Iron Feathers'): ((8, 8), 1),
    (836, 'Iron Defense'): ((), 1),
    (836, 'Rollout'): ((), 3),
    (837, 'Tool Drop'): ((), 3),
    (837, 'Triple Draw'): ((), 1),
    (838, 'Find a Friend'): ((), 1),
    (838, 'Gnaw'): ((8,), 0),
    (839, 'Hyper Beam'): ((8, 8, 8), 0),
    (840, 'Coated Attack'): ((8, 8, 8), 0),
    (841, 'Ball Roll'): ((), 1),
    (842, 'Round'): ((), 2),
    (842, 'Seismic Toss'): ((), 3),
    (843, 'Astonish'): ((), 2),
    (844, 'Dual Tail'): ((), 3),
    (844, 'Slap'): ((), 2),
    (845, 'Energizing Sketch'): ((), 1),
    (845, 'Hook'): ((), 2),
    (846, 'Surprise Attack'): ((), 1),
    (847, 'Slash'): ((), 1),
    (848, 'Kick'): ((), 2),
    (848, 'Run Around'): ((), 1),
    (849, 'Gale Thrust'): ((), 1),
    (849, 'Spiky Hopper'): ((), 2),
    (850, 'Bug Bite'): ((1,), 0),
    (851, 'Tackle'): ((1,), 0),
    (852, 'Energy Straw'): ((1,), 0),
    (852, 'Stun Spore'): ((1,), 0),
    (853, 'Trading Places'): ((1,), 0),
    (854, 'Twilight Poison'): ((1, 1), 0),
    (855, 'Flare Fall'): ((2, 2), 0),
    (856, 'Combustion'): ((2,), 1),
    (857, 'Power Stomp'): ((2,), 3),
    (857, 'Roasting Burn'): ((2,), 0),
    (858, 'Ram'): ((), 2),
    (859, 'Hydro Pump'): ((), 3),
    (860, 'Chilly'): ((3,), 0),
    (861, 'Absolute Snow'): ((3,), 2),
    (861, 'Resentful Refrain'): ((3,), 0),
    (862, 'Call for Family'): ((), 1),
    (862, 'Icy Snow'): ((3,), 1),
    (863, 'Flop'): ((), 1),
    (863, 'Sheer Cold'): ((3,), 2),
    (864, 'Blizzard'): ((3,), 2),
    (864, 'Snow Coating'): ((), 2),
    (865, 'Icicle'): ((3,), 0),
    (866, 'Cold Cyclone'): ((3, 3), 0),
    (867, 'Frosty Typhoon'): ((3, 3, 3), 0),
    (867, 'Ice Shot'): ((3,), 0),
    (868, 'Disaster Shock'): ((4, 4, 4), 0),
    (868, 'Split Bomb'): ((4, 4), 0),
    (869, 'Pouncing Trap'): ((4,), 0),
    (870, 'Zap Kick'): ((4,), 1),
    (871, 'Powerful Bolt'): ((4,), 2),
    (872, 'Fast Flight'): ((4,), 0),
    (872, 'Thunder Blast'): ((4, 4), 1),
    (873, 'Focused Wish'): ((5,), 0),
    (874, 'Double-Edge'): ((5,), 2),
    (874, 'Tri Kinesis'): ((), 2),
    (875, 'Ascension'): ((5,), 0),
    (876, 'Assassin’s Magic'): ((5,), 1),
    (877, 'Gadget Show'): ((), 2),
    (877, 'Roto Call'): ((), 1),
    (878, 'Splashing Dodge'): ((), 1),
    (879, 'Corner'): ((5,), 2),
    (879, 'Horrifying Revenge'): ((), 1),
    (880, 'Phantasmal Barrage'): ((5, 5), 1),
    (880, 'Spooky Shot'): ((5,), 0),
    (881, 'Relentless Burrowing'): ((6,), 0),
    (882, 'Mud Shot'): ((6,), 1),
    (883, 'Collect'): ((), 1),
    (883, 'Gentle Slap'): ((6,), 0),
    (884, 'Seventh Kick'): ((6,), 0),
    (885, 'Reckless Charge'): ((), 1),
    (886, 'Somersault Dive'): ((6, 6), 1),
    (887, 'Mud-Slap'): ((6,), 0),
    (888, 'Guard Press'): ((6,), 0),
    (888, 'Power Gem'): ((6,), 2),
    (889, 'Bulky Bump'): ((6,), 3),
    (889, 'Tar Cannon'): ((6,), 0),
    (890, 'Light Punch'): ((6,), 0),
    (890, 'Settle the Score'): ((6, 6), 1),
    (891, 'Hammer In'): ((7,), 2),
    (891, 'Rocket Feathers'): ((), 2),
    (892, 'Gnaw'): ((7,), 0),
    (893, 'Claw Slash'): ((7,), 1),
    (893, 'Gnaw'): ((7,), 0),
    (894, 'Punk Smash'): ((7,), 2),
    (894, 'Scarring Shout'): ((7,), 1),
    (895, 'Knock Off'): ((7,), 1),
    (896, 'Outlaw Leg'): ((7, 7), 1),
    (897, 'Corkscrew Punch'): ((7,), 0),
    (897, 'Master’s Punch'): ((7, 7), 1),
    (898, 'Mochi Rush'): ((7,), 0),
    (899, 'Push Down'): ((8,), 0),
    (900, 'Rapid Draw'): ((8,), 0),
    (901, 'Double-Edged Slash'): ((8, 8), 0),
    (902, 'Headbutt'): ((3, 4), 0),
    (903, 'Tail Snap'): ((3, 4), 0),
    (904, 'Ryuno Glide'): ((3, 4, 4), 0),
    (905, 'Breakthrough Assault'): ((4,), 1),
    (905, 'Dragon Claw'): ((2, 4), 1),
    (906, 'Rampaging Thunder'): ((2, 4, 4), 1),
    (906, 'Shred'): ((), 3),
    (907, 'Bite'): ((5, 7), 0),
    (907, 'Knickknack Carrying'): ((), 1),
    (908, 'Agility'): ((), 2),
    (908, 'Enhanced Blade'): ((5, 7), 0),
    (909, 'Reckless Charge'): ((1,), 0),
    (910, 'Poison Spray'): ((1,), 1),
    (911, 'Bloom Powder'): ((1, 1), 1),
    (912, 'Vine Slap'): ((1,), 0),
    (913, 'Leafy Cyclone'): ((1,), 1),
    (913, 'Melt'): ((1,), 0),
    (914, 'Flower Garden Rondo'): ((1,), 1),
    (914, 'Solar Beam'): ((1, 1), 1),
    (915, 'Bind'): ((1,), 1),
    (916, 'Cut Up'): ((), 1),
    (916, 'Slashing Strike'): ((), 2),
    (917, 'Growl'): ((), 1),
    (917, 'Seed Bomb'): ((1, 1), 0),
    (918, 'Leaf Step'): ((1, 1), 0),
    (919, 'Giant Bouquet'): ((), 3),
    (920, 'Wood Hammer'): ((1, 1), 2),
    (921, 'Coated Attack'): ((1,), 0),
    (922, 'Tons of Treading'): ((), 1),
    (923, 'Magical Leaf'): ((), 2),
    (924, 'Rising Bloom'): ((), 2),
    (925, 'Headbutt Bounce'): ((), 2),
    (926, 'Fire Claws'): ((2,), 1),
    (927, 'Heat Blast'): ((2,), 1),
    (928, 'Explosion Y'): ((2, 2), 1),
    (929, 'Searing Flame'): ((2,), 1),
    (930, 'Steady Firebreathing'): ((2,), 0),
    (931, 'Super Singe'): ((2, 2), 1),
    (932, 'Crimson Blast'): ((2, 2), 1),
    (933, 'Ram'): ((), 1),
    (933, 'Steady Firebreathing'): ((2,), 1),
    (934, 'Billowing Heat Wave'): ((2,), 0),
    (934, 'Heat Blast'): ((), 3),
    (935, 'Heat Burn'): ((2,), 1),
    (936, 'Heat Breath'): ((2,), 1),
    (937, 'Slight Intrusion'): ((3, 3), 0),
    (938, 'Crunch'): ((3, 3), 0),
    (939, 'Mortal Crunch'): ((3, 3), 1),
    (940, 'Crazy Headbutt'): ((3,), 2),
    (940, 'Damage Beat'): ((3,), 0),
    (941, 'Powder Snow'): ((3,), 0),
    (942, 'Ice Ball'): ((3, 3), 0),
    (942, 'Lunge Out'): ((3,), 0),
    (943, 'Frigid Fangs'): ((3,), 0),
    (943, 'Megaton Fall'): ((3, 3), 0),
    (944, 'Ice Prison'): ((3,), 3),
    (944, 'Regi Charge'): ((), 1),
    (945, 'Rain Splash'): ((3,), 1),
    (946, 'Spit Shot'): ((), 3),
    (946, 'Water Gun'): ((3,), 0),
    (947, 'Aerial Ace'): ((3,), 0),
    (948, 'Tail Smack'): ((4,), 0),
    (948, 'Tiny Bolt'): ((4,), 1),
    (949, 'Quick Blow'): ((4,), 0),
    (949, 'Strong Volt'): ((4, 4), 1),
    (950, 'Lightning Ball'): ((4,), 0),
    (951, 'Hundred-Hitting Ball'): ((), 3),
    (952, 'Drill Peck'): ((4,), 2),
    (952, 'Follow-Up Bolt'): ((4,), 1),
    (953, 'Thunder Wave'): ((4,), 0),
    (953, 'Thunderbolt'): ((4, 4), 1),
    (954, 'Piercing Gaze'): ((), 2),
    (954, 'Volt Strike'): ((4, 4), 0),
    (955, 'Collect'): ((), 1),
    (955, 'Static Shock'): ((4,), 0),
    (956, 'Combat Thunder'): ((4,), 1),
    (957, 'Hadron Spark'): ((4, 4), 1),
    (957, 'Slashing Claw'): ((4,), 0),
    (958, 'Magical Shot'): ((5,), 2),
    (958, 'Metronome'): ((), 2),
    (959, 'Pound'): ((), 2),
    (960, 'Draining Kiss'): ((), 2),
    (961, 'Flop'): ((5,), 1),
    (961, 'Hide'): ((), 1),
    (962, 'Energized Balloon'): ((), 3),
    (963, 'Spooky Shot'): ((5,), 0),
    (964, 'Cosmic Beatdown'): ((5,), 0),
    (965, 'Ram'): ((5,), 0),
    (966, 'Magical Shot'): ((5,), 1),
    (966, 'Tackle'): ((5,), 0),
    (967, 'Pleasant Aroma'): ((), 1),
    (967, 'Stampede'): ((), 1),
    (968, 'Wonder Shine'): ((), 2),
    (969, 'Crunch'): ((5,), 2),
    (969, 'Scream'): ((), 1),
    (970, 'Energy Feather'): ((5,), 0),
    (971, 'Adjusted Horn'): ((5,), 1),
    (972, 'Low Kick'): ((6,), 1),
    (972, 'Spin and Draw'): ((), 1),
    (973, 'Hammer In'): ((6, 6), 1),
    (973, 'Megaton Fall'): ((6, 6), 2),
    (974, 'Punch'): ((6,), 1),
    (975, 'Big Bite'): ((6,), 0),
    (975, 'Flopping Trap'): ((6,), 2),
    (976, 'Counter Jewel'): ((6, 6), 1),
    (977, 'Crabhammer'): ((), 3),
    (977, 'Vise Grip'): ((), 2),
    (978, 'Coordinated Throwing'): ((6,), 1),
    (979, 'Impact Blow'): ((6, 6), 1),
    (979, 'Orichalcum Fang'): ((6,), 1),
    (980, 'Spinning Attack'): ((7,), 2),
    (980, 'Spit Poison'): ((7,), 0),
    (981, 'Poison Ring'): ((7,), 0),
    (981, 'Spinning Attack'): ((7,), 2),
    (982, 'Dastardly Jab'): ((7,), 1),
    (982, 'Sludge Bomb'): ((7,), 2),
    (983, 'Flap'): ((), 2),
    (984, 'Bone Shot'): ((), 2),
    (984, 'Vulture Claw'): ((7,), 2),
    (985, 'Filch'): ((), 1),
    (985, 'Knuckle Impact'): ((7, 7), 1),
    (986, 'Poison Spray'): ((7,), 0),
    (986, 'Relentless Punches'): ((7, 7, 7), 0),
    (987, 'Bite'): ((8,), 2),
    (987, 'Call for Family'): ((), 1),
    (988, 'Protecting Steel'): ((8,), 3),
    (988, 'Regi Charge'): ((), 1),
    (989, 'Shield Attack'): ((8,), 1),
    (990, 'Spiky Rolling'): ((8,), 1),
    (990, 'Stun Needle'): ((8,), 0),
    (991, 'Hammer In'): ((8, 8), 1),
    (991, 'Iron Bash'): ((8, 8, 8, 8), 1),
    (992, 'Confront'): ((8, 8), 0),
    (992, 'Duralubeam'): ((8, 8, 8), 0),
    (993, 'Rock Tomb'): ((), 4),
    (994, 'Dragon Pulse'): ((6, 8), 0),
    (995, 'Bring Down the Axe'): ((6,), 0),
    (995, 'Dragon Pulse'): ((6, 8), 0),
    (996, 'Rising Lunge'): ((), 1),
    (997, 'Work Rush'): ((), 3),
    (998, 'Take It Easy'): ((), 1),
    (999, 'Slashing Claw'): ((), 2),
    (1000, 'Cat Kick'): ((), 1),
    (1001, 'Cat Kick'): ((), 1),
    (1001, 'Energy Crush'): ((), 2),
    (1002, 'Spike Draw'): ((), 1),
    (1002, 'Wild Scissors'): ((), 3),
    (1003, 'Glide'): ((), 1),
    (1003, 'Minor Errand-Running'): ((), 1),
    (1004, 'Flap'): ((), 1),
    (1004, 'Razor Wing'): ((), 2),
    (1005, 'Facade'): ((), 1),
    (1005, 'Feathery Strike'): ((), 3),
    (1006, 'Ear Force'): ((), 3),
    (1006, 'Kaleidowaltz'): ((), 1),
    (1007, 'Peck the Wound'): ((), 2),
    (1008, 'Brave Bird'): ((), 3),
    (1008, 'Clutch'): ((), 2),
    (1009, 'Dozing Draw'): ((), 1),
    (1010, 'Dragon Strike'): ((), 3),
    (1010, 'Gentle Slap'): ((), 2),
    (1011, 'Gooey Thread'): ((1,), 0),
    (1012, 'Poison Ring'): ((1,), 0),
    (1013, 'Leaf Step'): ((1,), 0),
    (1013, 'Send Flowers'): ((1,), 0),
    (1014, 'Reckless Charge'): ((1,), 0),
    (1015, 'Solar Cutter'): ((1,), 0),
    (1016, 'Regal Command'): ((1,), 0),
    (1016, 'Solar Coiling'): ((1, 1, 1), 0),
    (1017, 'Gnaw'): ((1,), 0),
    (1018, 'Hide'): ((1,), 0),
    (1019, 'Blow Through'): ((1,), 0),
    (1020, 'Find a Friend'): ((1,), 0),
    (1020, 'Tackle'): ((), 3),
    (1021, 'Feather Shot'): ((), 3),
    (1021, 'Leafage'): ((1,), 0),
    (1022, 'Crushing Arrow'): ((1,), 3),
    (1023, 'Flare'): ((2, 2), 0),
    (1024, 'Fire Wing'): ((2, 2), 0),
    (1025, 'Fire Claws'): ((2,), 0),
    (1026, 'Dire Nails'): ((2, 2), 0),
    (1026, 'Nasty Plot'): ((2,), 0),
    (1027, 'Heat Breath'): ((2, 2), 1),
    (1028, 'Rain Splash'): ((3,), 0),
    (1028, 'Wave Splash'): ((3, 3), 0),
    (1029, 'Wave Splash'): ((3, 3), 0),
    (1030, 'Water Gun'): ((3,), 0),
    (1031, 'Jetting Blow'): ((3,), 0),
    (1031, 'Nebula Beam'): ((), 3),
    (1032, 'Icy Wind'): ((3,), 1),
    (1033, 'Freezing Chill'): ((3, 3), 1),
    (1034, 'Powerful Steam'): ((3, 3), 1),
    (1034, 'Strength'): ((3,), 1),
    (1035, 'Double Scratch'): ((4,), 0),
    (1036, 'Static Shock'): ((4,), 1),
    (1037, 'Incessant Onslaught'): ((4,), 1),
    (1037, 'Strong Volt'): ((4,), 2),
    (1038, 'Tail Generator'): ((4,), 0),
    (1038, 'Thunder Shock'): ((4,), 1),
    (1039, 'Flop'): ((5, 5), 0),
    (1039, 'Follow Me'): ((5,), 0),
    (1040, 'Shooting Moons'): ((5, 5), 0),
    (1041, 'Double Eater'): ((5,), 1),
    (1042, 'Nap'): ((), 1),
    (1042, 'Stampede'): ((5,), 0),
    (1043, 'Perplex'): ((5,), 0),
    (1043, 'Psychic'): ((5,), 0),
    (1044, 'Ram'): ((5,), 0),
    (1044, 'Sweet Scent'): ((), 1),
    (1045, 'Draining Kiss'): ((5,), 1),
    (1046, 'Rolling Rocks'): ((6, 6), 0),
    (1047, 'Obliterating Nose'): ((6, 6, 6), 1),
    (1047, 'Rolling Rocks'): ((6, 6), 0),
    (1048, 'Bite'): ((6,), 2),
    (1048, 'Sand Attack'): ((6,), 0),
    (1049, 'Heavy Impact'): ((6, 6), 2),
    (1049, 'Twister Spewing'): ((6, 6), 1),
    (1050, 'Rock Tumble'): ((6, 6), 0),
    (1050, 'Screw Knuckle'): ((6, 6), 1),
    (1051, 'Double Draw'): ((6,), 0),
    (1051, 'Scratch'): ((6, 6), 0),
    (1052, 'Hammer In'): ((6, 6), 1),
    (1053, 'Get Angry'): ((6,), 1),
    (1054, 'Wreak Havoc'): ((6,), 1),
    (1055, 'Vengeful Kick'): ((6,), 0),
    (1056, 'Gaia Wave'): ((6, 6, 6), 0),
    (1056, 'Nullifying Zero'): ((6, 6, 6, 6, 6), 0),
    (1057, 'Surprise Attack'): ((7,), 0),
    (1058, 'Haunt'): ((7,), 0),
    (1059, 'Mind Jack'): ((7,), 0),
    (1060, 'Poison Jab'): ((7, 7), 0),
    (1061, 'Hazardous Tail'): ((7, 7, 7), 0),
    (1061, 'Wrack Down'): ((7, 7), 0),
    (1062, 'Dark Strike'): ((7, 7), 1),
    (1062, 'Soul Destroyer'): ((7, 7), 1),
    (1063, 'Rising Blade'): ((7, 7), 1),
    (1063, 'Strafe'): ((7,), 0),
    (1064, 'Sonic Ripper'): ((8, 8), 1),
    (1065, 'Cut'): ((), 1),
    (1066, 'Weaponized Swords'): ((), 2),
    (1067, 'Metal Slash'): ((8,), 3),
    (1067, 'Slash'): ((), 3),
    (1068, 'Memory Lock'): ((8,), 0),
    (1069, 'Take Down'): ((), 1),
    (1070, 'Retaliatory Incisors'): ((), 1),
    (1070, 'Scrape Off'): ((), 1),
    (1071, 'Tuck Tail'): ((), 3),
    (1072, 'Collapse'): ((), 4),
    (1072, 'Gormandizer'): ((), 1),
    (1073, 'Smash Kick'): ((), 1),
    (1074, 'Earthquake'): ((), 1),
    (1074, 'Whap Down'): ((), 3),
    (1075, 'Chirp'): ((), 1),
    (1075, 'Peck'): ((), 2),
    (1076, 'Hand Trim'): ((), 1),
    (1076, 'Headbutt'): ((), 1),
    (1180, 'Geobuster'): ((6, 6, 6, 6), 0),
}


# ---- 相手のスタジアム/ツールはメタ上位のカード単位で持つ ----
# スタジアムとツールには「属性」が無く効果がカード固有なので、相手側でも
# カード単位が要る。ただし無限ではなくメタ上位14種/11種で99.6%/100%を覆う。
# **メタが動いたら tools/meta_zones.py を再生成すること。**
META_STADIUM = [1245, 1246, 1247, 1249, 1250, 1252, 1256, 1257, 1259, 1260, 1261, 1262, 1264, 1266]
META_TOOL = [1155, 1156, 1159, 1161, 1166, 1167, 1173, 1174, 1175, 1176, 1177]

# 有効な特徴グループ（None なら全部。tools/featgen_core.py の _GRULES 参照）
FEAT_GROUPS = {'resource', 'slot_attr', 'misc', 'base', 'threat2'}

# ---- 相手側の判定に使う固定ID（自デッキと無関係。094から継承） ----
BOSS_ORDERS = 1182
TR_ARTICUNO = 414
MIST_IDS = (11, 20)          # Mist Energy / Rock Fighting Energy（PH無効化）

# ---- 自デッキから導出した定数 ----
SPECIES = (66, 140, 305, 343, 741, 742, 743)
SPECIES_NAME = {66: 'dudunsparce', 140: 'fezandipiti', 305: 'dunsparce', 343: 'shaymin', 741: 'abra', 742: 'kadabra', 743: 'alakazam'}
LINE_OF = {66: 'dunsparce', 140: 'fezandipiti', 305: 'dunsparce', 343: 'shaymin', 741: 'abra', 742: 'abra', 743: 'abra'}        # cardId -> 進化ライン名（最下段の種の名前）
LINE_NAMES = ('abra', 'dunsparce', 'fezandipiti', 'shaymin')
PRE_EVO = (305, 741, 742)        # 同デッキ内に進化先を持つ種
ENERGY_NAME = {5: 'basic_p_en', 13: 'enriching_', 19: 'telepath_p'}      # 全エネ（基本+特殊）の cardId -> 短縮名
BASIC_ENERGY_NAME = {5: 'basic_p_en'}
TOOL_NAME = {}     # ポケモンに付けるツール（ATTACHで来る）
STADIUM_NAME = {1266: 'nighttime_'}
ATTACH_BUCKET = {66: 'dunsparce_f', 140: 'fezandipiti_f', 305: 'dunsparce_p', 343: 'shaymin_f', 741: 'abra_p', 742: 'abra_p', 743: 'abra_f'}   # エネ貼り先の種 -> クラス分類のバケット
CLS_PLAY = {66: 'play_dudunsparce', 140: 'play_fezandipiti', 305: 'play_dunsparce', 343: 'play_shaymin', 741: 'play_abra', 742: 'play_kadabra', 743: 'play_alakazam', 1079: 'use_rare_candy', 1081: 'use_enhanced_h', 1086: 'use_buddy_budd', 1097: 'use_night_stre', 1129: 'use_sacred_ash', 1152: 'use_poké_pad', 1182: 'use_orders', 1184: 'use_aid', 1197: 'use_machinatio', 1225: 'use_hilda', 1231: 'use_dawn', 1266: 'use_nighttime_'}
CLS_EVO = {66: 'evo_dudunsparce', 742: 'evo_kadabra', 743: 'evo_alakazam'}
CLS_AB = {66: 'ab_dudunsparce', 140: 'ab_fezandipiti', 305: 'ab_dunsparce', 343: 'ab_shaymin', 741: 'ab_abra', 742: 'ab_kadabra', 743: 'ab_alakazam', 1266: 'ab_nighttime_'}        # ポケモンの特性 + 効果を持つスタジアム
CLS_ATK = {76: 'atk_dudunsparce_0', 183: 'atk_fezandipiti_0', 423: 'atk_dunsparce_0', 424: 'atk_dunsparce_1', 477: 'atk_shaymin_0', 1070: 'atk_abra_0', 1071: 'atk_kadabra_0', 1072: 'atk_alakazam_0'}      # attackId -> クラス名（技の打ち分け）
CLASSES = ('ab_abra', 'ab_alakazam', 'ab_dudunsparce', 'ab_dunsparce', 'ab_fezandipiti', 'ab_kadabra', 'ab_nighttime_', 'ab_other', 'ab_shaymin', 'ab_stadium', 'atk_abra_0', 'atk_alakazam_0', 'atk_dudunsparce_0', 'atk_dunsparce_0', 'atk_dunsparce_1', 'atk_fezandipiti_0', 'atk_kadabra_0', 'atk_other', 'atk_shaymin_0', 'attach_basic_p_en_abra_f', 'attach_basic_p_en_abra_p', 'attach_basic_p_en_dunsparce_f', 'attach_basic_p_en_dunsparce_p', 'attach_basic_p_en_fezandipiti_f', 'attach_basic_p_en_oth', 'attach_basic_p_en_shaymin_f', 'attach_enriching__abra_f', 'attach_enriching__abra_p', 'attach_enriching__dunsparce_f', 'attach_enriching__dunsparce_p', 'attach_enriching__fezandipiti_f', 'attach_enriching__oth', 'attach_enriching__shaymin_f', 'attach_telepath_p_abra_f', 'attach_telepath_p_abra_p', 'attach_telepath_p_dunsparce_f', 'attach_telepath_p_dunsparce_p', 'attach_telepath_p_fezandipiti_f', 'attach_telepath_p_oth', 'attach_telepath_p_shaymin_f', 'end', 'evo_alakazam', 'evo_dudunsparce', 'evo_kadabra', 'other', 'play_abra', 'play_alakazam', 'play_dudunsparce', 'play_dunsparce', 'play_fezandipiti', 'play_kadabra', 'play_shaymin', 'retreat', 'use_aid', 'use_buddy_budd', 'use_dawn', 'use_enhanced_h', 'use_hilda', 'use_machinatio', 'use_night_stre', 'use_nighttime_', 'use_orders', 'use_poké_pad', 'use_rare_candy', 'use_sacred_ash')


def _load_deck_counts() -> dict[int, int]:
    try:
        _d = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _d = os.getcwd()
    for path in (os.path.join(_d, "deck.csv"),
                 "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            counts: dict[int, int] = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.split(",")[0].isdigit():
                        continue
                    cid = int(line.split(",")[0])
                    counts[cid] = counts.get(cid, 0) + 1
            return counts
    raise FileNotFoundError("deck.csv not found")


# **104のキー集合を保つ**（枚数0のカードもキーを残す）。deck.csv から
# 再導出すると pool_/left_ のキーが消えて特徴次元が黙って縮む（EXP-107）。
DECK_COUNTS = {5: 2, 13: 1, 19: 4, 66: 3, 140: 1, 305: 4, 343: 0, 741: 4, 742: 4, 743: 4, 1079: 4, 1081: 3, 1086: 4, 1097: 1, 1129: 1, 1152: 4, 1182: 4, 1184: 1, 1197: 3, 1225: 4, 1231: 4, 1266: 0}
DECK_IDS = sorted(DECK_COUNTS)  # 特徴キーの固定順


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
META_CARDS = [2, 6, 11, 13, 14, 15, 18, 19, 89, 90, 92, 93, 104, 119, 120, 121, 235, 341, 342, 344, 345, 379, 380, 381, 387, 400, 401, 414, 431, 646, 647, 648, 741, 742, 743, 756, 860, 1071, 1081, 1134, 1142, 1147, 1173, 1198, 1216, 1217, 1218, 1219, 1220, 1245, 1259, 1261]
META_ARCH = {"Marnie's Grimmsnarl ex": [104, 646, 647, 648, 860, 1219, 1259], 'Mega Kangaskhan ex': [11, 14, 18, 344, 345, 756, 1147], 'Alakazam': [13, 19, 741, 742, 743, 1081], "Team Rocket's Spidops": [15, 400, 401, 414, 431, 1134, 1216, 1217, 1218, 1220], 'Dragapult ex': [2, 119, 120, 121, 235, 1071, 1198], "Cynthia's Garchomp ex": [6, 341, 342, 379, 380, 381, 387, 1142, 1173, 1261], 'Thwackey': [89, 90, 92, 93, 1245]}
ARCH_SLUG = {a: "".join(ch for ch in a.lower().replace(" ", "_")
                        if ch.isalnum() or ch == "_")[:14]
             for a in META_ARCH}
# 自デッキの各カードから進化できる先（選択肢の「取ると進化がつながるか」判定用）
META_ARCH_NAMES = ['Alakazam', "Cynthia's Garchomp ex", 'Dragapult ex', "Marnie's Grimmsnarl ex", 'Mega Kangaskhan ex', "Team Rocket's Spidops", 'Thwackey']


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
    for cid in META_CARDS:
        f[f"opp_seen_{cid}"] = float(op_seen.get(cid, 0))
    # 系統ごとの「看板カードを何割見たか」。1枚でも見えれば強い証拠になる
    for a in META_ARCH_NAMES:
        ids = META_ARCH[a]
        hit = sum(1 for cid in ids if op_seen.get(cid))
        f[f"oparch_{ARCH_SLUG[a]}"] = hit / max(1, len(ids))

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
SELECT_DIM = 5          # その決定自体の要件（下記 logs_vector 参照）
LOGS_DIM = N_LOGTYPE * 2 + SELECT_DIM
OPT_DIM = 29            # option_vector の次元（24 + 可能化4 + 状態異常種別1）
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
    if sel is not None:
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
        _num094(getattr(opt, "specialConditionType", None), -1.0),
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
