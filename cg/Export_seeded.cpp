// Export_seeded.cpp — CRN(共通乱数)用のシード付きBattleStartを追加したローカルビルド用
// ラッパー (I-122 / EXP-079)。
//
// 配布ソース (pokemon-tcg-ai-battle/data/ptcg_engine/ptcgProgram 22) は一切変更せず、
// Export.cpp を #include で取り込んだ上で追加APIのみを定義する。
// エンジンは元々 GameConfig.seed + mt19937 + deviceRand フラグの決定化機構を内蔵しており
// (ShuffleDeck/コイン/対象シャッフルは全て deviceRand=false で game.rng に分岐)、
// 公式の ApiBattleStart が deviceRand=true 固定かつ rng を random_device で上書きしている
// だけなので、本パッチは「シードを外から渡す入口」を足すだけで全乱数が決定化される。
//
// ビルド (CLAUDE.md の再ビルド手順はこのファイルを対象にする):
//   cd リポジトリルート
//   SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
//   clang++ -std=c++20 -O2 -shared -fPIC -isysroot $SDK \
//     -isystem $SDK/usr/include/c++/v1 \
//     -I "pokemon-tcg-ai-battle/data/ptcg_engine/ptcgProgram 22" \
//     -o cg/libcg.dylib cg/Export_seeded.cpp
//
// 既存シンボル(BattleStart等)は Export.cpp のまま無変更＝追加ビルドは完全後方互換。
// 提出物には含めない（提出時のcgは配布オリジナルを同梱、CLAUDE.md参照）。

#include "Export.cpp"

// ApiBattleStart (Api.h) と同一のデッキ検証+初期化。相違点は3つだけ:
//  (1) config.seed を引数で指定  (2) config.deviceRand = false
//  (3) 初期化後に seed_seq で game.rng を上書きしない
static StartData ApiBattleStartSeeded(int* cards, unsigned int seed) {
	ApiData* data = new ApiData();
	data->apiDataType = 1;

	GameConfig config = {};
	config.seed = (seed == 0) ? 1u : seed; // 0 は Game::init が random_device に置換するため回避
	config.recordLog = true;
	config.deviceRand = false;
	for (int i = 0; i < 2; i++) {
		std::unordered_map<std::u8string, int> nameCount;
		bool aceSpec = false;
		bool basic = false;
		for (int j = 0; j < DECK_SIZE; j++) {
			CardId id = cards[i * DECK_SIZE + j];
			if (!CardTable.contains(id)) {
				delete data;
				return { nullptr, i, 1 };
			}

			const CardMaster& master = CardTable.at(id);
			if (master.aceSpec) {
				if (aceSpec) {
					delete data;
					return { nullptr, i, 4 };
				} else {
					aceSpec = true;
				}
			}

			if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
				basic = true;
			}

			int& count = nameCount[master.name];
			count++;
			if (count > DECK_SAME_CARD_MAX) {
				if (master.cardType != CardType::BasicEnergy) {
					delete data;
					return { nullptr, i, 2 };
				}
			}

			config.decks[i].cards[j] = cards[i * DECK_SIZE + j];
		}
		if (!basic) {
			delete data;
			return { nullptr, i, 3 };
		}
	}

	data->init(config);
	data->start();
	data->next();
	return { data, -1, 0 };
}

extern "C" {

	GAME_API StartData BattleStartSeeded(int* cards, unsigned int seed) {
		return ApiBattleStartSeeded(cards, seed);
	}

}

// ============================================================================
// 特権情報API（リーグRLのCritic用。設計メモは cg/Export_seeded_patch.txt）
//
// 相手の隠れ領域（手札・山札・サイド）の cardId 列だけを返す。
// GetBattleData は ToJsonApi で視点が手番プレイヤーに固定され、PlayerJson が
// `pi == myPlayerIndex` のときしか hand を出さないため、既存APIでは取得できない。
// 観戦者モード(playerIndex==2)なら取れるが山札+カード名まで吐くので生成ループには重い。
//
// **踏んではいけない罠（実装済みの回避）**:
//  (1) state.nextLogStart() を呼ばない — ログカーソルを進める副作用があり、
//      本来の観測が受け取るログを食い潰す
//  (2) data->jsonBuilder を使わない — GetBattleData の戻り値がそのバッファを
//      指しているので、共有すると直前の観測を上書きしうる
// ============================================================================

static thread_local JsonBuilder g_hiddenBuilder;

static void AppendCardIds(const State& state, JsonBuilder& j, const CardList& list) {
	j.append('[');
	for (int i = 0; i < list.size(); i++) {
		j.comma(i);
		j.append((int)state.getCard(list[i]).cardId);
	}
	j.append(']');
}

extern "C" {

	// playerIndex の隠れ領域を {"hand":[id...],"deck":[id...],"prize":[id...]} で返す。
	// json のみ有効（data/count は未使用）。selectPlayer には現在の手番を入れる。
	GAME_API SerialData GetHiddenData(ApiData* data, int playerIndex) {
		if (data == nullptr || data->apiDataType != 1) {
			return {};
		}
		if (playerIndex < 0 || playerIndex > 1) {
			return {};
		}
		const State& state = data->state;
		const PlayerState& ps = state.players[playerIndex];
		JsonBuilder& j = g_hiddenBuilder;
		j.clear();
		j.append('{');
		j.appendKey("hand");
		AppendCardIds(state, j, ps.hand);
		j.appendCommaKey("deck");
		AppendCardIds(state, j, ps.deck);
		j.appendCommaKey("prize");
		AppendCardIds(state, j, ps.prize);
		j.append('}');
		return { j.buf.c_str(), nullptr, 0, state.selectPlayer };
	}

}
