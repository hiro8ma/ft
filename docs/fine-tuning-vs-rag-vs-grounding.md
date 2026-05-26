# いつ fine-tuning を選ぶか — RAG / grounding との使い分け意思決定

## TL;DR

モデルを賢くする手法は「重みをどこまで触るか」で 3 層に並ぶ。fine-tuning（重みを更新）> RAG（外部知識を検索して prompt に注入）> grounding（与えた文脈だけに答えを縛る）の順で改変コストが下がる。**まず grounding、足りなければ RAG、それでも足りない時だけ fine-tuning** が基本の優先順位。このリポは最深層の fine-tuning を実験する場として、3 択の一番奥を担う。

## 3 手法を 1 枚で

| 手法 | 何を触るか | 改変対象 | 反映の速さ | 主な効果 |
|---|---|---|---|---|
| **grounding** | 重みは触らない | prompt に渡す文脈 | 即時 | 与えた文脈の範囲内に回答を縛り、hallucination を抑える |
| **RAG** | 重みは触らない | 検索インデックス + 検索ロジック | index 更新で即時 | 最新・大量の外部知識を動的に注入する |
| **fine-tuning** | 重みを更新する | 学習データ + adapter | 再学習が必要（週〜月） | 振る舞い・口調・出力形式・領域語彙をモデル自体に焼き込む |

3 層は排他ではなく積み重ね。grounding は RAG の出力先（検索した文脈に回答を縛る）でもあり、fine-tuning したモデルの上に RAG を載せる構成も普通。

## 4 軸の意思決定マトリクス

「どれを選ぶか」は次の 4 軸で判断する。

| 軸 | grounding 有利 | RAG 有利 | fine-tuning 有利 |
|---|---|---|---|
| **レイテンシ** | 検索なしで最速 | 検索往復で +数十〜百 ms | 推論のみで速い（学習は事前に完了） |
| **データ** | 文脈をその場で渡せる少量 | 頻繁に更新される大量の外部知識 | 量より質の高い教師データが安定供給できる |
| **精度** | 文脈準拠の正確さで足りる | 事実・出典の正確さが要る | 振る舞い・形式・専門語彙の一貫性が要る |
| **拡張性** | 文脈長の上限で頭打ち | index をスケールさせれば伸びる | 領域追加ごとに adapter 学習が要る |

### 軸ごとの読み方

- **レイテンシ** — RAG は検索往復が乗る。最速を求めるなら grounding か fine-tuning。fine-tuning はコストを学習時に前払いし、推論はモデル単体で完結する。
- **データ** — 「毎日変わる事実」は RAG（重みに焼くと陳腐化する）。「めったに変わらない振る舞い・形式」は fine-tuning。その場で渡せる少量なら grounding で済む。
- **精度** — 「正しい事実を引く」課題は RAG。「正しい振る舞いをする」課題は fine-tuning。両者は別問題で、混同すると手法を取り違える。
- **拡張性** — RAG は index を足すだけで知識が増える。fine-tuning は領域が増えるたびに学習が要るが、adapter を切り替える運用で複数領域を捌ける。

## 優先順位の原則

1. **grounding を試す** — prompt と文脈設計だけで解けるなら、学習も検索基盤も不要。最も安く速い。
2. **足りなければ RAG** — 知識が prompt に収まらない / 頻繁に更新される場合。重みは触らず index を育てる。
3. **それでも足りなければ fine-tuning** — 振る舞い・出力形式・領域語彙が prompt 指示だけでは安定しない場合。重みに焼き込む。

「知識を足したい」のか「振る舞いを変えたい」のかで分岐するのが要点。知識不足を fine-tuning で解こうとすると、学習コストの割に陳腐化が早く割に合わない。

## このリポが「fine-tuning を選んだ具体例」

このリポは MLX + LoRA で軽量モデルを fine-tune する実験場で、上の 3 択の最深層に該当する。fine-tuning を選ぶ判断が効く局面を、実際の実験要素で示す。

| 実験要素 | 4 軸のどこに効くか |
|---|---|
| LoRA / DoRA で adapter を学習（`train` / `train-dora`） | 重み更新で振る舞い・領域語彙を焼き込む（精度軸の「振る舞いの一貫性」） |
| データ量別の学習曲線（25 / 50 / 100% で loss 計測） | 「質の高い教師データが供給できるか」のデータ軸を定量で確認 |
| base vs FT の出力比較（`eval` / `chat`） | fine-tuning を選んだ妥当性の検証（prompt 指示で足りたか） |
| adapter 切替（`adapters-*` のディレクトリ別管理） | 領域追加を adapter 単位で捌く拡張性軸 |
| ローカル推論で完結（外部 API 呼び出しなし） | 機密データを外に出さず推論する選択肢（grounding/RAG にない利点） |

このリポが扱うのは「外部 API に送れない機密データを、領域特化した軽量モデルで処理する」局面。これは RAG / grounding では代替しにくく、fine-tuning を選ぶ典型例になる。

## FT × RAG ハイブリッド（RAFT）

fine-tuning と RAG は二者択一ではなく、組み合わせると強い。

- **RAFT（Retrieval-Augmented Fine-Tuning）** — 検索で引いた文脈と一緒に、ノイズ文書も混ぜて「正しい文脈だけを使って答える」訓練をする。RAG で文脈を渡されたときに余計な情報へ引きずられにくいモデルを作る発想。
- 役割分担は **fine-tuning で「文脈の使い方・出力形式」を、RAG で「最新の事実」を** 担う。重みには「振る舞い」を焼き、知識は外部 index に逃がす。
- このリポで学習した adapter を、検索基盤を持つエージェント側で RAG と併用すれば、ローカル軽量モデルでも「最新知識 + 領域特化の振る舞い」を両立できる。

## 参考

- Retrieval Augmented Generation (RAG) vs Fine-Tuning: https://www.ibm.com/think/topics/rag-vs-fine-tuning
- RAFT — Adapting Language Model to Domain Specific RAG: https://arxiv.org/abs/2403.10131
- Grounding for generative AI: https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview
