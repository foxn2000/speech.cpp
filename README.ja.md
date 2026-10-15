<p align="center"><img src="docs/images/banner.png" alt="speech.cpp" width="768"></p>

<p align="center">
  音声対話のための音声合成、音声認識、発話区間検出を、ggml の上で C++ で動かします
</p>

<p align="center">
  <a href="#モデル">モデル</a> ·
  <a href="#インストール">インストール</a> ·
  <a href="#クイックスタート">クイックスタート</a> ·
  <a href="#ドキュメント">ドキュメント</a> ·
  <a href="docs/c-api.md">C API</a>
</p>

<p align="center">
  <a href="README.md">English</a> · 日本語
</p>

<p align="center">
  <a href="https://github.com/nyosegawa/speech.cpp/actions/workflows/build.yml"><img src="https://img.shields.io/github/actions/workflow/status/nyosegawa/speech.cpp/build.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/nyosegawa/speech.cpp/releases/latest"><img src="https://img.shields.io/github/v/release/nyosegawa/speech.cpp?style=flat-square" alt="最新のリリース"></a>
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%C2%B7%20Metal-lightgrey?style=flat-square" alt="Apple Silicon の macOS で Metal を使う">
  <img src="https://img.shields.io/badge/Windows-x64%20%C2%B7%20Vulkan-lightgrey?style=flat-square" alt="x64 の Windows で Vulkan を使う">
  <img src="https://img.shields.io/badge/Linux-x64%20%C2%B7%20Vulkan%20or%20CPU-lightgrey?style=flat-square" alt="x64 の Linux で Vulkan か CPU を使う">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT ライセンス"></a>
</p>

speech.cpp は、1つのコマンド `speech` と1つの共有ライブラリ `libspeech` からなり、どちらも1つの C API の上に作られています。[ggml](https://github.com/ggml-org/ggml) を通して CPU、Metal、Vulkan で動きます。このコマンドは、テキストを読み上げ、録音を文字に起こし、録音の中で人が話している区間を見つけ、マイクの音声を話している間に文字に起こします(`speech asr --live`)。また、OpenAI の音声 API と Realtime の文字起こしを HTTP で提供し、リクエストの間もモデルを読み込んだままにしておきたいプログラムのためにワーカープロセスを動かします。[ASIST](https://github.com/nyosegawa/asist) もこのワーカーを使っています。

**最初の音声までの時間は 0.25 秒以内です。** Apple M5 では、Qwen3-TTS 0.6B はテキストを受け取ってから 0.035 秒で、Irodori-TTS v4.1-Small-MF は 0.22 秒で最初の音声を送り、残りは再生している間に続けて届きます。

**実行ファイルが1つあれば、ほかに何もインストールする必要はありません。** Python も CUDA も使いません。`speech` はモデルを名前で取得し(コーデックを含めて1つの GGUF ファイルです)、Metal、Vulkan、CPU のいずれかで動かします。

**OpenAI の API を自分のコンピュータで使えます。** `speech serve` は OpenAI の音声 API と Realtime の文字起こしに応答するので、OpenAI のクライアントはベース URL を変えるだけで動きます。

**公式の実装と照らし合わせています。** どのモデルも、すべての段階を同じ入力から公式の実装と比べて確かめています。

<p align="center">
  <img src="docs/images/page-speak.png" alt="speech serve のページの Speak タブ。Qwen3-TTS 0.6B が読み上げた文の波形と、最初の音声までの時間" width="720">
  <br>
  <sub><code>speech serve --open</code> のページです。カタログからモデルを選び、取得して試せます。</sub>
</p>

<table>
  <tr>
    <td width="50%"><img src="docs/images/page-transcribe.png" alt="Transcribe タブ。Silero VAD が発話を見つけた区間ごとに、同じ音声を Qwen3-ASR 0.6B が文字に起こしたもの"><br><sub>Transcribe は、ファイルや録音を、人が話している区間ごとに文字に起こします。</sub></td>
    <td width="50%"><img src="docs/images/page-live.png" alt="話している最中の Live タブ。最初の文は確定し、次の文は話している途中なので灰色で表示されている"><br><sub>Live は、マイクの音声を話している間に文字に起こします。灰色のテキストは、まだ話している途中の発話です。</sub></td>
  </tr>
</table>

## モデル

| タスク | 名前 | 言語 | 得意なこと |
|---|---|---|---|
| 合成 | [`qwen3-tts-0.6b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF), [`qwen3-tts-1.7b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | de en es fr it ja ko pt ru zh | 名前のある9人の話者を持ち、0.08 秒の最初のフレームから音声をストリーミングします。1.7B モデルは話し方の指示に従います |
| 合成 | [`irodori-tts-mf`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF), [`irodori-tts`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF), [`irodori-tts-large`](https://huggingface.co/sakasegawa/Irodori-TTS-v4-Large-GGUF) | ja | 参照録音の声や、言葉で説明した声で日本語を話します。large はどちらにもより忠実です |
| 認識 | [`qwen3-asr-0.6b`](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF), [`qwen3-asr-1.7b`](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) | 30 言語 | 言語を自分で判定し、人名や用語のプロンプトを受け付け、20 分までの音声を一度に認識します |
| 認識 | [`reazonspeech-v2`](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) | ja | 何分もある録音の日本語を、句読点つきで認識します |
| 認識 | [`parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) | ja | 短い日本語の発話を速く認識し、時刻も出します |
| 認識 | [`parakeet-tdt-0.6b-v3`](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) | ヨーロッパの 25 言語 | 言語を自分で判定し、句読点、大文字、時刻つきで認識します |
| 検出 | [`silero-vad`](https://huggingface.co/sakasegawa/silero-vad-GGUF) | どの言語でも | 人が話している区間を見つけ、認識モデルに発話を1つずつ渡します。1.2 MB です |

コマンドはモデルを名前で受け取り、初めて使うときに、名前のリンク先の Hugging Face のリポジトリから変換済みのファイルを取得します。[docs/models.md](docs/models.md) には各モデルのサイズ、声、言語が載っています。ファミリーごとのページもあります。[Qwen3-TTS](docs/models/qwen3-tts.md)、[Irodori-TTS](docs/models/irodori-tts.md)、[FastConformer](docs/models/fastconformer.md)、[Qwen3-ASR](docs/models/qwen3-asr.md)、[Silero VAD](docs/models/silero-vad.md)。

## インストール

Apple silicon の macOS と x86-64 の Linux では、次のコマンドを実行します。

```sh
curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh
```

x64 の Windows では、PowerShell で次のコマンドを実行します。

```powershell
irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1 | iex
```

インストーラはリリースのアーカイブを SHA-256 で確かめ、`speech` をホームフォルダの下に置き、`PATH` に加えます。`speech update` を実行するか、インストーラをもう一度実行すると更新されます。Linux では、Vulkan のローダーが入っていれば Vulkan 版を、入っていなければ CPU 版をインストールします。Linux 版には glibc 2.34 以降が必要です。オプション、アンインストールの方法、各プラットフォームで必要なものは [docs/install.md](docs/install.md) にあります。Intel Mac やその他のシステムでは、[ソースからビルド](docs/install.md#build-from-source)してください。

## クイックスタート

1. 文を読み上げます。初回はモデルを取得します(1.21 GB)。

   ```sh
   speech tts qwen3-tts-0.6b --voice ryan -o hello.wav "Hello. It will be sunny in Tokyo tomorrow."
   ```

2. その音声を文字に戻します(初回は 0.84 GB を取得します)。

   ```sh
   speech asr qwen3-asr-0.6b hello.wav
   ```

   ```
   Hello, it will be sunny in Tokyo tomorrow.
   ```

3. モデルを選び、読み上げと文字起こしを試せるページを開きます。

   ```sh
   speech serve --open
   ```

`speech models` は、すべてのモデルと、言語ごとに最初に使うとよいモデルを表示します。

## コマンドライン

| サブコマンド | すること |
|---|---|
| `speech tts` | テキストを読み上げ、WAVE ファイルか標準出力に書き出します |
| `speech asr` | WAVE ファイルの音声、または話している最中のマイクの音声を文字に起こします |
| `speech vad` | WAVE ファイルの中で人が話している区間を書き出します |
| `speech voice` | 参照録音から Irodori-TTS か Drama 3 の声のファイルを作ります |
| `speech info` | モデルファイルに書かれているモデルの情報を表示します |
| `speech devices` | モデルを動かせるデバイスを一覧にします |
| `speech models`, `pull`, `rm` | モデルの一覧表示、取得、削除をします |
| `speech quantize` | F32 の重みのモデルファイルを F16、Q8_0、またはより小さい型で書き出します |
| `speech update` | 最新のリリースをインストールするか、新しいリリースが出ているかどうかを表示します |
| `speech serve` | OpenAI の音声 API と、モデルを試すページを HTTP で提供します |
| `speech worker` | プログラムのために、標準入力と標準出力の JSON Lines でモデルを提供します |

```sh
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o - < story.txt | ffplay -nodisp -autoexit -
speech asr reazonspeech-v2 --timestamps meeting.wav
speech asr qwen3-asr-1.7b --language ja --prompt "Claude Code、渋谷" meeting.wav
speech asr reazonspeech-v2 --vad silero-vad --live
speech vad silero-vad --max-speech-duration-s 10 meeting.wav
```

リクエストのオプションはすべてフラグとして指定でき、モデルが受け付けるオプションは `speech info MODEL` で確かめられます。すべてのサブコマンドとオプションは [docs/cli.md](docs/cli.md) に載っています。

## サーバーとページ

`speech serve` は合成モデル、認識モデル、検出モデルを1つずつ保持し、OpenAI の音声 API と同じように`POST /v1/audio/speech` と `POST /v1/audio/transcriptions` に応答します。Web アプリ、スクリプト、curl から使えます。

```sh
speech serve qwen3-tts-0.6b qwen3-asr-0.6b
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "Hello.", "voice": "ryan"}' -o hello.wav
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@hello.wav
```

サーバーは 127.0.0.1 で待ち受け、認証はありません。エンドポイント、形式、エラー、ページについては [docs/server.md](docs/server.md) を見てください。

## ライブラリを使う

プログラムはリリースのアーカイブから `libspeech` をリンクし、`speech.h` をインクルードします。

```c
speech_model * model = NULL;
speech_request * request = NULL;
speech_model_load("Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf", NULL, &model);
speech_request_new(model, &request);
speech_request_set_text(request, "Hello.");
speech_request_set_string(request, SPEECH_OPT_VOICE, "ryan");
speech_synthesize(request, on_audio, NULL);   /* on_audio receives the samples as they are made */
```

どの関数もステータスを返し、その分類から何が失敗したかがわかります。API の全体は [docs/c-api.md](docs/c-api.md) にあります。

ライブラリをリンクしたくないプログラムは、ASIST と同じように `speech worker MODEL` を起動して JSON Lines でやりとりできます。標準入力に1行に1つずつリクエストを書くと、標準出力にその音声かテキストと、1つの終了メッセージが返ります。プロトコルは [docs/worker.md](docs/worker.md) にあります。

## 精度と速度

移植したすべての段階を、公式の実装から書き出したテンソルと比べています。認識モデルは、確認に使うすべての入力について、F32 でも F16 でも公式の実装と同じテキストを出します。

[speech-bench](https://github.com/nyosegawa/speech-bench) を使い、2026-10-14 に Apple M5(Metal)と RTX 2080(Vulkan)で、リクエストを1つずつ処理して測りました。合成は日本語の 20 文をシード 1 と 2 で 3 回ずつ読み上げ(Irodori-TTS は1つの参照音声の声で読み上げます)、最初の音声までの時間の p50、RTFx(音声の秒数をかかった秒数で割った値で、大きいほど速い)、そして合成した音声を Qwen3-ASR 1.7B が聞き取ったときの CER を測りました。

| モデル | 最初の音声までの時間、M5 / RTX 2080 | RTFx | CER |
|---|---|---|---|
| `qwen3-tts-0.6b`, Q8_0 | 0.035 s / 0.024 s | 3.7 / 5.2 | 6.4% / 6.0% |
| `qwen3-tts-1.7b`, Q8_0 | 0.057 s / 0.032 s | 2.6 / 4.2 | 4.2% / 6.1% |
| `irodori-tts-mf`, F16 | 0.22 s / 0.12 s | 7.7 / 18 | 6.7% / 6.4% |
| `irodori-tts`, F16, 16 steps | 1.15 s / 0.52 s | 3.5 / 8.5 | 3.2% / 3.2% |
| `irodori-tts-large`, Q8_0, 16 steps | 4.44 s / 1.25 s | 0.98 / 3.3 | 5.5% / 5.5% |

認識は、Common Voice 8.0 の日本語テストセットの 4,483 発話(声の部分だけに切り詰めたもの)で測りました。括弧内は、同じ語の別の表記を正解と認めたときの CER で、CER は2台で同じでした。レイテンシの p50 も同じ発話で測りました。parakeet-tdt-0.6b-v3 は FLEURS の英語 300 発話で、3 回測りました。

| モデル | Common Voice ja の CER | レイテンシの p50、M5 / RTX 2080 |
|---|---|---|
| `parakeet-tdt_ctc-0.6b-ja`, Q8_0 | 7.9% (3.0%) | 0.045 s / 0.050 s |
| `reazonspeech-v2`, Q8_0 | 12.0% (7.1%) | 0.063 s / 0.092 s |
| `qwen3-asr-0.6b`, Q8_0 | 11.9% (7.0%) | 0.127 s / 0.073 s |
| `qwen3-asr-1.7b`, Q8_0 | 9.5% (4.7%) | 0.312 s / 0.140 s |
| `parakeet-tdt-0.6b-v3`, F16 | FLEURS en の WER 8.8% | 0.077 s / 0.095 s |

詳しくは各モデルのページを見てください。

## ドキュメント

以下のページは英語で書かれています。

| ページ | 内容 |
|---|---|
| [Install](docs/install.md) | インストーラ、更新、アンインストール、必要なもの、リリースのアーカイブ、ソースからのビルド |
| [Command line](docs/cli.md) | すべてのサブコマンドとオプション |
| [Models](docs/models.md) | カタログ、モデルの名前、モデルを置く場所、声、言語 |
| [Qwen3-TTS](docs/models/qwen3-tts.md), [Irodori-TTS](docs/models/irodori-tts.md), [FastConformer](docs/models/fastconformer.md), [Qwen3-ASR](docs/models/qwen3-asr.md), [Silero VAD](docs/models/silero-vad.md) | 各ファミリーができること、オプション、精度、速度 |
| [Server and page](docs/server.md) | `speech serve` と OpenAI の音声 API |
| [Worker protocol](docs/worker.md) | プログラムから使う `speech worker` |
| [C API](docs/c-api.md) | `speech.h` |

## ライセンス

MIT ライセンスです。[LICENSE](LICENSE) を見てください。モデルの重みはそれぞれの作者のものです。

- Qwen3-TTS と Qwen3-ASR は Qwen チームのもので、Apache License 2.0 で提供されています。
- Irodori-TTS v4.1-Small と v4.1-Small-MF は Aratako 氏のもので、MIT License に加えて、モデルカードにある倫理上の制限がかかります。同意のない声のクローン、ディープフェイク、誤情報への利用は禁じられています。
- Irodori-TTS v4-Large は Aratako 氏のもので、テキストエンコーダーが Google の T5Gemma 2 から派生しているため、Gemma Terms of Use で提供されています。倫理上の制限は v4.1 と同じです。
- Irodori-TTS のファイルに含まれるコーデック Semantic-DACVAE-Japanese-32dim は Aratako 氏のもので、MIT License で提供されています。これは Meta の facebook/dacvae-watermarked(Apache License 2.0)から派生しています。そのモデルカードの本文は SAM License にも触れていますが、facebookresearch/dacvae で 2025-12-19 に Apache-2.0 に訂正されました。
- parakeet のモデルは NVIDIA のもので CC-BY-4.0、reazonspeech-nemo-v2 は reazon-research のもので Apache License 2.0 で提供されています。
- Silero VAD は Silero Team のもので、MIT License で提供されています。
