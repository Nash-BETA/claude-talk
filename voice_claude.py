"""マイクに話しかけると Claude が答え、その返答を読み上げる常駐プログラム。

流れ:
    マイク → 発話区間の検出 (webrtcvad) → ローカルで文字起こし (mlx-whisper)
    → 呼びかけ語の判定 → claude -p (stream-json で常駐) → say で読み上げ

音声はローカルで文字起こしし、Claude に送るのは呼びかけ語で始まる発話の文字だけ。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from collections import deque
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pykakasi
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 モノラル

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
# 「クロド」「グロード」「クロウド」のような聞き取りの揺れは、読みで比べて吸収する（reading() を参照）
DEFAULT_WAKE_WORDS = ["クロード", "claude"]
RESET_PHRASES = ["新しい会話", "会話をリセット"]
# 呼びかけ語だけを言ったあと、続きを呼びかけなしで受け付ける秒数（返答後の --followup とは別）
CALL_ONLY_WINDOW_SEC = 8.0
CUE_SOUND = Path("/System/Library/Sounds/Tink.aiff")

# NOTE: Whisper が無音や雑音から生成しがちな定型文。呼びかけ後の受付時間中に拾っても送らない。
HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "おやすみなさい",
    "おやすみなさい。",
    "チャンネル登録お願いします",
    "ありがとうございました。",
}

SYSTEM_PROMPT = """\
これは音声での会話です。
ユーザーの発話はマイクから文字起こししたもので、聞き間違いを含むことがあります。意味が通らないときは推測で進めず、短く聞き返してください。
返答は音声合成で読み上げられます。日本語の話し言葉で、原則 2〜3 文で答えてください。
箇条書き・表・見出し・コードブロック・URL・絵文字は使わないでください。CLAUDE.md などに書かれた出力形式の指示より、この音声向けの指示を優先してください。
メッセージの送信、ファイルの書き込みや削除、外部サービスへの登録など、取り消しにくい操作や外部に出る操作の前は、何をどうするかを一文で読み上げ、ユーザーが「はい」と答えてから実行してください。聞き間違いのまま実行するのを防ぐためです。
このセッションは確認画面を出せないため、承認が必要な操作は自動で拒否されます。拒否されたときは、その旨を短く伝えてください。
"""

# NOTE: 長い返答を最後まで読み上げると会話にならないので、冒頭だけ読んで続きは画面に任せる
MAX_SPEECH_CHARS = 300

# NOTE: UserPromptSubmit フックにブロックされると、ブロック文が is_error: false の返答として返ってくる。
#       文面はこの接頭辞で始まる（Claude Code 2.1.314 で実測）
HOOK_BLOCKED_PREFIX = "UserPromptSubmit operation blocked by hook"


def peak_dbfs(frame: bytes) -> float:
    """フレームの最大振幅を dBFS で返す。"""
    # NOTE: int16 のまま絶対値を取ると -32768 が桁あふれするので、int32 にしてから取る
    peak = int(np.abs(np.frombuffer(frame, dtype=np.int16).astype(np.int32)).max())
    return 20 * math.log10(max(peak, 1) / 32768)


def display_width(ch: str) -> int:
    # 端末の設定で幅が変わる文字（█ や絵文字など）は、折り返さないよう 2 桁として数える
    return 1 if unicodedata.east_asian_width(ch) in ("Na", "H") else 2


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class StatusLine:
    """ターミナルの最下行に今の状態を出し続ける。ログはその上に流す。

    マイクの音量に合わせてバーが伸び縮みするので、声を拾えているかが話しながら分かる。
    定期的な描画は _tick スレッドだけが行う。音声を処理するメインループは端末への書き込みを待たない。
    """

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    BAR_WIDTH = 12
    # この範囲の入力レベル (dBFS) を、バーの空〜満タンに対応させる。
    # 下限は室内の雑音（実測 -45〜-56 dBFS）でほとんど伸びない値にしている
    LEVEL_FLOOR_DB = -50.0
    LEVEL_CEIL_DB = -15.0

    def __init__(self) -> None:
        self.enabled = False
        self.lock = threading.Lock()
        self.state_since: tuple[State, float] = (State.IDLE, time.monotonic())
        self.level = 0.0
        self.followup_until = 0.0

    def start(self) -> None:
        self.enabled = True
        threading.Thread(target=self._tick, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            if self.enabled:
                self.enabled = False
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()

    def set_state(self, state: State) -> None:
        # 状態と開始時刻は 1 つの組で差し替える（描画スレッドが新しい状態と古い経過秒数を組み合わせないように）
        if state != self.state_since[0]:
            self.state_since = (state, time.monotonic())

    def open_followup(self, until: float) -> None:
        self.followup_until = until

    def update_level(self, frame: bytes) -> None:
        level = (peak_dbfs(frame) - self.LEVEL_FLOOR_DB) / (self.LEVEL_CEIL_DB - self.LEVEL_FLOOR_DB)
        # 上がるときはすぐ、下がるときはゆっくり（バーがちらつかないように）
        self.level = max(min(max(level, 0.0), 1.0), self.level * 0.85)

    def text(self) -> str:
        now = time.monotonic()
        state, since = self.state_since
        elapsed = now - since
        spinner = self.SPINNER[int(now * 10) % len(self.SPINNER)]
        filled = round(self.level * self.BAR_WIDTH)
        bar = "█" * filled + "·" * (self.BAR_WIDTH - filled)
        match state:
            case State.LISTENING:
                return f"👂 聞き取り中  {bar}  {elapsed:.1f} 秒"
            case State.TRANSCRIBING:
                return f"{spinner} 文字起こし中"
            case State.THINKING:
                return f"{spinner} 考え中  {elapsed:.0f} 秒"
            case State.SPEAKING:
                return "🔊 読み上げ中"
            case State.IDLE:
                rest = self.followup_until - now
                label = f"続けてどうぞ（あと {math.ceil(rest)} 秒）" if rest > 0 else "待機中"
                return f"🎙  {label}  {bar}"
        raise ValueError(f"未知の状態: {state!r}")

    def render(self) -> None:
        with self.lock:
            if not self.enabled:
                return
            sys.stdout.write("\r\x1b[2K" + self._fit(self.text()))
            sys.stdout.flush()

    @staticmethod
    def _fit(text: str) -> str:
        # 端末の幅を超えて折り返すと、\x1b[2K で消せない行が残って描画のたびに増えるので切り詰める
        columns = shutil.get_terminal_size((80, 24)).columns - 1
        width = 0
        for i, ch in enumerate(text):
            width += display_width(ch)
            if width > columns:
                return text[:i]
        return text

    def _tick(self) -> None:
        while self.enabled:
            time.sleep(0.1)
            self.render()


status_line = StatusLine()


def log(label: str, text: str = "") -> None:
    line = f"[{datetime.now():%H:%M:%S}] {label}{': ' + text if text else ''}"
    # 最下行の状態表示をいったん消してからログを出し、その下に描き直す
    with status_line.lock:
        if status_line.enabled:
            sys.stdout.write("\r\x1b[2K")
        print(line, flush=True)
    status_line.render()


class Utterance(NamedTuple):
    pcm: bytes
    started_at: float  # 話し始めと判定した時刻 (time.monotonic())


class Segmenter:
    """30ms ごとのフレームを受け取り、発話のまとまりを返す。"""

    def __init__(
        self,
        vad_level: int,
        end_silence_ms: int = 1200,
        preroll_ms: int = 600,
        max_ms: int = 30000,
        min_ms: int = 400,
    ):
        self.vad = webrtcvad.Vad(vad_level)
        # 話し始めの判定に使う、直近 300ms の「声かどうか」
        self.start_ring: deque[bool] = deque(maxlen=10)
        # NOTE: 話し始めと判定する頃には、頭の子音（「ク」のような息だけの音）が過ぎていることがある。
        #       判定より前の音も取り込んで、「ロード」のように頭が欠けるのを防ぐ
        self.preroll: deque[bytes] = deque(maxlen=preroll_ms // FRAME_MS)
        self.end_ring: deque[bool] = deque(maxlen=end_silence_ms // FRAME_MS)
        self.max_frames = max_ms // FRAME_MS
        self.min_frames = min_ms // FRAME_MS
        self.reset()

    def reset(self) -> None:
        self.triggered = False
        self.started_at = 0.0
        self.voiced: list[bytes] = []
        self.start_ring.clear()
        self.preroll.clear()
        self.end_ring.clear()

    def push(self, frame: bytes) -> Utterance | None:
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        if not self.triggered:
            self.preroll.append(frame)
            self.start_ring.append(is_speech)
            if sum(self.start_ring) >= 0.8 * self.start_ring.maxlen:
                self.triggered = True
                self.started_at = time.monotonic()
                self.voiced = list(self.preroll)
                self.start_ring.clear()
                self.preroll.clear()
            return None

        self.voiced.append(frame)
        self.end_ring.append(is_speech)
        silent = len(self.end_ring) == self.end_ring.maxlen and sum(self.end_ring) <= 0.1 * self.end_ring.maxlen
        if not silent and len(self.voiced) < self.max_frames:
            return None

        frames, started_at = self.voiced, self.started_at
        self.reset()
        if len(frames) < self.min_frames:
            return None
        return Utterance(b"".join(frames), started_at)


class Transcriber:
    def __init__(self, model: str):
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self.model = model

    def __call__(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language="ja",
            condition_on_previous_text=False,
            verbose=None,
        )
        return result.get("text", "").strip()


_kakasi = pykakasi.kakasi()
_O_ROW = set("オコソトノホモヨロゴゾドボポョォ")
_SMALL_KANA = str.maketrans("ァィゥェォャュョッヮ", "アイウエオヤユヨツワ")
_SEP = r"\s、,。.!！?？・"


def reading(text: str) -> str:
    """読みをカタカナにそろえ、聞き取りで揺れやすい部分を落とす。

    「クロード」「クロド」「グロード」「クロウド」はどれも「クロト」になる。
    「クラウド」（クラウト）や「クロス」は別の読みのまま残る。
    """
    kana = "".join(item["kana"] for item in _kakasi.convert(unicodedata.normalize("NFKC", text)))
    out: list[str] = []
    for ch in kana.lower():
        if ch in "ー〜～":  # 長音
            continue
        if ch in "ウオ" and out and out[-1] in _O_ROW:  # ロウ・ロオ → ロ
            continue
        out.append(ch)
    s = "".join(out).translate(_SMALL_KANA)
    # 濁点・半濁点を外す（グ → ク、ド → ト）
    return "".join(c for c in unicodedata.normalize("NFD", s) if c not in "゙゚")


class WakeMatcher:
    """「クロード、〜」のような呼びかけを判定し、呼びかけ語より後ろの文言を取り出す。

    呼びかけ語は読みで比べるので、「クロド」「グロード」「クロウド」のような聞き取りの揺れも拾う。
    """

    MAX_PREFIX = 4  # 呼びかけ語の前に置ける一言の長さ（「えっと」「あのー」など）
    PREFIX_SEARCH = MAX_PREFIX + 3  # 一言のあとに続く区切り（「、」や空白）のぶんを足した、呼びかけ語の開始位置の上限
    MAX_WORD = 8  # 呼びかけ語として調べる文字数の最小の上限（「Claude」の 6 文字が入る長さ）
    # NOTE: 漢字を含む並びは呼びかけ語として扱わない。「黒と白」「苦労とは」「玄人好み」も読みはクロトになるが、
    #       呼びかけではない。名前は文字起こしでカタカナか英字になるので、取りこぼしはほぼ無い
    KANJI = re.compile(r"[一-鿿々〆]")
    # 決まった前置き付きの呼びかけ（「ねえクロード」）は、これまでどおり直後に区切りが無くても受け付ける
    CALLER = re.compile(rf"^(?:ねえ|ねぇ|ヘイ|hey|おい)?[{_SEP}]*$", re.IGNORECASE)
    # NOTE: それ以外の前置き（ひらがなの一言）は、呼びかけ語の直後が区切りか「さん」「くん」のときだけ受け付ける。
    #       「さっきクロードに聞いたら」のように、会話の中で名前が出ただけの発話を拾わないため
    FILLER = re.compile(rf"^[ぁ-んー]{{1,{MAX_PREFIX}}}[{_SEP}]*$")
    AFTER_CALL = re.compile(rf"^(?:$|[{_SEP}]|さん|くん)")
    HONORIFIC = re.compile(rf"^(?:さん|くん)?[{_SEP}]*")

    def __init__(self, words: list[str]):
        self.targets = {reading(w) for w in words}
        # 長い呼びかけ語（--wake で指定したもの）も調べられるよう、表記の揺れで伸びるぶん（+2）を足す
        self.max_word = max(self.MAX_WORD, max(len(w) for w in words) + 2)

    def match(self, text: str) -> str | None:
        """呼びかけでなければ None、呼びかけだけなら空文字を返す。"""
        text = unicodedata.normalize("NFKC", text).strip()
        for i in range(min(len(text), self.PREFIX_SEARCH)):
            by_caller = bool(self.CALLER.match(text[:i]))
            if not by_caller and not self.FILLER.match(text[:i]):
                continue
            for j in range(i + 1, min(len(text), i + self.max_word) + 1):
                segment = text[i:j]
                if self.KANJI.search(segment):
                    break  # これより長い並びも漢字を含む
                if reading(segment) not in self.targets:
                    continue
                rest = text[j:]
                if by_caller or self.AFTER_CALL.match(rest):
                    return self.HONORIFIC.sub("", rest, count=1).strip()
        return None


def to_speech(text: str) -> str:
    """Markdown の記号など、読み上げに向かない部分を取り除き、長ければ冒頭だけにする。"""
    # [auto-review:done] のようなフック向けの印。[12:30] のような時刻や [text:x](url) のリンクは残す
    text = re.sub(r"\[[A-Za-z][A-Za-z0-9_-]*:[A-Za-z][A-Za-z0-9_-]*\](?!\()", "", text)
    text = re.sub(r"```.*?```", "（コードは画面に出しています）", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "リンク", text)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:#+|[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~>#]", "", text)
    text = "".join(c for c in text if unicodedata.category(c) != "So")  # 絵文字などの記号
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) <= MAX_SPEECH_CHARS:
        return text
    head = text[:MAX_SPEECH_CHARS]
    cut = max(head.rfind(c) for c in "。？！?!\n")
    head = head[:cut + 1] if cut > 0 else head + "……。"
    return head.rstrip() + "続きは画面に出しています。"


class ClaudeSession:
    """claude -p を stream-json で常駐させ、1 プロセスで会話を続ける。

    NOTE: 毎回 claude -p を起動すると 1 往復 5〜8 秒かかるが、常駐させると 2 回目以降は約 1 秒になる
          (--safe-mode・sonnet での実測)。
    """

    def __init__(
        self,
        workdir: Path,
        model: str | None,
        permission_mode: str,
        tools: str | None,
        safe_mode: bool,
        timeout: float,
    ):
        self.workdir = workdir
        self.model = model
        self.permission_mode = permission_mode
        self.tools = tools
        self.safe_mode = safe_mode
        self.timeout = timeout
        self.session_id: str | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.events: queue.Queue[dict | None] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=20)

    def _command(self, resume: str | None) -> list[str]:
        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", self.permission_mode,
            "--append-system-prompt", SYSTEM_PROMPT,
        ]
        if self.safe_mode:
            # 個人設定 (CLAUDE.md・フック・MCP・Skill) を読み込まない。軽く速いが、普段の Claude とは別物になる
            cmd.append("--safe-mode")
        if self.tools is not None:
            # NOTE: 空文字は「すべての組み込みツールを無効にする」の意味なので、未指定 (None) と区別する
            cmd += ["--tools", self.tools]
        if self.model:
            cmd += ["--model", self.model]
        if resume:
            cmd += ["--resume", resume]
        return cmd

    def start(self, resume: str | None = None) -> None:
        self.proc = subprocess.Popen(
            self._command(resume),
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            # NOTE: フック側で音声セッションを見分けるための目印。
            #       短いプロンプトを止める UserPromptSubmit フックなどは、これを見て素通しさせる
            env={**os.environ, "CLAUDE_TALK": "1"},
            # 打ち切るときに Bash ツールが起動した子プロセスごと止められるよう、プロセスグループを分ける
            start_new_session=True,
        )
        # 再起動前のプロセスのイベントが混ざらないよう、キューはプロセスごとに作り直す
        self.events = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self.proc, self.events), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()

    @staticmethod
    def _read_stdout(proc: subprocess.Popen[str], events: queue.Queue[dict | None]) -> None:
        for line in proc.stdout:
            try:
                events.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        events.put(None)

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        for line in proc.stderr:
            self.stderr_tail.append(line.rstrip())

    def ask(self, text: str) -> str:
        # NOTE: 送る前にプロセスが落ちていたときだけ、同じ会話で再起動して送り直す。
        #       送ったあとに落ちた場合は送り直さない（送信や書き込みが二重に実行されうるため）
        for _ in range(2):
            if self.proc is None or self.proc.poll() is not None:
                self.start(resume=self.session_id)
            try:
                message = {"type": "user", "message": {"role": "user", "content": text}}
                self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except BrokenPipeError:
                self.proc = None
                continue

            reply = self._wait_result()
            if reply is None:
                self.proc = None
                raise RuntimeError("claude が途中で終了しました: " + " / ".join(self.stderr_tail))
            return reply
        raise RuntimeError("claude を起動できませんでした: " + " / ".join(self.stderr_tail))

    def _wait_result(self) -> str | None:
        # NOTE: 上限は最後のイベントからの経過時間。ツールを使う長い作業でも、動いている間は打ち切らない
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError
            try:
                event = self.events.get(timeout=remaining)
            except queue.Empty:
                continue
            if event is None:
                return None
            deadline = time.monotonic() + self.timeout
            if event.get("session_id"):
                self.session_id = event["session_id"]
            if event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(str(event.get("result") or event.get("subtype")))
                return event.get("result") or ""

    def reset(self) -> None:
        self.close()
        self.session_id = None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            # Bash ツールが起動した子プロセスも残さないよう、プロセスグループごと止める
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.proc.pid, sig)
                    self.proc.wait(timeout=3)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.proc = None


class Speaker:
    def __init__(self, voice: str, rate: int | None, enabled: bool):
        self.voice = voice
        self.rate = rate
        self.enabled = enabled

    def say(self, text: str) -> None:
        if not self.enabled or not text:
            return
        cmd = ["say", "-v", self.voice, "-f", "-"]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        # 本文は標準入力で渡す (先頭が "-" の文がオプションとして解釈されないように)
        subprocess.run(cmd, input=text, text=True, check=False)

    def cue(self) -> None:
        if self.enabled and CUE_SOUND.exists():
            subprocess.Popen(["afplay", str(CUE_SOUND)])


class VoiceAssistant:
    def __init__(self, args: argparse.Namespace):
        self.transcribe = Transcriber(args.whisper_model)
        self.wake = WakeMatcher(args.wake or DEFAULT_WAKE_WORDS)
        self.claude = ClaudeSession(
            workdir=Path(args.workdir).expanduser(),
            model=args.model,
            permission_mode=args.permission_mode,
            tools=args.tools,
            safe_mode=args.safe_mode,
            timeout=args.timeout,
        )
        self.speaker = Speaker(args.voice, args.rate, not args.no_speak)
        self.followup_sec = args.followup
        self.debug = args.debug
        # 呼びかけなしで受け付ける発話の、話し始めの時刻の範囲
        self.followup_from = 0.0
        self.followup_until = 0.0

    def warm_up(self) -> None:
        log("準備中", "文字起こしモデルを読み込んでいます（初回はダウンロードに数分かかります）")
        self.transcribe(b"\0" * FRAME_BYTES * 20)
        self.claude.start()

    def handle(self, pcm: bytes, started_at: float) -> bool:
        """発話を 1 つ処理する。読み上げや効果音を出したときは True を返す。"""
        heard_at = time.monotonic()
        status_line.set_state(State.TRANSCRIBING)
        try:
            return self._handle(pcm, started_at, heard_at)
        finally:
            status_line.set_state(State.IDLE)

    def _say(self, text: str) -> None:
        status_line.set_state(State.SPEAKING)
        self.speaker.say(text)

    def _open_followup(self, since: float, seconds: float) -> None:
        self.followup_from = since
        self.followup_until = time.monotonic() + seconds
        status_line.open_followup(self.followup_until)

    def _in_followup(self, started_at: float) -> bool:
        # NOTE: 文字起こしが終わった時刻ではなく、話し始めた時刻で判定する。
        #       画面の「あと N 秒」のうちに話し始めた発話を、話し終わるのが遅れたせいで取りこぼさないように。
        #       started_at はフレームを「処理した」時刻。マイクから「取得した」時刻に変えるときは、
        #       followup_from も同じ基準にそろえること（そろえないと、下限で正しい続きを落とす）
        return self.followup_from <= started_at <= self.followup_until

    def _handle(self, pcm: bytes, started_at: float, heard_at: float) -> bool:
        if self.debug:
            log("発話を検出", f"{len(pcm) / 2 / SAMPLE_RATE:.1f} 秒。文字起こし中")
        text = self.transcribe(pcm)
        if not text or text in HALLUCINATIONS:
            if self.debug:
                log("無視（文字起こし結果なし）", text)
            return False

        command = self.wake.match(text)
        if command is None:
            if not self._in_followup(started_at):
                if self.debug:
                    log("無視（呼びかけなし）", text)
                return False
            command = text
        log("あなた", text)

        if command == "":
            # NOTE: 呼びかけ語だけのときは返事をしない。「クロード、」のあとの間で発話が区切られても、
            #       続きを話している最中に声を重ねず、その間に拾った音も捨てずに聞き続けるため。
            #       受け付けていることは最下行の「続けてどうぞ」で示す
            self._open_followup(since=heard_at, seconds=CALL_ONLY_WINDOW_SEC)
            if not status_line.enabled:
                log("受付中", f"{CALL_ONLY_WINDOW_SEC:.0f} 秒以内に続けて話してください")
            return False
        if any(p in command for p in RESET_PHRASES):
            self.claude.reset()
            self.claude.start()
            log("会話をリセットしました")
            self._say("新しい会話を始めます")
            return True

        self.speaker.cue()
        status_line.set_state(State.THINKING)
        try:
            reply = self.claude.ask(command)
        except TimeoutError:
            reply = "時間内に返事がありませんでした。途中まで実行された可能性があるので、画面を確認してください。"
        except RuntimeError as e:
            log("エラー", str(e))
            reply = "エラーが起きました。画面を確認してください。"
        log("Claude", reply)
        if reply.lstrip().startswith(HOOK_BLOCKED_PREFIX):
            self._say("フックに止められました。画面を確認してください。")
        else:
            self._say(to_speech(reply))
        self._open_followup(since=time.monotonic(), seconds=self.followup_sec)
        return True

    def close(self) -> None:
        self.claude.close()


def run_microphone(assistant: VoiceAssistant, segmenter: Segmenter, device: str | int | None) -> None:
    import sounddevice as sd

    frames: queue.Queue[bytes] = queue.Queue()

    def callback(indata, _frames, _time, _status) -> None:
        frames.put(bytes(indata))

    buffer = bytearray()
    peak_db = -math.inf
    meter_at = time.monotonic() + 3
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES, dtype="int16", channels=1, device=device, callback=callback
    ):
        log("待機中", f"{sd.query_devices(device, 'input')['name']} で「クロード、〜」と話しかけてください（Ctrl+C で終了）")
        while True:
            buffer += frames.get()
            while len(buffer) >= FRAME_BYTES:
                frame = bytes(buffer[:FRAME_BYTES])
                del buffer[:FRAME_BYTES]
                if assistant.debug:
                    # 3 秒ごとに入力の最大レベルを出す。話しても -40 dBFS 程度より上がらなければマイクに声が届いていない
                    peak_db = max(peak_db, peak_dbfs(frame))
                    if time.monotonic() >= meter_at:
                        log("入力レベル", f"{peak_db:.0f} dBFS")
                        peak_db = -math.inf
                        meter_at = time.monotonic() + 3
                status_line.update_level(frame)
                utterance = segmenter.push(frame)
                if utterance is None:
                    status_line.set_state(State.LISTENING if segmenter.triggered else State.IDLE)
                    continue
                if not assistant.handle(utterance.pcm, utterance.started_at):
                    # 音を出していなければ、処理中に拾った音（呼びかけの続きなど）はそのまま聞き続ける
                    continue
                # 考え中・読み上げ中に拾った音 (自分の読み上げ音声を含む) は捨てる
                buffer.clear()
                while not frames.empty():
                    frames.get_nowait()
                segmenter.reset()


def run_wav(assistant: VoiceAssistant, segmenter: Segmenter, path: str) -> None:
    """マイクの代わりに WAV ファイル (16kHz / モノラル / 16bit) を流す。動作確認用。"""
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            sys.exit("WAV は 16kHz / モノラル / 16bit にしてください（afconvert -f WAVE -d LEI16@16000 -c 1 で変換できます）")
        data = w.readframes(w.getnframes())
    # 末尾に、話し終わりの判定（--end-silence）より長い無音を足して、最後の発話の終わりを確定させる
    data += b"\0" * FRAME_BYTES * (segmenter.end_ring.maxlen + 10)
    for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
        utterance = segmenter.push(data[i:i + FRAME_BYTES])
        if utterance is not None:
            assistant.handle(utterance.pcm, utterance.started_at)


def bounded(low: float, high: float):
    """argparse の type に渡す、範囲を検査する数値の変換関数を返す。"""

    def parse(value: str) -> float:
        number = float(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"{low:g}〜{high:g} の範囲で指定してください（指定値: {value}）")
        return number

    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="マイクで話しかけると Claude が音声で答える")
    p.add_argument("--workdir", default=str(Path.cwd()), help="Claude Code を起動するディレクトリ（既定: 起動したディレクトリ）")
    p.add_argument("--model", help="Claude のモデル（既定: Claude Code の設定どおり）。速さ優先なら sonnet")
    p.add_argument(
        "--permission-mode",
        default="auto",
        # NOTE: Claude Code 2.1.314 時点の選択肢。打ち間違いを起動時に弾くために列挙している
        choices=["auto", "manual", "acceptEdits", "dontAsk", "plan", "bypassPermissions"],
        help="権限モード（既定: %(default)s）。承認が必要な操作は確認画面を出せないため自動で拒否される。"
        "bypassPermissions は聞き間違いがそのまま無確認で実行されるので勧めない",
    )
    p.add_argument("--tools", help="組み込みツールを絞る（例: Read,Grep,Glob）。MCP のツールは対象外。既定は絞らない")
    p.add_argument("--safe-mode", action="store_true", help="CLAUDE.md・フック・MCP・Skill を読み込まない（軽く速い）")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="文字起こしモデル（既定: %(default)s）")
    p.add_argument(
        "--wake",
        action="append",
        help="呼びかけ語。複数指定可（既定: クロード / claude）。読みで比べるので「クロド」「グロード」などの揺れも拾う",
    )
    p.add_argument(
        "--followup",
        type=bounded(0, 60),
        default=8.0,
        help="返答後、呼びかけなしで話し始められる秒数。0 で無効（既定: %(default)s）",
    )
    p.add_argument(
        "--end-silence",
        type=bounded(0.3, 5.0),
        default=1.2,
        help="この秒数黙ったら話し終わりとみなす（既定: %(default)s）。短くすると返事が早くなるが、"
        "「クロード、」のあとの間で区切られやすくなる",
    )
    p.add_argument("--voice", default="Kyoko", help="say の声（既定: %(default)s）")
    p.add_argument("--rate", type=int, help="読み上げ速度（1 分あたりの語数）")
    p.add_argument("--vad-level", type=int, default=2, choices=range(4), help="発話検出の厳しさ 0〜3（既定: %(default)s）")
    p.add_argument(
        "--timeout", type=float, default=600.0, help="最後の出力からこの秒数、何も返ってこなければ打ち切る（既定: %(default)s）"
    )
    p.add_argument("--device", help="マイクのデバイス番号または名前")
    p.add_argument("--list-devices", action="store_true", help="オーディオデバイスの一覧を表示して終了")
    p.add_argument("--wav", help="マイクの代わりに WAV ファイルを流す（動作確認用）")
    p.add_argument("--no-speak", action="store_true", help="読み上げずに画面表示だけにする")
    p.add_argument("--debug", action="store_true", help="呼びかけなしで無視した発話も表示する")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    assistant = VoiceAssistant(args)
    segmenter = Segmenter(args.vad_level, end_silence_ms=int(args.end_silence * 1000))
    try:
        assistant.warm_up()
        if args.wav:
            run_wav(assistant, segmenter, args.wav)
        else:
            # ログをファイルやパイプに流すときは、状態表示の制御文字を混ぜない
            if sys.stdout.isatty():
                status_line.start()
            run_microphone(assistant, segmenter, device)
    except KeyboardInterrupt:
        log("終了します")
    finally:
        status_line.stop()
        assistant.close()


if __name__ == "__main__":
    main()
