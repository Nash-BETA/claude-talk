"""voice_claude の、マイク・Claude・音声合成・メニューバーの実物に依存しない部分のテスト。"""
import os
import time

import pytest

import voice_claude as vc


@pytest.fixture
def assistant(monkeypatch):
    # 文字起こしモデルや claude を起動せずに、判定と応答の流れだけを試す
    monkeypatch.setattr(vc, "Transcriber", lambda model: None)
    a = vc.VoiceAssistant(vc.parse_args([]))
    a.said, a.asked = [], []
    a.speaker.say = a.said.append
    a.speaker.cue = lambda: None
    a.claude.ask = lambda text: a.asked.append(text) or "了解です。"
    a.claude.start = lambda resume=None: None
    a.claude.reset = lambda: None
    return a


def hear(assistant, text, started_at=None):
    assistant.transcribe = lambda pcm: text
    return assistant.handle(b"\0" * 3200, time.monotonic() if started_at is None else started_at)


class TestWakeMatcher:
    matcher = vc.WakeMatcher(vc.DEFAULT_WAKE_WORDS)

    @pytest.mark.parametrize(
        ("text", "command"),
        [
            ("クロード、今何時？", "今何時?"),
            ("クロード", ""),
            # 聞き取りの揺れ
            ("クロド、天気は", "天気は"),
            ("グロード今日の予定", "今日の予定"),
            ("クロウド、メモして", "メモして"),
            ("くろーど、天気は", "天気は"),
            ("Claude, hello", "hello"),
            # 呼びかけの前の一言
            ("ねえクロード、天気は", "天気は"),
            ("あの、クロード、天気は", "天気は"),
            ("えっとクロードさん、教えて", "教えて"),
            ("クロード・リポジトリの略称で", "リポジトリの略称で"),
            # 決まった前置き（ねえ・おい など）のあとは、呼びかけ語の直後に区切りが無くても受け付ける仕様
            ("ねえクロードに聞いたら", "に聞いたら"),
        ],
    )
    def test_accepts_calls(self, text, command):
        assert self.matcher.match(text) == command

    @pytest.mark.parametrize(
        "text",
        [
            # 会話の中で名前が出ただけ
            "さっきクロードに聞いたら",
            "今日はクロード",
            "あのクロードが言ってた",
            # 読みが近い別の言葉
            "クラウドに上げておいて",
            "クロスチェックして",
            "今日はいい天気ですね",
            # 読みは「クロト」になるが漢字を含む
            "黒と白どっちがいい",
            "苦労とは思わない",
            "九郎と行った",
            "玄人好みの味",
            # 前置きが長すぎる（5 文字）
            "えっとねー、クロード、天気は",
        ],
    )
    def test_ignores_non_calls(self, text):
        assert self.matcher.match(text) is None

    def test_matches_long_wake_word(self):
        assert vc.WakeMatcher(["アシスタントちゃん"]).match("アシスタントちゃん、天気は") == "天気は"


class TestToSpeech:
    @pytest.mark.parametrize(
        ("text", "speech"),
        [
            ("ログの [12:30] の行で", "ログの [12:30] の行で"),
            ("詳細は [doc:v2](https://x.y) を見て", "詳細は doc:v2 を見て"),
            ("直しました。[auto-review:done]", "直しました。"),
            ("📝 Tips 候補です ✅", "Tips 候補です"),
        ],
    )
    def test_removes_unspeakable_parts(self, text, speech):
        assert vc.to_speech(text) == speech

    def test_cuts_long_reply_at_sentence_end(self):
        speech = vc.to_speech("これは質問ですか？" * 60)
        assert speech.endswith("ですか？続きは画面に出しています。")
        assert len(speech) <= vc.MAX_SPEECH_CHARS + len("続きは画面に出しています。")


class TestStatusLine:
    def test_rounds_up_remaining_followup_seconds(self):
        status = vc.StatusLine()
        status.open_followup(time.monotonic() + 0.4)
        assert "あと 1 秒" in status.text()

    def test_rejects_unknown_state(self):
        status = vc.StatusLine()
        status.state_since = ("listning", time.monotonic())
        with pytest.raises(ValueError):
            status.text()

    def test_fits_terminal_width(self, monkeypatch):
        monkeypatch.setattr(vc.shutil, "get_terminal_size", lambda fallback: os.terminal_size((20, 24)))
        fitted = vc.StatusLine._fit("🎙  続けてどうぞ（あと 6 秒）  ████")
        assert sum(vc.display_width(c) for c in fitted) <= 19

    @pytest.mark.parametrize(
        ("level", "followup", "text"),
        [
            (0.0, 0.0, "🎙 待機中 ▁"),
            (1.0, 0.4, "🎙 続けてどうぞ 1秒 █"),
        ],
    )
    def test_menu_text_format(self, level, followup, text):
        status = vc.StatusLine()
        status.level = level
        if followup:
            status.open_followup(time.monotonic() + followup)
        assert status.menu_text() == text

    @pytest.mark.parametrize("state", list(vc.State))
    def test_both_displays_handle_every_state(self, state):
        # 状態を足したときに、片方の表示だけ直し忘れていないか
        status = vc.StatusLine()
        status.set_state(state)
        assert status.text() and status.menu_text()

    def test_menu_text_rejects_unknown_state(self):
        status = vc.StatusLine()
        status.state_since = ("listning", time.monotonic())
        with pytest.raises(ValueError):
            status.menu_text()


class FakeStdin:
    """補助プロセスの標準入力の代わり。broken にすると、終了したプロセスへの書き込みと同じく失敗する。"""

    def __init__(self):
        self.lines, self.broken, self.closed = [], False, False

    def write(self, text):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if self.broken:
            raise BrokenPipeError
        self.lines.append(text)

    def flush(self):
        pass

    def close(self):
        self.closed = True
        if self.broken:
            raise BrokenPipeError


class FakeProc:
    """補助プロセスの代わり。hang にすると、標準入力を閉じても終了しないプロセスになる。"""

    def __init__(self):
        self.stdin, self.hang, self.killed = FakeStdin(), False, False

    def wait(self, timeout=None):
        if self.hang and not self.killed:
            raise vc.subprocess.TimeoutExpired("menubar.py", timeout)
        return 0

    def kill(self):
        self.killed = True


@pytest.fixture
def menubar(monkeypatch):
    # 本物のメニューバーの項目を出さないよう、補助プロセスを起動しない
    monkeypatch.setattr(vc.subprocess, "Popen", lambda *a, **kw: FakeProc())
    return vc.MenuBar()


class TestMenuBar:
    def test_sends_only_changes(self, menubar):
        for text in ["🎙 待機中 ▁", "🎙 待機中 ▁", "👂 聞き取り中 ▅"]:
            assert menubar.show(text) is True
        assert menubar.proc.stdin.lines == ["🎙 待機中 ▁\n", "👂 聞き取り中 ▅\n"]

    def test_reports_exited_helper(self, menubar):
        menubar.proc.stdin.broken = True
        assert menubar.show("🎙 待機中 ▁") is False

    def test_reports_send_after_close(self, menubar):
        menubar.close()
        assert menubar.show("🎙 待機中 ▁") is False

    def test_close_tolerates_exited_helper(self, menubar):
        menubar.proc.stdin.broken = True
        menubar.close()
        assert menubar.proc.killed is False

    def test_close_kills_helper_that_does_not_exit(self, menubar):
        menubar.proc.hang = True
        menubar.close()
        assert menubar.proc.killed is True


class TestDraw:
    def test_stops_sending_when_helper_exited(self, menubar, capsys):
        status = vc.StatusLine()
        status.menubar = menubar
        menubar.proc.stdin.broken = True
        status.draw()
        assert status.menubar is None and not status.visible
        assert "状態を表示できなくなりました" in capsys.readouterr().out

    def test_keeps_quiet_when_stopped_while_sending(self, menubar, capsys):
        status = vc.StatusLine()
        status.menubar = menubar

        # 送っている最中に終了処理が始まり、補助プロセスが閉じられた場合
        def write_during_shutdown(text):
            status.stop()
            menubar.close()
            raise ValueError("I/O operation on closed file")

        menubar.proc.stdin.write = write_during_shutdown
        status.draw()
        assert capsys.readouterr().out == ""


class TestSegmenter:
    SPEECH = b"\x01" * vc.FRAME_BYTES
    SILENCE = b"\0" * vc.FRAME_BYTES

    def feed(self, segmenter, frames):
        results = [segmenter.push(frame) for frame in frames]
        return [r for r in results if r is not None]

    def test_returns_utterance_with_preroll_and_start_time(self):
        segmenter = vc.Segmenter(vad_level=2)
        # webrtcvad の代わりに、フレームの中身で「声かどうか」を決める
        segmenter.vad = type("FakeVad", (), {"is_speech": lambda self, frame, rate: frame[0] != 0})()
        before = time.monotonic()
        utterances = self.feed(segmenter, [self.SILENCE] * 30 + [self.SPEECH] * 20 + [self.SILENCE] * 60)
        assert len(utterances) == 1
        # 判定前の 0.6 秒 (20 フレーム) + 判定後の声 12 フレーム + 話し終わりの判定までの 36 フレーム
        assert len(utterances[0].pcm) == 68 * vc.FRAME_BYTES
        assert utterances[0].started_at >= before


class TestFollowup:
    def test_call_only_does_not_speak_and_accepts_continuation(self, assistant):
        assert hear(assistant, "クロード") is False
        assert assistant.said == []
        assert hear(assistant, "明日の予定を教えて") is True
        assert assistant.asked == ["明日の予定を教えて"]

    def test_call_only_window_is_independent_of_followup_option(self, assistant):
        assistant.followup_sec = 0
        hear(assistant, "クロード")
        assert hear(assistant, "明日の予定を教えて") is True

    def test_reply_is_spoken(self, assistant):
        assert hear(assistant, "クロード、今何時？") is True
        assert assistant.asked == ["今何時?"]
        assert assistant.said == ["了解です。"]

    def test_judges_by_start_time(self, assistant):
        # 受付時間はすでに過ぎているが、受付時間のうちに話し始めた発話は受け付ける
        now = time.monotonic()
        assistant.followup_from, assistant.followup_until = now - 5, now - 1
        assert hear(assistant, "明日の予定を教えて", started_at=now - 2) is True

    @pytest.mark.parametrize("offset", [-6, -0.5])
    def test_ignores_speech_started_outside_followup(self, assistant, offset):
        now = time.monotonic()
        assistant.followup_from, assistant.followup_until = now - 5, now - 1
        assert hear(assistant, "明日の予定を教えて", started_at=now + offset) is False
        assert assistant.asked == []


class TestArgs:
    @pytest.mark.parametrize("argv", [["--end-silence", "-1"], ["--end-silence", "0"], ["--followup", "-3"]])
    def test_rejects_out_of_range_values(self, argv):
        with pytest.raises(SystemExit):
            vc.parse_args(argv)
