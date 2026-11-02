"""claude-talk の状態をメニューバーに出す補助プロセス。voice_claude.py から起動される。

標準入力から 1 行ずつ表示する文字列を受け取り、メニューバーの項目に出す。
標準入力が閉じたら（本体が終了したら）終了する。
"""
import signal
import sys
import threading

try:
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSStatusBar, NSVariableStatusItemLength
    from PyObjCTools import AppHelper
except ImportError:
    sys.exit("メニューバーの表示には pyobjc-framework-Cocoa が必要です（uv sync で入ります）")


def main() -> None:
    # Ctrl+C は本体が受ける。本体は終了するときに標準入力を閉じて、こちらに終了を伝える
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    app = NSApplication.sharedApplication()
    # Dock にアイコンを出さず、メニューバーの項目だけにする
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    # ⌘ を押しながらドラッグして動かした位置を覚えさせる。
    # NOTE: bundle を持たない Python のプロセスで、次回の起動まで位置が保存されるかは確かめていない
    item.setAutosaveName_("claude-talk")
    button = item.button()
    button.setToolTip_("claude-talk")
    button.setTitle_("🎙 準備中")

    def read_stdin() -> None:
        # NOTE: 読み取りが例外で終わっても必ず終了する。SIGINT を無視しているので、残ると kill でしか消せない
        try:
            # AppKit の部品はメインスレッドからしか触れないので、書き換えはメインスレッドに頼む
            for line in sys.stdin:
                AppHelper.callAfter(button.setTitle_, line.rstrip("\n"))
        finally:
            AppHelper.callAfter(app.terminate_, None)

    threading.Thread(target=read_stdin, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()
