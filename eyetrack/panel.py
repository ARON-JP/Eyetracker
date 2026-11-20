"""Dark, compact settings panel for gaze tracking and overlay controls."""

from __future__ import annotations

import base64
import tkinter as tk
from tkinter import colorchooser, messagebox

import cv2

from .config import Config, list_profiles
from .imgenc import encode_png
from .tracker import Sample
from .win32 import VK_CODES, vk_code

BG = "#0b1020"
SIDEBAR = "#0a0e1b"
CARD = "#131a2d"
CARD_ALT = "#182139"
FG = "#edf0fa"
MUT = "#929bb4"
ACCENT = "#9277ff"
LINE = "#252e47"
FONT = ("Meiryo UI", 9)
FONT_B = ("Meiryo UI", 9, "bold")
FONT_H = ("Meiryo UI", 19, "bold")
PREVIEW_W, PREVIEW_H = 336, 252
HOTKEY_LABELS = (
    ("calibrate", "キャリブレーション"),
    ("reset_frontal", "正面リセット"),
    ("edge_calibration", "画面端を簡易補正"),
    ("toggle_overlay", "オーバーレイ切替"),
    ("toggle_preview", "プレビュー切替"),
    ("toggle_panel", "パネル表示切替"),
    ("quit", "終了"),
)


class Panel:
    def __init__(self, root: tk.Tk, app):
        self.root = root
        self.app = app
        self.cfg: Config = app.config
        root.title("GazeDot — Eye Tracking")
        root.configure(bg=BG)
        root.geometry("1140x780")
        root.minsize(1000, 680)
        root.resizable(True, True)
        self._photos = []
        self._last_status = None
        self._page_name = ""
        self._capturing_hotkey: str | None = None
        self._hotkey_buttons: dict[str, tk.Button] = {}

        shell = tk.Frame(root, bg=BG)
        shell.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(shell, bg=SIDEBAR, width=220, padx=18, pady=20)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="GazeDot", bg=SIDEBAR, fg=ACCENT, font=FONT_H).pack(anchor="w")
        tk.Label(
            self.sidebar, text="EYE TRACKING", bg=SIDEBAR, fg=MUT,
            font=("Meiryo UI", 8, "bold"),
        ).pack(anchor="w", pady=(1, 25))
        tk.Label(
            self.sidebar, text="設定", bg=SIDEBAR, fg="#66718d",
            font=("Meiryo UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 7))
        self.nav_buttons = {}
        for key, label in (
            ("tracking", "視線トラッキング"),
            ("overlay", "オーバーレイ"),
            ("stability", "描画の安定性"),
            ("hotkeys", "ショートカット"),
        ):
            button = tk.Button(
                self.sidebar, text=f"  {label}", anchor="w", command=lambda k=key: self._show_page(k),
                bg=SIDEBAR, fg=MUT, activebackground=CARD_ALT, activeforeground=FG,
                relief="flat", bd=0, padx=10, pady=11, font=FONT, cursor="hand2",
            )
            button.pack(fill="x", pady=2)
            self.nav_buttons[key] = button
        tk.Label(
            self.sidebar, text="BEAM INSPIRED\n独自の視線校正と姿勢補正",
            bg=SIDEBAR, fg="#67718a", font=("Meiryo UI", 8), justify="left",
        ).pack(side="bottom", anchor="w", pady=10)

        self.content = tk.Frame(shell, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)
        header = tk.Frame(self.content, bg=BG, padx=28, pady=18)
        header.pack(fill="x")
        self.lbl_page_title = tk.Label(header, text="", bg=BG, fg=FG, font=FONT_H)
        self.lbl_page_title.pack(anchor="w")
        self.lbl_page_subtitle = tk.Label(header, text="", bg=BG, fg=MUT, font=FONT)
        self.lbl_page_subtitle.pack(anchor="w", pady=(4, 0))

        status = tk.Frame(self.content, bg=CARD, padx=14, pady=10)
        status.pack(fill="x", padx=28, pady=(0, 10))
        self.lbl_face = tk.Label(status, text="● 顔: --", bg=CARD, fg=MUT, font=FONT_B)
        self.lbl_face.pack(side="left")
        self.lbl_fps = tk.Label(status, text="FPS --", bg=CARD, fg=MUT, font=FONT)
        self.lbl_fps.pack(side="left", padx=(14, 0))
        self.lbl_err = tk.Label(status, text="未キャリブレーション", bg=CARD, fg=MUT, font=FONT)
        self.lbl_err.pack(side="right")
        self.lbl_tracking = tk.Label(status, text="追跡待機中", bg=CARD, fg=MUT, font=FONT)
        self.lbl_tracking.pack(anchor="w", pady=(6, 0))

        body = tk.Frame(self.content, bg=BG)
        body.pack(fill="both", expand=True, padx=(28, 12), pady=(0, 20))
        self.scroll_canvas = tk.Canvas(body, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.page_host = tk.Frame(self.scroll_canvas, bg=BG)
        self._page_window = self.scroll_canvas.create_window(
            (0, 0), window=self.page_host, anchor="nw"
        )
        self.page_host.bind(
            "<Configure>",
            lambda _event: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            ),
        )
        self.scroll_canvas.bind(
            "<Configure>",
            lambda event: self.scroll_canvas.itemconfigure(
                self._page_window, width=event.width
            ),
        )
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._show_page("tracking")

    def _on_mousewheel(self, event) -> None:
        if self.root.winfo_exists():
            self.scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _show_page(self, name: str) -> None:
        pages = {
            "tracking": ("視線トラッキング", "カメラ・モニターを設定し、画面全体を校正します。"),
            "overlay": ("オーバーレイ", "透明な円形トレイルの外観をカスタマイズします。"),
            "stability": ("描画の安定性", "更新レートと微小な視線揺れへの反応を調整します。"),
            "hotkeys": ("ショートカット", "アプリを使いながら実行できるキーを割り当てます。"),
        }
        self._page_name = name
        title, subtitle = pages[name]
        self.lbl_page_title.configure(text=title)
        self.lbl_page_subtitle.configure(text=subtitle)
        for key, button in self.nav_buttons.items():
            selected = key == name
            button.configure(
                bg=CARD_ALT if selected else SIDEBAR,
                fg=ACCENT if selected else MUT,
            )
        for child in self.page_host.winfo_children():
            child.destroy()
        self._hotkey_buttons.clear()
        if name == "tracking":
            self._build_tracking()
        elif name == "overlay":
            self._build_overlay()
        elif name == "stability":
            self._build_stability()
        else:
            self._build_hotkeys()
        self.scroll_canvas.yview_moveto(0)

    def _card(self, parent, title: str, hint: str | None = None):
        frame = tk.Frame(parent, bg=CARD, padx=18, pady=15)
        frame.pack(fill="x", pady=(0, 12), padx=(0, 10))
        tk.Label(frame, text=title, bg=CARD, fg=FG, font=FONT_B).pack(anchor="w")
        if hint:
            tk.Label(
                frame, text=hint, bg=CARD, fg=MUT, font=FONT, wraplength=680,
                justify="left",
            ).pack(anchor="w", pady=(4, 10))
        return frame

    def _button(self, parent, text, cmd, primary=False, width=None):
        button = tk.Button(
            parent, text=text, command=cmd, bg=ACCENT if primary else CARD_ALT,
            fg="#ffffff" if primary else FG,
            activebackground="#7860db" if primary else LINE,
            activeforeground=FG, relief="flat", bd=0, padx=12, pady=8,
            font=FONT_B, cursor="hand2",
        )
        if width:
            button.configure(width=width)
        return button

    def _slider(
        self, parent, label, lo, hi, var, cmd, value_format=None, resolution=1,
    ):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=5)
        tk.Label(row, text=label, bg=CARD, fg=FG, font=FONT).pack(anchor="w")
        controls = tk.Frame(row, bg=CARD)
        controls.pack(fill="x", pady=(2, 0))
        value = tk.Label(controls, text="", bg=CARD, fg=ACCENT, font=FONT_B, width=8, anchor="e")
        value.pack(side="right")
        fmt = value_format or (lambda x: str(int(float(x))))
        value.configure(text=fmt(var.get()))
        tk.Scale(
            controls, from_=lo, to=hi, orient="horizontal", variable=var,
            showvalue=0, resolution=resolution, bg=CARD, fg=FG,
            troughcolor=LINE, highlightthickness=0, bd=0, sliderrelief="flat",
            activebackground=ACCENT, length=430,
            command=lambda raw: (
                value.configure(text=fmt(raw)),
                cmd(float(raw)),
            ),
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        return value

    def _check(self, parent, label, var, cmd):
        return tk.Checkbutton(
            parent, text=label, variable=var, command=cmd, bg=CARD, fg=FG,
            selectcolor=CARD_ALT, activebackground=CARD, activeforeground=FG,
            highlightthickness=0, bd=0, font=FONT, cursor="hand2",
        )

    def _build_tracking(self) -> None:
        welcome = tk.Frame(self.page_host, bg="#2858bd", padx=18, pady=14)
        welcome.pack(fill="x", pady=(0, 12), padx=(0, 10))
        tk.Label(
            welcome, text="ⓘ  ここから始める", bg="#2858bd", fg="#ffffff",
            font=FONT_B,
        ).pack(anchor="w")
        tk.Label(
            welcome,
            text="カメラを画面中央・目線に近い高さへ固定し、表示モニターを選んで中央から画面端まで校正してください。"
                 "顔の位置や向きを変えた後は、正面リセットまたは画面端の簡易補正を使えます。",
            bg="#2858bd", fg="#eef3ff", font=FONT, wraplength=680, justify="left",
        ).pack(anchor="w", pady=(7, 0))
        mon = self._card(self.page_host, "ディスプレイ", "視線を表示し、校正するモニターを選びます。")
        self._mon_labels = [f"{i + 1}: {m.label()}" for i, m in enumerate(self.app.monitors)]
        self.var_monitor = tk.StringVar(
            value=self._mon_labels[min(self.cfg.monitor, len(self._mon_labels) - 1)]
        )
        menu = tk.OptionMenu(mon, self.var_monitor, *self._mon_labels, command=self._on_monitor)
        menu.configure(bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
                       highlightthickness=0, bd=0, font=FONT, anchor="w")
        menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        menu.pack(fill="x")

        cal = self._card(
            self.page_host, "キャリブレーション",
            "中央から始めて、端に近い点を順に見つめます。座る位置を保ち、各点をクリックして開始してください。",
        )
        row = tk.Frame(cal, bg=CARD)
        row.pack(fill="x")
        self.cfg.calibration_points = 9 if self.cfg.calibration_points == 9 else 5
        self.var_points = tk.StringVar(value=f"{self.cfg.calibration_points}点")
        point_menu = tk.OptionMenu(
            row, self.var_points, "5点", "9点", command=self._on_calibration_points
        )
        point_menu.configure(bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
                             highlightthickness=0, bd=0, font=FONT)
        point_menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        point_menu.pack(side="left", padx=(0, 8))
        self._button(
            row, f"キャリブレーション開始 ({self.cfg.hotkeys.get('calibrate', 'F9')})",
            lambda: self.app.start_calibration(int(self.var_points.get().replace("点", ""))),
            primary=True,
        ).pack(side="left")
        self._button(row, "精度チェック", self.app.start_validation).pack(side="left", padx=8)
        actions = tk.Frame(cal, bg=CARD)
        actions.pack(fill="x", pady=(10, 0))
        reset_key = self.cfg.hotkeys.get("reset_frontal", "ALT+C")
        edge_key = self.cfg.hotkeys.get("edge_calibration", "F5")
        self._button(
            actions, f"正面リセット ({reset_key})",
            self.app.reset_frontal_baseline,
        ).pack(side="left")
        self._button(actions, f"画面端を簡易補正 ({edge_key})", self.app.start_edge_calibration).pack(
            side="left", padx=8
        )

        face = self._card(
            self.page_host, "顔の向き", "姿勢の変化が大きいときは一時停止します。座り直した後に正面リセットを使ってください。",
        )
        self.var_max_yaw = tk.DoubleVar(value=self.cfg.filter.max_yaw)
        self.var_max_pitch = tk.DoubleVar(value=self.cfg.filter.max_pitch)
        self.var_max_roll = tk.DoubleVar(value=self.cfg.filter.max_roll)
        self.var_head_smooth = tk.DoubleVar(value=self.cfg.filter.head_responsiveness)
        self.var_preview = tk.BooleanVar(value=self.cfg.show_preview)
        self._slider(face, "左右の顔向き許容", 10, 75, self.var_max_yaw, self._on_max_yaw, lambda x: f"{int(float(x))}°")
        self._slider(face, "上下の顔向き許容", 10, 60, self.var_max_pitch, self._on_max_pitch, lambda x: f"{int(float(x))}°")
        self._slider(face, "顔の傾き許容", 10, 60, self.var_max_roll, self._on_max_roll, lambda x: f"{int(float(x))}°")
        self._slider(
            face, "頭部トラッキングのスムージング", 0, 100,
            self.var_head_smooth, self._on_head_smooth,
            lambda x: f"{int(float(x))}",
        )

        self._build_camera_card()

        profile = self._card(self.page_host, "キャリブレーションプロファイル")
        names = sorted(set(list_profiles()) | {self.cfg.profile})
        self.var_profile = tk.StringVar(value=self.cfg.profile)
        self.om_profile = tk.OptionMenu(profile, self.var_profile, *names)
        self.om_profile.configure(bg=CARD_ALT, fg=FG, activebackground=LINE,
                                  activeforeground=FG, highlightthickness=0,
                                  bd=0, font=FONT, width=14, anchor="w")
        self.om_profile["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        self.om_profile.pack(side="left")
        self._button(profile, "読込", self._load_selected).pack(side="left", padx=8)
        self._button(profile, "保存", self._save_current).pack(side="left")

        self._check(profile, "カメラプレビュー (F8)", self.var_preview, self._on_preview).pack(
            side="right"
        )
        self.canvas = tk.Canvas(
            profile, width=PREVIEW_W, height=PREVIEW_H, bg="#080c16",
            highlightthickness=1, highlightbackground=LINE, bd=0,
        )
        self.canvas.pack(anchor="w", pady=(12, 0))
        self._img_item = self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2)
        self.canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H - 14, text="プレビュー: F8",
            fill="#68738d", font=FONT,
        )

    def _build_camera_card(self) -> None:
        camera = self._card(
            self.page_host, "入力 / カメラ配置",
            "カメラをモニター上部中央、目線に近い高さで固定します。位置・距離を変えたら正面リセット後に再校正してください。",
        )
        row = tk.Frame(camera, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text="カメラ", bg=CARD, fg=FG, font=FONT).pack(side="left")
        self.var_camera = tk.StringVar(value=str(self.cfg.camera.index))
        camera_menu = tk.OptionMenu(
            row, self.var_camera, *(str(i) for i in range(8))
        )
        camera_menu.configure(
            bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
            highlightthickness=0, bd=0, font=FONT, width=8,
        )
        camera_menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        camera_menu.pack(side="left", padx=(8, 18))

        self.var_resolution = tk.StringVar(
            value=f"{self.cfg.camera.width}x{self.cfg.camera.height}"
        )
        resolution_menu = tk.OptionMenu(
            row, self.var_resolution, "320x240", "640x480", "1280x720", "1920x1080"
        )
        resolution_menu.configure(
            bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
            highlightthickness=0, bd=0, font=FONT,
        )
        resolution_menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        resolution_menu.pack(side="left", padx=(0, 18))
        self.var_camera_fps = tk.StringVar(value=str(self.cfg.camera.fps))
        fps_menu = tk.OptionMenu(row, self.var_camera_fps, "15", "30", "60")
        fps_menu.configure(
            bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
            highlightthickness=0, bd=0, font=FONT,
        )
        fps_menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        fps_menu.pack(side="left")
        tk.Label(row, text="FPS", bg=CARD, fg=MUT, font=FONT).pack(side="left", padx=5)

        options = tk.Frame(camera, bg=CARD)
        options.pack(fill="x", pady=(10, 0))
        self.var_mirror = tk.BooleanVar(value=self.cfg.camera.mirror)
        self._check(options, "ミラー表示", self.var_mirror, lambda: None).pack(side="left")
        tk.Label(options, text="追跡する目", bg=CARD, fg=FG, font=FONT).pack(
            side="left", padx=(18, 5)
        )
        eye_modes = {"両目": "both", "左目": "left", "右目": "right"}
        self._eye_mode_values = {label: mode for label, mode in eye_modes.items()}
        selected_eye = next(
            (label for label, mode in eye_modes.items() if mode == self.cfg.camera.eye_mode),
            "両目",
        )
        self.var_eye_mode = tk.StringVar(value=selected_eye)
        eye_menu = tk.OptionMenu(options, self.var_eye_mode, *eye_modes)
        eye_menu.configure(
            bg=CARD_ALT, fg=FG, activebackground=LINE, activeforeground=FG,
            highlightthickness=0, bd=0, font=FONT,
        )
        eye_menu["menu"].configure(bg=CARD_ALT, fg=FG, activebackground=ACCENT)
        eye_menu.pack(side="left")
        self._check(options, "カメラプレビュー (F8)", self.var_preview, self._on_preview).pack(
            side="right"
        )
        tk.Label(
            camera,
            text="カメラ共有方式はWindowsドライバーに任せて自動選択します。",
            bg=CARD, fg=MUT, font=FONT,
        ).pack(anchor="w", pady=(8, 0))

        diagram = tk.Canvas(
            camera, width=480, height=142, bg="#505050",
            highlightthickness=0, bd=0,
        )
        diagram.pack(fill="x", pady=(12, 8))
        diagram.create_rectangle(46, 28, 434, 124, outline="#d0d0d0", width=2)
        diagram.create_text(240, 76, text="モニター", fill="#ededed", font=FONT_B)
        diagram.create_line(240, 4, 240, 24, fill=ACCENT, width=4, arrow="last")
        diagram.create_oval(231, 2, 249, 20, fill=ACCENT, outline="")
        diagram.create_text(
            240, 18, text="カメラを上部中央・目線の高さに設置",
            fill="#ffffff", font=FONT,
        )
        self._button(
            camera, "カメラ設定を適用", self._apply_camera_settings, primary=True
        ).pack(anchor="w", pady=(2, 0))

    def _build_overlay(self) -> None:
        appearance = self._card(
            self.page_host, "外観", "透明な内部と紫色のアンチエイリアス輪郭。円が重なった部分の内側の輪郭とグローは描画しません。",
        )
        row = tk.Frame(appearance, bg=CARD)
        row.pack(fill="x", pady=(0, 5))
        tk.Label(row, text="色 (hex)", bg=CARD, fg=FG, font=FONT).pack(side="left")
        self.var_color = tk.StringVar(value=self.cfg.overlay.color.upper())
        color_entry = tk.Entry(
            row, textvariable=self.var_color, width=9, bg=CARD_ALT, fg=FG,
            insertbackground=FG, relief="flat", justify="center", font=FONT,
        )
        color_entry.pack(side="left", padx=8)
        color_entry.bind("<Return>", self._validate_color_entry)
        color_entry.bind("<FocusOut>", self._validate_color_entry)
        self.btn_color = self._button(row, "色を選ぶ", self._pick_color)
        self.btn_color.configure(bg=self.cfg.overlay.color)
        self.btn_color.pack(side="left")
        self.var_overlay = tk.BooleanVar(value=self.cfg.show_overlay)
        self._check(row, "有効 (F10)", self.var_overlay, self._on_overlay).pack(side="left", padx=16)
        self.var_shadow = tk.BooleanVar(value=self.cfg.overlay.shadow)
        self._check(row, "ネオングロー", self.var_shadow, self._on_shadow).pack(side="left")

        self.var_size = tk.DoubleVar(
            value=min(100.0, max(10.0, self.cfg.overlay.radius / 1.2))
        )
        self.var_thickness = tk.DoubleVar(value=self.cfg.overlay.thickness)
        self.var_opacity = tk.DoubleVar(value=round(self.cfg.overlay.opacity * 100))
        self.var_glow = tk.DoubleVar(value=round(self.cfg.overlay.glow_intensity * 100))
        self.var_glow_radius = tk.DoubleVar(value=self.cfg.overlay.glow_radius)
        self._slider(appearance, "半径", 10, 100, self.var_size, self._on_size, lambda x: f"{int(float(x))}%")
        self._slider(appearance, "輪郭の太さ", 1, 8, self.var_thickness, self._on_thickness, lambda x: f"{int(float(x))} px")
        self._slider(appearance, "不透明度", 10, 100, self.var_opacity, self._on_opacity, lambda x: f"{int(float(x))}%")
        self._slider(appearance, "グロー強度", 0, 100, self.var_glow, self._on_glow, lambda x: f"{int(float(x))}%")
        self._slider(appearance, "グロー半径", 4, 36, self.var_glow_radius, self._on_glow_radius, lambda x: f"{int(float(x))} px")

        capture = self._card(
            self.page_host, "録画と配信",
            "キャプチャ除外は対応するWindowsアプリでのみ有効です。OBSでは、ソース側でキャプチャ方法を確認してください。",
        )
        self.var_capture_excluded = tk.BooleanVar(
            value=self.cfg.overlay.exclude_from_capture
        )
        self._check(
            capture, "画面キャプチャからオーバーレイを除外",
            self.var_capture_excluded, self._on_capture_excluded,
        ).pack(anchor="w")
        self._button(capture, "セットアップのヘルプ", self._show_capture_help).pack(
            anchor="w", pady=(8, 0)
        )

        trail = self._card(
            self.page_host, "モーショントレイル",
            "視線の速度に応じて円のサイズが変わり、古い円は徐々に消えます。円の重なりは外周をまとめて描画します。",
        )
        self.var_trail_duration = tk.DoubleVar(value=self.cfg.overlay.trail_duration_ms)
        self.var_trail_speed = tk.DoubleVar(value=round(self.cfg.overlay.trail_speed_scale * 100))
        self._slider(trail, "トレイルの持続時間", 80, 1200, self.var_trail_duration,
                     self._on_trail_duration, lambda x: f"{int(float(x))} ms", 20)
        self._slider(trail, "速度によるサイズ変化", 0, 150, self.var_trail_speed,
                     self._on_trail_speed, lambda x: f"{int(float(x))}%")
        self._button(trail, "テスト表示", self.app.test_overlay).pack(anchor="w", pady=(8, 0))

    def _build_stability(self) -> None:
        card = self._card(
            self.page_host, "更新と揺れの抑制",
            "トラッカーが取得した新しいフレームだけを処理します。更新頻度を下げすぎると追従が遅れるため、30 Hz前後を推奨します。",
        )
        self.var_update_hz = tk.DoubleVar(value=self.cfg.filter.update_hz)
        self.var_deadzone = tk.DoubleVar(value=self.cfg.filter.deadzone_px)
        self.var_smooth = tk.DoubleVar(value=self.cfg.filter.responsiveness)
        self._slider(card, "最大描画更新頻度", 10, 60, self.var_update_hz,
                     self._on_update_hz, lambda x: f"{int(float(x))} Hz")
        self._slider(card, "微小な動きを無視", 0, 12, self.var_deadzone,
                     self._on_deadzone, lambda x: f"{float(x):.1f} px", 0.1)
        self._slider(card, "視線トラッキングのスムージング", 0, 100, self.var_smooth,
                     self._on_smooth, lambda x: f"{int(float(x))}%")
        tk.Label(
            card,
            text="大きな視線移動は低遅延で追従し、小さな揺れだけをデッドゾーンで抑えます。",
            bg=CARD, fg=MUT, font=FONT, wraplength=680, justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _build_hotkeys(self) -> None:
        card = self._card(
            self.page_host, "割り当て可能なキー",
            "割り当てボタンを押してキーを入力します。Ctrl / Alt / Shiftとの組み合わせも使えます。重複する割り当てはできません。",
        )
        self._capturing_hotkey = None
        for action, label in HOTKEY_LABELS:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=CARD, fg=FG, font=FONT, anchor="w").pack(
                side="left", fill="x", expand=True
            )
            button = self._button(
                row, self.cfg.hotkeys.get(action, "--"),
                lambda key=action: self._capture_hotkey(key), width=10,
            )
            button.pack(side="right")
            button.bind("<KeyPress>", lambda event, key=action: self._on_hotkey_key(key, event))
            self._hotkey_buttons[action] = button

    def _on_monitor(self, _value=None) -> None:
        self.cfg.monitor = self._mon_labels.index(self.var_monitor.get())
        self.app.on_config_changed(reset_filter=True)

    def _on_calibration_points(self, value: str) -> None:
        self.cfg.calibration_points = int(value.replace("点", ""))
        self.cfg.save()

    def _capture_hotkey(self, action: str) -> None:
        if self._capturing_hotkey is not None:
            previous = self._capturing_hotkey
            self._hotkey_buttons[previous].configure(text=self.cfg.hotkeys.get(previous, "--"))
        self._capturing_hotkey = action
        button = self._hotkey_buttons[action]
        button.configure(text="入力中")
        button.focus_set()

    def _on_hotkey_key(self, action: str, event: tk.Event) -> str:
        if self._capturing_hotkey != action:
            return "break"
        special = {
            "Escape": "ESC", "space": "SPACE", "Tab": "TAB", "Home": "HOME",
            "End": "END", "Insert": "INSERT", "Delete": "DELETE", "Pause": "PAUSE",
            "Scroll_Lock": "SCROLL",
        }
        key = special.get(str(event.keysym), str(event.keysym).upper())
        ascii_key = len(key) == 1 and key.isascii() and key.isalnum()
        if not (ascii_key or key in VK_CODES) or vk_code(key) is None:
            messagebox.showwarning(
                "ショートカット",
                "F1〜F12、英数字、Esc などのキーを割り当ててください。",
                parent=self.root,
            )
            self._capturing_hotkey = None
            self._hotkey_buttons[action].configure(text=self.cfg.hotkeys.get(action, "--"))
            return "break"
        modifiers = []
        if event.state & 0x4:
            modifiers.append("CTRL")
        if event.state & 0x8:
            modifiers.append("ALT")
        if event.state & 0x1:
            modifiers.append("SHIFT")
        key_value = "+".join((*modifiers, key)) if modifiers else key
        conflict = next(
            (other for other, assigned in self.cfg.hotkeys.items()
             if other != action
             and self._normalize_hotkey(str(assigned)) == self._normalize_hotkey(key_value)),
            None,
        )
        if conflict is not None:
            messagebox.showwarning(
                "ショートカット",
                f"このキーは「{dict(HOTKEY_LABELS).get(conflict, conflict)}」に割り当て済みです。",
                parent=self.root,
            )
            self._capturing_hotkey = None
            self._hotkey_buttons[action].configure(text=self.cfg.hotkeys.get(action, "--"))
            return "break"
        self._capturing_hotkey = None
        self.app.set_hotkey(action, key_value)
        self._hotkey_buttons[action].configure(text=key_value)
        return "break"

    @staticmethod
    def _normalize_hotkey(value: str) -> str:
        tokens = [token.strip().upper() for token in value.split("+") if token.strip()]
        modifiers = [name for name in ("CTRL", "ALT", "SHIFT", "WIN") if name in tokens]
        main = next((token for token in reversed(tokens) if token not in modifiers), "")
        return "+".join((*modifiers, main))

    def _on_size(self, value: float) -> None:
        self.cfg.overlay.radius = value * 1.2
        self.app.on_config_changed()

    def _on_thickness(self, value: float) -> None:
        self.cfg.overlay.thickness = value
        self.app.on_config_changed()

    def _on_opacity(self, value: float) -> None:
        self.cfg.overlay.opacity = min(1.0, max(0.0, value / 100.0))
        self.app.on_config_changed()

    def _on_glow(self, value: float) -> None:
        self.cfg.overlay.glow_intensity = min(1.0, max(0.0, value / 100.0))
        self.app.on_config_changed()

    def _on_glow_radius(self, value: float) -> None:
        self.cfg.overlay.glow_radius = value
        self.app.on_config_changed()

    def _on_trail_duration(self, value: float) -> None:
        self.cfg.overlay.trail_duration_ms = int(value)
        self.app.on_config_changed()

    def _on_trail_speed(self, value: float) -> None:
        self.cfg.overlay.trail_speed_scale = value / 100.0
        self.app.on_config_changed()

    def _on_smooth(self, value: float) -> None:
        self.cfg.filter.responsiveness = int(value)
        self.app.on_config_changed()

    def _on_head_smooth(self, value: float) -> None:
        self.cfg.filter.head_responsiveness = int(value)
        self.app.on_config_changed()

    def _on_deadzone(self, value: float) -> None:
        self.cfg.filter.deadzone_px = value
        self.app.on_config_changed()

    def _on_update_hz(self, value: float) -> None:
        self.cfg.filter.update_hz = int(value)
        self.app.on_config_changed()

    def _on_max_yaw(self, value: float) -> None:
        self.cfg.filter.max_yaw = value
        self.app.on_config_changed()

    def _on_max_pitch(self, value: float) -> None:
        self.cfg.filter.max_pitch = value
        self.app.on_config_changed()

    def _on_max_roll(self, value: float) -> None:
        self.cfg.filter.max_roll = value
        self.app.on_config_changed()

    def _on_shadow(self) -> None:
        self.cfg.overlay.shadow = bool(self.var_shadow.get())
        self.app.on_config_changed()

    def _on_overlay(self) -> None:
        self.cfg.show_overlay = bool(self.var_overlay.get())
        self.app.on_config_changed()

    def _on_preview(self) -> None:
        self.cfg.show_preview = bool(self.var_preview.get())
        self.app.on_config_changed()

    def _apply_camera_settings(self) -> None:
        try:
            width, height = (
                int(value) for value in self.var_resolution.get().split("x", 1)
            )
            self.app.apply_camera_settings(
                index=int(self.var_camera.get()),
                width=width,
                height=height,
                fps=int(self.var_camera_fps.get()),
                mirror=bool(self.var_mirror.get()),
                eye_mode=self._eye_mode_values[self.var_eye_mode.get()],
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            messagebox.showerror("カメラ設定", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "カメラ設定",
            "カメラ設定を適用しました。追跡する目やカメラを変更した場合は、再キャリブレーションしてください。",
            parent=self.root,
        )

    def _on_capture_excluded(self) -> None:
        enabled = bool(self.var_capture_excluded.get())
        self.cfg.overlay.exclude_from_capture = enabled
        try:
            self.app.on_config_changed()
        except RuntimeError as exc:
            self.cfg.overlay.exclude_from_capture = not enabled
            self.var_capture_excluded.set(not enabled)
            messagebox.showerror("画面キャプチャ", str(exc), parent=self.root)

    def _show_capture_help(self) -> None:
        messagebox.showinfo(
            "セットアップガイド",
            "カメラをモニター上部中央、目線に近い高さに固定してください。\n\n"
            "1. カメラ映像で顔と両目が明るく見えることを確認\n"
            "2. カメラ・モニターを選び、設定を適用\n"
            "3. 正面を向いてキャリブレーション\n"
            "4. 頭の位置が変わったら正面リセット\n\n"
            "画面キャプチャ除外はWindowsのキャプチャ方式によって非対応の場合があります。",
            parent=self.root,
        )

    def _pick_color(self) -> None:
        _rgb, color = colorchooser.askcolor(
            color=self.cfg.overlay.color, parent=self.root, title="インジケーターの色"
        )
        if color:
            self.cfg.overlay.color = color
            self.var_color.set(color.upper())
            self.btn_color.configure(bg=color)
            self.app.on_config_changed()

    def _validate_color_entry(self, _event=None) -> str:
        value = self.var_color.get().strip()
        if not value.startswith("#"):
            value = f"#{value}"
        digits = value[1:]
        if len(digits) not in (3, 6) or any(
            char not in "0123456789abcdefABCDEF" for char in digits
        ):
            messagebox.showwarning(
                "色の設定", "色は #RGB または #RRGGBB の形式で入力してください。",
                parent=self.root,
            )
            self.var_color.set(self.cfg.overlay.color.upper())
            return "break"
        self.cfg.overlay.color = value.upper()
        self.btn_color.configure(bg=value)
        self.app.on_config_changed()
        return "break"

    def _load_selected(self) -> None:
        name = self.var_profile.get()
        if self.app.load_profile(name):
            self.cfg.profile = name
            self.cfg.save()
            self._last_status = None
        else:
            messagebox.showwarning("GazeDot", f"プロファイル '{name}' を読み込めません", parent=self.root)

    def _save_current(self) -> None:
        name = self.var_profile.get().strip() or "default"
        self.cfg.profile = name
        if self.app.save_profile(name):
            names = sorted(set(list_profiles()) | {name})
            menu = self.om_profile["menu"]
            menu.delete(0, "end")
            for profile in names:
                menu.add_command(label=profile, command=lambda value=profile: self.var_profile.set(value))
            self.var_profile.set(name)
        else:
            messagebox.showwarning("GazeDot", "保存するキャリブレーションがありません", parent=self.root)

    def sync_toggles(self) -> None:
        if hasattr(self, "var_overlay"):
            self.var_overlay.set(self.cfg.show_overlay)
        if hasattr(self, "var_preview"):
            self.var_preview.set(self.cfg.show_preview)

    def refresh(self, face: bool, fps: float, info: str, tracking: str) -> None:
        state = (face, round(fps, 1), info, tracking)
        if state == self._last_status:
            return
        self._last_status = state
        self.lbl_face.configure(
            text="● 顔: 検出" if face else "● 顔: 未検出",
            fg="#43d89c" if face else "#ff7581",
        )
        self.lbl_fps.configure(text=f"FPS {fps:.1f}" if fps else "FPS --")
        self.lbl_err.configure(text=info)
        self.lbl_tracking.configure(
            text=tracking, fg=ACCENT if tracking.startswith("視線:") else MUT
        )

    def update_preview(self, sample: Sample) -> None:
        if self._page_name != "tracking" or sample.frame is None:
            return
        try:
            if self.root.state() != "normal":
                return
        except tk.TclError:
            return
        small = cv2.resize(sample.frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        png = encode_png(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        photo = tk.PhotoImage(data=base64.b64encode(png))
        self._photos = [photo]
        self.canvas.itemconfigure(self._img_item, image=photo)
        self.canvas.coords(self._img_item, PREVIEW_W // 2, PREVIEW_H // 2)
        self.canvas.delete("lm")
        if sample.points is not None:
            for x, y in sample.points:
                px, py = float(x) * PREVIEW_W, float(y) * PREVIEW_H
                self.canvas.create_oval(
                    px - 2, py - 2, px + 2, py + 2, outline="#73e8ff", tags="lm"
                )

    def toggle(self) -> None:
        try:
            if self.root.state() == "iconic":
                self.root.deiconify()
                self.root.lift()
                self.root.focus_force()
            else:
                self.root.iconify()
        except tk.TclError:
            pass
