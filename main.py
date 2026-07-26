# -*- coding: utf-8 -*-
"""
main.py
Кроссплатформенное приложение (Windows/Linux/macOS/Android/iOS) на Kivy --
рабочее пространство в духе Adobe Animate/Flash, где пользователь играет роль
Аниматора, а Viktim -- полностью автономный живой стикмен со своим мозгом.

РЕВИЗИЯ: белый фон Stage (как в реальном Flash), Viktim -- тёмно-серый (не
чёрный). Полный тулбокс Flash (14+ инструментов), регулируемые размер/цвет
кисти (Алан мог их менять -- теперь и здесь можно), Stroke/Fill цвета,
Zoom/Hand через Scatter-трансформацию сцены. Ластик может физически повредить
тело Viktim (утрата конечности) -- порождает фантомную боль (см. viktim.py),
угасающую через корковый ремаппинг, а не по таймеру.

Запуск (десктоп): pip install -r requirements.txt && python main.py
Android: buildozer -v android debug
iOS: toolchain build python3 kivy && toolchain create AdobeBeckerApp .
"""

import math
import os
import json

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.graphics import Color, Line, Ellipse

from workspace import Workspace
from scene_objects import PALETTE
from viktim import Viktim
from timeline import Timeline
import world_physics as wp

BRAIN_STEPS_PER_FRAME = 8
MAX_VIKTIMS = 6
VIKTIM_COLOR = (0.28, 0.28, 0.28, 1)          # тёмно-серый (не чёрный)
VIKTIM_BLOCKED_COLOR = (0.7, 0.2, 0.2, 1)     # заморожен/спутан -- визуальный акцент
LIMB_PARTS = ("head", "arm_l", "arm_r", "leg_l", "leg_r")


class Toolbar(BoxLayout):
    """Строка 1: инструменты рисования Flash + рука аниматора + клон."""

    def __init__(self, app, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=38, **kwargs)
        self.app = app
        tools = [
            ("select", "Arrow"), ("subselect", "Subsel."), ("lasso_select", "Lasso"),
            ("pen", "Pen"), ("text", "Text"), ("line", "Line"),
            ("rect", "Rect"), ("circle", "Oval"), ("pencil", "Pencil"), ("brush", "Brush"),
            ("free_transform", "Transform"), ("ink_bottle", "Ink"), ("paint_bucket", "Bucket"),
            ("eyedropper", "Dropper"), ("eraser_tool", "Eraser"),
            ("hand", "✋Hand"), ("zoom_in", "Zoom+"), ("zoom_out", "Zoom-"),
            ("clone", "Клон"),
        ]
        for tool, label in tools:
            btn = Button(text=label, font_size=9)
            if tool == "zoom_in":
                btn.bind(on_release=lambda i: app.zoom(1.2))
            elif tool == "zoom_out":
                btn.bind(on_release=lambda i: app.zoom(1 / 1.2))
            else:
                btn.bind(on_release=lambda inst, t=tool: self.app.set_tool(t))
            self.add_widget(btn)


class ColorBar(BoxLayout):
    """Строка 2: Stroke/Fill цвета + размер кисти (Алан мог их менять -- и здесь можно)."""

    def __init__(self, app, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=34, **kwargs)
        self.app = app
        self.add_widget(Label(text="Stroke:", size_hint=(0.08, 1), font_size=10))
        for color in PALETTE:
            swatch = Button(background_normal="", background_color=color, text="", size_hint=(0.05, 1))
            swatch.bind(on_release=lambda inst, c=color: app.set_stroke_color(c))
            self.add_widget(swatch)
        self.add_widget(Label(text="Fill:", size_hint=(0.06, 1), font_size=10))
        for color in PALETTE:
            swatch = Button(background_normal="", background_color=color, text="", size_hint=(0.05, 1))
            swatch.bind(on_release=lambda inst, c=color: app.set_fill_color(c))
            self.add_widget(swatch)
        self.add_widget(Label(text="Size:", size_hint=(0.06, 1), font_size=10))
        self.size_slider = Slider(min=1, max=16, value=3, size_hint=(0.25, 1))
        self.size_slider.bind(value=lambda i, v: app.set_brush_size(v))
        self.add_widget(self.size_slider)


class ObjectToolbar(BoxLayout):
    """Строка 3: физические предметы сцены (никакой пометки хорошо/плохо) + история + файл."""

    def __init__(self, app, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=36, **kwargs)
        self.app = app
        objs = [
            ("anvil", "Наковальня"), ("shuriken", "Сюрикен"), ("lasso", "Лассо"),
            ("plasma_gun", "Плазм.пушка"), ("frame_cannon", "Пушка-кадр"), ("sword", "Меч"),
            ("paintbrush", "Кисть-предм."), ("eraser", "Ластик-предм."), ("pen", "Перо"),
            ("magnifying_glass", "Лупа"),
            ("timeline_frame", "Кадр таймл.*"), ("toolbar_fragment", "Обломок UI*"),
            ("ice", "Заморозка**"), ("portal", "Портал**"),
        ]
        for tool, label in objs:
            btn = Button(text=label, font_size=8)
            btn.bind(on_release=lambda inst, t=tool: self.app.set_tool(t))
            self.add_widget(btn)

        for label, cb in (("⎌Undo", lambda i: app.workspace.undo()),
                          ("⎌Redo", lambda i: app.workspace.redo()),
                          ("Очистить", lambda i: app.workspace.clear()),
                          ("💾Сцена", lambda i: app.save_scene()),
                          ("📂Сцена", lambda i: app.load_scene()),
                          ("🧠💾", lambda i: app.save_brain()),
                          ("🧠📂", lambda i: app.load_brain())):
            b = Button(text=label, font_size=9)
            b.bind(on_release=cb)
            self.add_widget(b)


class TimelineBar(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=30, **kwargs)
        self.app = app
        self.rec_btn = Button(text="⏺ REC", font_size=10, size_hint=(0.2, 1))
        self.rec_btn.bind(on_release=self._toggle_rec)
        self.add_widget(self.rec_btn)
        self.slider = Slider(min=0, max=1, value=1, size_hint=(0.55, 1))
        self.slider.bind(value=lambda i, v: app.scrub_timeline(v))
        self.add_widget(self.slider)
        live_btn = Button(text="▶ Live", font_size=10, size_hint=(0.15, 1))
        live_btn.bind(on_release=lambda i: app.resume_live())
        self.add_widget(live_btn)
        hud_btn = Button(text="HUD", font_size=10, size_hint=(0.1, 1))
        hud_btn.bind(on_release=lambda i: app.toggle_hud())
        self.add_widget(hud_btn)

    def _toggle_rec(self, *a):
        tl = self.app.active_timeline()
        if tl is None:
            return
        tl.recording = not tl.recording
        self.rec_btn.text = f"⏺ {'REC' if tl.recording else 'OFF'}"


class BrainHUD(Label):
    """Отладочная панель (нет в оригинальной анимации) -- скрыта по умолчанию."""

    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, None), height=0, halign="left", valign="middle",
                          font_size=10, color=(0, 0, 0, 1), **kwargs)
        self.visible = False
        self.bind(size=self._update_text_size)

    def _update_text_size(self, *a):
        self.text_size = self.size

    def set_visible(self, flag):
        self.visible = flag
        self.height = 62 if flag else 0
        if not flag:
            self.text = ""

    def refresh(self, out, viktim):
        if not self.visible or out is None:
            return
        nm = out["neuromodulators"]
        day_time = f"д.{out['day']} {out['hour']:05.2f}ч {'ночь' if out['is_night'] else 'день'}"
        axes = (
            f"DA={nm['dopamine']:.2f} 5HT={nm['serotonin']:.2f} NE={nm['norepinephrine']:.2f} "
            f"CORT={nm['cortisol']:.2f} MEL={nm['melatonin']:.2f} ADO={nm['adenosine']:.2f} "
            f"OXT={nm['oxytocin']:.2f} AVP={nm['vasopressin']:.2f} ACh={nm['acetylcholine']:.2f} "
            f"END={nm['endorphin']:.2f} TEST={nm['testosterone']:.2f} GABA={nm['gaba']:.2f}"
        )
        held = " [держит предмет]" if viktim.held_object is not None else ""
        missing = [p for p in LIMB_PARTS if not viktim.limbs[p]]
        limb_note = f" | утрачено: {','.join(missing)} (remap={out['remap_count']})" if missing else ""
        mem_note = (f" | LTM: {out['long_term_memories']} следов, "
                    f"{out['consolidated_memories']} закреплено в коре")
        if out.get("memories_replayed_tonight"):
            mem_note += f", реигрывалось этой ночью: {out['memories_replayed_tonight']}"
        self.text = (
            f"[{viktim.state}{held}]{limb_note}{mem_note} {day_time} · cpg={out['critical_period_gain']:.2f} "
            f"· спайков={out['spikes_total']}\n{axes}"
        )


class SceneRenderer:
    def __init__(self, workspace):
        self.workspace = workspace

    def draw(self, viktims, timelines, cursor_pos, playhead_frame=None):
        wsp = self.workspace
        wsp.canvas.after.clear()
        with wsp.canvas.after:
            for tl in timelines:
                for i, ghost in enumerate(tl.onion_skin(3)):
                    Color(0.2, 0.2, 0.9, 0.08 + 0.08 * i)
                    for (x1, y1, x2, y2) in ghost.segs:
                        Line(points=[x1, y1, x2, y2], width=1.5)

            if playhead_frame is not None:
                Color(0.9, 0.1, 0.1, 0.7)
                for (x1, y1, x2, y2) in playhead_frame.segs:
                    Line(points=[x1, y1, x2, y2], width=2.2)
                return

            for v in viktims:
                segs, head_center, head_r = v.skeleton_points()
                color = VIKTIM_BLOCKED_COLOR if v.state in ("frozen", "entangled") else VIKTIM_COLOR
                Color(*color)
                for (x1, y1, x2, y2) in segs:
                    Line(points=[x1, y1, x2, y2], width=2.2)
                if head_center is not None:  # голова могла быть стёрта ластиком
                    Color(0.95, 0.85, 0.7, 1)
                    Ellipse(pos=(head_center[0] - head_r, head_center[1] - head_r),
                            size=(head_r * 2, head_r * 2))
                    Color(*color)
                    Line(circle=(head_center[0], head_center[1], head_r), width=1.5)

            if cursor_pos is not None:
                cx, cy = cursor_pos
                Color(0.1, 0.4, 0.9, 0.9)
                Line(points=[cx - 8, cy, cx + 8, cy], width=2)
                Line(points=[cx, cy - 8, cx, cy + 8], width=2)


class AdobeBeckerApp(App):
    def build(self):
        self.title = "Adobe Becker Style Studio -- Viktim"
        root = BoxLayout(orientation="vertical", padding=[0, 24, 0, 0])

        self.workspace = Workspace(size_hint=(1, 1))
        self.workspace.on_hand_down = self.hand_down
        self.workspace.on_hand_move = self.hand_move
        self.workspace.on_hand_up = self.hand_up
        self.workspace.on_clone_down = self.clone_at
        self.workspace.on_eraser_tool_hit = self.eraser_hit
        self.workspace.get_aim_target = self.nearest_viktim_pos

        self.hud = BrainHUD()
        root.add_widget(Toolbar(self))
        root.add_widget(ColorBar(self))
        root.add_widget(ObjectToolbar(self))

        # Scatter даёт Zoom (масштаб) и Hand (панорамирование) без переписывания
        # всей системы координат физики -- Kivy сам транслирует touch-координаты
        # в локальное пространство Workspace при трансформации Scatter.
        self.scatter = Scatter(do_rotation=False, do_scale=False, do_translation=False)
        self.scatter.add_widget(self.workspace)
        root.add_widget(self.scatter)

        root.add_widget(TimelineBar(self))
        root.add_widget(self.hud)

        self.viktims = [Viktim(x=300, y=300, seed=42)]
        self.timelines = [Timeline()]
        self._held_index = None
        self._scrub_fraction = None
        self._cursor_pos = None

        Window.bind(mouse_pos=self._on_mouse_pos)
        Window.bind(size=self._sync_workspace_size)
        self._sync_workspace_size(Window, Window.size)

        self.renderer = SceneRenderer(self.workspace)
        Clock.schedule_interval(self.tick, 1.0 / 30.0)
        return root

    def _sync_workspace_size(self, window, size):
        # Workspace живёт внутри Scatter, поэтому фиксируем ему реальный размер экрана,
        # иначе физический мир (координаты Viktim/объектов) не будет совпадать с окном
        w, h = size
        toolbars_h = 38 + 34 + 36 + 30  # высоты трёх верхних тулбаров + таймлайна
        self.workspace.size = (w, max(200, h - toolbars_h - 64))
        self.workspace.pos = (0, 0)

    def _on_mouse_pos(self, window, pos):
        local = self.scatter.to_widget(*pos)
        if self.workspace.collide_point(*local):
            self._cursor_pos = local

    # ---------------- инструменты ----------------
    def set_tool(self, tool):
        self.workspace.tool = tool
        self.scatter.do_translation = (tool == "hand")

    def set_stroke_color(self, color):
        self.workspace.stroke_color = color
        self.workspace.current_color = color

    def set_fill_color(self, color):
        self.workspace.fill_color = color

    def set_brush_size(self, size):
        self.workspace.brush_width = float(size)

    def zoom(self, factor):
        self.scatter.scale = max(0.4, min(3.0, self.scatter.scale * factor))

    def toggle_hud(self):
        self.hud.set_visible(not self.hud.visible)

    def nearest_viktim_pos(self):
        if not self.viktims:
            return None
        return (self.viktims[0].x, self.viktims[0].y)

    # ---------------- "Рука аниматора" ----------------
    def hand_down(self, x, y):
        best, best_d = None, 35.0
        for i, v in enumerate(self.viktims):
            d = math.hypot(v.x - x, v.y - y)
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            self._held_index = best
            self.viktims[best].grabbed = True
            self._cursor_pos = (x, y)
            return True
        return False

    def hand_move(self, x, y):
        self._cursor_pos = (x, y)
        if self._held_index is not None:
            v = self.viktims[self._held_index]
            v.x, v.y = x, y

    def hand_up(self, x, y):
        if self._held_index is not None:
            self.viktims[self._held_index].grabbed = False
            self._held_index = None

    # ---------------- ластик -> повреждение тела (фантомная боль, см. viktim.py) ----------------
    def eraser_hit(self, x, y):
        for v in self.viktims:
            segs, head_center, head_r = v.skeleton_points()
            if head_center is not None and math.hypot(head_center[0] - x, head_center[1] - y) < head_r + 14:
                v.damage_limb("head")
                return
            cursor_i = 0
            for part in ("arm_l", "arm_r", "leg_l", "leg_r"):
                if not v.limbs[part]:
                    continue
                for (x1, y1, x2, y2) in segs[cursor_i:cursor_i + 2]:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    if math.hypot(mx - x, my - y) < 16:
                        v.damage_limb(part)
                        return
                cursor_i += 2

    # ---------------- клонирование ----------------
    def clone_at(self, x, y):
        if len(self.viktims) >= MAX_VIKTIMS:
            return
        self.viktims.append(Viktim(x=x, y=y, seed=None))
        self.timelines.append(Timeline())

    # ---------------- таймлайн ----------------
    def active_timeline(self):
        return self.timelines[0] if self.timelines else None

    def scrub_timeline(self, fraction):
        self._scrub_fraction = fraction

    def resume_live(self):
        self._scrub_fraction = None
        for tl in self.timelines:
            tl.resume_live()

    # ---------------- сохранение/загрузка сцены ----------------
    def _scene_file_path(self):
        return os.path.join(self.user_data_dir, "scene.json")

    def save_scene(self):
        try:
            with open(self._scene_file_path(), "w", encoding="utf-8") as f:
                f.write(self.workspace.export_scene())
        except OSError:
            pass

    def load_scene(self):
        path = self._scene_file_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.workspace.import_scene(f.read())
        except (OSError, json.JSONDecodeError):
            pass

    # ---------------- ДОЛГОВРЕМЕННАЯ ПАМЯТЬ: сохранение/загрузка мозга между сессиями ----------------
    # Отдельно от "сцены" (рисунков/предметов) -- это буквально личность и весь
    # накопленный опыт Viktim (веса синапсов, гормональные базовые уровни,
    # гиппокампальные энграммы, возраст/критический период, утраченные
    # конечности). Без этого перезапуск приложения стирал бы всё, чему он
    # научился -- именно так реализована настоящая долговременная память.
    def _brain_file_path(self, idx=0):
        return os.path.join(self.user_data_dir, f"viktim_brain_{idx}.json")

    def save_brain(self):
        for i, v in enumerate(self.viktims):
            try:
                v.save_state(self._brain_file_path(i))
            except OSError:
                pass

    def load_brain(self):
        if not self.viktims:
            return
        path = self._brain_file_path(0)
        if not os.path.exists(path):
            return
        try:
            self.viktims[0].load_state(path)
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    # ---------------- игровой цикл ----------------
    def tick(self, dt_s):
        dt_ms = dt_s * 1000.0
        w, h = self.workspace.size

        # ВАЖНО: resolve_physics считает физику (и решает, что уничтожено), но
        # мы намеренно НЕ убираем уничтоженные объекты из списка ДО того, как
        # Viktim.update этот же кадр их "увидит" -- иначе объект, который только
        # что причинил боль, уже исчезает из восприятия ДО формирования следа
        # долговременной памяти, и энграмм кодируется с пустым сенсорным
        # паттерном (не с чем ассоциировать боль). Убираем их уже ПОСЛЕ.
        removed = wp.resolve_physics(self.workspace.scene_objects, self.viktims, dt_ms, (w, h))
        self.workspace._full_redraw()

        n_viktims = max(1, len(self.viktims))
        steps = max(2, BRAIN_STEPS_PER_FRAME // n_viktims)

        last_out = None
        for v, tl in zip(self.viktims, self.timelines):
            for _ in range(steps):
                last_out = v.update(dt_ms / steps, self.workspace.scene_objects,
                                     cursor_pos=self._cursor_pos, canvas_size=(w, h))
            v.x = max(20, min(max(21, w - 20), v.x))
            v.y = max(60, min(max(61, h - 20), v.y))
            tl.maybe_capture(dt_ms, v)

        if removed:
            removed_ids = {id(o) for o in removed}
            self.workspace.scene_objects = [o for o in self.workspace.scene_objects
                                             if id(o) not in removed_ids]

        playhead_frame = None
        if self._scrub_fraction is not None:
            tl0 = self.active_timeline()
            if tl0:
                playhead_frame = tl0.scrub_to(self._scrub_fraction)

        self.renderer.draw(self.viktims, self.timelines, self._cursor_pos, playhead_frame)
        if last_out is not None:
            self.hud.refresh(last_out, self.viktims[-1])


if __name__ == "__main__":
    AdobeBeckerApp().run()
