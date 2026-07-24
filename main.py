# -*- coding: utf-8 -*-
"""
main.py
Кроссплатформенное приложение (Windows / Linux / macOS / Android / iOS) на Kivy,
в стиле рабочего пространства Adobe Animate, со стикменом Viktim (Alan Becker style),
управляемым живой нейросетью из brain.py + viktim.py.

Запуск на десктопе:
    pip install -r requirements.txt
    python main.py

Сборка под Android:
    pip install buildozer
    buildozer init          # создаст buildozer.spec (пример уже приложен)
    buildozer -v android debug

Сборка под iOS:
    pip install kivy-ios
    toolchain build python3 kivy
    toolchain create AdobeBeckerApp /path/to/adobe_becker_app
    # далее открыть сгенерированный Xcode-проект и собрать как обычно
"""

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse

from workspace import Workspace
from viktim import Viktim

BRAIN_STEPS_PER_FRAME = 4  # мозг тактируется чаще, чем кадры экрана (для корректной динамики AdEx)


class Toolbar(BoxLayout):
    def __init__(self, workspace, hud, **kwargs):
        super().__init__(orientation="horizontal", size_hint=(1, None), height=48, **kwargs)
        self.workspace = workspace
        self.hud = hud
        for tool, label in [("brush", "Кисть"), ("star", "Награда"),
                            ("eraser", "Угроза"), ("select", "Выбор")]:
            btn = Button(text=label)
            btn.bind(on_release=lambda inst, t=tool: setattr(self.workspace, "tool", t))
            self.add_widget(btn)
        clear_btn = Button(text="Очистить сцену")
        clear_btn.bind(on_release=lambda inst: self.workspace.clear())
        self.add_widget(clear_btn)


class BrainHUD(Label):
    """Небольшая панель состояния мозга/гормонов Viktim -- отладочный вид, как в редакторах Adobe."""
    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, None), height=28, halign="left", valign="middle", **kwargs)
        self.bind(size=self._update_text_size)

    def _update_text_size(self, *a):
        self.text_size = self.size

    def refresh(self, out, state):
        self.text = (
            f"[state={state}]  DA(дофамин)={out['dopamine']:.2f}  "
            f"5-HT(серотонин)={out['serotonin']:.2f}  NE(норадреналин)={out['norepinephrine']:.2f}  "
            f"CORT(кортизол)={out['cortisol']:.2f}  спайков/кадр={out['spikes_total']}"
        )


class VictimRenderer:
    """Отвечает за отрисовку скелета Viktim на канвасе workspace (отдельная canvas-группа)."""
    def __init__(self, workspace):
        self.workspace = workspace
        self.instructions = []

    def draw(self, viktim: Viktim):
        wsp = self.workspace
        wsp.canvas.after.clear()
        segs, head_center, head_r = viktim.skeleton_points()
        with wsp.canvas.after:
            Color(0.05, 0.05, 0.05, 1)
            for (x1, y1, x2, y2) in segs:
                Line(points=[x1, y1, x2, y2], width=2.2)
            Color(0.95, 0.85, 0.7, 1)
            Ellipse(pos=(head_center[0] - head_r, head_center[1] - head_r),
                    size=(head_r * 2, head_r * 2))
            Color(0.05, 0.05, 0.05, 1)
            Line(circle=(head_center[0], head_center[1], head_r), width=1.5)


class AdobeBeckerApp(App):
    def build(self):
        self.title = "Adobe Becker Style Studio -- Viktim"
        root = BoxLayout(orientation="vertical")

        self.workspace = Workspace(size_hint=(1, 1))
        self.hud = BrainHUD()
        toolbar = Toolbar(self.workspace, self.hud)

        root.add_widget(toolbar)
        root.add_widget(self.workspace)
        root.add_widget(self.hud)

        self.viktim = Viktim(x=300, y=300, seed=42)
        self.renderer = VictimRenderer(self.workspace)

        Clock.schedule_interval(self.tick, 1.0 / 30.0)
        return root

    def tick(self, dt_s):
        dt_ms = dt_s * 1000.0
        objects = self.workspace.objects_as_dicts()
        out = None
        # несколько тактов мозга на кадр рендера -- нейроны живут в своём, более быстром времени
        for _ in range(BRAIN_STEPS_PER_FRAME):
            out = self.viktim.update(dt_ms / BRAIN_STEPS_PER_FRAME, objects)

        # держим Viktim в пределах видимой области холста
        w, h = self.workspace.size
        self.viktim.x = max(20, min(w - 20, self.viktim.x))
        self.viktim.y = max(60, min(h - 20, self.viktim.y))

        self.renderer.draw(self.viktim)
        self.hud.refresh(out, self.viktim.state)


if __name__ == "__main__":
    AdobeBeckerApp().run()
