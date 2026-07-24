# -*- coding: utf-8 -*-
"""
workspace.py
Рабочая область в духе Adobe Animate: холст, инструменты (кисть/фигуры/ластик),
слои объектов сцены. Viktim "видит" все объекты этого workspace через Viktim.perceive().
"""

from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.properties import StringProperty
import random


class SceneObject:
    """Объект на сцене, который Viktim может воспринять (угроза/награда/нейтрально)."""
    def __init__(self, x, y, tag, radius=10):
        self.x, self.y, self.tag, self.radius = x, y, tag, radius


class Workspace(Widget):
    """
    Холст рисования. Инструменты:
      - 'brush'  : свободное рисование линий (как кисть Adobe Animate)
      - 'star'   : ставит объект-награду (звезда) под курсором
      - 'eraser' : ставит объект-угрозу (ластик) под курсором
      - 'select' : ничего не рисует, просто панорамирование сцены
    """
    tool = StringProperty("brush")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strokes = []       # список Line-инструкций (визуальные штрихи, декоративные)
        self.scene_objects = [] # список SceneObject, которые видит мозг Viktim
        self._current_line = None
        self.bind(size=self._redraw_bg, pos=self._redraw_bg)

    def _redraw_bg(self, *args):
        pass  # фон рисуется в kv/App через canvas.before

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        if self.tool == "brush":
            with self.canvas:
                Color(0.1, 0.1, 0.1, 1)
                self._current_line = Line(points=[touch.x, touch.y], width=2)
            self.strokes.append(self._current_line)
        elif self.tool == "star":
            self._add_scene_object(touch.x, touch.y, "star", (1, 0.85, 0.1, 1))
        elif self.tool == "eraser":
            self._add_scene_object(touch.x, touch.y, "eraser", (0.9, 0.15, 0.15, 1))
        return True

    def on_touch_move(self, touch):
        if self._current_line and self.tool == "brush":
            self._current_line.points += [touch.x, touch.y]
            return True
        return False

    def on_touch_up(self, touch):
        self._current_line = None
        return True

    def _add_scene_object(self, x, y, tag, color):
        obj = SceneObject(x, y, tag)
        self.scene_objects.append(obj)
        with self.canvas:
            Color(*color)
            r = 10
            Ellipse(pos=(x - r, y - r), size=(r * 2, r * 2))
        # ограничиваем количество объектов, чтобы сцена не разрасталась бесконечно
        if len(self.scene_objects) > 40:
            self.scene_objects.pop(0)

    def objects_as_dicts(self):
        return [{"x": o.x, "y": o.y, "tag": o.tag} for o in self.scene_objects]

    def clear(self):
        self.canvas.clear()
        self.strokes.clear()
        self.scene_objects.clear()
