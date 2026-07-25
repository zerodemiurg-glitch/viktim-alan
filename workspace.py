# -*- coding: utf-8 -*-
"""
workspace.py
Рабочая область в духе Adobe Animate/Flash. Фон по умолчанию белый (как в
реальном Flash Stage). Полный набор инструментов из тулбокса Flash (14+):
arrow(select)/subselect/line/lasso_select/pen/text/oval/rectangle/pencil/
brush/free_transform/ink_bottle/paint_bucket/eyedropper/eraser_tool/hand/zoom,
плюс отдельные Stroke/Fill цвета и регулируемый размер кисти/карандаша --
именно так, как Алан мог менять размер и цвет своей кисти в реальном Flash.

Инструменты спавна физических предметов сцены (без ярлыков "плохо"/"хорошо",
см. scene_objects.py): наковальня, снаряд, лассо, кисть/ластик/перо/лупа-предметы,
меч, плазменная пушка, пушка из кадров, а также обломки самого интерфейса
(кадр таймлайна, фрагмент панели инструментов) -- Victim может их вырвать
и построить из них укрытие, как в оригинале.
"""

from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.core.text import Label as CoreLabel
from kivy.properties import StringProperty, NumericProperty
import json
import math

from scene_objects import PRESETS, COLORS, PALETTE, SceneObject, PROJECTILE_TAGS

SPAWNABLE = set(PRESETS.keys())
PROJECTILE_SPEED = 220.0


class Workspace(Widget):
    tool = StringProperty("brush")
    brush_width = NumericProperty(3.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_objects = []
        self.strokes = []          # (kind, data, stroke_color, width)
        self.text_items = []       # (x, y, text, color)
        self.selected_index = None
        self.stroke_color = PALETTE[0]
        self.fill_color = PALETTE[2]
        self.current_color = self.stroke_color  # обратная совместимость

        self._undo_stack = []
        self._redo_stack = []
        self._draw_start = None
        self._current_line = None
        self._pen_points = []   # для Pen-инструмента (полилиния по клику)

        # хуки, назначаемые app (main.py) -- Workspace не знает про список Viktim
        self.on_hand_down = None
        self.on_hand_move = None
        self.on_hand_up = None
        self.on_clone_down = None
        self.on_eraser_tool_hit = None   # fn(x, y) -- ластик касается тела Viktim -> повреждение конечности
        self.get_aim_target = None       # fn() -> (x, y) | None -- куда целить снаряд при спавне

        with self.canvas.before:
            Color(1, 1, 1, 1)  # фон Stage -- белый, как в реальном Flash/Adobe Animate
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(size=self._redraw_bg, pos=self._redraw_bg)

    # ---------------- история (undo/redo) ----------------
    def _snapshot(self):
        return (list(self.scene_objects), list(self.strokes), list(self.text_items))

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self.scene_objects, self.strokes, self.text_items = self._undo_stack.pop()
        self._full_redraw()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self.scene_objects, self.strokes, self.text_items = self._redo_stack.pop()
        self._full_redraw()

    # ---------------- рендер ----------------
    def _redraw_bg(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    def _full_redraw(self):
        self.canvas.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        with self.canvas:
            for item in self.strokes:
                kind, data, color, width = item
                Color(*color)
                if kind == "line":
                    Line(points=data, width=width)
                elif kind == "rect":
                    x1, y1, x2, y2 = data
                    Line(rectangle=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)), width=width)
                elif kind == "circle":
                    cx, cy, r = data
                    Line(circle=(cx, cy, r), width=width)
                elif kind == "filled_rect":
                    x1, y1, x2, y2 = data
                    Rectangle(pos=(min(x1, x2), min(y1, y2)), size=(abs(x2 - x1), abs(y2 - y1)))
            for (x, y, text, color) in self.text_items:
                label = CoreLabel(text=text, font_size=16)
                label.refresh()
                Color(*color)
                Rectangle(texture=label.texture, pos=(x, y), size=label.texture.size)
            for obj in self.scene_objects:
                if obj.held_by is not None:
                    continue
                color = COLORS.get(obj.tag, (0.5, 0.5, 0.5, 1))
                Color(*color)
                r = obj.radius
                Ellipse(pos=(obj.x - r, obj.y - r), size=(r * 2, r * 2))

    # ---------------- ввод ----------------
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        t = self.tool

        if t in ("brush", "pencil"):
            self._push_undo()
            width = self.brush_width if t == "brush" else max(1.0, self.brush_width * 0.4)
            with self.canvas:
                Color(*self.stroke_color)
                self._current_line = Line(points=[touch.x, touch.y], width=width)
            self.strokes.append(("line", self._current_line.points, self.stroke_color, width))

        elif t in ("line", "rect", "circle", "free_transform"):
            self._draw_start = (touch.x, touch.y)

        elif t == "pen":
            self._pen_points += [touch.x, touch.y]
            if len(self._pen_points) >= 4:
                self._push_undo()
                with self.canvas:
                    Color(*self.stroke_color)
                    Line(points=self._pen_points, width=self.brush_width)
                self.strokes.append(("line", list(self._pen_points), self.stroke_color, self.brush_width))

        elif t == "text":
            self._push_undo()
            self.text_items.append((touch.x, touch.y, "Victim", self.fill_color))
            self._full_redraw()

        elif t in ("eraser_tool",):
            self._push_undo()
            self._erase_near(touch.x, touch.y)
            if self.on_eraser_tool_hit:
                self.on_eraser_tool_hit(touch.x, touch.y)

        elif t in ("select", "subselect", "lasso_select"):
            self.selected_index = self._pick_object(touch.x, touch.y)

        elif t == "resize":
            pass

        elif t == "ink_bottle":
            self._recolor_nearest_stroke(touch.x, touch.y, self.stroke_color)

        elif t == "paint_bucket":
            self._recolor_nearest_stroke(touch.x, touch.y, self.fill_color)

        elif t == "eyedropper":
            picked = self._pick_color_near(touch.x, touch.y)
            if picked is not None:
                self.stroke_color = picked
                self.current_color = picked

        elif t in SPAWNABLE:
            self._push_undo()
            obj = SceneObject(touch.x, touch.y, t)
            if t in PROJECTILE_TAGS:
                target = self.get_aim_target() if self.get_aim_target else None
                if target is not None:
                    dx, dy = target[0] - touch.x, target[1] - touch.y
                    dist = math.hypot(dx, dy) + 1e-6
                    obj.vx = dx / dist * PROJECTILE_SPEED
                    obj.vy = dy / dist * PROJECTILE_SPEED
            self.scene_objects.append(obj)
            if len(self.scene_objects) > 60:
                self.scene_objects.pop(0)
            self._full_redraw()

        elif t == "hand" and self.on_hand_down:
            self._hand_active = bool(self.on_hand_down(touch.x, touch.y))

        elif t == "clone" and self.on_clone_down:
            self.on_clone_down(touch.x, touch.y)

        return True

    def on_touch_move(self, touch):
        t = self.tool
        if t in ("brush", "pencil") and self._current_line is not None:
            self._current_line.points += [touch.x, touch.y]
            width = self.strokes[-1][3]
            self.strokes[-1] = ("line", self._current_line.points, self.stroke_color, width)
            return True

        if t in ("select", "subselect") and self.selected_index is not None:
            obj = self.scene_objects[self.selected_index]
            obj.x, obj.y = touch.x, touch.y
            self._full_redraw()
            return True

        if t in ("resize", "free_transform") and self.selected_index is not None:
            obj = self.scene_objects[self.selected_index]
            dist = math.hypot(touch.x - obj.x, touch.y - obj.y)
            obj.radius = max(4, min(60, dist))
            self._full_redraw()
            return True

        if t == "hand" and getattr(self, "_hand_active", False) and self.on_hand_move:
            self.on_hand_move(touch.x, touch.y)
            return True

        return False

    def on_touch_up(self, touch):
        t = self.tool
        if t == "hand" and getattr(self, "_hand_active", False) and self.on_hand_up:
            self.on_hand_up(touch.x, touch.y)
            self._hand_active = False

        if t in ("line", "rect", "circle") and self._draw_start:
            self._push_undo()
            x1, y1 = self._draw_start
            with self.canvas:
                Color(*self.stroke_color)
                if t == "line":
                    Line(points=[x1, y1, touch.x, touch.y], width=self.brush_width)
                    self.strokes.append(("line", [x1, y1, touch.x, touch.y], self.stroke_color, self.brush_width))
                elif t == "rect":
                    Line(rectangle=(min(x1, touch.x), min(y1, touch.y),
                                     abs(touch.x - x1), abs(touch.y - y1)), width=self.brush_width)
                    self.strokes.append(("rect", (x1, y1, touch.x, touch.y), self.stroke_color, self.brush_width))
                elif t == "circle":
                    r = math.hypot(touch.x - x1, touch.y - y1)
                    Line(circle=(x1, y1, r), width=self.brush_width)
                    self.strokes.append(("circle", (x1, y1, r), self.stroke_color, self.brush_width))
            self._draw_start = None

        self._current_line = None
        return True

    # ---------------- утилиты ----------------
    def finish_pen(self):
        """Двойное нажатие кнопки Pen в тулбаре завершает полилинию."""
        self._pen_points = []

    def _pick_object(self, x, y, radius=25):
        best, best_d = None, radius
        for i, obj in enumerate(self.scene_objects):
            d = math.hypot(obj.x - x, obj.y - y)
            if d < best_d:
                best, best_d = i, d
        return best

    def _recolor_nearest_stroke(self, x, y, color, radius=30):
        best_i, best_d = None, radius
        for i, (kind, data, c, w) in enumerate(self.strokes):
            if kind == "line":
                pts = data
                for j in range(0, len(pts) - 1, 2):
                    d = math.hypot(pts[j] - x, pts[j + 1] - y)
                    if d < best_d:
                        best_i, best_d = i, d
            else:
                cx = data[0] if kind == "circle" else (data[0] + data[2]) / 2
                cy = data[1] if kind == "circle" else (data[1] + data[3]) / 2
                d = math.hypot(cx - x, cy - y)
                if d < best_d:
                    best_i, best_d = i, d
        if best_i is not None:
            kind, data, _, w = self.strokes[best_i]
            self.strokes[best_i] = (kind, data, color, w)
            self._full_redraw()

    def _pick_color_near(self, x, y, radius=30):
        for obj in self.scene_objects:
            if math.hypot(obj.x - x, obj.y - y) < radius:
                return COLORS.get(obj.tag)
        best_c, best_d = None, radius
        for (kind, data, color, w) in self.strokes:
            if kind == "line":
                pts = data
                for j in range(0, len(pts) - 1, 2):
                    d = math.hypot(pts[j] - x, pts[j + 1] - y)
                    if d < best_d:
                        best_d, best_c = d, color
        return best_c

    def _erase_near(self, x, y, radius=20):
        self.scene_objects = [o for o in self.scene_objects
                               if o.held_by is not None or math.hypot(o.x - x, o.y - y) > radius]

        def far_enough(item):
            kind, data = item[0], item[1]
            if kind == "line":
                pts = data
                return not any(math.hypot(pts[i] - x, pts[i + 1] - y) < radius
                               for i in range(0, len(pts) - 1, 2))
            if kind == "rect":
                x1, y1, x2, y2 = data
                return math.hypot((x1 + x2) / 2 - x, (y1 + y2) / 2 - y) > radius
            if kind == "circle":
                cx, cy, r = data
                return math.hypot(cx - x, cy - y) > radius
            return True

        self.strokes = [s for s in self.strokes if far_enough(s)]
        self._full_redraw()

    def clear(self):
        self._push_undo()
        self.scene_objects.clear()
        self.strokes.clear()
        self.text_items.clear()
        self._full_redraw()

    # ---------------- сохранение/загрузка сцены ----------------
    def export_scene(self):
        return json.dumps({
            "objects": [o.to_dict() for o in self.scene_objects],
            "strokes": [[k, list(d) if not isinstance(d, list) else d, list(c), w] for k, d, c, w in self.strokes],
        })

    def import_scene(self, text):
        data = json.loads(text)
        self.scene_objects = [SceneObject.from_dict(d) for d in data.get("objects", [])]
        self.strokes = [(s[0], s[1], tuple(s[2]), s[3]) for s in data.get("strokes", [])]
        self._full_redraw()
