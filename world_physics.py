# -*- coding: utf-8 -*-
"""
world_physics.py
Вся механика взаимодействия объектов и тела Viktim -- ЗАКОНЫ МИРА, а не
решения/знания стикмена. Viktim ничего "не знает" про то, что ластик стирает,
а лассо спутывает -- он просто физически соприкасается с объектами, и мир
детерминированно на это реагирует. Единственное, что доходит до мозга Viktim
(brain.py через viktim.py) -- это СЫРЫЕ ПОСЛЕДСТВИЯ: боль от удара, спутанность
движения, облегчение от исчезновения опасности рядом. Что из этого "хорошо",
а что "плохо", и как на это реагировать в будущем -- целиком выучивается
нейросетью через пластичность (plasticity.py), а не прописано здесь.

resolve_physics() вызывается один раз за кадр из main.py.
"""

import math


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by) + 1e-6


def _closest_point_on_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return x1, y1, math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return cx, cy, math.hypot(px - cx, py - cy)


def _stroke_segments(kind, data):
    """Возвращает список отрезков (x1,y1,x2,y2), из которых состоит нарисованный штрих."""
    if kind == "line":
        pts = data
        return [(pts[i], pts[i + 1], pts[i + 2], pts[i + 3]) for i in range(0, len(pts) - 3, 2)]
    if kind == "rect":
        x1, y1, x2, y2 = data
        xa, ya, xb, yb = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        return [(xa, ya, xb, ya), (xb, ya, xb, yb), (xb, yb, xa, yb), (xa, yb, xa, ya)]
    return []


def step_objects(objects, dt_ms, canvas_h):
    """Движение: гравитация для тяжёлых объектов, инерция для летящих снарядов."""
    dt_s = dt_ms / 1000.0
    for o in objects:
        if o.held_by is not None:
            continue  # объект, который держит Viktim, двигается вместе с ним (см. main.py)
        if o.gravity:
            o.vy -= 260.0 * dt_s  # "падает" (экранные Y растут вверх в Kivy -> ускорение вниз отрицательно)
        o.x += o.vx * dt_s
        o.y += o.vy * dt_s


def resolve_physics(objects, viktims, dt_ms, canvas_size):
    """
    Возвращает список объектов, которые нужно удалить из сцены в этом кадре
    (уничтожены столкновением с землёй, стёрты, использованы), и попутно
    заполняет одноразовые физические события в каждом Viktim
    (v._physical_pain, v._physical_relief, v.entangled, v.held_object_props).
    """
    w, h = canvas_size
    to_remove = set()

    step_objects(objects, dt_ms, h)

    # объекты, упавшие за пределы холста, просто убираем (наковальня разбилась о пол)
    for o in objects:
        if o.gravity and o.y < 10:
            to_remove.add(id(o))
        # снаряды, улетевшие далеко за пределы видимой сцены -- убираем, чтобы
        # не копились вечно невидимыми (наковальня/лассо и т.п. неподвижны, их не трогаем)
        margin = 150
        if (o.vx != 0 or o.vy != 0) and not o.gravity:
            if o.x < -margin or o.x > w + margin or o.y < -margin or o.y > h + margin:
                to_remove.add(id(o))

    for v in viktims:
        # держимый объект следует за телом (условная "рука")
        if v.held_object is not None and v.held_object.held_by is v:
            v.held_object.x, v.held_object.y = v.x + 14 * v.facing, v.y + 10

        pain = 0.0
        relief = 0.0

        for o in objects:
            if id(o) in to_remove:
                continue
            d = _dist(v.x, v.y, o.x, o.y)
            touching = d < (v.body_radius + o.radius)

            # --- хватательный рефлекс: контакт с лёгким предметом, руки свободны ---
            if touching and o.grabbable and o.held_by is None and v.held_object is None:
                o.held_by = v
                v.held_object = o
                continue  # схваченный в этот же кадр предмет ни с кем не сталкивается

            if not touching or o.held_by is v:
                continue

            # --- физический удар (импульс = масса * относительная скорость) ---
            rel_speed = math.hypot(o.vx - v.vx, o.vy - v.vy)
            impact = o.mass * max(rel_speed, 40.0) / 400.0  # 40 -- минимальная "скорость" контакта

            # --- удерживаемый Viktim объект может физически стереть/пережечь то,
            #     с чем СЕЙЧАС соприкасается тело (не "решение", а механика материала) ---
            erased = False
            if v.held_object is not None:
                held = v.held_object
                if held.erase_radius > 0 and d < held.erase_radius:
                    to_remove.add(id(o))
                    erased = True
                elif held.burns_through and o.is_entangling and v.entangled_by is o:
                    to_remove.add(id(o))
                    v.entangled_by = None
                    erased = True

            if erased:
                relief += 0.8  # опасность рядом исчезла -- физическое облегчение (безусловный рефлекс)
                continue

            if o.is_entangling and v.entangled_by is None:
                v.entangled_by = o
                pain += 0.3  # само сдавливание неприятно (врождённая реакция на сковывание)
            elif impact > 0.05:
                pain += min(1.0, impact)
                to_remove.add(id(o))  # объект, ударивший Viktim, разрушается при ударе (наковальня, снаряд)

            # --- песочные дополнения (честно не из оригинала) ---
            if getattr(o, "freezes", False):
                v.freeze_timer = max(v.freeze_timer, 2500.0)
                pain += 0.4
            if getattr(o, "teleports", False):
                v.x = float(w * 0.5 + (hash((id(v), int(v.x))) % 200 - 100))
                v.y = float(max(60, min(h - 30, v.y)))
                pain += 0.2

        if v.entangled_by is not None and id(v.entangled_by) in to_remove:
            v.entangled_by = None

        v._physical_pain = min(1.0, pain)
        v._physical_relief = min(1.0, relief)

    return [o for o in objects if id(o) in to_remove]


def apply_stroke_collisions(strokes, viktims, bump_pain=0.4):
    """
    Нарисованные пользователем (в роли Аниматора) линии/фигуры/кисть -- это
    ЗАКОН МИРА ровно в той же степени, что и наковальня или лассо: реальный
    физический барьер. Раньше Viktim проходил сквозь них незамеченным --
    теперь тело физически выталкивается из линии при контакте, а сам факт
    столкновения -- ощутимый "удар" (bump_pain), который его нервная система
    заметит на следующем шаге (см. viktim._physical_pain), а не молчаливо
    игнорируемое препятствие.

    Вызывается ПОСЛЕ Viktim.update() (когда позиция уже сдвинулась) -- штрихи
    не являются SceneObject и не участвуют в основном resolve_physics().
    """
    for v in viktims:
        bumped = False
        for item in strokes:
            kind, data = item[0], item[1]

            if kind == "circle":
                cx, cy, r = data
                dist_to_center = _dist(v.x, v.y, cx, cy)
                d = abs(dist_to_center - r)
                if d < v.body_radius:
                    nx = (v.x - cx) / (dist_to_center + 1e-6)
                    ny = (v.y - cy) / (dist_to_center + 1e-6)
                    # выталкиваем на ближайшую сторону окружности (внутрь или наружу -- смотря откуда пришёл)
                    target_r = r if dist_to_center >= r else r
                    push = v.body_radius - d
                    v.x += nx * push
                    v.y += ny * push
                    bumped = True
                continue

            for (x1, y1, x2, y2) in _stroke_segments(kind, data):
                # дешёвая предварительная проверка по bounding-box перед точным расчётом
                if (min(x1, x2) - v.body_radius > v.x or max(x1, x2) + v.body_radius < v.x or
                        min(y1, y2) - v.body_radius > v.y or max(y1, y2) + v.body_radius < v.y):
                    continue
                px, py, d = _closest_point_on_segment(v.x, v.y, x1, y1, x2, y2)
                if d < v.body_radius:
                    nx, ny = v.x - px, v.y - py
                    n_len = math.hypot(nx, ny) + 1e-6
                    push = v.body_radius - d
                    v.x += (nx / n_len) * push
                    v.y += (ny / n_len) * push
                    bumped = True

        if bumped:
            v._physical_pain = max(v._physical_pain, bump_pain)
