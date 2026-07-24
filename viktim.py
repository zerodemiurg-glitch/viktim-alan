# -*- coding: utf-8 -*-
"""
viktim.py
Стикмен "Viktim" (как в анимациях Alan Becker: Animator vs. Animation).
Тело -- простой скелет (голова, туловище, 2 руки, 2 ноги), управляемый
не заранее прописанной анимацией, а живым выходом мозга (brain.py).

Viktim "видит" рабочее пространство приложения: получает список объектов
(курсор, нарисованные фигуры, инструменты) и кодирует ближайшую угрозу/
привлекательный объект в сенсорный сигнал для мозга. Мозг решает, куда
двигаться и как -- убегать, подходить, бездействовать -- через популяционный
моторный выход, который декодируется в углы суставов и скорость перемещения.
"""

import math
import numpy as np
from brain import Brain, N_HEMI

# Порог классификации объектов workspace (простая эвристика, а не хардкод-анимация):
# каждому объекту присвоен "valence" от -1 (угроза, например ластик/нож) до +1 (приятное, например звезда)
THREAT_TAGS = {"eraser", "knife", "fire", "enemy"}
REWARD_TAGS = {"star", "food", "friend", "coin"}


class Bone:
    __slots__ = ("length", "angle")

    def __init__(self, length, angle=0.0):
        self.length = length
        self.angle = angle  # относительно родителя, в радианах


class Viktim:
    def __init__(self, x=200, y=200, seed=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.facing = 1  # 1 = вправо, -1 = влево

        self.brain = Brain(seed=seed)

        # Скелет: длины сегментов в px (пропорции классического Alan Becker stickman)
        self.head_r = 12
        self.torso = Bone(40, angle=-math.pi / 2)
        self.arm_l_upper = Bone(20, angle=math.radians(20))
        self.arm_l_lower = Bone(18, angle=math.radians(10))
        self.arm_r_upper = Bone(20, angle=math.radians(-20))
        self.arm_r_lower = Bone(18, angle=math.radians(-10))
        self.leg_l_upper = Bone(22, angle=math.radians(10))
        self.leg_l_lower = Bone(20, angle=math.radians(5))
        self.leg_r_upper = Bone(22, angle=math.radians(-10))
        self.leg_r_lower = Bone(20, angle=math.radians(-5))

        self.state = "idle"  # idle | flee | approach | walk
        self.energy = 1.0
        self._t = 0.0

    # ---------- Восприятие рабочего пространства ----------
    def perceive(self, workspace_objects):
        """
        workspace_objects: список dict {'x','y','tag'} -- всё, что нарисовано/лежит
        в текущем workspace приложения (см. workspace.py).
        Возвращает (threat_level, reward_level, position_signal[30]) для мозга.
        """
        if not workspace_objects:
            return 0.0, 0.0, np.zeros(30)

        threat_level = 0.0
        reward_level = 0.0
        nearest_threat = None
        nearest_reward = None
        best_td = best_rd = 1e9

        for obj in workspace_objects:
            dx = obj["x"] - self.x
            dy = obj["y"] - self.y
            dist = math.hypot(dx, dy) + 1e-6
            tag = obj.get("tag", "")
            if tag in THREAT_TAGS and dist < best_td:
                best_td = dist
                nearest_threat = (dx, dy, dist)
            elif tag in REWARD_TAGS and dist < best_rd:
                best_rd = dist
                nearest_reward = (dx, dy, dist)

        VISION_RANGE = 400.0
        if nearest_threat:
            dx, dy, dist = nearest_threat
            threat_level = max(0.0, 1.0 - dist / VISION_RANGE)
        if nearest_reward:
            dx, dy, dist = nearest_reward
            reward_level = max(0.0, 1.0 - dist / VISION_RANGE)

        # Популяционное кодирование направления: 15 "рецептивных полей" по кругу на полушарие
        position_signal = np.zeros(30)
        ref_obj = nearest_threat or nearest_reward
        if ref_obj:
            dx, dy, dist = ref_obj
            angle = math.atan2(dy, dx)
            for hemi in range(2):
                base = hemi * 15
                for i in range(15):
                    pref_angle = -math.pi + i * (2 * math.pi / 15)
                    diff = abs((angle - pref_angle + math.pi) % (2 * math.pi) - math.pi)
                    tuning = math.exp(-(diff ** 2) / (2 * (0.5 ** 2)))
                    position_signal[base + i] = tuning * max(0.0, 1.0 - dist / VISION_RANGE)
        return threat_level, reward_level, position_signal

    # ---------- Шаг симуляции ----------
    def update(self, dt_ms, workspace_objects):
        threat, reward, pos_signal = self.perceive(workspace_objects)
        out = self.brain.step(threat, reward, pos_signal)
        self._t += dt_ms

        motor = out["motor"]  # 30 значений (15 левое полушарие + 15 правое), 0..~
        # Разбиваем моторный выход на функциональные каналы:
        # 0-4 скорость ног (локомоция), 5-9 руки, 10-14 ориентация/поворот головы-тела
        loco = motor[0:5].mean()
        arms = motor[5:10]
        turn = motor[10:15].mean() - motor[25:30].mean()

        # Поведенческое состояние определяется гормональным фоном, а не хардкодом if/else анимации
        cortisol, dopamine = out["cortisol"], out["dopamine"]
        if cortisol > 0.5 and threat > 0.15:
            self.state = "flee"
            direction = -1 if turn >= 0 else 1
            speed = 60 + 120 * cortisol
        elif dopamine > 0.55 and reward > 0.1:
            self.state = "approach"
            direction = 1 if turn >= 0 else -1
            speed = 40 + 80 * dopamine
        elif loco > 0.05:
            self.state = "walk"
            direction = 1 if turn >= 0 else -1
            speed = 30 + 100 * loco
        else:
            self.state = "idle"
            direction = 0
            speed = 0

        self.facing = direction if direction != 0 else self.facing
        self.vx = direction * speed
        self.x += self.vx * (dt_ms / 1000.0)

        # Процедурная походка: фаза шага зависит от локомоторной активности мозга, не таймера анимации
        phase = self._t * (0.002 + 0.02 * loco)
        swing = math.sin(phase) * (0.3 + 0.6 * loco)
        self.leg_l_upper.angle = math.radians(10) + swing
        self.leg_r_upper.angle = math.radians(-10) - swing
        self.leg_l_lower.angle = math.radians(5) + max(0, swing) * 0.5
        self.leg_r_lower.angle = math.radians(-5) - min(0, swing) * 0.5

        self.arm_l_upper.angle = math.radians(20) + arms[0] * 1.5 - swing * 0.5
        self.arm_r_upper.angle = math.radians(-20) - arms[1] * 1.5 + swing * 0.5

        return out  # отдаём наружу нейро/гормональное состояние для отладочного HUD

    # ---------- Геометрия скелета для отрисовки ----------
    def skeleton_points(self):
        """Возвращает список отрезков (x1,y1,x2,y2) в мировых координатах для рендера в workspace."""
        fx = self.facing
        hip = (self.x, self.y)

        def project(origin, bone, mirror=False):
            a = bone.angle * (1 if not mirror else -1)
            end = (origin[0] + math.cos(a) * bone.length * fx,
                   origin[1] + math.sin(a) * bone.length)
            return end

        segs = []
        neck = (hip[0], hip[1] - self.torso.length)
        segs.append((hip[0], hip[1], neck[0], neck[1]))  # позвоночник
        head_center = (neck[0], neck[1] - self.head_r)

        elbow_l = project(neck, self.arm_l_upper)
        hand_l = project(elbow_l, self.arm_l_lower)
        elbow_r = project(neck, self.arm_r_upper)
        hand_r = project(elbow_r, self.arm_r_lower)
        segs.append((neck[0], neck[1], elbow_l[0], elbow_l[1]))
        segs.append((elbow_l[0], elbow_l[1], hand_l[0], hand_l[1]))
        segs.append((neck[0], neck[1], elbow_r[0], elbow_r[1]))
        segs.append((elbow_r[0], elbow_r[1], hand_r[0], hand_r[1]))

        knee_l = project(hip, self.leg_l_upper)
        foot_l = project(knee_l, self.leg_l_lower)
        knee_r = project(hip, self.leg_r_upper)
        foot_r = project(knee_r, self.leg_r_lower)
        segs.append((hip[0], hip[1], knee_l[0], knee_l[1]))
        segs.append((knee_l[0], knee_l[1], foot_l[0], foot_l[1]))
        segs.append((hip[0], hip[1], knee_r[0], knee_r[1]))
        segs.append((knee_r[0], knee_r[1], foot_r[0], foot_r[1]))

        return segs, head_center, self.head_r
