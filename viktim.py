# -*- coding: utf-8 -*-
"""
viktim.py
Стикмен "Viktim". РЕВИЗИЯ: убраны все ветвления вида "если держит X и рядом Y,
то атаковать" -- такого знания у него быть не может. Всё, что доходит до его
мозга -- это СЫРЫЕ физические последствия, посчитанные в world_physics.py:
боль от удара, дискомфорт от спутывания, физическое облегчение, когда
опасность рядом исчезает. Что из этого хорошо, а что плохо, и как действовать
в будущем -- целиком решает нейросеть (brain.py) через 7 уровней пластичности,
а не код этого файла.

Физические ограничения тела (frozen/entangled/grabbed) -- это то, что
происходит С НИМ (мир так устроен), а не решения, которые он принимает --
поэтому они напрямую блокируют движение, в отличие от поведенческих
состояний (idle/walk/flee/approach/sleep), которые лишь ОПИСЫВАЮТ постфактум
то, что уже решила нейросеть, и не участвуют в принятии решений.
"""

import math
import numpy as np
from brain import Brain

VISION_RANGE = 400.0

# --- Соматотопическая карта тела в МОТОРНОЙ коре (контралатерально, как в реальном
# мозге: левое полушарие управляет ПРАВОЙ стороной тела и наоборот). Индексы --
# позиции в конкатенированном 40-мерном motor-векторе (motor[0:20] = выход левого
# полушария, motor[20:40] = выход правого). Голова -- билатеральная структура,
# получает вклад от обоих полушарий. Используется только для фантомной боли
# (см. Viktim.update) -- при потере части тела эти же нейроны продолжают получать
# "приказ двигаться", но никогда не получают сенсорного подтверждения.
BODY_PART_MOTOR_IDX = {
    "head":  [0, 1, 20, 21],
    "arm_r": [2, 3, 4, 5],      # контралатерально -- левое полушарие (motor[0:20])
    "leg_r": [6, 7, 8, 9],
    "arm_l": [22, 23, 24, 25],  # контралатерально -- правое полушарие (motor[20:40])
    "leg_l": [26, 27, 28, 29],
}
# те же части тела, но как ЛОКАЛЬНЫЕ индексы внутри MOTOR-региона (80-99) одного
# полушария -- нужно для коркового ремаппинга (plasticity.apply_cortical_remapping,
# вызывается из brain.py, который работает с локальной нумерацией нейронов).
_MOTOR_START = 80
BODY_PART_LOCAL_IDX = {
    "head_left":  [_MOTOR_START + 0, _MOTOR_START + 1],
    "head_right": [_MOTOR_START + 0, _MOTOR_START + 1],
    "arm_r": [_MOTOR_START + 2, _MOTOR_START + 3, _MOTOR_START + 4, _MOTOR_START + 5],   # локально в LEFT
    "leg_r": [_MOTOR_START + 6, _MOTOR_START + 7, _MOTOR_START + 8, _MOTOR_START + 9],   # локально в LEFT
    "arm_l": [_MOTOR_START + 2, _MOTOR_START + 3, _MOTOR_START + 4, _MOTOR_START + 5],   # локально в RIGHT
    "leg_l": [_MOTOR_START + 6, _MOTOR_START + 7, _MOTOR_START + 8, _MOTOR_START + 9],   # локально в RIGHT
}
PHANTOM_WEIGHT = 0.5  # сила фантомного сигнала до какого-либо ремаппинга


class Bone:
    __slots__ = ("length", "angle")

    def __init__(self, length, angle=0.0):
        self.length = length
        self.angle = angle


class Viktim:
    def __init__(self, x=200, y=200, seed=None, hours_per_real_second=24.0 / 600.0):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.facing = 1
        self.body_radius = 16.0

        self.brain = Brain(seed=seed, hours_per_real_second=hours_per_real_second)

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

        # --- физическое состояние тела (заполняется world_physics.py, не самим Viktim) ---
        self.held_object = None       # SceneObject | None -- что сейчас в "руке"
        self.entangled_by = None      # SceneObject | None -- что сейчас его спутывает
        self.freeze_timer = 0.0       # мс -- временный паралич (песочный эффект, не из оригинала)
        self.grabbed = False          # True пока его тащит "рука аниматора" (пользователь)
        self._physical_pain = 0.0     # выставляется world_physics.py каждый кадр
        self._physical_relief = 0.0   # выставляется world_physics.py каждый кадр

        # --- утрата частей тела и фантомная боль (см. damage_limb / update) ---
        self.limbs = {"head": True, "arm_l": True, "arm_r": True, "leg_l": True, "leg_r": True}
        self._remap_count_at_injury = {}  # part -> remap_count мозга в момент травмы (для угасания)

        self.state = "idle"  # чисто описательная метка ПОСТФАКТУМ, ни на что не влияет
        self._t = 0.0

    def damage_limb(self, part):
        """
        Вызывается извне (main.py), когда инструмент "Ластик" физически стирает
        часть тела Viktim. Конечность пропадает НЕМЕДЛЕННО (физика), но
        соответствующие нейроны моторной коры продолжают существовать и пытаться
        ей управлять -- рассинхронизация эфферентной команды и отсутствующей
        сенсорной обратной связи и есть фантомная боль (см. update()).
        """
        if part in self.limbs and self.limbs[part]:
            self.limbs[part] = False
            self._remap_count_at_injury[part] = self.brain.remap_count

    # ---------- Восприятие мира: только сырые физические величины ----------
    def perceive(self, objects, cursor_pos):
        """
        Возвращает (threat_level, reward_level, position_signal[40]).
        Никакой классификации объектов по типу -- только:
          - салиентность ближайшего физического тела (близость + скорость
            сближения, т.е. "надвигающийся объект" -- врождённый рефлекс
            избегания, есть почти у всех животных, это не выученная категория);
          - отдельно направление на курсор (визуально отличимая, узнаваемая
            сущность -- рука аниматора, которую Viktim видит независимо от
            того, хватают его сейчас или нет).
        """
        best_obj, best_score = None, 0.0
        for o in objects:
            if o.held_by is self:
                continue
            dx, dy = o.x - self.x, o.y - self.y
            dist = math.hypot(dx, dy) + 1e-6
            rel_speed = math.hypot(o.vx - self.vx, o.vy - self.vy)
            score = max(0.0, 1.0 - dist / VISION_RANGE) + 0.5 * min(1.0, rel_speed / 300.0)
            if score > best_score:
                best_score, best_obj = score, o

        position_signal = np.zeros(40)
        if best_obj is not None:
            dx, dy = best_obj.x - self.x, best_obj.y - self.y
            dist = math.hypot(dx, dy) + 1e-6
            angle = math.atan2(dy, dx)
            for hemi in range(2):
                base = hemi * 20
                for i in range(10):
                    pref = -math.pi + i * (2 * math.pi / 10)
                    diff = abs((angle - pref + math.pi) % (2 * math.pi) - math.pi)
                    tuning = math.exp(-(diff ** 2) / (2 * (0.6 ** 2)))
                    position_signal[base + i] = tuning * max(0.0, 1.0 - dist / VISION_RANGE)

        if cursor_pos is not None:
            cx, cy = cursor_pos
            dx, dy = cx - self.x, cy - self.y
            dist = math.hypot(dx, dy) + 1e-6
            angle = math.atan2(dy, dx)
            for hemi in range(2):
                base = hemi * 20 + 10
                for i in range(10):
                    pref = -math.pi + i * (2 * math.pi / 10)
                    diff = abs((angle - pref + math.pi) % (2 * math.pi) - math.pi)
                    tuning = math.exp(-(diff ** 2) / (2 * (0.6 ** 2)))
                    position_signal[base + i] = tuning * max(0.0, 1.0 - dist / VISION_RANGE)

        looming = best_score * 0.35  # мягкая врождённая настороженность к быстро приближающимся телам
        return looming, position_signal

    # ---------- Шаг симуляции ----------
    def update(self, dt_ms, objects, cursor_pos=None, canvas_size=None):
        looming, pos_signal = self.perceive(objects, cursor_pos)

        # --- фантомная боль: мозг продолжает "приказывать" утраченной части тела
        # двигаться (эфферентная команда есть), но сенсорного подтверждения того,
        # что она подвинулась, никогда не приходит -- это рассогласование сам
        # мозг воспринимает как боль (Ramachandran, реальный клинический механизм
        # фантомных болей). Угасает НЕ по таймеру, а по числу прошедших раундов
        # коркового ремаппинга (plasticity.apply_cortical_remapping) с момента
        # травмы -- т.е. чем больше соседние карты "вторглись" на депривированную
        # территорию, тем меньше остаётся нерешённого рассогласования.
        phantom_pain = 0.0
        prev_motor = getattr(self, "_last_motor", None)
        if prev_motor is not None:
            for part, intact in self.limbs.items():
                if intact:
                    continue
                idx = BODY_PART_MOTOR_IDX[part]
                efferent_command = float(np.mean(prev_motor[idx]))
                rounds_since_injury = max(0, self.brain.remap_count - self._remap_count_at_injury.get(part, 0))
                decay = 1.0 / (1.0 + 0.15 * rounds_since_injury)  # угасает с ремаппингом, не со временем
                phantom_pain += efferent_command * PHANTOM_WEIGHT * decay
        phantom_pain = min(0.9, phantom_pain)

        deprived_local = {"left": [], "right": []}
        for part, intact in self.limbs.items():
            if intact:
                continue
            if part in ("arm_r", "leg_r"):
                deprived_local["left"] += BODY_PART_LOCAL_IDX[part]
            elif part in ("arm_l", "leg_l"):
                deprived_local["right"] += BODY_PART_LOCAL_IDX[part]
            elif part == "head":
                deprived_local["left"] += BODY_PART_LOCAL_IDX["head_left"]
                deprived_local["right"] += BODY_PART_LOCAL_IDX["head_right"]

        entangle_discomfort = 0.35 if self.entangled_by is not None else 0.0
        threat_level = min(1.0, self._physical_pain + entangle_discomfort + looming + phantom_pain)
        reward_level = self._physical_relief
        # ПРИМЕЧАНИЕ: pain/relief намеренно НЕ обнуляются здесь -- world_physics.resolve_physics()
        # вызывается один раз за кадр и сам присваивает (а не накапливает) свежее значение,
        # поэтому событие естественным образом "живёт" ровно один кадр и гаснет само,
        # даже если мозг за этот кадр делает несколько под-шагов интегрирования (см. main.py).

        out = self.brain.step(threat_level, reward_level, pos_signal, deprived_local=deprived_local)
        self._last_motor = out["motor"]
        self._t += dt_ms

        physically_blocked = self.grabbed or self.freeze_timer > 0 or self.entangled_by is not None
        if self.freeze_timer > 0:
            self.freeze_timer = max(0.0, self.freeze_timer - dt_ms)

        if physically_blocked:
            self.vx = 0.0
            self.state = "grabbed" if self.grabbed else ("frozen" if self.freeze_timer > 0 else "entangled")
            return out

        motor = out["motor"]
        loco = motor[0:7].mean()
        turn = motor[14:20].mean() - motor[34:40].mean()

        nm = out["neuromodulators"]
        cortisol, dopamine = nm["cortisol"], nm["dopamine"]
        sleep_pressure = out["sleep_pressure"]
        is_night = out["is_night"]

        # состояние -- ТОЛЬКО ярлык постфактум для HUD/анимации, не участвует в принятии решений
        if is_night and sleep_pressure > 0.5 and threat_level < 0.1:
            self.state = "sleep"
            direction, speed = 0, 0
        elif threat_level > 0.2 or cortisol > 0.45 or nm["norepinephrine"] > 0.45:
            # немедленный рефлекс бегства/отдёргивания -- реагирует на СЕЙЧАС ощущаемую
            # боль/опасность (быстро, через threat_level и NE), а не ждёт, пока накопится
            # медленный кортизол -- тот отражает уже закрепившуюся тревожность ПОСЛЕ события,
            # а не то, что должно триггерить сам рефлекс бегства
            self.state = "flee"
            direction = -1 if turn >= 0 else 1
            speed = 60 + 120 * max(cortisol, threat_level)
        elif dopamine > 0.55 and loco > 0.03:
            self.state = "approach"
            direction = 1 if turn >= 0 else -1
            speed = 40 + 80 * dopamine
        elif loco > 0.05:
            self.state = "walk"
            direction = 1 if turn >= 0 else -1
            speed = 30 + 100 * loco
        else:
            self.state = "idle"
            direction, speed = 0, 0

        self.facing = direction if direction != 0 else self.facing
        self.vx = direction * speed
        self.x += self.vx * (dt_ms / 1000.0)

        if self.state == "sleep":
            breathing = math.sin(self._t * 0.003) * 0.05
            self.torso.angle = -math.radians(10) + breathing
            for b in (self.leg_l_upper, self.leg_r_upper, self.leg_l_lower, self.leg_r_lower,
                      self.arm_l_upper, self.arm_r_upper):
                b.angle *= 0.9
        else:
            self.torso.angle = -math.pi / 2
            phase = self._t * (0.002 + 0.02 * loco)
            swing = math.sin(phase) * (0.3 + 0.6 * loco)
            self.leg_l_upper.angle = math.radians(10) + swing
            self.leg_r_upper.angle = math.radians(-10) - swing
            self.leg_l_lower.angle = math.radians(5) + max(0, swing) * 0.5
            self.leg_r_lower.angle = math.radians(-5) - min(0, swing) * 0.5
            arms = motor[7:14]
            self.arm_l_upper.angle = math.radians(20) + arms[0] * 1.5 - swing * 0.5
            self.arm_r_upper.angle = math.radians(-20) - arms[1] * 1.5 + swing * 0.5

        return out

    # ---------- Геометрия скелета для отрисовки ----------
    def skeleton_points(self):
        fx = self.facing if self.facing != 0 else 1
        hip = (self.x, self.y)

        def project(origin, bone):
            a = bone.angle
            return (origin[0] + math.cos(a) * bone.length * fx,
                    origin[1] + math.sin(a) * bone.length)

        segs = []
        neck_drop = self.torso.length * (0.4 if self.state == "sleep" else 1.0)
        neck = (hip[0], hip[1] - neck_drop)
        segs.append((hip[0], hip[1], neck[0], neck[1]))
        head_center = (neck[0], neck[1] - self.head_r) if self.limbs["head"] else None

        elbow_l = project(neck, self.arm_l_upper)
        hand_l = project(elbow_l, self.arm_l_lower)
        elbow_r = project(neck, self.arm_r_upper)
        hand_r = project(elbow_r, self.arm_r_lower)
        if self.limbs["arm_l"]:
            segs += [(neck[0], neck[1], elbow_l[0], elbow_l[1]),
                     (elbow_l[0], elbow_l[1], hand_l[0], hand_l[1])]
        if self.limbs["arm_r"]:
            segs += [(neck[0], neck[1], elbow_r[0], elbow_r[1]),
                     (elbow_r[0], elbow_r[1], hand_r[0], hand_r[1])]

        knee_l = project(hip, self.leg_l_upper)
        foot_l = project(knee_l, self.leg_l_lower)
        knee_r = project(hip, self.leg_r_upper)
        foot_r = project(knee_r, self.leg_r_lower)
        if self.limbs["leg_l"]:
            segs += [(hip[0], hip[1], knee_l[0], knee_l[1]),
                     (knee_l[0], knee_l[1], foot_l[0], foot_l[1])]
        if self.limbs["leg_r"]:
            segs += [(hip[0], hip[1], knee_r[0], knee_r[1]),
                     (knee_r[0], knee_r[1], foot_r[0], foot_r[1])]

        if self.held_object is not None and self.limbs["arm_r"]:
            # чисто визуально: что-то есть в руке; никакого поведенческого смысла это не несёт
            segs.append((hand_r[0], hand_r[1], hand_r[0] + 10 * fx, hand_r[1] + 4))

        return segs, head_center, self.head_r
