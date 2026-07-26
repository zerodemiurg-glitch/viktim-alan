# -*- coding: utf-8 -*-
"""
memory.py
ДОЛГОВРЕМЕННАЯ ПАМЯТЬ Viktim -- отдельная от синаптических весов система,
реализующая ту же архитектуру, что и в реальном мозге (Complementary Learning
Systems, McClelland/O'Reilly; hippocampal indexing theory, Teyler/DiScenna):

  1) Гиппокамп быстро и "дёшево" кодирует ОТДЕЛЬНЫЙ ЭПИЗОД (сенсорный паттерн +
     валентность момента: сильная боль или сильное облегчение) в компактный
     "энграмм" -- это быстрая, но нестабильная память.
  2) Во время сна (см. brain.py -- та же фаза, что запускает уровень 7
     пластичности, apply_sleep_consolidation) гиппокамп ПОВТОРНО ПРОИГРЫВАЕТ
     (replay) самые значимые энграммы обратно в кору -- ровно так, как в
     реальном мозге происходят гиппокампальные sharp-wave ripples во сне,
     обеспечивающие системную консолидацию. Реиграется сенсорный паттерн,
     как будто он снова воспринимается -- и та же Хеббовская/STDP пластичность
     (уровень 2, уже реализована) закрепляет соответствующие корковые пути.
  3) Каждый повтор увеличивает "прочность" энграмма. Энграммы, которые никогда
     не повторяются, медленно угасают (забывание) и в итоге удаляются --
     ровно как нестабильные гиппокампальные следы, которые не были
     консолидированы. Энграммы с высокой прочностью считаются
     "консолидированными в кору" -- воспоминание стало частью личности,
     а не отдельным эпизодом.
"""

import time
import numpy as np

MAX_ENGRAMS = 60
ENCODE_THRESHOLD = 0.45     # порог валентности (боль или облегчение) для формирования нового следа
DECAY_PER_SLEEP = 0.12      # угасание прочности энграмма, если он НЕ был повторён в эту "ночь"
CONSOLIDATED_STRENGTH = 3.0  # прочность, после которой след считается перенесённым в кору


class Engram:
    """Один эпизод долговременной памяти."""
    __slots__ = ("sensory", "valence", "context", "age_hours", "strength", "replays")

    def __init__(self, sensory, valence, context, age_hours):
        self.sensory = sensory          # сжатый сенсорный паттерн (numpy-массив, копия position_signal)
        self.valence = valence          # -1..+1 (боль .. облегчение)
        self.context = context          # dict: день/ночь, утраченные конечности и т.п. на момент события
        self.age_hours = age_hours      # возраст мозга (в часах) на момент формирования следа
        self.strength = 1.0
        self.replays = 0

    def to_dict(self):
        return {
            "sensory": self.sensory.tolist(), "valence": self.valence,
            "context": self.context, "age_hours": self.age_hours,
            "strength": self.strength, "replays": self.replays,
        }

    @staticmethod
    def from_dict(d):
        e = Engram(np.array(d["sensory"], dtype=float), d["valence"], d["context"], d["age_hours"])
        e.strength = d["strength"]
        e.replays = d["replays"]
        return e


class LongTermMemory:
    """
    Хранилище энграммов + вся логика энкодинга/replay/забывания.
    Не хранит ни одного синаптического веса сама -- это ЧИСТО эпизодический
    индекс (гиппокампальный), консолидация в фактические синаптические веса
    коры происходит через replay -> brain.step с реинжектированным
    sensory-паттерном -> обычная Хеббовская пластичность делает остальное.
    """

    def __init__(self):
        self.engrams = []

    def maybe_encode(self, sensory_pattern, pain, relief, context, age_hours):
        """Вызывается каждый шаг мозга; формирует новый след только при сильном событии."""
        valence = relief - pain
        if abs(valence) < ENCODE_THRESHOLD:
            return None
        if np.linalg.norm(sensory_pattern) < 1e-6:
            return None  # нет узнаваемого сенсорного контекста -- нечего запоминать
        e = Engram(sensory_pattern.copy(), float(np.clip(valence, -1, 1)), dict(context), age_hours)
        self.engrams.append(e)
        if len(self.engrams) > MAX_ENGRAMS:
            # вытесняем самый слабый и наименее консолидированный след (как реальное забывание)
            self.engrams.sort(key=lambda x: x.strength)
            self.engrams.pop(0)
        return e

    def nightly_replay(self, n_replay=5):
        """
        Вызывается ОДИН раз за ночь (см. brain.py, та же точка входа, что и
        apply_sleep_consolidation). Возвращает список (sensory, valence) для
        реинжекции в мозг -- самые сильные/важные следы повторяются чаще,
        как в реальности значимые воспоминания "прокручиваются" во сне активнее.
        """
        if not self.engrams:
            return []
        ranked = sorted(self.engrams, key=lambda e: e.strength * (1 + abs(e.valence)), reverse=True)
        chosen = ranked[:n_replay]
        for e in chosen:
            e.strength += 0.4
            e.replays += 1

        not_chosen = [e for e in self.engrams if e not in chosen]
        for e in not_chosen:
            e.strength = max(0.0, e.strength - DECAY_PER_SLEEP)
        self.engrams = [e for e in self.engrams if e.strength > 0.05]

        return [(e.sensory, e.valence) for e in chosen]

    def consolidated_count(self):
        return sum(1 for e in self.engrams if e.strength >= CONSOLIDATED_STRENGTH)

    def to_dict(self):
        return {"engrams": [e.to_dict() for e in self.engrams]}

    def load_dict(self, d):
        self.engrams = [Engram.from_dict(x) for x in d.get("engrams", [])]
