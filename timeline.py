# -*- coding: utf-8 -*-
"""
timeline.py
Таймлайн покадровой анимации в духе Adobe Animate + "onion skinning"
(полупрозрачные силуэты предыдущих кадров) -- классический приём, которым
пользуется сам Аниматор в роликах Alan Becker, когда рисует позы кадр за кадром.

Здесь таймлайн не подменяет "живой" мозг Viktim -- он лишь ЗАПИСЫВАЕТ то,
что мозг уже сделал (скелет по кадрам), и позволяет прокручивать/воспроизводить
запись позже, как показ повтора. Само поведение Viktim по-прежнему целиком
определяется brain.py.
"""

import json


class Keyframe:
    __slots__ = ("t", "segs", "head", "head_r", "state")

    def __init__(self, t, segs, head, head_r, state):
        self.t = t
        self.segs = segs
        self.head = head
        self.head_r = head_r
        self.state = state

    def to_dict(self):
        return {"t": self.t, "segs": self.segs, "head": self.head,
                "head_r": self.head_r, "state": self.state}

    @staticmethod
    def from_dict(d):
        return Keyframe(d["t"], d["segs"], tuple(d["head"]), d["head_r"], d["state"])


class Timeline:
    """Хранит запись поз одного Viktim. max_frames ограничивает память (кольцевой буфер)."""

    def __init__(self, max_frames=1800, capture_every_ms=33.0):
        self.frames = []
        self.max_frames = max_frames
        self.capture_every_ms = capture_every_ms
        self._acc_ms = 0.0
        self.recording = True
        self.playhead = None  # None = live-режим (следовать за живым мозгом), иначе индекс кадра

    def maybe_capture(self, dt_ms, viktim):
        if not self.recording:
            return
        self._acc_ms += dt_ms
        if self._acc_ms < self.capture_every_ms:
            return
        self._acc_ms = 0.0
        segs, head, head_r = viktim.skeleton_points()
        kf = Keyframe(t=len(self.frames), segs=segs, head=head, head_r=head_r, state=viktim.state)
        self.frames.append(kf)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def onion_skin(self, n_ghosts=4):
        """Последние n_ghosts кадров (не считая текущего) для полупрозрачной отрисовки."""
        if len(self.frames) < 2:
            return []
        return self.frames[-(n_ghosts + 1):-1]

    def scrub_to(self, fraction):
        """fraction в [0,1] -- переместить playhead по записанной истории (пауза live-режима)."""
        if not self.frames:
            self.playhead = None
            return None
        idx = int(fraction * (len(self.frames) - 1))
        self.playhead = idx
        return self.frames[idx]

    def resume_live(self):
        self.playhead = None

    def export_json(self):
        return json.dumps([f.to_dict() for f in self.frames])

    def import_json(self, text):
        data = json.loads(text)
        self.frames = [Keyframe.from_dict(d) for d in data]
