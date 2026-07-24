# -*- coding: utf-8 -*-
"""
brain.py
Мозг стикмана Viktim: 150 AdEx-нейронов, разделённых на два полушария (75 + 75),
соединённых ~15000 синапсами (внутриполушарные + межполушарные "мозолистое тело").

Критичные нейромедиаторные/гормональные оси (упрощённые, но функциональные leaky-переменные,
управляемые активностью соответствующих нейронных популяций):
  - Дофамин   (DA)  -- мотивация / вознаграждение / приближение
  - Серотонин (5-HT) -- настроение / торможение агрессии / стабильность
  - Норадреналин (NE) -- возбуждение / реакция "бей-беги" / общий gain
  - Кортизол  (CORT) -- ось HPA, хронический стресс, подавляет 5-HT, повышает NE-чувствительность

Популяции нейронов по функции (внутри каждого полушария, индексы локальные):
  0-14   Sensory   (визуальный вход из workspace: позиция/тип ближайшего объекта)
  15-19  Threat    (детектор угрозы, миндалевидное тело)
  20-24  Reward    (детектор "приятного" объекта, дофаминовый вход)
  25-59  Assoc     (ассоциативные/интернейроны, свободная рекуррентная сеть)
  60-74  Motor     (моторный выход -> суставы стикмана)
"""

import numpy as np
from neuron import AdExPopulation, SynapseMatrix

N_HEMI = 75
N_TOTAL = N_HEMI * 2  # 150

SENSORY = slice(0, 15)
THREAT = slice(15, 20)
REWARD = slice(20, 25)
ASSOC = slice(25, 60)
MOTOR = slice(60, 75)

TOTAL_SYNAPSES_TARGET = 15000


def _build_hemisphere_connectivity(rng, n=N_HEMI, n_synapses=6500):
    """Случайная разреженная связность внутри одного полушария с учётом функциональных зон."""
    pre, post, w, exc = [], [], [], []
    # 80% возбуждающих / 20% тормозных нейронов (правило Дейла)
    excitatory_mask = rng.random(n) < 0.8

    zones = [SENSORY, THREAT, REWARD, ASSOC, MOTOR]
    # базовые "разрешённые" направления потока информации (упрощённая кортикальная иерархия)
    flow = {
        "SENSORY": [ASSOC, THREAT, REWARD],
        "THREAT": [ASSOC, MOTOR],
        "REWARD": [ASSOC, MOTOR],
        "ASSOC": [ASSOC, MOTOR, THREAT, REWARD],
        "MOTOR": [ASSOC],
    }
    names = ["SENSORY", "THREAT", "REWARD", "ASSOC", "MOTOR"]

    edges = 0
    attempts = 0
    while edges < n_synapses and attempts < n_synapses * 20:
        attempts += 1
        src_name = names[rng.integers(0, len(names))]
        src_slice = dict(zip(names, zones))[src_name]
        dst_slice = flow[src_name][rng.integers(0, len(flow[src_name]))]
        p = rng.integers(src_slice.start, src_slice.stop)
        q = rng.integers(dst_slice.start, dst_slice.stop)
        if p == q:
            continue
        pre.append(p)
        post.append(q)
        weight = rng.uniform(0.05, 0.4) if excitatory_mask[p] else rng.uniform(0.05, 0.6)
        w.append(weight)
        exc.append(bool(excitatory_mask[p]))
        edges += 1

    return (np.array(pre), np.array(post), np.array(w), np.array(exc), excitatory_mask)


class Hemisphere:
    def __init__(self, rng, n_synapses):
        pre, post, w, exc, excitatory_mask = _build_hemisphere_connectivity(rng, N_HEMI, n_synapses)
        self.pop = AdExPopulation(N_HEMI, excitatory_mask)
        self.syn = SynapseMatrix(N_HEMI, pre, post, w, exc)
        self.excitatory_mask = excitatory_mask


class Neuromodulators:
    """Упрощённые эндокринные/нейромедиаторные оси с затуханием к базовому уровню (гомеостаз)."""

    def __init__(self):
        self.dopamine = 0.3     # 0..1
        self.serotonin = 0.5
        self.norepinephrine = 0.2
        self.cortisol = 0.2
        self._base = dict(dopamine=0.3, serotonin=0.5, norepinephrine=0.2, cortisol=0.15)

    def update(self, dt_ms, reward_activity, threat_activity, assoc_activity):
        tau = 2000.0  # мс, медленный гомеостатический возврат к базе
        # Дофамин растёт от активности reward-популяции, гейтит уверенность/приближение
        self.dopamine += dt_ms * (reward_activity * 1.5 - (self.dopamine - self._base['dopamine']) / tau)
        # Норадреналин растёт от угрозы, быстрый ответ "бей-беги"
        self.norepinephrine += dt_ms * (threat_activity * 2.0 - (self.norepinephrine - self._base['norepinephrine']) / (tau * 0.3))
        # Кортизол -- медленная ось HPA, растёт при устойчивой угрозе, спадает медленно
        self.cortisol += dt_ms * (threat_activity * 0.3 - (self.cortisol - self._base['cortisol']) / (tau * 4))
        # Серотонин подавляется кортизолом/угрозой, восстанавливается сам
        target_5ht = self._base['serotonin'] - 0.6 * self.cortisol
        self.serotonin += dt_ms * ((target_5ht - self.serotonin) / (tau * 0.6))

        self.dopamine = float(np.clip(self.dopamine, 0, 1))
        self.serotonin = float(np.clip(self.serotonin, 0, 1))
        self.norepinephrine = float(np.clip(self.norepinephrine, 0, 1))
        self.cortisol = float(np.clip(self.cortisol, 0, 1))

    @property
    def arousal_gain(self):
        """Общий множитель чувствительности нейронов -- йеркс-додсоновская U-образная зависимость."""
        ne = self.norepinephrine
        return 0.7 + 1.0 * ne - 0.6 * (ne ** 2)

    @property
    def excitatory_boost(self):
        """Дофамин усиливает возбуждающую передачу (мотивация/подход к цели)."""
        return 0.7 + 0.8 * self.dopamine

    @property
    def aggression_damping(self):
        """Серотонин демпфирует импульсивные/агрессивные моторные реакции."""
        return 1.2 - 0.7 * self.serotonin


class Brain:
    """Полный мозг Viktim: два полушария + межполушарные связи + нейромодуляторные оси."""

    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)
        n_intra = 6500  # x2 полушария = 13000
        n_inter = TOTAL_SYNAPSES_TARGET - n_intra * 2  # ~2000 "мозолистое тело"

        self.left = Hemisphere(rng, n_intra)
        self.right = Hemisphere(rng, n_intra)

        # Межполушарные связи (corpus callosum): в основном ассоциативные и моторные зоны
        pre_l, post_r, w_ir, exc_ir = [], [], [], []
        for _ in range(n_inter // 2):
            p = rng.integers(ASSOC.start, MOTOR.stop)
            q = rng.integers(ASSOC.start, MOTOR.stop)
            pre_l.append(p); post_r.append(q)
            w_ir.append(rng.uniform(0.05, 0.25))
            exc_ir.append(True)
        pre_r, post_l, w_ri, exc_ri = [], [], [], []
        for _ in range(n_inter - n_inter // 2):
            p = rng.integers(ASSOC.start, MOTOR.stop)
            q = rng.integers(ASSOC.start, MOTOR.stop)
            pre_r.append(p); post_l.append(q)
            w_ri.append(rng.uniform(0.05, 0.25))
            exc_ri.append(True)

        self.cc_l2r = SynapseMatrix(N_HEMI, np.array(pre_l), np.array(post_r), np.array(w_ir), np.array(exc_ir))
        self.cc_r2l = SynapseMatrix(N_HEMI, np.array(pre_r), np.array(post_l), np.array(w_ri), np.array(exc_ri))

        self.neuromod = Neuromodulators()
        self.dt = self.left.pop.dt
        self.total_synapses = n_intra * 2 + n_inter
        self._motor_smooth = np.zeros(15 * 2)  # сглаженный моторный выход (оба полушария)

    def inject_sensory(self, threat_level, reward_level, position_signal):
        """
        threat_level, reward_level: 0..1
        position_signal: массив длиной 15*2 (по 15 на полушарие) -- напр. направление/дистанция
        до ближайшего объекта, закодированные популяционно (population coding).
        Возвращает (I_ext_left, I_ext_right) размерности N_HEMI каждая, в nA.
        """
        I_left = np.zeros(N_HEMI)
        I_right = np.zeros(N_HEMI)

        sens = np.asarray(position_signal, dtype=float)
        sens_l = sens[:15]
        sens_r = sens[15:]

        I_left[SENSORY] = sens_l * 0.6
        I_right[SENSORY] = sens_r * 0.6

        I_left[THREAT] = threat_level * 1.2
        I_right[THREAT] = threat_level * 1.2
        I_left[REWARD] = reward_level * 1.0
        I_right[REWARD] = reward_level * 1.0

        return I_left, I_right

    def step(self, threat_level=0.0, reward_level=0.0, position_signal=None):
        if position_signal is None:
            position_signal = np.zeros(30)
        I_ext_l, I_ext_r = self.inject_sensory(threat_level, reward_level, position_signal)

        gain = self.neuromod.arousal_gain
        exc_boost = self.neuromod.excitatory_boost
        motor_damp = self.neuromod.aggression_damping

        spikes_l = self.left.pop.spiked
        spikes_r = self.right.pop.spiked

        I_syn_l = self.left.syn.step(spikes_l, self.dt, neuromod_gain=exc_boost)
        I_syn_r = self.right.syn.step(spikes_r, self.dt, neuromod_gain=exc_boost)

        I_cc_to_r = self.cc_l2r.step(spikes_l, self.dt, neuromod_gain=exc_boost)
        I_cc_to_l = self.cc_r2l.step(spikes_r, self.dt, neuromod_gain=exc_boost)

        noise = 0.01 + 0.02 * self.neuromod.cortisol  # кортизол повышает нейронный "шум"/тревожность

        spikes_l = self.left.pop.step(I_syn_l + I_cc_to_l, I_ext_l, gain=gain, noise_std=noise)
        spikes_r = self.right.pop.step(I_syn_r + I_cc_to_r, I_ext_r, gain=gain, noise_std=noise)

        reward_activity = float(np.mean([spikes_l[REWARD].mean(), spikes_r[REWARD].mean()]))
        threat_activity = float(np.mean([spikes_l[THREAT].mean(), spikes_r[THREAT].mean()]))
        assoc_activity = float(np.mean([spikes_l[ASSOC].mean(), spikes_r[ASSOC].mean()]))
        self.neuromod.update(self.dt, reward_activity, threat_activity, assoc_activity)

        motor_now = np.concatenate([spikes_l[MOTOR], spikes_r[MOTOR]]).astype(float)
        alpha = 0.05  # сглаживание для плавности анимации
        self._motor_smooth = (1 - alpha) * self._motor_smooth + alpha * motor_now / motor_damp

        return {
            "motor": self._motor_smooth.copy(),
            "dopamine": self.neuromod.dopamine,
            "serotonin": self.neuromod.serotonin,
            "norepinephrine": self.neuromod.norepinephrine,
            "cortisol": self.neuromod.cortisol,
            "spikes_total": int(spikes_l.sum() + spikes_r.sum()),
        }
