# -*- coding: utf-8 -*-
"""
brain.py
Мозг Viktim v3: 500 AdEx-нейронов (250 + 250 в двух полушариях), каждое
полушарие разделено на 5 функциональных регионов по 50 нейронов; ~37500
синапсов (внутрирегиональные/межрегиональные внутри полушария + межполушарные
"мозолистое тело", сохранена та же плотность связности ~75 исходящих
синапсов/нейрон, что и в версии на 200 нейронов); 12 нейромедиаторных/
гормональных осей; циркадные часы с ощущением времени суток; семь уровней
пластичности (см. plasticity.py); ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (см. memory.py) --
гиппокампальные энграммы значимых событий, реиграемые во сне для системной
консолидации в кору, + сохранение/загрузка всего состояния мозга на диск
между сессиями (см. Brain.save_state/load_state) -- то есть память,
переживающая перезапуск приложения, а не только веса синапсов в рамках сессии.

Регионы (индексы локальные, 0..249, одинаковы для обоих полушарий):
  SENSORY  0-49    сенсорная/зрительная кора      -- вход из workspace
  LIMBIC   50-99   миндалина + прилежащее ядро     -- 50-74 угроза, 75-99 награда
  PFC      100-149 префронтальная ассоциативная    -- интеграция, принятие решений
  HIPPO    150-199 гиппокамп                        -- память контекста + ощущение времени
  MOTOR    200-249 моторная кора                    -- выход к суставам стикмана
"""

import math
import json
import numpy as np
from neuron import AdExPopulation, SynapseMatrix
import plasticity as plx
from memory import LongTermMemory

N_HEMI = 250
N_TOTAL = N_HEMI * 2  # 500

REGIONS = {
    "SENSORY": slice(0, 50),
    "LIMBIC":  slice(50, 100),
    "PFC":     slice(100, 150),
    "HIPPO":   slice(150, 200),
    "MOTOR":   slice(200, 250),
}
THREAT = slice(50, 75)    # подзона LIMBIC
REWARD = slice(75, 100)   # подзона LIMBIC
REGION_NAMES = list(REGIONS.keys())

FLOW_MAP = {
    "SENSORY": ["PFC", "LIMBIC", "HIPPO"],
    "LIMBIC":  ["PFC", "MOTOR", "HIPPO"],
    "PFC":     ["PFC", "MOTOR", "LIMBIC", "HIPPO"],
    "HIPPO":   ["PFC", "LIMBIC"],
    "MOTOR":   ["PFC"],
}

TOTAL_SYNAPSES_TARGET = 37500  # та же плотность (~75 исходящих/нейрон), что и в версии на 200 нейронов
POSITION_SIGNAL_LEN = 100       # 50 на полушарие (25 "объект" + 25 "курсор" -- см. viktim.py)
TIME_CODE_UNITS = 50


def _build_hemisphere_connectivity(rng, n_synapses):
    pre, post, w, exc = [], [], [], []
    excitatory_mask = rng.random(N_HEMI) < 0.8

    edges = 0
    attempts = 0
    while edges < n_synapses and attempts < n_synapses * 20:
        attempts += 1
        src_name = REGION_NAMES[rng.integers(0, len(REGION_NAMES))]
        src_slice = REGIONS[src_name]
        dst_name = FLOW_MAP[src_name][rng.integers(0, len(FLOW_MAP[src_name]))]
        dst_slice = REGIONS[dst_name]
        p = rng.integers(src_slice.start, src_slice.stop)
        q = rng.integers(dst_slice.start, dst_slice.stop)
        if p == q:
            continue
        pre.append(p)
        post.append(q)
        w.append(rng.uniform(0.05, 0.4) if excitatory_mask[p] else rng.uniform(0.05, 0.6))
        exc.append(bool(excitatory_mask[p]))
        edges += 1

    return (np.array(pre), np.array(post), np.array(w), np.array(exc), excitatory_mask)


class Hemisphere:
    def __init__(self, rng, n_synapses):
        pre, post, w, exc, excitatory_mask = _build_hemisphere_connectivity(rng, n_synapses)
        self.pop = AdExPopulation(N_HEMI, excitatory_mask)
        self.syn = SynapseMatrix(N_HEMI, pre, post, w, exc)
        self.excitatory_mask = excitatory_mask

    def region_activity(self, region_slice):
        return float(self.pop.spiked[region_slice].mean())

    def syn_state(self):
        return dict(pre=self.syn.pre.tolist(), post=self.syn.post.tolist(),
                    w=self.syn.w.tolist(), active=self.syn.active.astype(int).tolist(),
                    sign=self.syn.sign.tolist())

    def load_syn_state(self, d):
        self.syn.pre = np.array(d["pre"])
        self.syn.post = np.array(d["post"])
        self.syn.w = np.array(d["w"], dtype=float)
        self.syn.active = np.array(d["active"], dtype=bool)
        self.syn.sign = np.array(d["sign"], dtype=float)


class Neuromodulators:
    """
    12 нейромедиаторных/гормональных осей. Каждая -- затухающая к базовому
    уровню переменная (гомеостаз), управляемая активностью соответствующих
    регионов мозга и взаимодействующая с другими осями (как в реальной
    нейроэндокринной системе -- ни одна ось не работает в изоляции).
    """

    def __init__(self):
        self.base = dict(
            dopamine=0.30, serotonin=0.50, norepinephrine=0.20, cortisol=0.15,
            melatonin=0.10, adenosine=0.20, oxytocin=0.30, vasopressin=0.20,
            acetylcholine=0.40, endorphin=0.20, testosterone=0.30, gaba=0.35,
        )
        for k, v in self.base.items():
            setattr(self, k, v)

    def update(self, dt_ms, *, reward_act, threat_act, sensory_act, assoc_act,
               hippo_act, motor_act, night_signal, resting):
        """
        Единая устойчивая схема для ВСЕХ 12 осей: dx/dt = (target - x) / tau,
        где target всегда сам является ограниченной величиной (0..1), собранной
        из базового уровня + вклада активности регионов + влияния других осей.
        """
        b = self.base

        target_da = b['dopamine'] + 0.9 * reward_act
        self.dopamine += dt_ms * (target_da - self.dopamine) / 1500.0

        target_5ht = b['serotonin'] - 0.5 * self.cortisol + 0.25 * self.oxytocin
        self.serotonin += dt_ms * (target_5ht - self.serotonin) / 3000.0

        target_ne = b['norepinephrine'] + 0.9 * threat_act + 0.15 * self.acetylcholine
        self.norepinephrine += dt_ms * (target_ne - self.norepinephrine) / 700.0

        target_cort = b['cortisol'] + 0.8 * threat_act * (1.0 - 0.5 * self.oxytocin)
        self.cortisol += dt_ms * (target_cort - self.cortisol) / 9000.0

        target_mel = night_signal * (1.0 - 0.5 * self.norepinephrine)
        self.melatonin += dt_ms * (target_mel - self.melatonin) / 4000.0

        if resting and night_signal > 0.3:
            target_ado, tau_ado = 0.05, 3000.0
        else:
            target_ado, tau_ado = 1.0, 60000.0
        self.adenosine += dt_ms * (target_ado - self.adenosine) / tau_ado

        target_oxt = b['oxytocin'] + 0.7 * reward_act - 0.2 * threat_act
        self.oxytocin += dt_ms * (target_oxt - self.oxytocin) / 5000.0

        target_avp = b['vasopressin'] + 0.7 * threat_act
        self.vasopressin += dt_ms * (target_avp - self.vasopressin) / 4000.0

        target_ach = 0.2 + 0.75 * sensory_act * (1.0 - self.melatonin) + 0.1 * threat_act
        self.acetylcholine += dt_ms * (target_ach - self.acetylcholine) / 1500.0

        target_end = b['endorphin'] + 0.6 * self.cortisol
        self.endorphin += dt_ms * (target_end - self.endorphin) / 6000.0

        target_test = b['testosterone'] + 0.35 * threat_act + 0.25 * reward_act
        self.testosterone += dt_ms * (target_test - self.testosterone) / 10000.0

        target_gaba = 0.2 + 0.35 * self.serotonin + 0.4 * self.adenosine - 0.25 * self.norepinephrine
        self.gaba += dt_ms * (target_gaba - self.gaba) / 2500.0

        for name in self.base:
            setattr(self, name, float(np.clip(getattr(self, name), 0.0, 1.0)))

    @property
    def arousal_gain(self):
        ne = self.norepinephrine
        return (0.7 + 1.0 * ne - 0.6 * ne ** 2) * (1.0 - 0.3 * self.gaba)

    @property
    def excitatory_boost(self):
        return (0.7 + 0.8 * self.dopamine) * (0.8 + 0.4 * self.acetylcholine)

    @property
    def aggression_damping(self):
        return (1.2 - 0.7 * self.serotonin) * (1.2 - 0.5 * self.gaba) / (1.0 + 0.3 * self.testosterone)

    @property
    def is_night(self):
        return self.melatonin > 0.35

    @property
    def sleep_pressure(self):
        return self.adenosine

    def snapshot(self):
        return {k: round(getattr(self, k), 3) for k in self.base}

    def to_dict(self):
        return {k: getattr(self, k) for k in self.base}

    def load_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class CircadianClock:
    """Циркадные часы Viktim -- даёт мозгу настоящее "ощущение времени суток"."""

    def __init__(self, hours_per_real_second=24.0 / 600.0, start_hour=8.0):
        self.hours_per_ms = hours_per_real_second / 1000.0
        self.hour = start_hour
        self.day_count = 0
        self.age_hours_total = 0.0

    def step(self, dt_ms):
        prev_hour = self.hour
        self.hour = (self.hour + dt_ms * self.hours_per_ms) % 24.0
        self.age_hours_total += dt_ms * self.hours_per_ms
        wrapped = self.hour < prev_hour
        if wrapped:
            self.day_count += 1
        return wrapped

    @property
    def night_signal(self):
        return max(0.0, math.cos(2 * math.pi * (self.hour - 3.0) / 24.0))

    def time_of_day_code(self, n_units=TIME_CODE_UNITS):
        prefs = np.linspace(0, 24, n_units, endpoint=False)
        diff = np.abs((self.hour - prefs + 12) % 24 - 12)
        return np.exp(-(diff ** 2) / (2 * (2.0 ** 2)))

    def to_dict(self):
        return dict(hour=self.hour, day_count=self.day_count, age_hours_total=self.age_hours_total)

    def load_dict(self, d):
        self.hour = d["hour"]
        self.day_count = d["day_count"]
        self.age_hours_total = d["age_hours_total"]


class Brain:
    def __init__(self, seed=None, hours_per_real_second=24.0 / 600.0):
        rng = np.random.default_rng(seed)
        self._rng = rng
        n_intra = 17500
        n_inter = TOTAL_SYNAPSES_TARGET - n_intra * 2

        self.left = Hemisphere(rng, n_intra)
        self.right = Hemisphere(rng, n_intra)

        def build_cc(n):
            pre, post, w, exc = [], [], [], []
            for _ in range(n):
                p = rng.integers(REGIONS["PFC"].start, REGIONS["MOTOR"].stop)
                q = rng.integers(REGIONS["PFC"].start, REGIONS["MOTOR"].stop)
                pre.append(p); post.append(q); w.append(rng.uniform(0.05, 0.25)); exc.append(True)
            return np.array(pre), np.array(post), np.array(w), np.array(exc)

        pre, post, w, exc = build_cc(n_inter // 2)
        self.cc_l2r = SynapseMatrix(N_HEMI, pre, post, w, exc)
        pre, post, w, exc = build_cc(n_inter - n_inter // 2)
        self.cc_r2l = SynapseMatrix(N_HEMI, pre, post, w, exc)

        self.neuromod = Neuromodulators()
        self.clock = CircadianClock(hours_per_real_second=hours_per_real_second)
        self.longterm_memory = LongTermMemory()
        self.dt = self.left.pop.dt
        self.total_synapses = n_intra * 2 + n_inter
        motor_len = REGIONS["MOTOR"].stop - REGIONS["MOTOR"].start
        self._motor_smooth = np.zeros(motor_len * 2)  # оба полушария
        self._step_count = 0
        self._was_night = False
        self.remap_count = 0

    # ---------- сенсорный вход ----------
    def inject_sensory(self, threat_level, reward_level, position_signal, time_code):
        I_left = np.zeros(N_HEMI)
        I_right = np.zeros(N_HEMI)

        pos = np.asarray(position_signal, dtype=float)
        half = POSITION_SIGNAL_LEN // 2
        I_left[REGIONS["SENSORY"]] = pos[:half] * 0.6
        I_right[REGIONS["SENSORY"]] = pos[half:] * 0.6

        I_left[THREAT] = threat_level * 1.2
        I_right[THREAT] = threat_level * 1.2
        I_left[REWARD] = reward_level * 1.0
        I_right[REWARD] = reward_level * 1.0

        I_left[REGIONS["HIPPO"]] += time_code * 0.35
        I_right[REGIONS["HIPPO"]] += time_code * 0.35

        return I_left, I_right

    def _replay_engram(self, sensory, valence, n_ticks=25):
        """
        Реиграть один энграмм долговременной памяти во время сна -- внутренне
        сгенерированная активность, повторяющая исходный сенсорный паттерн
        (аналог гиппокампальных sharp-wave ripples), которая через ту же
        Хеббовскую/STDP пластичность (уровень 2) закрепляет соответствующие
        корковые пути СИЛЬНЕЕ, чем в момент бодрствования -- системная
        консолидация памяти (Complementary Learning Systems).
        """
        threat_level = max(0.0, -valence)
        reward_level = max(0.0, valence)
        zero_time = np.zeros(TIME_CODE_UNITS)
        exc_boost = self.neuromod.excitatory_boost
        for _ in range(n_ticks):
            I_ext_l, I_ext_r = self.inject_sensory(threat_level, reward_level, sensory, zero_time)
            spikes_l = self.left.pop.spiked
            spikes_r = self.right.pop.spiked
            I_syn_l = self.left.syn.step(spikes_l, self.dt, neuromod_gain=exc_boost)
            I_syn_r = self.right.syn.step(spikes_r, self.dt, neuromod_gain=exc_boost)
            self.left.pop.step(I_syn_l, I_ext_l, gain=self.neuromod.arousal_gain * 0.6)
            self.right.pop.step(I_syn_r, I_ext_r, gain=self.neuromod.arousal_gain * 0.6)

        cpg = plx.critical_period_gain(self.clock.age_hours_total) * 1.5  # усиленная пластичность во сне
        for hemi in (self.left, self.right):
            plx.apply_hebbian_plasticity(hemi.syn, hemi.pop, hemi.pop,
                                          self.neuromod.acetylcholine, self.neuromod.dopamine, cpg)

    # ---------- основной шаг ----------
    def step(self, threat_level=0.0, reward_level=0.0, position_signal=None, deprived_local=None):
        if position_signal is None:
            position_signal = np.zeros(POSITION_SIGNAL_LEN)

        self.clock.step(self.dt)
        time_code = self.clock.time_of_day_code(TIME_CODE_UNITS)
        I_ext_l, I_ext_r = self.inject_sensory(threat_level, reward_level, position_signal, time_code)

        gain = self.neuromod.arousal_gain
        exc_boost = self.neuromod.excitatory_boost
        motor_damp = self.neuromod.aggression_damping

        spikes_l = self.left.pop.spiked
        spikes_r = self.right.pop.spiked

        I_syn_l = self.left.syn.step(spikes_l, self.dt, neuromod_gain=exc_boost)
        I_syn_r = self.right.syn.step(spikes_r, self.dt, neuromod_gain=exc_boost)
        I_cc_to_r = self.cc_l2r.step(spikes_l, self.dt, neuromod_gain=exc_boost)
        I_cc_to_l = self.cc_r2l.step(spikes_r, self.dt, neuromod_gain=exc_boost)

        noise = 0.01 + 0.02 * self.neuromod.cortisol + 0.015 * self.neuromod.adenosine

        spikes_l = self.left.pop.step(I_syn_l + I_cc_to_l, I_ext_l, gain=gain, noise_std=noise)
        spikes_r = self.right.pop.step(I_syn_r + I_cc_to_r, I_ext_r, gain=gain, noise_std=noise)

        reward_act = float(np.mean([self.left.region_activity(REWARD), self.right.region_activity(REWARD)]))
        threat_act = float(np.mean([self.left.region_activity(THREAT), self.right.region_activity(THREAT)]))
        sensory_act = float(np.mean([self.left.region_activity(REGIONS["SENSORY"]),
                                      self.right.region_activity(REGIONS["SENSORY"])]))
        assoc_act = float(np.mean([self.left.region_activity(REGIONS["PFC"]),
                                    self.right.region_activity(REGIONS["PFC"])]))
        hippo_act = float(np.mean([self.left.region_activity(REGIONS["HIPPO"]),
                                    self.right.region_activity(REGIONS["HIPPO"])]))
        motor_act = float(np.mean([self.left.region_activity(REGIONS["MOTOR"]),
                                    self.right.region_activity(REGIONS["MOTOR"])]))
        resting = motor_act < 0.02

        self.neuromod.update(
            self.dt, reward_act=reward_act, threat_act=threat_act, sensory_act=sensory_act,
            assoc_act=assoc_act, hippo_act=hippo_act, motor_act=motor_act,
            night_signal=self.clock.night_signal, resting=resting,
        )

        # ---------- 7 уровней пластичности ----------
        cpg = plx.critical_period_gain(self.clock.age_hours_total)

        for hemi in (self.left, self.right):
            plx.apply_hebbian_plasticity(
                hemi.syn, hemi.pop, hemi.pop,
                ach_level=self.neuromod.acetylcholine, dopamine_level=self.neuromod.dopamine,
                critical_period_gain=cpg,
            )
        plx.apply_hebbian_plasticity(self.cc_l2r, self.left.pop, self.right.pop,
                                      self.neuromod.acetylcholine, self.neuromod.dopamine, cpg)
        plx.apply_hebbian_plasticity(self.cc_r2l, self.right.pop, self.left.pop,
                                      self.neuromod.acetylcholine, self.neuromod.dopamine, cpg)

        plx.apply_homeostatic_scaling(self.left.syn, self.left.pop, self.dt)
        plx.apply_homeostatic_scaling(self.right.syn, self.right.pop, self.dt)

        self._step_count += 1
        if self._step_count % 3000 == 0:
            deprived_local = deprived_local or {}
            all_motor = np.arange(REGIONS["MOTOR"].start, REGIONS["MOTOR"].stop)
            for side, hemi in (("left", self.left), ("right", self.right)):
                dep = np.array(deprived_local.get(side, []), dtype=int)
                if len(dep) > 0:
                    donors = np.setdiff1d(all_motor, dep)
                    donors = np.concatenate([donors, np.arange(REGIONS["PFC"].start, REGIONS["PFC"].stop)])
                    n = plx.apply_cortical_remapping(hemi.syn, self._rng, dep, donors)
                    if n:
                        self.remap_count += 1
                else:
                    plx.apply_structural_plasticity(hemi.syn, self._rng, REGIONS, FLOW_MAP, REGION_NAMES)

        # уровень 7 (циркадная консолидация сна + replay долговременной памяти)
        is_night_now = self.neuromod.is_night
        replayed = 0
        if is_night_now and not self._was_night and self.neuromod.sleep_pressure > 0.4:
            plx.apply_sleep_consolidation(self.left.syn)
            plx.apply_sleep_consolidation(self.right.syn)
            for sensory, valence in self.longterm_memory.nightly_replay():
                self._replay_engram(sensory, valence)
                replayed += 1
        self._was_night = is_night_now

        motor_now = np.concatenate([spikes_l[REGIONS["MOTOR"]], spikes_r[REGIONS["MOTOR"]]]).astype(float)
        alpha = 0.05
        self._motor_smooth = (1 - alpha) * self._motor_smooth + alpha * motor_now / motor_damp

        return {
            "motor": self._motor_smooth.copy(),
            "neuromodulators": self.neuromod.snapshot(),
            "hour": round(self.clock.hour, 2),
            "day": self.clock.day_count,
            "is_night": is_night_now,
            "sleep_pressure": round(self.neuromod.sleep_pressure, 3),
            "critical_period_gain": round(float(cpg), 3),
            "remap_count": self.remap_count,
            "spikes_total": int(spikes_l.sum() + spikes_r.sum()),
            "memories_replayed_tonight": replayed,
            "long_term_memories": len(self.longterm_memory.engrams),
            "consolidated_memories": self.longterm_memory.consolidated_count(),
        }

    # ---------- ДОЛГОВРЕМЕННАЯ ПАМЯТЬ: сохранение/загрузка между сессиями ----------
    def to_dict(self):
        return {
            "version": 3,
            "left_syn": self.left.syn_state(),
            "right_syn": self.right.syn_state(),
            "cc_l2r": dict(pre=self.cc_l2r.pre.tolist(), post=self.cc_l2r.post.tolist(),
                           w=self.cc_l2r.w.tolist(), sign=self.cc_l2r.sign.tolist()),
            "cc_r2l": dict(pre=self.cc_r2l.pre.tolist(), post=self.cc_r2l.post.tolist(),
                           w=self.cc_r2l.w.tolist(), sign=self.cc_r2l.sign.tolist()),
            "neuromod": self.neuromod.to_dict(),
            "clock": self.clock.to_dict(),
            "remap_count": self.remap_count,
            "longterm_memory": self.longterm_memory.to_dict(),
        }

    def load_dict(self, d):
        self.left.load_syn_state(d["left_syn"])
        self.right.load_syn_state(d["right_syn"])
        self.cc_l2r.pre = np.array(d["cc_l2r"]["pre"])
        self.cc_l2r.post = np.array(d["cc_l2r"]["post"])
        self.cc_l2r.w = np.array(d["cc_l2r"]["w"], dtype=float)
        self.cc_l2r.sign = np.array(d["cc_l2r"]["sign"], dtype=float)
        self.cc_r2l.pre = np.array(d["cc_r2l"]["pre"])
        self.cc_r2l.post = np.array(d["cc_r2l"]["post"])
        self.cc_r2l.w = np.array(d["cc_r2l"]["w"], dtype=float)
        self.cc_r2l.sign = np.array(d["cc_r2l"]["sign"], dtype=float)
        self.neuromod.load_dict(d["neuromod"])
        self.clock.load_dict(d["clock"])
        self.remap_count = d.get("remap_count", 0)
        self.longterm_memory.load_dict(d.get("longterm_memory", {}))

    def save_state(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def load_state(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.load_dict(json.load(f))
