# -*- coding: utf-8 -*-
"""
brain.py
Мозг Viktim v2: 200 AdEx-нейронов (100 + 100 в двух полушариях), каждое
полушарие разделено на 5 функциональных регионов по 20 нейронов; ~15000
синапсов (внутрирегиональные/межрегиональные внутри полушария + межполушарные
"мозолистое тело"); 12 нейромедиаторных/гормональных осей; циркадные часы
с ощущением времени суток; семь уровней пластичности (см. plasticity.py).

Регионы (индексы локальные, 0..99, одинаковы для обоих полушарий):
  SENSORY  0-19   сенсорная/зрительная кора      -- вход из workspace
  LIMBIC   20-39  миндалина + прилежащее ядро     -- 20-29 угроза, 30-39 награда
  PFC      40-59  префронтальная ассоциативная    -- интеграция, принятие решений
  HIPPO    60-79  гиппокамп                        -- память контекста + ощущение времени
  MOTOR    80-99  моторная кора                    -- выход к суставам стикмана
"""

import math
import numpy as np
from neuron import AdExPopulation, SynapseMatrix
import plasticity as plx

N_HEMI = 100
N_TOTAL = N_HEMI * 2  # 200

REGIONS = {
    "SENSORY": slice(0, 20),
    "LIMBIC":  slice(20, 40),
    "PFC":     slice(40, 60),
    "HIPPO":   slice(60, 80),
    "MOTOR":   slice(80, 100),
}
THREAT = slice(20, 30)   # подзона LIMBIC
REWARD = slice(30, 40)   # подзона LIMBIC
REGION_NAMES = list(REGIONS.keys())

FLOW_MAP = {
    "SENSORY": ["PFC", "LIMBIC", "HIPPO"],
    "LIMBIC":  ["PFC", "MOTOR", "HIPPO"],
    "PFC":     ["PFC", "MOTOR", "LIMBIC", "HIPPO"],
    "HIPPO":   ["PFC", "LIMBIC"],
    "MOTOR":   ["PFC"],
}

TOTAL_SYNAPSES_TARGET = 15000


def _build_hemisphere_connectivity(rng, n_synapses=6500):
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
        Это гарантирует релаксацию к физиологически осмысленному значению,
        а не "убегание" от него, как было бы при раздельной форсирующей
        add-компоненте без привязки к tau.
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

        # аденозин ("давление сна", process S): медленно растёт в бодрствовании
        # пропорционально прошедшему времени, быстро сбрасывается во время отдыха ночью
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

    # ---- производные величины, используемые остальной частью мозга ----
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


class CircadianClock:
    """Циркадные часы Viktim -- даёт мозгу настоящее "ощущение времени суток"."""

    def __init__(self, hours_per_real_second=24.0 / 600.0, start_hour=8.0):
        # по умолчанию: один симулированный день = 10 реальных минут сессии
        self.hours_per_ms = hours_per_real_second / 1000.0
        self.hour = start_hour
        self.day_count = 0
        self.age_hours_total = 0.0  # для критического периода (уровень 7 пластичности), не оборачивается

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
        # пик "ночи" около 03:00, плавно спадает к нулю днём
        return max(0.0, math.cos(2 * math.pi * (self.hour - 3.0) / 24.0))

    def time_of_day_code(self, n_units=20):
        """Популяционное кодирование часа суток -- 'time cells' для гиппокампа."""
        prefs = np.linspace(0, 24, n_units, endpoint=False)
        diff = np.abs((self.hour - prefs + 12) % 24 - 12)
        return np.exp(-(diff ** 2) / (2 * (2.0 ** 2)))


class Brain:
    def __init__(self, seed=None, hours_per_real_second=24.0 / 600.0):
        rng = np.random.default_rng(seed)
        self._rng = rng
        n_intra = 6500
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
        self.dt = self.left.pop.dt
        self.total_synapses = n_intra * 2 + n_inter
        self._motor_smooth = np.zeros(20 * 2)
        self._step_count = 0
        self._was_night = False
        self.remap_count = 0  # число раундов коркового ремаппинга (для угасания фантомной боли)

    # ---------- сенсорный вход ----------
    def inject_sensory(self, threat_level, reward_level, position_signal, time_code):
        I_left = np.zeros(N_HEMI)
        I_right = np.zeros(N_HEMI)

        pos = np.asarray(position_signal, dtype=float)
        I_left[REGIONS["SENSORY"]] = pos[:20] * 0.6
        I_right[REGIONS["SENSORY"]] = pos[20:] * 0.6

        I_left[THREAT] = threat_level * 1.2
        I_right[THREAT] = threat_level * 1.2
        I_left[REWARD] = reward_level * 1.0
        I_right[REWARD] = reward_level * 1.0

        # "ощущение времени": популяционный код часа суток подаётся в гиппокамп
        I_left[REGIONS["HIPPO"]] += time_code * 0.35
        I_right[REGIONS["HIPPO"]] += time_code * 0.35

        return I_left, I_right

    # ---------- основной шаг ----------
    def step(self, threat_level=0.0, reward_level=0.0, position_signal=None, deprived_local=None):
        """
        deprived_local: опционально {"left": [индексы], "right": [индексы]} --
        локальные индексы нейронов MOTOR-региона, представлявших утраченную
        часть тела Viktim (см. viktim.py). Если задано, структурная
        пластичность (уровень 6) в эту итерацию делает не случайный, а
        целенаправленный корковый ремаппинг -- см. plasticity.apply_cortical_remapping.
        """
        if position_signal is None:
            position_signal = np.zeros(40)

        wrapped_day = self.clock.step(self.dt)
        time_code = self.clock.time_of_day_code(20)
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

        # --- активность по регионам (для нейромодуляторов и поведения) ---
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
        cpg = plx.critical_period_gain(self.clock.age_hours_total)  # часть уровня 7

        # уровни 2+3+5 (Hebb/STDP + метапластичность BCM + нейромодуляторный гейтинг)
        for hemi in (self.left, self.right):
            plx.apply_hebbian_plasticity(
                hemi.syn, hemi.pop, hemi.pop,
                ach_level=self.neuromod.acetylcholine, dopamine_level=self.neuromod.dopamine,
                critical_period_gain=cpg,
            )
        # межполушарные синапсы тоже пластичны
        plx.apply_hebbian_plasticity(self.cc_l2r, self.left.pop, self.right.pop,
                                      self.neuromod.acetylcholine, self.neuromod.dopamine, cpg)
        plx.apply_hebbian_plasticity(self.cc_r2l, self.right.pop, self.left.pop,
                                      self.neuromod.acetylcholine, self.neuromod.dopamine, cpg)

        # уровень 4 (гомеостатическое масштабирование) -- каждый шаг, но с медленной dt/tau динамикой
        plx.apply_homeostatic_scaling(self.left.syn, self.left.pop, self.dt)
        plx.apply_homeostatic_scaling(self.right.syn, self.right.pop, self.dt)

        # уровень 6 (структурная пластичность) -- редко, раз в ~несколько симулированных минут
        self._step_count += 1
        if self._step_count % 3000 == 0:
            deprived_local = deprived_local or {}
            all_motor = np.arange(REGIONS["MOTOR"].start, REGIONS["MOTOR"].stop)
            for side, hemi in (("left", self.left), ("right", self.right)):
                dep = np.array(deprived_local.get(side, []), dtype=int)
                if len(dep) > 0:
                    donors = np.setdiff1d(all_motor, dep)  # соседние живые части тела + PFC-связка
                    donors = np.concatenate([donors, np.arange(REGIONS["PFC"].start, REGIONS["PFC"].stop)])
                    n = plx.apply_cortical_remapping(hemi.syn, self._rng, dep, donors)
                    if n:
                        self.remap_count += 1
                else:
                    plx.apply_structural_plasticity(hemi.syn, self._rng, REGIONS, FLOW_MAP, REGION_NAMES)

        # уровень 7 (циркадная консолидация сна) -- один раз за переход в ночную фазу
        is_night_now = self.neuromod.is_night
        if is_night_now and not self._was_night and self.neuromod.sleep_pressure > 0.4:
            plx.apply_sleep_consolidation(self.left.syn)
            plx.apply_sleep_consolidation(self.right.syn)
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
        }
