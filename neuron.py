# -*- coding: utf-8 -*-
"""
neuron.py
AdEx-нейрон (Brette & Gerstner, 2005) + синаптическая матрица с УРОВНЕМ 1
пластичности (кратковременная, Tsodyks-Markram: facilitation/depression)
встроенным прямо в шаг интегрирования, т.к. она работает на масштабе мс
одновременно с самим спайком.

Остальные уровни пластичности (2 - STDP/Hebb, 3 - метапластичность BCM,
4 - гомеостатическое масштабирование, 5 - нейромодуляторный гейтинг,
6 - структурная пластичность, 7 - системная/циркадная консолидация)
реализованы в plasticity.py и оркестрируются из brain.py.
"""

import numpy as np


class AdExPopulation:
    """Векторизованная популяция AdEx-нейронов с трассировкой активности для BCM/гомеостаза."""

    def __init__(self, n, excitatory_mask, dt_ms=0.5):
        self.n = n
        self.dt = dt_ms
        self.excitatory = excitatory_mask.astype(bool)

        self.C = np.full(n, 200.0)
        self.gL = np.full(n, 10.0)
        self.EL = np.full(n, -70.0)
        self.VT = np.full(n, -50.0)
        self.DeltaT = np.full(n, 2.0)
        self.a = np.full(n, 2.0)
        self.tau_w = np.full(n, 30.0)
        self.b = np.full(n, 0.02)
        self.Vr = np.full(n, -58.0)
        self.Vpeak = np.full(n, 0.0)

        inhib = ~self.excitatory
        self.a[inhib] = 0.5
        self.b[inhib] = 0.005
        self.tau_w[inhib] = 10.0
        self.VT[inhib] = -47.0

        self.V = self.EL.copy() + np.random.uniform(-2, 2, n)
        self.w = np.zeros(n)
        self.spiked = np.zeros(n, dtype=bool)

        self.refractory_ms = 2.0
        self.refractory_timer = np.zeros(n)

        # --- для уровня 2/3 пластичности: экспоненциальные следы спайков ---
        self.tau_trace = 20.0  # мс
        self.spike_trace = np.zeros(n)          # быстрый след (Hebb/STDP)

        # --- для уровня 3 (метапластичность BCM) и уровня 4 (гомеостаз) ---
        self.tau_avg_rate = 5000.0               # мс, медленное скользящее среднее частоты
        self.avg_rate = np.full(n, 0.01)          # доля спайков/шаг, скользящее среднее
        self.target_rate = np.full(n, 0.02)       # гомеостатическая целевая активность

    def step(self, I_syn, I_ext=0.0, gain=1.0, noise_std=0.0):
        active = self.refractory_timer <= 0
        I_total = (I_syn + I_ext) * gain
        if noise_std > 0:
            I_total = I_total + np.random.normal(0, noise_std, self.n)

        expo = self.gL * self.DeltaT * np.exp(
            np.clip((self.V - self.VT) / self.DeltaT, -10, 30)
        )
        dV = (-self.gL * (self.V - self.EL) + expo + I_total * 1000.0 - self.w) / self.C
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w

        self.V = np.where(active, self.V + dV * self.dt, self.Vr)
        self.w = np.where(active, self.w + dw * self.dt, self.w)

        self.spiked = active & (self.V >= self.Vpeak)
        self.V = np.where(self.spiked, self.Vr, np.clip(self.V, -100, self.Vpeak))
        self.w = np.where(self.spiked, self.w + self.b * 1000.0, self.w)

        self.refractory_timer = np.where(
            self.spiked, self.refractory_ms, np.maximum(self.refractory_timer - self.dt, 0)
        )

        # обновление следов активности (используются уровнями пластичности 2,3,4)
        self.spike_trace *= np.exp(-self.dt / self.tau_trace)
        self.spike_trace += self.spiked.astype(float)
        self.avg_rate += (self.dt / self.tau_avg_rate) * (self.spiked.astype(float) - self.avg_rate)

        return self.spiked


class SynapseMatrix:
    """
    Разреженная COO-матрица синапсов.
    Содержит УРОВЕНЬ 1 пластичности (кратковременная, Tsodyks-Markram) прямо
    в step(): каждый синапс имеет ресурс x (доступный нейротрансмиттер) и
    коэффициент использования u (вероятность высвобождения), которые
    динамически меняются при каждом пресинаптическом спайке -- это даёт
    кратковременную депрессию/фасилитацию независимо от долговременного веса w.

    Долговременный вес w изменяется отдельно, через apply_plasticity()
    (уровни 2/3/5, см. plasticity.py), и структурно перестраивается через
    структурную пластичность (уровень 6, plasticity.py).
    """

    def __init__(self, n_post, pre_idx, post_idx, weights, is_excitatory,
                 tau_syn_ms=5.0, region_of=None):
        self.n_post = n_post
        self.pre = pre_idx
        self.post = post_idx
        self.w = weights.astype(np.float64)
        self.sign = np.where(is_excitatory, 1.0, -1.0)
        self.is_excitatory = np.asarray(is_excitatory, dtype=bool)
        self.tau_syn = tau_syn_ms
        self.I = np.zeros(n_post)
        self.active = np.ones(len(pre_idx), dtype=bool)  # для структурной пластичности (ур.6)

        # region_of: массив region_id для каждого нейрона популяции (используется
        # структурной пластичностью, чтобы восстанавливать связи в биологически
        # правдоподобных направлениях, а не абсолютно случайно)
        self.region_of = region_of

        n_syn = len(pre_idx)
        # --- Tsodyks-Markram параметры кратковременной пластичности ---
        # возбуждающие синапсы -- преимущественно депрессирующие (типично для E->E коры),
        # тормозные -- слабо фасилитирующие (типично для части интернейронов)
        self.U0 = np.where(self.is_excitatory, 0.25, 0.15)
        self.tau_rec = np.where(self.is_excitatory, 300.0, 100.0)   # мс, восстановление ресурса
        self.tau_facil = np.where(self.is_excitatory, 50.0, 400.0)  # мс, фасилитация
        self.u = self.U0.copy()
        self.x = np.ones(n_syn)

        # накопитель "дневного" использования синапса -- нужен уровню 7 (консолидация сна)
        self.daily_usage = np.zeros(n_syn)

    def step(self, presyn_spikes, dt, neuromod_gain=1.0):
        # пассивное восстановление ресурсов (уровень 1, кратковременная пластичность)
        self.x += dt * (1.0 - self.x) / self.tau_rec
        self.u += dt * (self.U0 - self.u) / self.tau_facil

        self.I *= np.exp(-dt / self.tau_syn)

        fired = presyn_spikes[self.pre] & self.active
        if fired.any():
            release = self.u[fired] * self.x[fired]           # эффективная доля высвобождения
            self.x[fired] -= release                           # истощение ресурса (депрессия)
            self.u[fired] += self.U0[fired] * (1.0 - self.u[fired])  # фасилитация от Ca2+ притока

            contrib = self.w[fired] * self.sign[fired] * neuromod_gain * release
            np.add.at(self.I, self.post[fired], contrib)
            np.add.at(self.daily_usage, np.nonzero(fired)[0], np.abs(contrib))
        return self.I
