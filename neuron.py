# -*- coding: utf-8 -*-
"""
neuron.py
Adaptive Exponential Integrate-and-Fire (AdEx) neuron.
Brette & Gerstner (2005) model. Векторизовано через numpy для 150 нейронов.

dV/dt = ( -gL*(V-EL) + gL*DeltaT*exp((V-VT)/DeltaT) + I_syn + I_ext - w ) / C
dw/dt = ( a*(V-EL) - w ) / tau_w
spike:  V > Vpeak  ->  V = Vr ; w += b
"""

import numpy as np


class AdExPopulation:
    """
    Векторизованная популяция AdEx-нейронов (используется для всех 150 клеток мозга).
    Каждый нейрон может иметь свой набор параметров (RS / FS / бурстинг), заданный
    массивами -- это позволяет иметь как возбуждающие "regular spiking", так и
    тормозные "fast spiking" клетки в одной популяции.
    """

    def __init__(self, n, excitatory_mask, dt_ms=0.5):
        self.n = n
        self.dt = dt_ms  # мс
        self.excitatory = excitatory_mask.astype(bool)  # True = глутамат, False = ГАМК

        # Параметры по умолчанию (регулярно-спайкующие возбуждающие клетки, pF/nS/mV/ms)
        self.C = np.full(n, 200.0)       # pF
        self.gL = np.full(n, 10.0)       # nS
        self.EL = np.full(n, -70.0)      # mV
        self.VT = np.full(n, -50.0)      # mV
        self.DeltaT = np.full(n, 2.0)    # mV
        self.a = np.full(n, 2.0)         # nS
        self.tau_w = np.full(n, 30.0)    # ms
        self.b = np.full(n, 0.02)        # nA -> используем nA=1000 pA согласовано с I в nA*? см. ниже
        self.Vr = np.full(n, -58.0)      # mV
        self.Vpeak = np.full(n, 0.0)     # mV

        # Тормозные (ГАМК) нейроны -> fast-spiking: меньше адаптации, более узкий рефрактер
        inhib = ~self.excitatory
        self.a[inhib] = 0.5
        self.b[inhib] = 0.005
        self.tau_w[inhib] = 10.0
        self.VT[inhib] = -47.0

        self.V = self.EL.copy() + np.random.uniform(-2, 2, n)
        self.w = np.zeros(n)
        self.spiked = np.zeros(n, dtype=bool)

        # рефрактерный период (мс) после спайка, чтобы не залипать в экспоненте
        self.refractory_ms = 2.0
        self.refractory_timer = np.zeros(n)

    def step(self, I_syn, I_ext=0.0, gain=1.0, noise_std=0.0):
        """
        Один шаг интегрирования (self.dt мс).
        I_syn, I_ext -- в nA (наноамперы), одна на нейрон (I_syn) или скаляр/массив (I_ext).
        gain -- множитель чувствительности (модулируется норадреналином/кортизолом).
        Возвращает булев массив self.spiked.
        """
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
        return self.spiked


class SynapseMatrix:
    """
    Разреженная матрица синапсов между нейронами одной популяции.
    Хранится в COO-формате (pre, post, weight, sign) для 15000 связей.
    Ток от каждого пресинаптического спайка складывается в экспоненциально
    затухающий синаптический ток на постсинаптическом нейроне.
    """

    def __init__(self, n, pre_idx, post_idx, weights, is_excitatory, tau_syn_ms=5.0):
        self.n = n
        self.pre = pre_idx
        self.post = post_idx
        self.w = weights.astype(np.float64)
        self.sign = np.where(is_excitatory, 1.0, -1.0)
        self.tau_syn = tau_syn_ms
        self.I = np.zeros(n)  # синаптический ток на каждый постсинаптический нейрон

    def step(self, presyn_spikes, dt, neuromod_gain=1.0):
        # экспоненциальный распад тока
        self.I *= np.exp(-dt / self.tau_syn)
        fired = presyn_spikes[self.pre]
        if fired.any():
            contrib = self.w[fired] * self.sign[fired] * neuromod_gain
            np.add.at(self.I, self.post[fired], contrib)
        return self.I

    def apply_stdp(self, pre_spike_trace, post_spike_trace, lr=0.0005, w_max=5.0):
        """Лёгкий STDP: усиление при корреляции pre->post, ослабление в обратную сторону."""
        dw = lr * (pre_spike_trace[self.pre] * post_spike_trace[self.post]
                    - 0.5 * post_spike_trace[self.pre] * pre_spike_trace[self.post])
        self.w = np.clip(self.w + dw, 0.0, w_max)
