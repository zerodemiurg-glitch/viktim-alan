# -*- coding: utf-8 -*-
"""
plasticity.py
Семь уровней пластичности мозга Viktim, по аналогии с иерархией пластичности
человеческого мозга (от миллисекунд до "дней жизни" сети):

  Уровень 1 -- Кратковременная синаптическая (STP, Tsodyks-Markram)
              -> реализована в neuron.SynapseMatrix.step() (мс-масштаб)
  Уровень 2 -- Долговременная Хеббовская / STDP (спайк-тайминг-зависимая)
  Уровень 3 -- Метапластичность (скользящий порог BCM: "пластичность самой пластичности")
  Уровень 4 -- Гомеостатическая пластичность (синаптическое масштабирование, Turrigiano)
  Уровень 5 -- Нейромодуляторный гейтинг (three-factor rule: eligibility x нейромодулятор)
  Уровень 6 -- Структурная пластичность (синаптогенез/прунинг связей)
  Уровень 7 -- Системная/циркадная консолидация (гомеостаз сна по Tononi & Cirelli
              + возрастное "закрытие критического периода")

Уровни 2,3,5 объединены в apply_hebbian_plasticity(), т.к. в реальном мозге они
работают на одном и том же синапсе одновременно и разделять их искусственно
в отдельные проходы было бы менее реалистично, чем биологически корректно.
"""

import numpy as np


def apply_hebbian_plasticity(syn, pre_pop, post_pop, ach_level, dopamine_level,
                              critical_period_gain, lr_base=0.0004, w_max=5.0):
    """
    Уровни 2 + 3 + 5 одновременно:

    2) STDP/Hebb: dw ~ pre_trace * post_spike (потенциация) - post_trace * pre_spike (депрессия)
    3) Метапластичность BCM: знак и сила пластичности зависят от скользящего среднего
       постсинаптической активности (theta) -- если нейрон "перевозбуждён" последнее время,
       порог LTP растёт, и то же событие может дать LTD вместо LTP (стабилизация сети).
    5) Нейромодуляторный гейтинг (three-factor rule): итоговая скорость обучения
       домножается на ацетилхолин (внимание/готовность к обучению) и дофамин
       (сигнал подкрепления) -- ровно так, как нейромодуляторы управляют LTP in vivo.
    """
    theta = np.clip(post_pop.avg_rate / (post_pop.target_rate + 1e-6), 0.2, 5.0)  # BCM-порог

    pre_trace = pre_pop.spike_trace[syn.pre]
    post_trace = post_pop.spike_trace[syn.post]
    post_spiked = post_pop.spiked[syn.post].astype(float)
    pre_spiked = syn_pre_spiked = pre_pop.spiked[syn.pre].astype(float)
    theta_post = theta[syn.post]

    # BCM-подобное правило: постсинаптическая активность выше своего порога -> LTP, ниже -> LTD
    ltp = pre_trace * post_spiked * np.maximum(0.0, 1.0 - theta_post * 0.5)
    ltd = post_trace * pre_spiked * theta_post * 0.5

    neuromod_gate = (0.25 + 0.9 * ach_level) * (0.4 + 0.9 * dopamine_level)
    lr = lr_base * neuromod_gate * critical_period_gain

    dw = lr * (ltp - ltd)
    syn.w = np.clip(syn.w + dw, 0.0, w_max)


def apply_homeostatic_scaling(syn, post_pop, dt_ms, tau_scale_ms=20000.0):
    """
    Уровень 4 -- гомеостатическая пластичность (синаптическое масштабирование).
    Медленно (масштаб десятков секунд симулированного времени) перенастраивает
    ВСЕ афферентные веса нейрона мультипликативно так, чтобы средняя частота
    его разрядов держалась около целевого уровня -- иначе Хеббовская пластичность
    (уровень 2) была бы нестабильна (runaway excitation / полная тишина).
    """
    rate_ratio = post_pop.target_rate / (post_pop.avg_rate + 1e-4)
    correction = 1.0 + (dt_ms / tau_scale_ms) * np.clip(rate_ratio - 1.0, -0.5, 0.5)
    factor_per_synapse = correction[syn.post]
    syn.w *= factor_per_synapse
    np.clip(syn.w, 0.0, 5.0, out=syn.w)


def apply_cortical_remapping(syn, rng, deprived_idx, donor_idx, n_new=25, w_init=(0.05, 0.2)):
    """
    Часть уровня 6 -- КОРКОВОЕ РЕМАППИНГ (Ramachandran, инвазия соседних карт
    в депривированную территорию после потери конечности -- документированный
    механизм угасания фантомной боли). deprived_idx -- локальные индексы
    нейронов, представлявших утраченную часть тела (они продолжают получать
    эфферентную команду "двигаться", но она никогда не подтверждается сенсорно
    -- отсюда фантомная боль, см. viktim.py). donor_idx -- индексы соседних
    ЖИВЫХ/активных зон (другие части тела, ассоциативная кора).

    В отличие от обычной структурной пластичности (которая ждёт, пока синапс
    сам ослабнет), ремаппинг ЦЕЛЕНАПРАВЛЕННО перехватывает синапсы, УЖЕ ведущие
    в депривированную зону (их пресинаптический источник всё равно бесполезен
    для неё сейчас), и переключает их источник на активные донорские зоны --
    именно так реальная кора "перепрограммирует" депривированную территорию,
    а не ждёт, пока там что-то спонтанно отмрёт.
    """
    if len(deprived_idx) == 0 or len(donor_idx) == 0:
        return 0
    deprived_set = set(int(i) for i in deprived_idx)
    candidates = np.array([i for i, p in enumerate(syn.post) if int(p) in deprived_set and syn.active[i]])
    if len(candidates) == 0:
        return 0
    chosen = rng.choice(candidates, size=min(n_new, len(candidates)), replace=False)
    for idx in chosen:
        syn.pre[idx] = donor_idx[rng.integers(0, len(donor_idx))]
        syn.w[idx] = rng.uniform(*w_init)
        syn.u[idx] = syn.U0[idx]
        syn.x[idx] = 1.0
        syn.daily_usage[idx] = 0.0
    return len(chosen)


def apply_structural_plasticity(syn, rng, region_slices, flow_map, region_names,
                                 prune_threshold=0.02, rewire_fraction=0.01):
    """
    Уровень 6 -- структурная пластичность (синаптогенез + прунинг).
    Каждый вызов (раз в несколько симулированных минут) часть слабых, почти
    неиспользуемых синапсов "отмирает" (как реальные дендритные шипики),
    а взамен на их месте формируются НОВЫЕ случайные связи между теми же
    типами регионов (мозг не отращивает связи куда попало -- он использует
    существующие функциональные "коридоры", отсюда flow_map).
    """
    weak = (syn.w < prune_threshold) & (syn.daily_usage < prune_threshold)
    candidates = np.nonzero(weak & syn.active)[0]
    if len(candidates) == 0:
        return 0

    n_rewire = max(1, int(len(syn.active) * rewire_fraction))
    chosen = rng.choice(candidates, size=min(n_rewire, len(candidates)), replace=False)

    syn.active[chosen] = False
    syn.w[chosen] = 0.0
    syn.u[chosen] = syn.U0[chosen]
    syn.x[chosen] = 1.0
    syn.daily_usage[chosen] = 0.0

    # синаптогенез: восстанавливаем столько же связей в разрешённых направлениях
    for idx in chosen:
        src_name = region_names[rng.integers(0, len(region_names))]
        src_slice = region_slices[src_name]
        dst_slice = region_slices[flow_map[src_name][rng.integers(0, len(flow_map[src_name]))]]
        p = rng.integers(src_slice.start, src_slice.stop)
        q = rng.integers(dst_slice.start, dst_slice.stop)
        syn.pre[idx] = p
        syn.post[idx] = q
        syn.w[idx] = rng.uniform(0.03, 0.15)
        syn.active[idx] = True
    return len(chosen)


def apply_sleep_consolidation(syn, downscale=0.985, boost=1.06, top_fraction=0.1):
    """
    Уровень 7 -- системная/циркадная консолидация (вызывается ОДИН раз за "ночь").

    Реализует гипотезу гомеостаза синапсов во сне (Tononi & Cirelli, synaptic
    homeostasis hypothesis, SHY): в течение "дня" синапсы в среднем усиливаются
    активностью (уровни 2-5), что накапливает шум и метаболическую нагрузку.
    Ночью происходит общее мультипликативное ослабление ВСЕХ синапсов
    (нормализация), кроме тех, что использовались больше всего за день --
    они "консолидируются" (усиливаются) -- аналог избирательного удержания
    важных воспоминаний при общем снижении фонового шума сети.
    """
    active_idx = np.nonzero(syn.active)[0]
    if len(active_idx) == 0:
        return
    usage = syn.daily_usage[active_idx]
    if usage.max() > 0:
        thresh = np.quantile(usage, 1.0 - top_fraction)
        important = active_idx[usage >= thresh]
    else:
        important = np.array([], dtype=int)

    syn.w *= downscale
    syn.w[important] *= boost / downscale  # компенсируем общий downscale для важных синапсов
    np.clip(syn.w, 0.0, 5.0, out=syn.w)
    syn.daily_usage[:] = 0.0


def critical_period_gain(age_hours, base_gain=0.3, extra_gain=1.7, tau_critical_hours=72.0):
    """
    Часть уровня 7 -- возрастное "закрытие критического периода".
    В начале жизни сети (первые часы/дни) пластичность существенно выше
    (как в развивающемся мозге ребёнка), затем экспоненциально снижается
    до базового "взрослого" уровня, что стабилизирует однажды выученные паттерны.
    """
    return base_gain + extra_gain * np.exp(-age_hours / tau_critical_hours)
