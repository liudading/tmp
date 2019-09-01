import os
import sys
sys.path.insert(0, '/home/lihepeng/Documents/Github/baselines/baselines')
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum

import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers, set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        low, high = self.env.unwrapped._feasible_action()

        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, low, high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_env(env_id, seed, train=True, logger_dir=None, reward_scale=1.0, mpi_rank=0, subrank=0):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = gym.make(env_id, **{"train":train})
    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env, 
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    env.seed(seed)
    env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_vec_env(env_id, seed, train=True, logger_dir=None, reward_scale=1.0, num_env=1):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id,
            seed,
            train=True,
            logger_dir=None,
            reward_scale=reward_scale,
            mpi_rank=mpi_rank,
            subrank=0
        )
    set_global_seeds(seed)
    return DummyVecEnv([make_thunk(i) for i in range(num_env)])

def optimize(env, plot=False):
    model = Model("dr_scheduling")

    price = env.price[env.t:env.t+env.T]
    temp = env.temp[env.t:env.t+env.T]

    # oven
    B_ov = {}
    T_ov = (env.ov.beta - env.ov.alpha) // env.dt
    ta_ov = (env.ov.alpha - env.init_time) // env.dt
    tb_ov = (env.ov.beta - env.init_time) // env.dt
    for t in range(T_ov):
        B_ov[t] = model.addVar(vtype='B', name='B_ov_%s' % t)
    model.addCons(quicksum([B_ov[t] for t in range(T_ov)]) == env.ov.DEMAND)
    model.addCons(quicksum([B_ov[t]*B_ov[t-1] for t in range(1,T_ov)]) == env.ov.DEMAND-1)

    # dishwasher
    B_dw = {}
    T_dw = (env.dw.beta - env.dw.alpha) // env.dt
    ta_dw = (env.dw.alpha - env.init_time) // env.dt
    tb_dw = (env.dw.beta - env.init_time) // env.dt
    for t in range(T_dw):
        B_dw[t] = model.addVar(vtype='B', name='B_dw_%s' % t)
    model.addCons(quicksum([B_dw[t] for t in range(T_dw)]) == env.dw.DEMAND)
    model.addCons(quicksum([B_dw[t]*B_dw[t-1] for t in range(1,T_dw)]) == env.dw.DEMAND-1)

    # washing machine
    B_wm = {}
    T_wm = (env.wm.beta - env.wm.alpha) // env.dt
    ta_wm = (env.wm.alpha - env.init_time) // env.dt
    tb_wm = (env.wm.beta - env.init_time) // env.dt
    for t in range(T_wm):
        B_wm[t] = model.addVar(vtype='B', name='B_wm_%s' % t)
    model.addCons(quicksum([B_wm[t] for t in range(T_wm)]) == env.wm.DEMAND)
    model.addCons(quicksum([B_wm[t]*B_wm[t-1] for t in range(1,T_wm)]) == env.wm.DEMAND-1)

    # clothes dryer
    B_cd = {}
    T_cd = (env.cd.beta - env.cd.alpha) // env.dt
    ta_cd = (env.cd.alpha - env.init_time) // env.dt
    tb_cd = (env.cd.beta - env.init_time) // env.dt
    for t in range(T_cd):
        B_cd[t] = model.addVar(vtype='B', name='B_cd_%s' % t)
    model.addCons(quicksum([B_cd[t] for t in range(T_cd)]) == env.cd.DEMAND)
    model.addCons(quicksum([B_cd[t]*B_cd[t-1] for t in range(1,T_cd)]) == env.cd.DEMAND-1)

    # refrigerator
    T_fg = (env.fg.beta - env.fg.alpha) // env.dt
    ta_fg = (env.fg.alpha - env.init_time) // env.dt
    tb_fg = (env.fg.beta - env.init_time) // env.dt
    P_fg = [env.fg.MAX_POWER] * T_fg

    # vaccum
    T_vc = (env.vc.beta - env.vc.alpha) // env.dt
    ta_vc = (env.vc.alpha - env.init_time) // env.dt
    tb_vc = (env.vc.beta - env.init_time) // env.dt
    P_vc = [env.vc.MAX_POWER] * T_vc

    # hair dryder
    T_hd = (env.hd.beta - env.hd.alpha) // env.dt
    ta_hd = (env.hd.alpha - env.init_time) // env.dt
    tb_hd = (env.hd.beta - env.init_time) // env.dt
    P_hd = [env.hd.MAX_POWER] * T_hd

    # tv
    T_tv = (env.tv.beta - env.tv.alpha) // env.dt
    ta_tv = (env.tv.alpha - env.init_time) // env.dt
    tb_tv = (env.tv.beta - env.init_time) // env.dt
    P_tv = [env.tv.MAX_POWER] * T_tv

    # notebook
    T_nb = (env.nb.beta - env.nb.alpha) // env.dt
    ta_nb = (env.nb.alpha - env.init_time) // env.dt
    tb_nb = (env.nb.beta - env.init_time) // env.dt
    P_nb = [env.nb.MAX_POWER] * T_nb

    # light
    T_lg = (env.lg.beta - env.lg.alpha) // env.dt
    ta_lg = (env.lg.alpha - env.init_time) // env.dt
    tb_lg = (env.lg.beta - env.init_time) // env.dt
    P_lg = [env.lg.MAX_POWER] * T_lg

    # ev
    P_ev, SOC = {}, {}
    T_ev = (env.ev.beta - env.ev.alpha) // env.dt
    ta_ev = (env.ev.alpha - env.init_time) // env.dt
    tb_ev = (env.ev.beta - env.init_time) // env.dt
    for t in range(T_ev):
        P_ev[t] = model.addVar(vtype='C', name='P_ev_%s' % t, lb=env.ev.MIN_POWER, ub=env.ev.MAX_POWER)
        SOC[t] = model.addVar(vtype='C', name='SOC_%s' % t, lb=env.ev.MIN_SOC, ub=env.ev.MAX_SOC)
        if t == 0:
            model.addCons(SOC[t] == env.ev.init_soc + (env.dt.seconds/3600) * P_ev[t] / env.ev.CAPACITY, name='SOC_DYN_0')
        else:
            model.addCons(SOC[t] == SOC[t-1] + (env.dt.seconds/3600) * P_ev[t] / env.ev.CAPACITY, name='SOC_DYN_(%s)' % t)
    model.addCons(SOC[T_ev-1] == 1.0, name='Target')

    # ewh
    P_wh, T_wh = {}, {}
    for t in range(env.T):
        Tin = 60
        T_air = 75
        d = 8.34
        Cp = 1.0069
        volume = 40
        SA = 24.1
        R = 15
        Q = 3412.1
        C = volume * d * Cp
        G = SA / R
        B = d * env.wh.flow_profile[t] * Cp
        R1 = 1/(G + B)
        coff = np.exp(-(env.dt.seconds/3600)/(R1*C))

        P_wh[t] = model.addVar(vtype='C', name='P_wh_%s' % t, lb=env.wh.MIN_POWER, ub=env.wh.MAX_POWER)
        T_wh[t] = model.addVar(vtype='C', name='T_wh_%s' % t, lb=env.wh.MIN_TEMP, ub=env.wh.MAX_TEMP)
        if t == 0:
            model.addCons(
                T_wh[t] == coff*sum(env.wh.state[1:])+(1-coff)*(G*R1*T_air+B*R1*Tin+P_wh[t]*(env.dt.seconds/3600)*Q*R1),
                name='Twh_DYN_0')
        else:
            model.addCons(
                T_wh[t] == coff*T_wh[t-1]+(1-coff)*(G*R1*T_air+B*R1*Tin+P_wh[t]*(env.dt.seconds/3600)*Q*R1),
                name='Twh_DYN_(%s)' % t)

    # hvac
    Req = 3.1965e-6 * 1.8
    Ca = 1.01 / 1.8
    Ma = 1778.369
    COP = 2
    delta_t = env.dt.seconds/3600
    P_ac, T_ac = {}, {}
    for t in range(env.T):
        P_ac[t] = model.addVar(vtype='C', name='P_ac_%s' % t, lb=env.ac.MIN_POWER, ub=env.ac.MAX_POWER)
        T_ac[t] = model.addVar(vtype='C', name='T_ac_%s' % t, lb=env.ac.MIN_TEMP-5, ub=env.ac.MAX_TEMP)
        if t == 0:
            model.addCons(
                T_ac[t] == (1-delta_t/(1000*Ma*Ca*Req))*sum(env.ac.state[1:]) + \
                    delta_t/(1000*Ma*Ca*Req)*temp[t] - (P_ac[t]*COP*delta_t)/(0.00027*Ma*Ca), 
                name='Tac_DYN_0')
        else:
            model.addCons(
                T_ac[t] == (1-delta_t/(1000*Ma*Ca*Req))*T_ac[t-1] + \
                    delta_t/(1000*Ma*Ca*Req)*temp[t] - (P_ac[t]*COP*delta_t)/(0.00027*Ma*Ca), 
                name='Tac_DYN_(%s)' % t)

    OBJ = {}
    for t in range(env.T):
        P = 0.0
        # oven
        if t >= ta_ov and t < tb_ov:
            P += B_ov[t-ta_ov] * env.ov.MAX_POWER
        # dishwasher
        if t >= ta_dw and t < tb_dw:
            P += B_dw[t-ta_dw] * env.dw.MAX_POWER
        # washing machine
        if t >= ta_wm and t < tb_wm:
            P += B_wm[t-ta_wm] * env.wm.MAX_POWER
        # clothes dryer
        if t >= ta_cd and t < tb_cd:
            P += B_cd[t-ta_cd] * env.cd.MAX_POWER
        # refrigerator
        if t >= ta_fg and t < tb_fg:
            P += P_fg[t-ta_fg]
        # vaccum
        if t >= ta_vc and t < tb_vc:
            P += P_vc[t-ta_vc]
        # hair dryer
        if t >= ta_hd and t < tb_hd:
            P += P_hd[t-ta_hd]
        # tv
        if t >= ta_tv and t < tb_tv:
            P += P_tv[t-ta_tv]
        # notebook
        if t >= ta_nb and t < tb_nb:
            P += P_nb[t-ta_nb]
        # light
        if t >= ta_lg and t < tb_lg:
            P += P_lg[t-ta_lg]
        # ev
        if t >= ta_ev and t < tb_ev:
            P += P_ev[t-ta_ev]
        # EWH
        P += P_wh[t]
        # HVAC
        P += P_ac[t]

        OBJ[t] = model.addVar(vtype='C', name='OBJ_%s' % t, lb=-1e10, ub=1e10)
        model.addCons(OBJ[t] == price[t] * P, name='Objective(%s)' % t)

    # Set objective function
    model.setObjective(quicksum([OBJ[t] for t in range(env.T)]), 'minimize')

    # Execute optimization
    model.hideOutput()
    # model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
    model.optimize()

    # Get results
    p_ov = np.hstack([
        np.zeros(ta_ov), 
        [model.getVal(B_ov[t])*env.ov.MAX_POWER for t in range(T_ov)], 
        np.zeros(env.T-tb_ov)
        ])
    p_dw = np.hstack([
        np.zeros(ta_dw), 
        [model.getVal(B_dw[t])*env.dw.MAX_POWER for t in range(T_dw)], 
        np.zeros(env.T-tb_dw)
        ])
    p_wm = np.hstack([
        np.zeros(ta_wm),
        [model.getVal(B_wm[t])*env.wm.MAX_POWER for t in range(T_wm)], 
        np.zeros(env.T-tb_wm)
        ])
    p_cd = np.hstack([
        np.zeros(ta_cd),
        [model.getVal(B_cd[t])*env.cd.MAX_POWER for t in range(T_cd)], 
        np.zeros(env.T-tb_cd)
        ])
    p_fg = np.hstack([
        np.zeros(ta_fg),
        P_fg, 
        np.zeros(env.T-tb_fg)
        ])
    p_vc = np.hstack([
        np.zeros(ta_vc),
        P_vc, 
        np.zeros(env.T-tb_vc)
        ])
    p_hd = np.hstack([
        np.zeros(ta_hd),
        P_hd, 
        np.zeros(env.T-tb_hd)
        ])
    p_tv = np.hstack([
        np.zeros(ta_tv),
        P_tv, 
        np.zeros(env.T-tb_tv)
        ])
    p_nb = np.hstack([
        np.zeros(ta_nb),
        P_nb, 
        np.zeros(env.T-tb_nb)
        ])
    p_lg = np.hstack([
        np.zeros(ta_lg),
        P_lg, 
        np.zeros(env.T-tb_lg)
        ])
    p_ev = np.hstack([
        np.zeros(ta_ev),
        [model.getVal(P_ev[t]) for t in range(T_ev)], 
        np.zeros(env.T-tb_ev)
        ])
    soc = np.hstack([
        np.zeros(ta_ev-1),
        [env.ev.init_soc]+[model.getVal(SOC[t]) for t in range(T_ev)], 
        np.zeros(env.T-tb_ev)
        ])
    p_wh = np.array([model.getVal(P_wh[t]) for t in range(env.T)])
    temp_wh = np.array([model.getVal(T_wh[t]) for t in range(env.T)])
    p_ac = np.array([model.getVal(P_ac[t]) for t in range(env.T)])
    temp_ac = np.array([model.getVal(T_ac[t]) for t in range(env.T)])
    f = model.getObjVal()
    p = p_ov + p_dw + p_wm + p_cd + \
        p_fg + p_vc + p_hd + p_tv + p_nb + p_lg + \
        p_ev + p_wh + p_ac

    # Render the scheduling result to the screen
    if plot is True:
        x = range(env.T+1)
        xticks = range(0, env.T+1, 12)
        xticklabels = [str(i) for i in range(8,24,2)] + [str(i) for i in range(0,9,2)]

        n_subfigs = 10
        plt.close()
        plt.figure(figsize=(10,15))

        ax1 = plt.subplot(n_subfigs,1,1)
        ax1.step(x, np.insert(price,0,price[:1]))
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(labels=xticklabels)
        ax1.set_xlim(0,env.T)

        ax2 = plt.subplot(n_subfigs,1,2)
        ax2.step(x, np.insert(p_ov,0,p_ov[:1]))
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels=xticklabels)
        ax2.axvline((env.ov.alpha-env.init_time)//env.dt, c='r')
        ax2.axvline((env.ov.beta-env.init_time)//env.dt, c='r')
        ax2.set_xlim(0,env.T)

        ax3 = plt.subplot(n_subfigs,1,3)
        ax3.step(x, np.insert(p_dw,0,p_dw[:1]))
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(labels=xticklabels)
        ax3.axvline((env.dw.alpha-env.init_time)//env.dt, c='r')
        ax3.axvline((env.dw.beta-env.init_time)//env.dt, c='r')
        ax3.set_xlim(0,env.T)

        ax4 = plt.subplot(n_subfigs,1,4)
        ax4.step(x, np.insert(p_wm,0,p_wm[:1]))
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(labels=xticklabels)
        ax4.axvline((env.wm.alpha-env.init_time)//env.dt, c='r')
        ax4.axvline((env.wm.beta-env.init_time)//env.dt, c='r')
        ax4.set_xlim(0,env.T)

        ax5 = plt.subplot(n_subfigs,1,5)
        ax5.step(x, np.insert(p_cd,0,p_cd[:1]))
        ax5.set_xticks(xticks)
        ax5.set_xticklabels(labels=xticklabels)
        ax5.axvline((env.cd.alpha-env.init_time)//env.dt, c='r')
        ax5.axvline((env.cd.beta-env.init_time)//env.dt, c='r')
        ax5.set_xlim(0,env.T)

        ax6 = plt.subplot(n_subfigs,1,6)
        ax6.step(x, np.insert(p_fg,0,p_fg[:1]), label='P_fg')
        ax6.step(x, np.insert(p_vc,0,p_vc[:1]), label='P_vc')
        ax6.step(x, np.insert(p_hd,0,p_hd[:1]), label='P_hd')
        ax6.step(x, np.insert(p_tv,0,p_tv[:1]), label='P_tv')
        ax6.step(x, np.insert(p_nb,0,p_nb[:1]), label='P_nb')
        ax6.step(x, np.insert(p_lg,0,p_lg[:1]), label='P_lg')
        ax6.set_xticks(xticks)
        ax6.set_xticklabels(labels=xticklabels)
        ax6.legend(ncol=6)
        ax6.set_xlim(0,env.T)

        ax7 = plt.subplot(n_subfigs,1,7)
        ax7.step(x, np.insert(p_ev,0,p_ev[:1]), label='P_ev')
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(labels=xticklabels)
        ax7.legend(ncol=6)
        ax7.set_xlim(0,env.T)
        ax71 = ax7.twinx()
        ax71.step(x, np.insert(soc,0,soc[:1]), color='r')

        ax8 = plt.subplot(n_subfigs,1,8)
        ax8.step(x, np.insert(p_wh,0,p_wh[:1]), label='P_wh')
        ax8.set_xticks(xticks)
        ax8.set_xticklabels(labels=xticklabels)
        ax8.legend()
        ax8.set_xlim(0,env.T)
        ax81 = ax8.twinx()
        ax81.step(x, np.insert(temp_wh,0,temp_wh[:1]), color='r')
        ax81.axhline(env.wh.MIN_TEMP, c='c')
        ax81.axhline(env.wh.MAX_TEMP, c='c')

        ax9 = plt.subplot(n_subfigs,1,9)
        ax9.step(x, np.insert(env.wh.flow_profile,0,env.wh.flow_profile[:1]))
        ax9.set_xticks(xticks)
        ax9.set_xticklabels(labels=xticklabels)
        ax9.set_xlim(0,env.T)
        ax91 = ax9.twinx()
        ax91.step(x, np.insert(temp,0,temp[:1]), color='r')

        ax10 = plt.subplot(n_subfigs,1,10)
        ax10.step(x, np.insert(p_ac,0,p_ac[:1]), label='P_ac')
        ax10.set_xticks(xticks)
        ax10.set_xticklabels(labels=xticklabels)
        ax10.set_xlim(0,env.T)
        ax10.legend()
        ax101 = ax10.twinx()
        ax101.step(x, np.insert(temp_ac,0,temp_ac[:1]), color='r')
        ax101.axhline(env.ac.MIN_TEMP, c='c')
        ax101.axhline(env.ac.MAX_TEMP, c='c')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
    
    return f, p

# Optimal
seed = 1314
env = make_env(
    env_id='SmartHome-v1',
    seed=seed,
    train=False,
    )
f_vals = []
for d in range(1,62):
    env.reset(day=d)
    f, p = optimize(env.unwrapped)
    f_vals.append(f)

print(np.mean(f_vals))
plt.plot(f_vals)
plt.show()

np.savetxt('/home/lihepeng/Documents/Github/tmp/dr/optimal/returns.txt', f_vals)