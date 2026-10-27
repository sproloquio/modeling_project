
# -*- coding: utf-8 -*-

"""
Simplified ODE model of the tumor microenvironment under radiotherapy (IR) and anti-PD-L1.

Two-variable reduction (C = tumor cells, T = CD8+ T cells) of the six-variable model of
Liao, Wieler & Lopez Gascon (2024), Mathematical Biosciences 374, 109218.
Dendritic cells (D), M2 macrophages and regulatory T cells (Tr) are frozen at their
wild-type steady-state values and TGF-beta is dropped, because it only acts on M2 and Tr.

Inhibition of T-cell activation by the PD-1/PD-L1 axis is modelled as in the source paper:
a Hill function of the product P*L (the proportionality constant of the complex is absorbed
into the half-inhibition constant bar_K_Q). This is a modelling assumption of the source paper:
it is adopted here, not tested.

Running this file as a script produces the figures in ``figures/``::

    python src/ode_model.py
"""

# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os


# --SECTION 1: TUMOR AND T-CELL DYNAMICS

# PARAMETERS

params = {
    # Tumor Cells params (C)
    'lambda_C': 0.5436,   # /day, growth rate
    'C_tilde': 0.9,       # carrying capacity
    'eta_T': 24.0,        # cm3/g/day, killing rate of C by T
    'd_C': 0.17,          # /day, natural death C
    'eta_R': 0.016557,    # 1/(Gy*day), killing rate of C by R

    # T CD8+ Cells (T)
    'lambda_T': 12.15,      # /day, activation rate
    'T_h': 4e-4,            # g/cm3, naive T-cells density
    'd_T': 0.18,            # /day, natural death T
    'K_D': 4e-5,            # g/cm3, half-saturation of D
    'bar_K_M2': 2.87e-5,    # g/cm3, inhibition on T by M2
    'bar_K_Tr': 1.25e-4,    # g/cm3, inhibition on T by Tr
    'bar_K_Q': 2.9247e-20,  # g2/cm6, inhibition on T by Q

    # Steady States
    'D_star': 4e-5,      # g/cm3, steady state of D in the wild type tumor
    'M2_star': 5.73e-5,  # g/cm3, steady state of M2 in the wild type tumor
    'Tr_star': 2.5e-4,   # g/cm3, steady state of Tr in the wild type tumor

    # Molecular parameters
    'rho_P': 2.49e-7,     # mass ratio between P to T
    'rho_L': 5.22e-7,     # mass ratio between L to T
    'epsilon': 1e-3,      # expression rate of L on C
    'lambda_LR': 0.5,     # 1/Gy, promotion rate of L by R
    'mu_TR': 0.06367,     # 1/(Gy*day), killing rate of T by R

    # Therapies default values
    'R': 0.0,    # Gy, value of R in absence of radiotherapy
    'f_q': 1.0   # Factor of drug efficacy, value in absence of anti-PD-L1
    # when active: R = 1.8; f_q = 0.1
}


# EQUATION SYSTEM
def tumor_system(t, y, p):
    C, T = y

    # PD-1/PD-L1 interaction
    P = p['rho_P'] * T
    L = p['rho_L'] * (T + p['epsilon'] * p['M2_star'] + p['epsilon'] * C) * (1.0 + p['lambda_LR'] * p['R'])
    PL = P * L

    # Inhibition functions (Hill functions)
    h_M2 = 1.0 / (1.0 + p['M2_star'] / p['bar_K_M2'])
    h_Tr = 1.0 / (1.0 + p['Tr_star'] / p['bar_K_Tr'])
    h_Q = 1.0 / (1.0 + (p['f_q'] * PL) / p['bar_K_Q'])

    # Activation by IL-12 (in T)
    activation_D = p['D_star'] / (p['K_D'] + p['D_star'])

    # EQUATIONS
    dCdt = p['lambda_C'] * C * (1.0 - C / p['C_tilde']) - p['eta_T'] * T * C - p['eta_R'] * p['R'] * C - p['d_C'] * C
    dTdt = p['lambda_T'] * p['T_h'] * activation_D * h_M2 * h_Tr * h_Q - (p['d_T'] + p['mu_TR'] * p['R']) * T

    return [dCdt, dTdt]


def plot_tumor_dynamics():
    # SET THERAPEUTIC SCENARIOS
    therapies = [
        {'name': 'Control', 'RT_active': False,  'AntiPDL1_active': False, 'color': 'black'},
        {'name': 'Anti-PD-L1', 'RT_active': False,  'AntiPDL1_active': True, 'color': 'green'},
        {'name': 'IR', 'RT_active': True,  'AntiPDL1_active': False, 'color': 'blue'},
        {'name': 'Combo', 'RT_active': True,  'AntiPDL1_active': True, 'color': 'red'}
    ]

    # INITIAL CONDITIONS
    y0 = [0.04, 5e-5]  # C and T initial values 10% of wild-type steady states for the Complete Model (Liao et al.)
    tmax = 30.0        # days
    t0 = 0.0
    dt = 0.1           # time interval between a value in the graph

    # set t_eval for solve_ivp
    t_eval = np.arange(t0, tmax + dt, dt)

    # empty plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # SOLVE AND PLOT
    for therapy in therapies:
        p_ther = params.copy()
        p_ther['R'] = 1.8 if therapy['RT_active'] else 0.0
        p_ther['f_q'] = 0.1 if therapy['AntiPDL1_active'] else 1.0

        # solve, using Runge-Kutta 45 integration
        sol = solve_ivp(fun=tumor_system, t_span=[t0, tmax], y0=y0, t_eval=t_eval, method='RK45', 
                        args=(p_ther,), atol=1e-10, rtol=1e-7)
        # control on solve_ivp
        if not sol.success:
            raise RuntimeError(f"Error during scenario integration {therapy['name']}: {sol.message}")
                
        # add the scenario (therapy) to the plot
        ax1.plot(sol.t, sol.y[0], label=therapy['name'], color=therapy['color'], linewidth=2.0)

    # PLOT SETTING AND SAVE
    ax1.set_title('FIG.1 — Evolution of Tumor Density by Case', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Time (days)', fontsize=13)
    ax1.set_ylabel('Tumor Density (g/cm³)', fontsize=13)
    ax1.set_ylim(0, 0.7)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(alpha=0.2)

    os.makedirs('figures', exist_ok=True)
    fig1.savefig('figures/fig1_tumor_dynamics.png')
    plt.close(fig1)


# --SECTION 2: PD-1/PD-L1 ANALYSIS

def plot_pdl1_analysis():
    # PARAMETERS
    rho_P = params['rho_P']
    rho_L = params['rho_L']
    epsilon = params['epsilon']
    lambda_LR = params['lambda_LR']
    bar_K_Q   = params['bar_K_Q']  

    M2_star   = params['M2_star']      
    C_ref     = 0.4          # g/cm3 (reference tumor density)
    T_ref     = 5e-4         # g/cm3 reference T-cell density of the complete model (Liao et al.)

    # Compute effective PD-1 x PD-L1 product (reduced by anti-PD-L1)
    def compute_Q(T, C, f_q, R):
        P = rho_P * T
        L = rho_L * (T + epsilon * M2_star + epsilon * C) * (1 + lambda_LR * R)
        return f_q * P * L   # effective P*L product
    
    # Hill inhibition function for PD-1/PD-L1 on T cell activation
    def inhibition_hQ(T, C, f_q, R):
        Q_eff = compute_Q(T, C, f_q, R)
        return 1.0 / (1.0 + Q_eff / bar_K_Q)

    # shared setup: T range
    T_vals = np.linspace(0, 2e-3, 500)   # range of T cell densities

    # --FIGURE 2 & 3: product formation and inhibition function 
    f_q_vals = [1.0, 0.5, 0.1, 0.0]
    colors_A = ['black', 'sienna', 'green', 'orange']
    labels_A = [r'No therapy ($f_q=1$)', r'$f_q=0.5$', r'Strong anti-PD-L1 ($f_q=0.1$)', r'Perfect blockade ($f_q=0$)']

    fig23, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 5))

    # PLOT FIG.2
    for f_q, col, lab in zip(f_q_vals, colors_A, labels_A):
        Q_vals = [compute_Q(T, C_ref, f_q, R=0) * 1e20 for T in T_vals]
        ax2.plot(T_vals * 1e4, Q_vals, color=col, lw=2, label=lab)

    ax2.set_xlabel(r'T cell density [$\times 10^{-4}$ g/cm$^3$]', fontsize=11)
    ax2.set_ylabel(r'Effective product $f_q \cdot P \cdot L$ [$\times 10^{-20}$ g$^2$/cm$^6$]', fontsize=11)
    ax2.set_title('FIG.2 — PD-1 x PD-L1 effective product', fontsize=12, fontweight='bold')
    ax2.axhline(bar_K_Q * 1e20, ls=':', color='purple', lw=1.5, label=r'$\bar{K}_Q$ (half-inhibition)')
    ax2.axvline(T_ref * 1e4, ls=':', color='darkslategray', lw=1.5, label=r'$T_{ref}$ (reference value complete model)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    # PLOT FIG.3
    R_scenarios = [(0.0, 'solid'), (1.8, 'dashed')]
    colors_B    = ['black', 'sienna', 'green']

    for f_q, col in zip([1.0, 0.5, 0.1], colors_B):
        for R_val, ls in R_scenarios:
            h_vals = [inhibition_hQ(T, C_ref, f_q, R_val) for T in T_vals]
            lbl = rf'$f_q={f_q}$, R={R_val} Gy'
            ax3.plot(T_vals * 1e4, h_vals, color=col, ls=ls, lw=2, label=lbl)
    # plot of the ideal situation  (f_q = 0), so a line in y=1
    h_vals_perfect_blockade = [inhibition_hQ(T, C_ref, 0.0, 0.0) for T in T_vals]
    ax3.plot(T_vals * 1e4, h_vals_perfect_blockade, color='orange', ls='-', lw=1.5, label=r'$f_q=0$ (perfect blockade)')
    ax3.set_xlabel(r'T cell density [$\times 10^{-4}$ g/cm$^3$]', fontsize=11)
    ax3.set_ylabel(r'Inhibition function  $h_Q$', fontsize=11)
    ax3.set_title('FIG.3 — T cell activation inhibition by PD-1/PD-L1\nsolid: R=0 Gy, dashed: R=1.8 Gy', 
                  fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.axhline(0.5, ls=':', color='purple', lw=1.5, alpha=0.7, label=r'$h_Q = 0.5$ (half-inhibition)')
    ax3.axvline(T_ref * 1e4, ls=':', color='darkslategray', lw=1.5, label=r'$T_{ref}$ (reference value complete model)')
    ax3.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax3.grid(alpha=0.25)

    fig23.tight_layout()

    # SAVE
    os.makedirs('figures', exist_ok=True)
    fig23.savefig('figures/fig23_pdl1_analysis.png', bbox_inches='tight')
    plt.close(fig23)


if __name__ == "__main__":
    print("Simulation...")
    plot_tumor_dynamics()
    print("Fig. 1 saved.")
    plot_pdl1_analysis()
    print("Fig. 2 and fig. 3 saved.")
    print("Simulation completed. Check the 'figures' folder.")
