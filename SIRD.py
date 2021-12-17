#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this project, you will use functions to simulate the epidemic of a virus
within a region. 

Due date: September 21 11:59 PM
Comment: 10/10 Good Job! 
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Section 1. set the model parameters
# =============================================================================
# 1.1 set the random seed to 380 (pseudo randomness)
np.random.seed(380)  # DO NOT CHANGE THE SEED

# 1.2 create a population vector with five areas, population in areas are:
# areaA: 18000. areaB: 22000, areaC: 50100, areaD:21010, areaE:25000
n_j = np.array([18000, 22000, 50100, 21010, 25000])

# 1.3 use the random integer generation function in numpy to create a 5 x 5
# Origination-Destination flow matrix
# Set the lower limit to 1000, upper limit to 3,000
od_matrix = np.random.randint(1000, 3000, size=(5, 5))

# 1.4 same modal share across all regione (1)
alpha_vec = np.full(len(n_j), 1)


# 1.6 same recover rate  across all regions
gamma_vec = np.full(len(n_j), 0.05)

# Fatality Rate
death_vec = np.full(len(n_j), 0.001)


# 1.8 Have 300 iterations
days = 300


def set_params(beta, od_flow_rate):
    # 1.5 same transmission rate across all regions
    beta_vec = np.full(len(n_j), beta)
    # 1.7 normal o-d flow
    od_flow = np.round(od_matrix) * od_flow_rate

    return beta_vec, od_flow

# =============================================================================
# Section 2. define the initial status table
# =============================================================================
# assume in the beginning, no recover or died yet,
# infected proportion in areas are:
# areaA: 1%; areaB: 0.5%; areaC:0.1%; arerD:0%, areaE:0%


def start_detect(n_j, detect, immune, dead):
    sir = np.zeros(shape=(5, 4))
    sir[:, 0] = n_j
    init_infect = np.round(sir[:, 0] * detect)
    init_immune = np.round(sir[:, 0] * immune)
    init_dead = np.round(sir[:, 0] * dead)

    # move infection to group I
    sir[:, 0] = sir[:, 0] - init_infect - init_immune
    sir[:, 1] = sir[:, 1] + init_infect
    sir[:, 2] = sir[:, 2] + init_immune
    sir[:, 3] = sir[:, 3] + init_dead

    return sir


sir = start_detect(n_j, np.array([0.01, 0.005, 0.001, 0, 0]), 0, 0)


# =============================================================================
# Section 3. Define a function to simulate the epidemic in this big region
# =============================================================================
# function input should include:
# n_j:              population vector
# initial status:   the initial status of the epidemic
# od_flow:          the 5 x 5 o-d matrix
# alpha_vec:        population density in each region
# beta_vec:         transmission rate in each region
# gamma_vec:        daily recover rate in each region
# days: total       iterations

# function return:
# susceptible_pop_norm: changes in the proportion of S group (aggregated)
# infected_pop_norm: changes in the proportion of  I group (aggregated)
# recovered_pop_norm: changes in the proportion of R group (aggregated)

def epidemic_sim(n_j, initial_status, od_flow, alpha_vec, beta_vec, gamma_vec, death_vec, days):
    """
    Parameters
        n_j : 5 x 1 array ==> total population in each area.
        initial_status : 5 x 3 array ==> initial status
        alpha : 5 x 1 array ==> modal share in each area
        beta : 5 x 1 array ==> transmission rate in each area
        fatality: 5 x 1 array ==> daily fatality rate in each area
        gamma : 5 x 1 array ==> recover rage in each area
        days: int ==> total iterations
    """

    # 3.1 make copy of the initial sir table
    sir_sim = initial_status.copy()

    # 3.2 create empty list to keep tracking the changes
    susceptible_pop_norm = []
    infected_pop_norm = []
    recovered_pop_norm = []
    dead_pop_norm = []

    # 3.3. use total_days as the interator
    total_days = np.linspace(1, days, days)

    for day in total_days:

        # 3.4 figure out where those infected people go
        # normalize the sir table by calculating the percentage of each group
        sir_percent = sir_sim / n_j[:, np.newaxis]

        # assuming the infected people travel to all ares with the same probability:
        infected_mat = np.array([sir_percent[:, 1], ] * len(n_j)).transpose()

        # od_infected gives the flow of infected people. i.e., where they go
        od_infected = np.round(infected_mat * od_flow)

        # "inflow infected" are those who will spread the disease to susceptible
        # total infected inflow in each area
        inflow_infected = od_infected.sum(axis=0)

        # consider population density
        inflow_infected = np.round(inflow_infected * alpha_vec)

        # 3.5 calculate new_infect
        new_infect = np.round(
            (sir_sim[:, 0] / (n_j + od_flow.sum(axis=0) - od_flow.sum(axis=1))) * inflow_infected * beta_vec)

        # 3.6 set upper limit of the infected group (total susceptible)
        new_infect = np.where(
            new_infect > sir_sim[:, 0], sir_sim[:, 0], new_infect
        )

        # 3.7 calculate total number of people recovered
        new_recovered = np.round(gamma_vec * sir_sim[:, 1])

        new_dead = np.round(death_vec * sir_sim[:, 1])

        new_dead = np.where(
            new_dead > sir_sim[:, 1], sir_sim[:, 1], new_dead
        )

        # 3.8 remove new infections from susceptible group
        sir_sim[:, 0] -= new_infect

        # 3.9 add new infections into infected group,
        # also remove recovers from the infected group
        # also remove deaths from the infected group
        sir_sim[:, 1] = sir_sim[:, 1] + new_infect - new_recovered - new_dead

        # 3.10 add recovers to the recover group
        sir_sim[:, 2] += new_recovered

        sir_sim[:, 3] += new_dead

        # 3.11 set lower limits of the groups (0 people)
        sir_sim = np.where(
            sir_sim < 0, 0, sir_sim
        )

        # 3.12 compute the normalized SIR matrix on aggregate level
        region_sum = sir_sim.sum(axis=0)

        region_sum_normalized = region_sum / n_j.sum(axis=0)

        s = region_sum_normalized[0]
        i = region_sum_normalized[1]
        r = region_sum_normalized[2]
        d = region_sum_normalized[3]

        susceptible_pop_norm.append(s)
        infected_pop_norm.append(i)
        recovered_pop_norm.append(r)
        dead_pop_norm.append(d)

    return [susceptible_pop_norm, infected_pop_norm, recovered_pop_norm, dead_pop_norm]


# 3.13 call the function to simulate the epidemic


# =============================================================================
# Section 4. define a function to visualize the simulation result
# =============================================================================

def transform_outcome(outcome):
    susceptible = np.array(outcome[0]) * 100
    infected = np.array(outcome[1]) * 100
    recovered = np.array(outcome[2]) * 100
    dead = np.array(outcome[3]) * 100

    return susceptible, infected, recovered, dead


def sir_simulation_plot(outcomes, days):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(221)
    bx = fig.add_subplot(222)
    cx = fig.add_subplot(223)
    dx = fig.add_subplot(224)

    days = np.linspace(1, days, days)

    susceptible_1, infected_1, recovered_1, dead_1 = transform_outcome(
        outcomes[0])
    susceptible_2, infected_2, recovered_2, dead_2 = transform_outcome(
        outcomes[1])
    susceptible_3, infected_3, recovered_3, dead_3 = transform_outcome(
        outcomes[2])
    susceptible_4, infected_4, recovered_4, dead_4 = transform_outcome(
        outcomes[3])

    ax.plot(days, susceptible_1, 'b', label='Susceptible')
    ax.plot(days, infected_1, 'r', label='Infected')
    ax.plot(days, recovered_1, 'g', label='Recovered')
    ax.plot(days, dead_1, 'black', label='Dead')
    ax.set_title('Policy 1')

    bx.plot(days, susceptible_2, 'b', label='Susceptible')
    bx.plot(days, infected_2, 'r', label='Infected')
    bx.plot(days, recovered_2, 'g', label='Recovered')
    bx.plot(days, dead_1, 'black', label='Dead')
    bx.set_title('Policy 2')

    cx.plot(days, susceptible_3, 'b', label='Susceptible')
    cx.plot(days, infected_3, 'r', label='Infected')
    cx.plot(days, recovered_3, 'g', label='Recovered')
    cx.plot(days, dead_1, 'black', label='Dead')
    cx.set_title('Policy 3')

    dx.plot(days, susceptible_4, 'b', label='Susceptible')
    dx.plot(days, infected_4, 'r', label='Infected')
    dx.plot(days, recovered_4, 'g', label='Recovered')
    dx.plot(days, dead_1, 'black', label='Dead')
    dx.set_title('Policy 4')

    plt.legend()
    plt.suptitle('SIR Model Simulation', fontsize=20)
    plt.show()


# #=============================================================================
# #Section 5. Policy evaluation
# #=============================================================================
# # Using the simulation model to evaluate the following policy targets
# # Visualize the results, organize the plots in a 2x2 figure, each outcome on
# # one subplot.
# #Policy 1. do nothing (this should have been done in )
beta_vec, od_flow = set_params(0.8, 1)
outcome_1 = epidemic_sim(n_j, sir, od_flow, alpha_vec,
                         beta_vec, gamma_vec, death_vec, days)

# #Policy 2. reduce the o-d flow by 50%, all other arguments stay unchanged
beta_vec, od_flow = set_params(0.8, 0.5)
outcome_2 = epidemic_sim(n_j, sir, od_flow, alpha_vec,
                         beta_vec, gamma_vec, death_vec, days)

# #Policy 3. reduce the o-d flow by 80%, all other arguments stay unchanged
beta_vec, od_flow = set_params(0.8, 0.2)
outcome_3 = epidemic_sim(n_j, sir, od_flow, alpha_vec,
                         beta_vec, gamma_vec, death_vec, days)

# #Policy 4. reduce the o-d flow by 80%, reduce beta by 50%, all other the same
beta_vec, od_flow = set_params(0.4, 0.2)
outcome_4 = epidemic_sim(n_j, sir, od_flow, alpha_vec,
                         beta_vec, gamma_vec, death_vec, days)

outcomes = [outcome_1, outcome_2, outcome_3, outcome_4]
sir_simulation_plot(outcomes, days)
