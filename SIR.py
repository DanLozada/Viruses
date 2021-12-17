#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly project 3. Silulate the virus epidemic with SIR Model
Due on: September 14, 11:59 PM

*You are encouraged to delete my hints and start from blank, programmers!* 
Comment: 10/10 Good Job
"""
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
# =============================================================================
# Section 1. set the model parameters
# =============================================================================
# 1.1 set the total population, let's say there are 3500 people on the island
n = 3500
# 1.2 set the initial infected group as 1% of the population (round to integer)
init_infect = round(0.001 * n)
# 1.3 set the initial recovered group as 0% of the population (round to integer)
init_immune = round(0 * n)
# 1.4 set the transmission rate as 0.8
beta = 0.8
# 1.5 set the recover rate as 0.05
gamma = 0.05


# =============================================================================
# Section 2. define the initial status table
# =============================================================================
# 2.1 generate a 1x3 matrix filled with 0, name is as sir
sir = np.zeros((1, 3))
# 2.2 replace the first column of the sir matrix with initial susceptible group
sir[0, 0] = n - init_infect - init_immune
# 2.3 replace the second column with initial infected group
sir[0, 1] = init_infect
# 2.4. replace the third column with initial recovered group
sir[0, 2] = init_immune
# up to here, you should have the initial sir matrix .

# =============================================================================
# Section 3. simulate the epidemic of the virus
# =============================================================================
# New Infected ⇒ 𝜷 I (S /N)
# ds/dt = -𝜷 I (S /N)
# dI/dt = -𝜷 (S / N) I - 𝛾 I
# dR/dt = 𝛾 I
# 3.1 make a copy of the initial status table
sir_sim = sir.copy()
# 3.2 create empty lists to keep tracking the changes
susceptible_pop_norm = []  # record of susceptible group
infected_pop_norm = []  # record of infected group
recovered_pop_norm = []  # record of recovered group

# 3.3 Set the time horizon to 100 days
days = 100
total_days = np.linspace(1, days, num=days)  # no need to change. done for you.

# 3.4 iterate through the 100 days
for day in total_days:

    # calculate new_infect
    new_infected = round(beta * sir_sim[0, 0]*sir_sim[0, 1]/(n))
    # set total susceptible number as the  upper limit of new_infected
    new_infect = np.where(
        new_infected > sir_sim[0, 0], sir_sim[0, 0], new_infected)
    # calculate new recovered number
    new_recovered = round(gamma * sir_sim[0, 1])
    # remove new infections from susceptible group
    sir_sim[0, 0] -= round(new_infected)
    # add new infections into infected group,
    # also remove recovers from the infected group
    sir_sim[0, 1] += round(new_infected - new_recovered)
    # add recovers to the recover group
    sir_sim[0, 2] += new_recovered
    # set lower limits of the groups (0 people)
    sir_sim = np.where(sir_sim < 0, 0, sir_sim)
    # normalize the SIR (i.e., calculate % of population),
    # and append to the record lists
    susceptible_pop_norm.append(sir_sim[0, 0]/n)
    infected_pop_norm.append(sir_sim[0, 1] / n)
    recovered_pop_norm.append(sir_sim[0, 2]/n)
    # 3.5 store the simulation outcome in a outcome list
outcome = [susceptible_pop_norm, infected_pop_norm, recovered_pop_norm]

# =============================================================================
# Section 4. Run the following code to visualize the simulation outcome.
# As long as you have section 1 - section 3 completed,
# and stored the simulation result in the outcome list,
# the following part is ready to go, no further change needed.
# =============================================================================

# define the plot function


def sir_simulation_plot(outcome, days):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    days = np.linspace(1, days, num=days)
    susceptible = np.array(outcome[0])*100
    infected = np.array(outcome[1])*100
    recovered = np.array(outcome[2])*100
    ax.plot(days, susceptible, label='susceptible', color='y')
    ax.plot(days, infected, label='infected', color='r')
    ax.plot(days, recovered, label='recovered', color='g')
    ax.set_xlabel('Days')
    ax.set_ylabel('Proportion of the population')
    ax.set_title("SIR Model Simulation")
    plt.legend()
    plt.show()


# call the function to plot the outcome
sir_simulation_plot(outcome, days=days)
