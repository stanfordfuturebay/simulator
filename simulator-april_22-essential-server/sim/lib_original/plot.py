import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import random as rd
import os, math
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lib.measures import (MeasureList, BetaMultiplierMeasure,
                      SocialDistancingForAllMeasure, BetaMultiplierMeasureByType,
                      SocialDistancingForPositiveMeasure, SocialDistancingByAgeMeasure, SocialDistancingForSmartTracing, ComplianceForAllMeasure)

import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

DPI = 200
NO_PLOT = False

matplotlib.rcParams.update({
    "figure.autolayout": False,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.8,
    "text.usetex": True,
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "legend.frameon": True,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
})


def days_to_datetime(arr, start_date):
    # timestamps
    ts = arr * 24 * 60 * 60 + pd.Timestamp(start_date).timestamp()
    return pd.to_datetime(ts, unit='s')


def lockdown_widget(lockdown_at, start_date, lockdown_label_y, ymax, lockdown_label, ax, ls='--', xshift=0.0):
    # Convert x-axis into posix timestamps and use pandas to plot as dates
    lckdn_x = days_to_datetime(lockdown_at, start_date=start_date)
    ax.plot([lckdn_x, lckdn_x], [0, ymax], linewidth=2.5, linestyle=ls,
            color='black', label='_nolegend_')
    lockdown_label_y = lockdown_label_y or ymax*0.4
    ax.text(x=lckdn_x - pd.Timedelta(2.1 + xshift, unit='d'),
            y=lockdown_label_y, s=lockdown_label, rotation=90)


def target_widget(show_target,start_date, ax):
    txx = np.linspace(0, show_target.shape[0] - 1, num=show_target.shape[0])
    txx = days_to_datetime(txx, start_date=start_date)
    ax.plot(txx, show_target, linewidth=4, linestyle='', marker='X', ms=6,
            color='red', label='COVID-19 case data')


class Plotter(object):
    """
    Plotting class
    """

    def __init__(self):

        # plot constants
        # check out https://colorhunt.co/

        self.color_expo = '#ffcc00'
        self.color_iasy = '#00a8cc'
        self.color_ipre = '#005082'
        self.color_isym = '#000839'

        self.color_testing = '#ffa41b'

        self.color_posi = '#21bf73'
        self.color_nega = '#fd5e53'

        self.color_all = '#ffa41b'
        self.color_positive = '#00a8cc'
        self.color_age = '#005082'
        self.color_tracing = '#000839'

        self.color_infected = '#000839'

        self.filling_alpha = 0.5

        self.color_different_scenarios = [
            '#dc2ade',
            '#21ff53',
            '#323edd',
            '#ff9021',  
            '#4d089a',
            '#cc0066',
            '#ff6666',
            '#216353',
            '#66cccc',
            '#ff2222'
        ]



        # sequential
        # self.color_different_scenarios = [
        #     # '#ffffcc',
        #     '#c7e9b4',
        #     '#7fcdbb',
        #     '#41b6c4',
        #     '#2c7fb8',
        #     '#253494',
        #     '#000000'
        # ]



        # 2D visualization
        self.density_alpha = 0.7

        self.marker_home = "^"
        self.marker_site = "o"

        self.color_home = '#000839'
        self.color_site = '#000000'

        self.size_home = 80
        self.size_site = 300



    def __is_state_at(self, sim, r, state, t):
        return (sim.state_started_at[state][r] <= t) & (sim.state_ended_at[state][r] > t)

    def __state_started_before(self, sim, r, state, t):
        return (sim.state_started_at[state][r] <= t)

    def __is_contained_at(self, sim, r, measure, t):
        contained = np.zeros(sim.n_people, dtype='bool')
        for i in range(sim.n_people):
            if measure == 'SocialDistancingForAllMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForAllMeasure, t=t, j=i)
            elif measure == 'SocialDistancingForSmartTracing':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForSmartTracing, t=t, j=i)
            elif measure == 'SocialDistancingByAgeMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingByAgeMeasure, t=t, age=sim.people_age[r, i])
            elif measure == 'SocialDistancingForPositiveMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForPositiveMeasure,
                                                                     t=t, j=i,
                                                                     state_posi_started_at=sim.state_started_at['posi'][r, :],
                                                                     state_posi_ended_at=sim.state_ended_at['posi'][r, :],
                                                                     state_resi_started_at=sim.state_started_at['resi'][r, :],
                                                                     state_dead_started_at=sim.state_started_at['dead'][r, :])
            else:
                raise ValueError('Social distancing measure unknown.')
        return contained

    def __comp_state_cumulative(self, sim, state, acc):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__state_started_before(sim, r, state, t))
                for r in range(sim.random_repeats)]
            ts.append(t/24.0)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)

    def __comp_state_over_time(self, sim, state, acc):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__is_state_at(sim, r, state, t))
                for r in range(sim.random_repeats)]
            ts.append(t/24.0)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)


    def __comp_contained_over_time(self, sim, measure, acc):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__is_contained_at(sim, r, measure, t))
                for r in range(sim.random_repeats)]
            ts.append(t/24.0)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)

    def __comp_checkins_in_a_day(self, sim, r, t):
        site_checkins = np.zeros(sim.n_sites, dtype='bool')

        for site in range(sim.n_sites):
            for indiv in range(sim.n_people):
                if ( (not sim.measure_list[r].is_contained_prob(SocialDistancingForAllMeasure, t=t, j=indiv)) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingForSmartTracing, t=t, j=indiv)) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingByAgeMeasure, t=t, age=sim.people_age[r, indiv])) and
                     (not sim.measure_list[r].is_contained_prob(SocialDistancingForPositiveMeasure,
                                                                t=t, j=indiv,
                                                                state_posi_started_at=sim.state_started_at['posi'][r, :],
                                                                state_posi_ended_at=sim.state_ended_at['posi'][r, :],
                                                                state_resi_started_at=sim.state_started_at['resi'][r, :],
                                                                state_dead_started_at=sim.state_started_at['dead'][r, :])) and
                     (not sim.state['dead'][r, indiv]) and
                    len(sim.mob[r].list_intervals_in_window_individual_at_site(indiv, site, t, t+24.0)) > 0):
                    site_checkins[site] += 1
        return site_checkins

    def plot_cumulative_infected(self, sim, title='Example', filename='daily_inf_0',
                                 figsize=(6, 5), errorevery=20, acc=1000, ymax=None,
                                 lockdown_label='Lockdown', lockdown_at=None,
                                 lockdown_label_y=None, show_target=None,
                                 test_lag=2, start_date='1970-01-01',
                                 subplot_adjust=None, legend_loc='upper right'):
        ''''
        Plots daily infected split by group
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, iasy_mu, iasy_sig = self.__comp_state_cumulative(sim, 'iasy', acc)
        # _,  ipre_mu, ipre_sig = self.__comp_state_cumulative(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_cumulative(sim, 'isym', acc)
        # _,  expo_mu, iexpo_sig = self.__comp_state_cumulative(sim, 'expo', acc)
        # _,  posi_mu, posi_sig = self.__comp_state_cumulative(sim, 'posi', acc)

        line_xaxis = np.zeros(ts.shape)
        line_iasy = iasy_mu
        line_isym = iasy_mu + isym_mu

        error_isym = np.sqrt(iasy_sig**2 + isym_sig**2)

        # Convert x-axis into posix timestamps and use pandas to plot as dates
        ts = days_to_datetime(ts, start_date=start_date)

        # lines
        ax.plot(ts, line_iasy, c='black', linestyle='-')
        ax.errorbar(ts, line_isym, yerr=error_isym, c='black', linestyle='-',
                    elinewidth=0.8, errorevery=errorevery, capsize=3.0)

        # filling
        ax.fill_between(ts, line_xaxis, line_iasy, alpha=self.filling_alpha, label='Asymptomatic',
                        edgecolor=self.color_iasy, facecolor=self.color_iasy, linewidth=0, zorder=0)
        ax.fill_between(ts, line_iasy, line_isym, alpha=self.filling_alpha, label='Symptomatic',
                        edgecolor=self.color_isym, facecolor=self.color_isym, linewidth=0, zorder=0)

        # limits
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        # extra
        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)
        if show_target is not None:
            target_widget(show_target, start_date, ax)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc=legend_loc, borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_infected(self, sim, title='Example', filename='daily_inf_0',
                            figsize=(6, 5), errorevery=20, acc=1000, ymax=None,
                            lockdown_label='Lockdown', lockdown_at=None,
                            lockdown_label_y=None, show_target=None,
                            lockdown_end=None,
                            test_lag=2, start_date='1970-01-01',
                            subplot_adjust=None, legend_loc='upper right'):
        ''''
        Plots daily infected split by group
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, iasy_mu, iasy_sig = self.__comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_over_time(sim, 'isym', acc)
        # _,  expo_mu, iexpo_sig = self.__comp_state_over_time(sim, 'expo', acc)
        # _,  posi_mu, posi_sig = self.__comp_state_over_time(sim, 'posi', acc)

        line_xaxis = np.zeros(ts.shape)
        line_iasy = iasy_mu
        line_ipre = iasy_mu + ipre_mu
        line_isym = iasy_mu + ipre_mu + isym_mu
        error_isym = np.sqrt(iasy_sig**2 + ipre_sig**2 + isym_sig**2)

        # Convert x-axis into posix timestamps and use pandas to plot as dates
        ts = days_to_datetime(ts, start_date=start_date)

        # lines
        ax.plot(ts, line_iasy,
                c='black', linestyle='-')
        ax.plot(ts, line_ipre,
                c='black', linestyle='-')
        ax.errorbar(ts, line_isym, yerr=error_isym, c='black', linestyle='-',
                    elinewidth=0.8, errorevery=errorevery, capsize=3.0)

        # filling
        ax.fill_between(ts, line_xaxis, line_iasy, alpha=self.filling_alpha, label='Asymptomatic',
                        edgecolor=self.color_iasy, facecolor=self.color_iasy, linewidth=0, zorder=0)
        ax.fill_between(ts, line_iasy, line_ipre, alpha=self.filling_alpha, label='Pre-symptomatic',
                        edgecolor=self.color_ipre, facecolor=self.color_ipre, linewidth=0, zorder=0)
        ax.fill_between(ts, line_ipre, line_isym, alpha=self.filling_alpha, label='Symptomatic',
                        edgecolor=self.color_isym, facecolor=self.color_isym, linewidth=0, zorder=0)

        # limits
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        # extra
        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)
        if lockdown_end is not None:
            lockdown_widget(lockdown_at=lockdown_end, start_date=start_date,
                            lockdown_label_y=lockdown_label_y, ymax=ymax,
                            lockdown_label='End of lockdown', ax=ax, ls='dotted')
        if show_target is not None:
            target_widget(show_target, start_date, ax)


        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc=legend_loc, borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_tested(self, sim, title='Example', filename='daily_tested_0', figsize=(10, 10), errorevery=20,
        acc=1000, ymax=None, test_lag=2):

        ''''
        Plots daily tested, positive daily tested, negative daily tested
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, posi_mu, posi_sig = self.__comp_state_over_time(sim, 'posi', acc)
        _,  nega_mu, nega_sig = self.__comp_state_over_time(sim, 'nega', acc)

        line_xaxis = np.zeros(ts.shape)
        line_posi = posi_mu
        line_nega = posi_mu + nega_mu

        error_posi = posi_sig
        error_nega = nega_sig + posi_sig

        # shift by `test_lag` to count the cases on the real dates, as the real data does
        T = posi_mu.shape[0]
        corr_posi, corr_sig_posi = np.zeros(T - test_lag), np.zeros(T - test_lag)
        corr_nega, corr_sig_nega = np.zeros(T - test_lag), np.zeros(T - test_lag)

        corr_posi[0 : T - test_lag] = posi_mu[test_lag : T]
        corr_sig_posi[0: T - test_lag] = posi_sig[test_lag: T]
        corr_nega[0 : T - test_lag] = nega_mu[test_lag : T]
        corr_sig_nega[0: T - test_lag] = nega_sig[test_lag: T]

        # lines
        ax.errorbar(ts[0 : T - test_lag], corr_posi, yerr=corr_sig_posi, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts[0: T - test_lag], corr_nega, yerr=corr_sig_nega, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')

        # filling
        ax.fill_between(ts[0: T - test_lag], line_xaxis[0: T - test_lag], corr_posi, alpha=self.filling_alpha, label=r'Positive tests',
                        edgecolor=self.color_posi, facecolor=self.color_posi, linewidth=0, zorder=0)
        ax.fill_between(ts[0: T - test_lag], corr_posi, corr_nega, alpha=self.filling_alpha, label=r'Negative tests',
                        edgecolor=self.color_nega, facecolor=self.color_nega, linewidth=0, zorder=0)
        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu + nega_mu)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def plot_daily_at_home(self, sim, title='Example', filename='daily_at_home_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None):

        ''''
        Plots daily tested, positive daily tested, negative daily tested
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, all_mu, all_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForAllMeasure', acc)
        _,  positive_mu, positive_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForPositiveMeasure', acc)
        _,  age_mu, age_sig = self.__comp_contained_over_time(sim, 'SocialDistancingByAgeMeasure', acc)
        _,  tracing_mu, tracing_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForSmartTracing', acc)

        _, iasy_mu, iasy_sig = self.__comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_over_time(sim, 'isym', acc)

        line_xaxis = np.zeros(ts.shape)

        line_all = all_mu
        line_positive = positive_mu
        line_age = age_mu
        line_tracing = tracing_mu

        line_infected = iasy_mu + ipre_mu + isym_mu

        error_all = all_sig
        error_positive = positive_sig
        error_age = age_sig
        error_tracing = tracing_sig

        error_infected = np.sqrt(np.square(iasy_sig) + np.square(ipre_sig) + np.square(isym_sig))

        # lines
        ax.errorbar(ts, line_infected, label=r'Total infected', errorevery=errorevery, c=self.color_infected, linestyle='--', yerr=error_infected)

        ax.errorbar(ts, line_all, yerr=error_all, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_positive, yerr=error_positive, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_age, yerr=error_age, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_tracing, yerr=error_tracing, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')

        # filling
        ax.fill_between(ts, line_xaxis, line_all, alpha=self.filling_alpha, label=r'SD for all',
                        edgecolor=self.color_all, facecolor=self.color_all, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_positive, alpha=self.filling_alpha, label=r'SD for positively tested',
                        edgecolor=self.color_positive, facecolor=self.color_positive, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_age, alpha=self.filling_alpha, label=r'SD for age group',
                        edgecolor=self.color_age, facecolor=self.color_age, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_tracing, alpha=self.filling_alpha, label=r'SD for traced contacts',
                        edgecolor=self.color_tracing, facecolor=self.color_tracing, linewidth=0, zorder=0)

        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max([all_mu, positive_mu, age_mu, tracing_mu])
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def __compute_Rt_over_time(self, sim, estimation_window):
        '''
        Computes Rt over time by counting infected "children" of nodes
        that got infectious in windows of size `estimation_window`
        over the whole time of simulation,

        Returns Rt as well as proportion accounted to by different
        types of infections (iasy, ipre, isym)
        '''

        ts = [0.0]
        Rt_mu, Rt_std = [0.0], [0.0]
        prop_iasy_mu, prop_ipre_mu, prop_isym_mu = [0.33], [0.33], [0.34]
        prop_iasy_std, prop_ipre_std, prop_isym_std = [0.0], [0.0], [0.0]

        acc = math.ceil(sim.max_time / estimation_window)
        for aa in range(acc - 1):

            # discrete time window
            t0 = aa * estimation_window
            t1 = (aa + 1) * estimation_window
            ts.append((t0 + t1) / 2)

            tmp_Rt, tmp_prop_iasy, tmp_prop_ipre, tmp_prop_isym = [], [], [], []
            for r in range(sim.random_repeats):

                # people that got infectious in this window
                became_iasy = (sim.state_started_at['iasy'][r] >= t0) & (
                    sim.state_started_at['iasy'][r] < t1)
                became_ipre = (sim.state_started_at['ipre'][r] >= t0) & (
                    sim.state_started_at['ipre'][r] < t1)
                became_isym = (sim.state_started_at['isym'][r] >= t0) & (
                    sim.state_started_at['isym'][r] < t1)

                idx_became_iasy = np.where(became_iasy)[0]
                idx_became_ipre = np.where(became_ipre)[0]
                idx_became_isym = np.where(became_isym)[0]

                # count children of people that got asymptomatic now
                iasy_count = idx_became_iasy.shape[0]
                iasy_children = sim.children_count_iasy[r, idx_became_iasy].sum()

                # count children of people that got presymptomatic now
                ipre_count = idx_became_ipre.shape[0]
                ipre_children = sim.children_count_ipre[r, idx_became_ipre].sum()

                # count children of people that got symptomatic now
                isym_count = idx_became_isym.shape[0]
                isym_children = sim.children_count_isym[r, idx_became_isym].sum()

                total = (iasy_children + ipre_children + isym_children)
                if total > 0:
                    tmp_Rt.append((iasy_children + ipre_children + isym_children) / (iasy_count + ipre_count + isym_count))
                    tmp_prop_iasy.append(iasy_children / total)
                    tmp_prop_ipre.append(ipre_children / total)
                    tmp_prop_isym.append(isym_children / total)
                else:
                    tmp_Rt.append(0.0)
                    tmp_prop_iasy.append(0.33)
                    tmp_prop_ipre.append(0.33)
                    tmp_prop_isym.append(0.34)

            Rt_mu.append(np.mean(tmp_Rt))
            prop_iasy_mu.append(np.mean(tmp_prop_iasy))
            prop_ipre_mu.append(np.mean(tmp_prop_ipre))
            prop_isym_mu.append(np.mean(tmp_prop_isym))

            Rt_std.append(np.std(tmp_Rt))
            prop_iasy_std.append(np.std(tmp_prop_iasy))
            prop_ipre_std.append(np.std(tmp_prop_ipre))
            prop_isym_std.append(np.std(tmp_prop_isym))

        Rt_mu, Rt_std = np.array(Rt_mu), np.array(Rt_std)
        prop_iasy_mu, prop_ipre_mu, prop_isym_mu = \
            np.array(prop_iasy_mu), np.array(prop_ipre_mu), np.array(prop_isym_mu)
        prop_iasy_std, prop_ipre_std, prop_isym_std = \
            np.array(prop_iasy_std), np.array(prop_ipre_std), np.array(prop_isym_std)
        return np.array(ts) / 24.0, (Rt_mu, Rt_std), (prop_iasy_mu, prop_ipre_mu, prop_isym_mu), (prop_iasy_std, prop_ipre_std, prop_isym_std)

    def plot_Rt_types(self, sim, title='Example', filename='reproductive_rate_inf_0', errorevery=20, estimation_window=7 * 24.0,
        figsize=(10, 10), lockdown_at=None, lockdown_end=None):

        '''
        Plots Rt split up by infection types (iasy, ipre, isym) over time,
        averaged over random restarts, using error bars for std-dev
        '''
        ts, (Rt_mu, Rt_std), \
            (prop_iasy_mu, prop_ipre_mu, prop_isym_mu), \
                (prop_iasy_std, prop_ipre_std, prop_isym_std) = \
                self.__compute_Rt_over_time(sim, estimation_window)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        line_xaxis = np.zeros(ts.shape)
        line_iasy = prop_iasy_mu * Rt_mu
        line_ipre = prop_iasy_mu * Rt_mu + prop_ipre_mu * Rt_mu
        line_isym = prop_iasy_mu * Rt_mu + prop_ipre_mu * Rt_mu + prop_isym_mu * Rt_mu

        error_iasy = prop_iasy_std * Rt_mu
        error_ipre = prop_iasy_std * Rt_mu + prop_ipre_std * Rt_mu
        error_isym = prop_iasy_std * Rt_mu + prop_ipre_std * Rt_mu + prop_isym_std * Rt_mu

        # lines
        ax.errorbar(ts, line_iasy, c='black', linestyle='-', yerr=error_iasy, elinewidth=0.8, errorevery=errorevery)
        ax.errorbar(ts, line_ipre, c='black', linestyle='-', yerr=error_ipre, elinewidth=0.8, errorevery=errorevery)
        ax.errorbar(ts, line_isym, c='black', linestyle='-', yerr=error_isym, elinewidth=0.8, errorevery=errorevery)

        # filling
        ax.fill_between(ts, line_xaxis, line_iasy, alpha=self.filling_alpha,
                        edgecolor='black', facecolor=self.color_iasy, linewidth=0,
                        label=r'$R_t$ due to asymptomatic $I^a(t)$', zorder=0)
        ax.fill_between(ts, line_iasy, line_ipre, alpha=self.filling_alpha,
                        edgecolor='black', facecolor=self.color_ipre, linewidth=0,
                        label=r'$R_t$ due to pre-symptomatic $I^p(t)$', zorder=0)
        ax.fill_between(ts, line_ipre, line_isym, alpha=self.filling_alpha,
                        edgecolor='black', facecolor=self.color_isym, linewidth=0,
                        label=r'$R_t$ due to symptomatic $I^s(t)$', zorder=0)

        # axis
        maxx = np.max(ts)
        ax.set_xlim((0, maxx))
        ymax = 1.5 * np.max(Rt_mu)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'$R_t$')

        if lockdown_at is not None:
            ax.plot(lockdown_at * np.ones(10), np.linspace(0, ymax, num=10),
                    linewidth=1, linestyle='--', color='black')

        if lockdown_end is not None:
            ax.plot(lockdown_end * np.ones(10), np.linspace(0, ymax, num=10),
                    linewidth=1, linestyle='dotted', color='black', label='End of restrictive measures')
            ax.set_xlim((0, max(maxx, lockdown_end + 2)))


        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def compare_total_infections(self, sims, titles, figtitle='Title',
        filename='compare_inf_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None,
        lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None,
        show_positives=False, test_lag=2, show_legend=True, legendYoffset=0.0, legend_is_left=False,
        subplot_adjust=None, start_date='1970-01-01', first_one_dashed=False):

        ''''
        Plots total infections for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = sims[i].max_time

            ts, iasy_mu, iasy_sig = self.__comp_state_over_time(sims[i], 'iasy', acc)
            _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sims[i], 'ipre', acc)
            _,  isym_mu, isym_sig = self.__comp_state_over_time(sims[i], 'isym', acc)
            _,  posi_mu, posi_sig = self.__comp_state_over_time(sims[i], 'posi', acc)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            ts = days_to_datetime(ts, start_date=start_date)

            line_xaxis = np.zeros(ts.shape)
            line_infected = iasy_mu + ipre_mu + isym_mu
            error_infected = np.sqrt(np.square(iasy_sig) + np.square(ipre_sig) + np.square(isym_sig))

            # lines
            if show_positives:
                ax.errorbar(ts, line_infected, yerr=error_infected, label='[Infected] ' + titles[i], errorevery=errorevery,
                           c=self.color_different_scenarios[i], linestyle='-')

                # shift by `test_lag` to count the cases on the real dates, as the real data does
                T = posi_mu.shape[0]
                corr_posi, corr_sig = np.zeros(T - test_lag), np.zeros(T - test_lag)
                corr_posi[0 : T - test_lag] = posi_mu[test_lag : T]
                corr_sig[0: T - test_lag] = posi_sig[test_lag: T]
                ax.errorbar(ts[:T - test_lag], corr_posi, yerr=corr_sig, label='[Tested positive]', errorevery=errorevery,
                            c=self.color_different_scenarios[i], linestyle='--', elinewidth=0.8)
            else:


                ax.errorbar(ts, line_infected, yerr=error_infected, label=titles[i], errorevery=errorevery, elinewidth=0.8,
                    capsize=3.0, c=self.color_different_scenarios[i], linestyle='--' if i == 0 and first_one_dashed else '-')



            # filling
            # ax.fill_between(ts, line_xaxis, line_infected, alpha=self.filling_alpha, zorder=0,
            #                edgecolor=self.color_different_scenarios[i], facecolor=self.color_different_scenarios[i], linewidth=0)




        # axis
        # ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax, xshift=0.5)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left', borderaxespad=0.5)
            else:
                leg = ax.legend(loc='upper right', borderaxespad=0.5)

            if legendYoffset != 0.0:
                # Get the bounding box of the original legend
                bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

                # Change to location of the legend. 
                bb.y0 += legendYoffset
                bb.y1 += legendYoffset
                leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def compare_total_fatalities_and_hospitalizations(self, sims, titles, figtitle=r'Hospitalizations and Fatalities',
                                 filename='compare_inf_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None, lockdown_at=None,
                                subplot_adjust=None, start_date='1970-01-01', first_one_dashed=False):
        ''''
        Plots total fatalities and hospitalizations for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # hospitalizations
        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = sims[i].max_time

            ts, hosp_mu, hosp_sig = self.__comp_state_over_time(
                sims[i], 'hosp', acc)

            ts, dead_mu, dead_sig = self.__comp_state_over_time(
                sims[i], 'dead', acc)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            ts = days_to_datetime(ts, start_date=start_date)

            # lines
            ax.errorbar(ts, hosp_mu, yerr=hosp_sig, label=titles[i], errorevery=errorevery,
                        c=self.color_different_scenarios[i], linestyle='-', elinewidth=0.8, capsize=3.0)

            ax.errorbar(ts, dead_mu, yerr=dead_sig, errorevery=errorevery,
                        c=self.color_different_scenarios[i], linestyle='--', elinewidth=0.8, capsize=3.0)

        # axis
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')


        if lockdown_at is not None:
            ax.plot(lockdown_at * np.ones(acc), np.linspace(0, ymax, num=acc),
                    linewidth=1, linestyle='--', color='black', zorder=10)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        # ax.legend(loc='upper right', borderaxespad=0.5)
        ax.legend(loc='upper left', borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {
            'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_2d_infections_at_time(self, sim, at_time, density_bandwidth=1.0, restart=0,
        title='Example', filename='2d_inf_0', figsize=(10, 10), acc=1000, ymax=None):

        '''
        Plots 2d visualization using mobility object. The bandwidth set by `density_bandwidth`
        determines the bandwidth of the RBF kernel in KDE used to generate the plot.
        Smaller means more affected by local changes. Set the colors and markers in the __init__ function
        '''
        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # infections
        r = restart
        is_expo = self.__is_state_at(sim, r, 'expo', at_time)
        is_iasy = self.__is_state_at(sim, r, 'iasy', at_time)
        is_ipre = self.__is_state_at(sim, r, 'ipre', at_time)
        is_isym = self.__is_state_at(sim, r, 'isym', at_time)
        is_infected = is_iasy | is_ipre | is_isym
        no_state = (1 - is_infected) & (1 - is_expo)

        idx_expo = np.where(is_expo)[0]
        idx_infected = np.where(is_infected)[0]
        idx_none = np.where(no_state)[0]

        # self.color_isym = 'red'
        # self.color_expo= 'yellow'


        ### sites
        site_loc = sim.site_loc
        ax.scatter(site_loc[:, 0], site_loc[:, 1], alpha=self.filling_alpha, label='public sites',
                   marker=self.marker_site, color=self.color_site, facecolors=self.color_site, s=self.size_site)


        ### home locations and their states
        home_loc = sim.home_loc
        # no state
        ax.scatter(home_loc[idx_none, 0], home_loc[idx_none, 1],
                   marker=self.marker_home, color=self.color_home,
                   facecolors='none', s=self.size_home)

        try:
            # expo
            ax.scatter(home_loc[idx_expo, 0], home_loc[idx_expo, 1],
                    marker=self.marker_home, color=self.color_home,
                    facecolors=self.color_expo, s=self.size_home, label='exposed households')
            sns.kdeplot(home_loc[idx_expo, 0], home_loc[idx_expo, 1], shade=True, alpha=self.density_alpha,
                        shade_lowest=False, cbar=False, ax=ax, color=self.color_expo, bw=density_bandwidth, zorder=0)

            # infected
            ax.scatter(home_loc[idx_infected, 0], home_loc[idx_infected, 1],
                    marker=self.marker_home, color=self.color_home,
                    facecolors=self.color_isym, s=self.size_home, label='infected households')
            sns.kdeplot(home_loc[idx_infected, 0], home_loc[idx_infected, 1], shade=True, alpha=self.density_alpha,
                        shade_lowest=False, cbar=False, ax=ax, color=self.color_isym, bw=density_bandwidth, zorder=0)

        except:
            print('KDE failed, likely no exposed and infected at this time. Try different timing.')
            plt.close()
            return

        # axis
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        plt.axis('off')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.85)

        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def compare_hospitalizations_over_time(self, sims, titles, figtitle='Hospitalizations', filename='compare_hosp_0',
        capacity_line_at=20, figsize=(10, 10), errorevery=20, acc=1000, ymax=None):
        ''''
        Plots total hospitalizations for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev.
        The value of `capacity_line_at` defines the y-intercept of the hospitalization capacity line
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = sims[i].max_time

            ts, line_hosp, error_sig = self.__comp_state_over_time(
                sims[i], 'hosp', acc)
            line_xaxis = np.zeros(ts.shape)

            # lines
            ax.errorbar(ts, line_hosp, yerr=error_sig, errorevery=errorevery,
                        c='black', linestyle='-', elinewidth=0.8)

            # filling
            ax.fill_between(ts, line_xaxis, line_hosp, alpha=self.filling_alpha, zorder=0,
                            label=r'Hospitalized under: ' + titles[i], edgecolor=self.color_different_scenarios[i],
                            facecolor=self.color_different_scenarios[i], linewidth=0)

        # capacity line
        ax.plot(ts, capacity_line_at * np.ones(ts.shape[0]), label=r'Max. hospitalization capacity',
                    c='red', linestyle='--', linewidth=4.0)

        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(line_hosp + error_sig)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(figtitle, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def plot_positives_vs_target(self, sim, targets, test_lag, title='Example',
        filename='inference_0', figsize=(6, 5), errorevery=1, acc=17, ymax=None,
        start_date='1970-01-01', lockdown_label='Lockdown', lockdown_at=None,
        lockdown_label_y=None, subplot_adjust=None):
        ''''
        Plots daily tested averaged over random restarts, using error bars for std-dev
        together with targets from inference
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig, ax = plt.subplots(figsize=figsize)

        # inference
        ts, posi_mu, posi_sig = self.__comp_state_over_time(sim, 'posi', acc)
        # shift by `test_lag` to count the cases on the real dates, as the real data does
        T = posi_mu.shape[0]
        corr_posi, corr_sig = np.zeros(T - test_lag), np.zeros(T - test_lag)
        corr_posi[0 : T - test_lag] = posi_mu[test_lag : T]
        corr_sig[0: T - test_lag] = posi_sig[test_lag: T]

        # Convert x-axis into posix timestamps and use pandas to plot as dates
        xx = days_to_datetime(ts[:T - test_lag], start_date=start_date)
        ax.plot(xx, corr_posi, c='k', linestyle='-',
                label='COVID-19 simulated case data')
        ax.fill_between(xx, corr_posi-corr_sig, corr_posi+corr_sig,
                        color='grey', alpha=0.1, linewidth=0.0)

        # target
        # txx = np.linspace(0, targets.shape[0] - 1, num=targets.shape[0])
        # # Convert x-axis into posix timestamps and use pandas to plot as dates
        # txx = days_to_datetime(txx, start_date=start_date)
        # ax.plot(txx, targets, linewidth=4, linestyle='', marker='X', ms=6,
        #         color='red', label='COVID-19 case data')
        target_widget(targets, start_date, ax)

        # axis
        #ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('Positive cases')

        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc='upper left', borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)#, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return
