from lib.inference import *
from lib.town_maps import MapIllustrator
from lib.town_data import generate_population, generate_sites, compute_distances
from lib.measures import (
    MeasureList,
    BetaMultiplierMeasure,
    BetaMultiplierMeasureByType,
    SocialDistancingForAllMeasure,
    SocialDistancingForKGroups,
    SocialDistancingByAgeMeasure,
    SocialDistancingForPositiveMeasure,
    ComplianceForAllMeasure,
    Interval)
from lib.data import collect_data_from_df
from lib.plot import Plotter
from lib.distributions import CovidDistributions
from lib.parallel import *
from bayes_opt import BayesianOptimization
from lib.dynamics import DiseaseModel
from lib.mobilitysim import MobilitySimulator
import multiprocessing
import re
import matplotlib
import warnings
import pickle
import seaborn
import math
import scipy as sp
import copy
import networkx as nx
import numpy as np
import pandas as pd
import sys
if '..' not in sys.path:
    sys.path.append('..')

if __name__ == '__main__':

    mob_settings = 'lib/tu_settings_20_10.pk'
    random_repeats = 40

    # cpus_used = multiprocessing.cpu_count()
    cpus_used = 23

    c = 0  # seed
    FIGSIZE = (8, 4)

    # #### Import town settings

    tuned_p_stay_home = 0.2
    tuned_site_multipliers = [0.0, 0.0, 0.5, 0.5, 0.5]

    # See town-generator.ipynb for an example on how to create own settings
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)
    np.random.seed(c)
    runstr = f'run{c}_'

    # General note for plotting: `errorevery` has to be set proportionally to `acc`, and best to keep `acc` as it is
    days_until_lockdown = 13  # March 10 - March 23
    days_after_lockdown = 20  # March 24 - April 12
    days_present = days_until_lockdown + days_after_lockdown + \
        2  # simulate 2 more days due to test lag
    days_future = 12 * 7  # projecting 12 weeks into the future

    case_downsample = 10
    new_cases_ = collect_data_from_df('LK Tübingen', 'new', until=days_present)
    resistant_cases_ = collect_data_from_df(
        'LK Tübingen', 'recovered', until=days_present)
    fatality_cases_ = collect_data_from_df(
        'LK Tübingen', 'fatality', until=days_present)


    # fatality rate per age group
    num_age_groups = fatality_cases_.shape[1]
    fatality_rates_by_age = (
        fatality_cases_[-1, :] / (new_cases_[-1, :] + fatality_cases_[-1, :] + resistant_cases_[-1, :]))

    print('Empirical fatality rates per age group:  ',
          fatality_rates_by_age.tolist())

    # Scale down cases based on number of people in simulation

    new_cases, resistant_cases, fatality_cases = (
        1/case_downsample * new_cases_,
        1/case_downsample * resistant_cases_,
        1/case_downsample * fatality_cases_)
    new_cases, resistant_cases, fatality_cases = np.ceil(
        new_cases), np.ceil(resistant_cases), np.ceil(fatality_cases)

    # Define initial seed count (based on infection counts on March 10)

    initial_seeds = {
        'expo': 1,
        'ipre': 1,
        'isym': 3,
        'iasy': 3,
    }
    present_seeds = {
        'expo': 3,
        'ipre': 1,
        'iasy': 0,
        'isym_notposi': 8,
        'isym_posi': 4,
        'resi_notposi': 78,
        'resi_posi': 110,
    }
    max_time_present = 24.0 * (days_present)
    max_time_future = 24.0 * (days_future)

    # #### Define standard testing parameters, same used for inference

    def standard_testing(max_time):
        standard_testing_params = {
            'testing_t_window': [0.0, max_time],  # in hours
            'testing_frequency': 24.0,     # in hours
            # in hours (actual and self-report delay)
            'test_reporting_lag': 48.0,
            'tests_per_batch': 10,       # assume 300 tests/day in LK Tübingen
            'test_smart_delta': 24.0 * 3,  # in hours
            'test_smart_duration': 24.0 * 7,  # in hours
            'test_smart_action': 'isolate',
            'test_smart_num_contacts': 10,
            'test_targets': 'isym',
            'test_queue_policy': 'fifo',
            'smart_tracing': None,
        }
        return standard_testing_params
    # #### Define distributions as estimated by literature
    #


    # instantiate correct distributions
    distributions = CovidDistributions(
        fatality_rates_by_age=fatality_rates_by_age)

    # #### Set epidemic parameters as inferred using Bayesian optimization

    # inferred parameters (see paper)
    inferred_params = {
        'betas': [1.1383] * 5,  # site infectivity by type
        'alpha': 0.3224,
        'mu': 0.2072
    }

    # Define function to run general type of experiment, fixing the above settings.

    def run(tparam, measure_list, t, local_seeds):

        # add standard measure of positives staying isolated
        measure_list += [
            SocialDistancingForPositiveMeasure(
                t_window=Interval(0.0, t), p_stay_home=1.0)
        ]
        measure_list = MeasureList(measure_list)

        # run simulations
        summary = launch_parallel_simulations(
            mob_settings,
            distributions,
            random_repeats, cpus_used,
            inferred_params, local_seeds, tparam, measure_list,
            max_time=t,
            num_people=mob.num_people,
            num_sites=mob.num_sites,
            site_loc=mob.site_loc,
            home_loc=mob.home_loc, verbose=False)
        return summary


    def save_summary(summary, filename):
        with open('summaries/' + filename, 'wb') as fp:
            pickle.dump(summary, fp)

    def load_summary(filename):
        with open('summaries/' + filename, 'rb') as fp:
            summary = pickle.load(fp)
        return summary


    """
    ========================= END OF HEADER =========================

    """

    # ## Only simulate the future from here onwards
    #
    #
    tuned_p_stay_home = 0.2
    tuned_site_multipliers = [0.0, 0.0, 0.5, 0.5, 0.5]

    # baseline (no measure for the future starting April 12)
    # future_baseline = run(standard_testing(max_time_future),
    #                       [], max_time_future, present_seeds)
    # save_summary(future_baseline, 'future_baseline_3.pk')

    # ### 4.3.4. Can we control the outbreak using only contact tracing and isolation?

    testing_params_SD_5 = standard_testing(max_time_future)
    # time window considered for inspecting contacts
    testing_params_SD_5['test_smart_delta'] = 24.0 * 3
    testing_params_SD_5['test_smart_action'] = 'isolate'
    testing_params_SD_5['test_targets'] = 'isym'

    # how many days selected people have to stay in isolation
    isolation_days = [7]
    # how many contacts are isolated in the `test_smart_delta` window
    contacts_isolated = [10, 25]
    duration_weeks_SD_5 = 12  # strategies tested for 12 weeks starting today

    summaries_SD_5 = dict()

    weeks = 1

    # baseline
    # same as tuned plus different isolation strategies for contact tracing
    m = [ # education close, social closes, rest reduced by 50%
        BetaMultiplierMeasureByType(
        t_window=Interval(0.0, 24.0 * weeks),
        beta_multiplier=tuned_site_multipliers),

        # 40 % less activite of all due to contact constraints
        SocialDistancingForAllMeasure(
        t_window=Interval(0.0, 24.0 * weeks),
        p_stay_home=tuned_p_stay_home)
    ]
    res = run(testing_params_SD_5, m,
              max_time_future, present_seeds)

    for days in isolation_days:
        for contacts in contacts_isolated:
            summaries_SD_5[(days, contacts)] = [res]

    # Isolation strategies
    for days in isolation_days:
        for j, contacts in enumerate(contacts_isolated):
            for policy in ['basic', 'advanced']:

                testing_params_SD_5['smart_tracing'] = policy
                testing_params_SD_5['test_smart_num_contacts'] = contacts

                # same as tuned plus different isolation strategies for contact tracing
                m = [SocialDistancingForSmartTracing(
                    t_window=Interval(
                        *testing_params_SD_5['testing_t_window']),
                    p_stay_home=1.0,
                    test_smart_duration=24.0 * days),

                    # education close, social closes, rest reduced by 50%
                    BetaMultiplierMeasureByType(
                        t_window=Interval(0.0, 24.0 * weeks),
                        beta_multiplier=tuned_site_multipliers),

                    # 40 % less activite of all due to contact constraints
                    SocialDistancingForAllMeasure(
                        t_window=Interval(0.0, 24.0 * weeks),
                        p_stay_home=tuned_p_stay_home)
                ]

                res = run(testing_params_SD_5, m,
                          max_time_future, present_seeds)
                summaries_SD_5[(days, contacts)].append(res)

                print(days, contacts, policy, ' done.')

            save_summary(summaries_SD_5, f'summaries_SD_5_and_measures_c{j}.pk')

    save_summary(summaries_SD_5, 'summaries_SD_5_and_measures.pk')
    # summaries_SD_5 = load_summary('summaries_SD_5.pk')
