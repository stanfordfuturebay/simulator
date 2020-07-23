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
from lib.runutils import *
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
import argparse
if '..' not in sys.path:
    sys.path.append('..')

    
    
if __name__ == '__main__':
    ### Parse Arguments ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_days', type=int, default=84, 
                        help="Integer number of days to run simulation. Default 12 weeks (84 days).")
    parser.add_argument('--outfile', type=str, default='summaries_SD_5', 
                        help="Name (without extension) for output pickle file")
    parser.add_argument('--num_workers',type=int, default=23, 
                        help="Number of parallel threads to run simultaneously, capped at (number of available CPUs - 1)")
    parser.add_argument('--random_repeats',type=int, default=40, 
                        help="Number of random realizations to run. Use at least 40 for stable results")
    parser.add_argument('--beta',type=float, default=0.5, 
                        help="Site infectivity parameter for all site types") # TODO set different betas for each site type
    parser.add_argument('--beta_household', type=float, default=0.5,
                        help="Infectivity within a household")
    parser.add_argument('--mob_settings', type=str, default='lib/mobility/Tubingen_settings_10.pk', 
                        help="Path to mobility settings pickle file")
    parser.add_argument('--seed', type=int, default=0,
                        help="Set random seed for reproducibility")
    parser.add_argument('--area', type=str, default='TU')
    parser.add_argument('--country', type=str, default='GER')
    args = parser.parse_args()
    print(args)
    

    random_repeats = args.random_repeats
    num_workers = min(args.num_workers,multiprocessing.cpu_count()-1)
    c = args.seed  # seed
    
    # mobility settings
    if args.area=='SF':
        from lib.settings.town_settings_sanfrancisco import *
        country='US'
    else:
        from lib.settings.town_settings_tubingen import *
        country='GER'
    mob_settings = args.mob_settings
    area = args.area
#     country = args.country
    
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)
    np.random.seed(c)
    runstr = f'run{c}_'


    #### Get case data and infer fatality rates by age group ####
    days_future = args.sim_days
    
    start_date = '2020-03-08'
    end_date = '2020-03-27'
    new_cases_ = collect_data_from_df(country=country, area=area, datatype='new',
        start_date_string=start_date, end_date_string=end_date)
    new_cases = np.ceil(
            (new_cases_ * mob.num_people_unscaled) /
            (mob.downsample * mob.region_population))


     # instantiate correct distributions
    distributions = CovidDistributions(country=country)   
    
    
    # Define initial seed count (based on infection counts on March 10)
    present_seeds = {
        'expo': 3,
        'ipre': 1,
        'iasy': 0,
        'isym_notposi': 8,
        'isym_posi': 4,
        'resi_notposi': 78,
        'resi_posi': 110,
    }
    max_time_future = 24.0 * (days_future)

    



    #### Set epidemic parameters as inferred using Bayesian optimization
    # inferred parameters (see paper)
#     beta = 1.1383
    beta = args.beta
    inferred_params = {
        'betas' : {
            'education': beta,
            'social': beta,
            'bus_stop': beta,
            'office': beta,
            'supermarket': beta}, 
        'beta_household' : args.beta_household
    }
    print(f'inferred_params: {inferred_params}')

    
    # Define function to run general type of experiment, fixing the above settings.
    def run(tparam, measure_list, t, local_seeds, dynamic_tracing=False):
        tic = time.perf_counter()
        # add standard measure of positives staying isolated
        measure_list +=  [
            SocialDistancingForPositiveMeasure(
                t_window=Interval(0.0, t), p_stay_home=1.0),

            SocialDistancingForPositiveMeasureHousehold(
                t_window=Interval(0.0, t), p_isolate=1.0)
        ]
        measure_list = MeasureList(measure_list)

        # run simulations
        summary = launch_parallel_simulations(
            mob_settings, 
            distributions, 
            random_repeats, num_workers, 
            inferred_params, local_seeds, tparam, measure_list, 
            max_time=t, 
            num_people=mob.num_people, 
            num_sites=mob.num_sites, 
            site_loc=mob.site_loc, 
            home_loc=mob.home_loc,
            dynamic_tracing=dynamic_tracing,
            verbose=False)
        toc = time.perf_counter()
        print(f'Elapsed time {(toc-tic):0.1f} seconds ({((toc-tic)/60):0.1f} minutes)')
        return summary





    """
    ========================= END OF HEADER =========================

    """


    #### 4.3.4. Can we control the outbreak using only contact tracing and isolation? ####
    daily_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    testing_params_SD_5 = standard_testing(max_time_future, daily_increase)
    testing_params_SD_5['test_smart_delta'] = 24.0 * 3
    testing_params_SD_5['test_smart_action'] = 'isolate'
    testing_params_SD_5['test_targets'] = 'isym'

    
    isolation_days = [7] # how many days selected people have to stay in isolation
    contacts_isolated = [25] # how many contacts are isolated in the `test_smart_delta` window

    
    summaries_SD_5 = dict()
    
    # Run baseline
    m = []
    res = run(testing_params_SD_5, m,
              max_time_future, present_seeds)

    # Add baseline to summaries for each (days, contacts)
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
                    test_smart_duration=24.0 * days)
                ]
                res = run(testing_params_SD_5, m,
                          max_time_future, present_seeds)
                summaries_SD_5[(days, contacts)].append(res)
                print(days, contacts, policy, ' done.')

    save_summary(summaries_SD_5, f'{args.outfile}.pk')
