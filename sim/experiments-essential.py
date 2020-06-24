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
    ComplianceForEssentialWorkers,
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
    parser.add_argument('--outfile', type=str, default='summaries_ST_essential', 
                        help="Name (without extension) for output pickle file")
    parser.add_argument('--num_workers',type=int, default=23, 
                        help="Number of parallel threads to run simultaneously, capped at (number of available CPUs - 1)")
    parser.add_argument('--random_repeats',type=int, default=40, 
                        help="Number of random realizations to run. Use at least 40 for stable results")
    parser.add_argument('--beta',type=float, default=1.1383, 
                        help="Site infectivity parameter for all site types") # TODO set different betas for each site type
    parser.add_argument('--alpha',type=float, default=0.3224, 
                        help="Proportion of cases that are asymptomatic")
    parser.add_argument('--mu', type=float, default=0.2072,
                        help="Relative infectivity of asymptomatic cases")
    parser.add_argument('--mob_settings', type=str, default='lib/tu_settings_20_10.pk', 
                        help="Path to mobility settings pickle file")
    parser.add_argument('--seed', type=int, default=0,
                        help="Set random seed for reproducibility")
    parser.add_argument('--p_compliance', type=float, nargs='*', default=[0,.25,.5,.75,1.0],
                        help="A list of values [0,1] representing either the proportion of the population who complies with contact tracing, or the proportion of the population that are essential workers, assuming all essential workers comply with contact tracing.")
    parser.add_argument('--only_essential',action='store_true', default=False,
                        help="Only run experiments for different proportions of essential workers, skip compliance levels for everyone")
    args = parser.parse_args()
    print(args)
    
    
    
#     mob_settings = 'lib/essen_settings_100_20.pk'
    mob_settings = args.mob_settings
    random_repeats = args.random_repeats
    num_workers = min(args.num_workers,multiprocessing.cpu_count()-1)
    c = args.seed  # seed

    # #### Import town settings
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)
    np.random.seed(c)
    runstr = f'run{c}_'

    #### Get case data and infer fatality rates by age group ####
    days_present = 33 # placeholder, get average fatality rates for March 10-April 12
    days_future = args.sim_days

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

    # instantiate correct distributions
    distributions = CovidDistributions(
        fatality_rates_by_age=fatality_rates_by_age)
    
    # Scale down cases based on number of people in simulation
    new_cases, resistant_cases, fatality_cases = (
        1/case_downsample * new_cases_,
        1/case_downsample * resistant_cases_,
        1/case_downsample * fatality_cases_)
    new_cases, resistant_cases, fatality_cases = np.ceil(
        new_cases), np.ceil(resistant_cases), np.ceil(fatality_cases)

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
    max_time_present = 24.0 * (days_present)
    max_time_future = 24.0 * (days_future)





    #### Set epidemic parameters as inferred using Bayesian optimization ####
    # inferred parameters (see paper)
    inferred_params = {
        'betas': [args.beta] * 5,  # site infectivity by type
        'alpha': args.alpha,
        'mu': args.mu
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
            random_repeats, num_workers,
            inferred_params, local_seeds, tparam, measure_list,
            max_time=t,
            num_people=mob.num_people,
            num_sites=mob.num_sites,
            site_loc=mob.site_loc,
            home_loc=mob.home_loc, verbose=False)
        return summary



    """
    ========================= END OF HEADER =========================
    """
    
    
    ### 4.3.5. Effects  of compliance on the efficacy of isolation for smart  tracing strategies    
    testing_params_SD_6 = standard_testing(max_time_future)
    testing_params_SD_6['test_smart_delta'] = 24.0 * 3    # time window considered for inspecting contacts
    testing_params_SD_6['test_smart_action'] = 'isolate'
    testing_params_SD_6['test_targets'] = 'isym'
    testing_params_SD_6['test_smart_num_contacts'] = 25
    isolation_days_SD_6 = 7  # how many days selected people have to stay in isolation

    summaries_SD_6 = dict()

    p_compliance = args.p_compliance    

    for policy in ['basic']:
        summaries_ = []
        testing_params_SD_6['smart_tracing'] = policy

        if args.only_essential is False:
            for p in p_compliance:

                m = [SocialDistancingForSmartTracing(
                        t_window=Interval(*testing_params_SD_6['testing_t_window']),
                        p_stay_home=1.0,
                        test_smart_duration=24.0 * isolation_days_SD_6),
                    ComplianceForAllMeasure(
                        t_window=Interval(*testing_params_SD_6['testing_t_window']),
                        p_compliance=p)
                ]
                res = run(testing_params_SD_6, m, max_time_future, present_seeds)
                summaries_.append(res)

                print('ComplianceForAll', p, ' done.')
            summaries_SD_6[('all', p)] = summaries_

        
        for p in p_compliance:

            mob.essential_workers = generate_sf_essential(p)
            
            m = [SocialDistancingForSmartTracing(
                    t_window=Interval(*testing_params_SD_6['testing_t_window']),
                    p_stay_home=1.0,
                    test_smart_duration=24.0 * isolation_days_SD_6),
                ComplianceForEssentialWorkers(
                    t_window=Interval(*testing_params_SD_6['testing_t_window']),
                    p_compliance=1.0)
            ]
            res = run(testing_params_SD_6, m, max_time_future, present_seeds)
            summaries_.append(res)

            print('ComplianceForEssentialWorkers', p, ' done.')
        summaries_SD_6[('essential',p)] = summaries_
        
    save_summary(summaries_SD_6, f'{args.outfile}.pk') 