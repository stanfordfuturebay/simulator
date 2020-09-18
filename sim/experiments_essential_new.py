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
    SocialDistancingForPositiveMeasureHousehold,
    ComplianceForAllMeasure,
    ComplianceForEssentialWorkers,
    SocialDistancingForNonEssential,
    SocialDistancingForSmartTracingHousehold,
    Interval)
from lib.data import collect_data_from_df
from lib.plot import Plotter
from lib.distributions import CovidDistributions
from lib.parallel import *
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
    parser.add_argument('--beta',type=float, default=0.1, 
                        help="Site infectivity parameter for all site types") # TODO set different betas for each site type
    parser.add_argument('--beta_household', type=float, default=0.0,
                        help="Infectivity within a household")
    parser.add_argument('--worker_type',choices=['education','office','social','supermarket','multi'],required=True,
                        help="Which type of essential worker to include in model, at specified percentage of population")
    parser.add_argument('--essential_pct',type=int,default=50,
                       help="Proportion of total population to set to be essential workers of type --worker_type")
    parser.add_argument('--downsample',type=int,default=100)
#     parser.add_argument('--mob_settings', type=str, default='lib/mobility/San_Francisco_settings_100.pk', 
#                         help="Path to mobility settings pickle file")
    parser.add_argument('--seed', type=int, default=0,
                        help="Set random seed for reproducibility")
    parser.add_argument('--area', type=str, default='SF')
    parser.add_argument('--country', type=str, default='US')
    parser.add_argument('--mob_settings',type=str, default=None)
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
#     mob_settings = args.mob_settings
    area = args.area

    mob_settings = f'lib/mobility/San_Francisco_settings_{args.downsample}_{args.worker_type}_{args.essential_pct}pct.pk' if not args.mob_settings else args.mob_settings
    
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)
    np.random.seed(c)
    runstr = f'run{c}_'


    #### Get case data  ####
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


    #### Set epidemic parameters as inferred using Bayesian optimization ####
    # inferred parameters (see paper)
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
                t_window=Interval(0.0, t), p_isolate=1.0),
            
            SocialDistancingForNonEssential(
                t_window=Interval(0.0,t), p_stay_home=0.5)
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

    # ### 4.3.5. Effects  of compliance on the efficacy of isolation for smart  tracing strategies
    daily_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    testing_params_SD_6 = standard_testing(max_time_future, daily_increase)
    testing_params_SD_6['test_smart_delta'] = 24.0 * 3     # time window considered for inspecting contacts
    testing_params_SD_6['test_smart_action'] = 'isolate'
    testing_params_SD_6['test_targets'] = 'isym'
    testing_params_SD_6['test_smart_num_contacts'] = 25
    testing_params_SD_6['smart_tracing'] = 'advanced'
    isolation_days_SD_6 = 7  # how many days selected people have to stay in isolation
    duration_weeks_SD_6 = 12  # strategies tested for 12 weeks starting today

    summaries_ = dict()
    
    # CT 0.0 compliance
    policy = 'None'
    p=0.0
    m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
         ComplianceForAllMeasure(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p),
         SocialDistancingForSmartTracingHousehold(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_isolate=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6)
        ]
    res = run(testing_params_SD_6, m, max_time_future, present_seeds)
    summaries_[(policy,p)] = res
    print(f'Completed {policy} {p}')
    
    # CT 0.5 distributed randomly
    policy = 'random'
    p=0.5
    m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
         ComplianceForAllMeasure(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p),
         SocialDistancingForSmartTracingHousehold(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_isolate=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6)         
        ]
    res = run(testing_params_SD_6, m, max_time_future, present_seeds)
    summaries_[(policy,p)] = res
    print(f'Completed {policy} {p}')
    
    # CT 0.5 distributed to essential workers first
    policy = 'essential'
    p=0.5
    m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
         ComplianceForEssentialWorkers(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p),
         SocialDistancingForSmartTracingHousehold(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_isolate=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6)         
        ]
    res = run(testing_params_SD_6, m, max_time_future, present_seeds)
    summaries_[(policy,p)] = res
    print(f'Completed {policy} {p}')
    
    # CT 1.0 (distributed to everyone)
    policy = 'None'
    p=1.0
    m = [SocialDistancingForSmartTracing(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_stay_home=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6),
         ComplianceForAllMeasure(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_compliance=p),
         SocialDistancingForSmartTracingHousehold(
            t_window=Interval(*testing_params_SD_6['testing_t_window']),
            p_isolate=1.0,
            test_smart_duration=24.0 * isolation_days_SD_6)         
        ]
    res = run(testing_params_SD_6, m, max_time_future, present_seeds)
    summaries_[(policy,p)] = res
    print(f'Completed {policy} {p}')    

    # save summary
    save_summary(summaries_, f'{args.outfile}.pk')
