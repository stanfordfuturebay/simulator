import pickle
import os
import numpy as np
import pandas as pd
from lib.town_data import generate_population
import pdb
import dateutil
import datetime
from lib.measures import *

#### Define standard testing parameters, same used for inference
def standard_testing(max_time, daily_increase):
    TO_HOURS = 24.0
#     daily_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    standard_testing_params = {
        'testing_t_window'    : [0.0, max_time], # in hours
        'testing_frequency'   : 1 * TO_HOURS,     # in hours
        'test_reporting_lag'  : 2 * TO_HOURS,     # in hours (actual and self-report delay)
        'tests_per_batch'     : int(daily_increase.max()), # test capacity based on empirical positive tests
        'test_fpr'            : 0.0, # test false positive rate
        'test_fnr'            : 0.0, # test false negative rate
        'test_smart_delta'    : 3 * TO_HOURS, # in hours
        'test_smart_duration' : 7 * TO_HOURS, # in hours
        'test_smart_action'   : 'isolate', 
        'test_smart_num_contacts'   : 10, 
        'test_targets'        : 'isym',
        'test_queue_policy'   : 'fifo',
        'smart_tracing'       : None, 
    }
    return standard_testing_params

def save_summary(summary, filename):
    with open('summaries/' + filename, 'wb') as fp:
        pickle.dump(summary, fp)

def load_summary(filename):
    with open('summaries/' + filename, 'rb') as fp:
        summary = pickle.load(fp)
    return summary


def generate_sf_essential(prop_essential_total):
    population_path='lib/data/population_sf/' # Directory containing FB population density files
    sites_path='lib/data/queries_sf/' # Directory containing OSM site files
    bbox = (37.7115, 37.8127, -122.5232, -122.3539) # Coordinate bounding box

    density_files=[]
    for root,dirs,files in os.walk(population_path):
        for f in files:
            if f.endswith(".csv"):
                density_files.append(population_path+f)   
    population_per_age_group = [194, 296, 154, 263, 1646, 835, 682, 181]     
    

    # proportion of all essential workers within each age group
    prop_essential_per_age_group = np.array([
        0,   # 0-4
        0,   # 5-14
        .04,  # 15-19
        .06,  # 20-24
        .45,  # 25-44
        .24,  # 45-59
        .20, # 60-79
        0])  # 

    # proportion of each age group that are essential workers
    essential_prop_per_age_group = (prop_essential_per_age_group*prop_essential_total) / (np.array(population_per_age_group) / sum(population_per_age_group))

    
    _, _, _, _, essential_workers = generate_population(density_files=density_files, bbox=bbox, population_per_age_group=population_per_age_group, tile_level=16, seed=42, essential_prop_per_age_group=essential_prop_per_age_group)
    
    return essential_workers



def num_people(summary):
    return len(summary.people_age[0])

def num_essential(summary):
    return summary.essential_workers[0].sum()

def num_nonessential(summary):
    return (summary.essential_workers[0]==False).sum()

def num_type_people(summary,wtype):
    return (summary.worker_types[0]==wtype).sum()


def num_infected(summary):
    has_been_infected_states = ['expo','ipre','isym','iasy','resi','dead']
    result = 0.0
    for state in has_been_infected_states:
        arr = summary.state[state]
        result += (arr.sum() / float(summary.random_repeats))
    return result

def num_nonessential_infected(summary):
    has_been_infected_states = ['expo','ipre','isym','iasy','resi','dead']
    result = 0.0
    for state in has_been_infected_states:
        arr = summary.state[state]
        temp = np.broadcast_to((summary.essential_workers[0]==False), (len(arr),len(summary.essential_workers[0])))
        result += (arr[temp].sum() / float(summary.random_repeats))
    return result

def num_essential_infected(summary):
    # individuals in any of these states are either currently infected or were once infected (dead or recovered)
    has_been_infected_states = ['expo','ipre','isym','iasy','resi','dead']
#     result = {}
    result = 0.0
    for state in has_been_infected_states:
        arr = summary.state[state]
        temp = np.broadcast_to((summary.essential_workers[0]==True), (len(arr),len(summary.essential_workers[0])))
        result += (arr[temp].sum() / float(summary.random_repeats))
#         result[state] = arr[summary.essential_workers[0]==True].sum() / float(summary.random_repeats)
    return result

def num_type_infected(summary,wtype):
    # individuals in any of these states are either currently infected or were once infected (dead or recovered)
    has_been_infected_states = ['expo','ipre','isym','iasy','resi','dead']
#     result = {}
    result = 0.0
    for state in has_been_infected_states:
        arr = summary.state[state]
        temp = np.broadcast_to((summary.worker_types[0]==wtype), (len(arr),len(summary.worker_types[0])))
        result += (arr[temp].sum() / float(summary.random_repeats))
#         result[state] = arr[summary.essential_workers[0]==True].sum() / float(summary.random_repeats)
    return result

def num_contacts_uncontained(summary):
    ncu_total = 0
    ncu_nonessential = 0
    ncu_essential = 0
    for i in range(len(summary.mob)):
        for contact in summary.mob[i]:
            if (contact.data['i_contained']==False) and (contact.data['j_contained']==False):
                ncu_total +=1
                if summary.essential_workers[0][contact.indiv_j]==True:
                    ncu_essential += 1
                else:
                    ncu_nonessential += 1
    ncu_total /= len(summary.mob)
    ncu_nonessential /= len(summary.mob)
    ncu_essential /= len(summary.mob)
    return ncu_total, ncu_nonessential, ncu_essential


def num_contacts_uncontained_new(summary):
    ncu_total = 0
    ncu_nonessential = 0
    ncu_education = 0
    ncu_office = 0
    ncu_social = 0
    ncu_supermarket = 0
    for i in range(len(summary.mob)):
        for contact in summary.mob[i]:
            if (contact.data['i_contained']==False) and (contact.data['j_contained']==False):
                ncu_total +=1
                wtype = summary.worker_types[0][contact.indiv_j]
                if wtype==0:
                    ncu_education += 1
                elif wtype==1:
                    ncu_office += 1
                elif wtype==2:
                    ncu_social += 1
                elif wtype==3:
                    ncu_supermarket += 1
                else:
                    ncu_nonessential += 1
    ncu_total /= len(summary.mob)
    ncu_nonessential /= len(summary.mob)
    ncu_education /= len(summary.mob)
    ncu_office /= len(summary.mob)
    ncu_social /= len(summary.mob)
    ncu_supermarket /= len(summary.mob)
    return ncu_total, ncu_nonessential, ncu_education, ncu_office, ncu_social, ncu_supermarket
    
def make_summary_df(summary):
    df = pd.DataFrame(index=['num_people','num_infected','pct_infected','num_contacts'],columns=['Total','Nonessential','Essential'])
    df.loc['num_people','Total'] = num_people(summary)
    df.loc['num_people','Nonessential'] = num_nonessential(summary)
    df.loc['num_people','Essential'] = num_essential(summary)
    df.loc['num_infected','Total'] = num_infected(summary)
    df.loc['num_infected','Nonessential'] = num_nonessential_infected(summary)
    df.loc['num_infected','Essential'] = num_essential_infected(summary)
    df.loc['pct_infected','Total'] = (float(df.loc['num_infected','Total']) / df.loc['num_people','Total'])
    df.loc['pct_infected','Nonessential'] = float(df.loc['num_infected','Nonessential']) / df.loc['num_people','Nonessential']
    df.loc['pct_infected','Essential'] = float(df.loc['num_infected','Essential']) / df.loc['num_people','Essential']
    df.loc['pct_infected'] = df.loc['pct_infected'].apply('{:.1%}'.format)
    df.loc['num_contacts', :] = num_contacts_uncontained(summary)
    return df


def make_summary_df_new(summary):
    df = pd.DataFrame(index=['num_people','num_infected','pct_infected','num_contacts'],columns=['Total','Nonessential','Education','Office','Social','Supermarket'])
    df.loc['num_people','Total'] = num_people(summary)
    df.loc['num_infected','Total'] = num_infected(summary)
    df.loc['pct_infected','Total'] = (float(df.loc['num_infected','Total']) / df.loc['num_people','Total'])
    for i, col in enumerate(df.columns[1:]):
        wtype = i-1
        df.loc['num_people',col] = num_type_people(summary,wtype)
        df.loc['num_infected',col] = num_type_infected(summary)
        df.loc['pct_infected',col] = float(df.loc['num_infected',col]) / df.loc['num_people',col]
    
    df.loc['num_contacts', :] = num_contacts_uncontained_new(summary)     
    df.loc['pct_infected'] = df.loc['pct_infected'].apply('{:.1%}'.format)
    return df
    
    
    
def beta_mult_measures_from_csv(filename, start_date_str, sim_days, site_dict):
    measures = []
    start_date = dateutil.parser.parse(start_date_str)
    end_date = start_date + datetime.timedelta(days=sim_days)
    df = pd.read_csv(filename)
    
    for date_str in df['date'].unique():
        date = dateutil.parser.parse(date_str)
        if date < start_date or date > end_date:
            continue
        ticks = (date - start_date).days * 24
        beta_mults = {}
        beta_home_mult = 0
        for i in range(len(site_dict)-1):    # exclude home gatherings
            site_type = site_dict[i]
            row = df.loc[(df['date']==date_str) & (df['model_category']==site_type)]
            mult = row['multiplier'].iloc[0]
            beta_mults[site_type] = mult
            beta_home_mult += mult
        beta_mults['home'] = beta_home_mult / float(len(site_dict)-1)
        print(beta_mults)
        measure = BetaMultiplierMeasureByType(t_window=Interval(ticks,ticks+(24*7)), beta_multiplier=beta_mults)
        measures.append(measure)
    
    return measures

    
def pstay_home_measures_from_csv(filename, start_date_str, sim_days, site_dict):
    measures = []
    start_date = dateutil.parser.parse(start_date_str)
    end_date = start_date + datetime.timedelta(days=sim_days)
    ticks_dur = sim_days *24
    df = pd.read_csv(filename)
    
    for index, row in df.iterrows():
        date_str = row['date']
        date = dateutil.parser.parse(date_str)
        start_ticks = (date - start_date).days * 24
        end_ticks = start_ticks + (24*7)
        
        if start_ticks < 0: start_ticks = 0  # trim overlaps with beginning of range
        if end_ticks <= 0: continue    # wholly before desired range
        if start_ticks >= ticks_dur: continue    # wholly after desired range
        
        p_stay_home = (1.0 - row['multiplier'])
        print(f'Initing p_stay_home={p_stay_home} for interval {date} - {date + datetime.timedelta(days=7)}')
        measure = SocialDistancingForAllMeasure(t_window=Interval(start_ticks,end_ticks), p_stay_home=p_stay_home)
        measures.append(measure)
    
    return measures    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    