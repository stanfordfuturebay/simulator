import pickle
import os
import numpy as np
from lib.town_data import generate_population

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
    
    
    