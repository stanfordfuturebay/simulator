import numpy as np

'''
Settings for town generation
'''

'''
TO DO:
Daily testing capacity  vs daily number of tests?
'''

town_name = 'San_Francisco' 

# Make sure to download country-specific population density data
# Source: Facebook's Data for Good program
# https://data.humdata.org/dataset/united-states-high-resolution-population-density-maps-demographic-estimates
# Number of people living within 30-meter grid tiles
population_path='lib/data/population/population_density_sf.csv' # Population density of SF extracted from the data (original data has 6 large files)

sites_path='lib/data/queries_sf/' # Directory containing OSM site query details
bbox = (37.7115, 37.8127, -122.5232, -122.3539) # Coordinate bounding box

# Population per age group in the region (matching the RKI age groups)
# Source: safegraph open census data
population_per_age_group = np.array([
    38715,  # 0-4
    59181,  # 5-14
    30824,  # 15-19
	52567,  # 20-24
	329257, # 25-44
    167051,  # 45-59
    136499,  # 60-79
    36188]) # 80+

town_population = 850282 
region_population = population_per_age_group.sum()

# !!!TODO!!!: Daily testing capacity vs daily number of tests?
# Roughly 100k tests per day in Germany: https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/15_20.pdf?__blob=publicationFile
# daily_tests_unscaled = int(100000 * town_population / 83000000)
# SF: rough estimate based on the daily number of tests in the past 5 weekts: https://data.sfgov.org/stories/s/d96w-cdge
daily_tests_unscaled = 1200

# Information about household structure (set to None if not available)
# Source for US: https://www.census.gov/data/tables/2019/demo/families/cps-2019.html
household_info = {
    'size_dist' : [28.37, 34.51, 15.07, 12.76, 5.78, 2.26, 1.25], # distribution of household sizes (1-7 people) from Table H1
    'soc_role' : { # Each element is a probability. Each column should add up to 1. Simplification based on the bureau data
        'children' : [1, 1, 1, 0, 0, 0, 0, 0], # age groups 0,1,2 (0-19) can be children (must be in a household with "parents")
        'parents' : [0, 0, 0, 1, 1, 1, 0, 0], # age groups 3,4,5 (20-59) can be parents (They do not necessarily have kids)
        'elderly' : [0, 0, 0, 0, 0, 0, 1, 1] # age groups 6,7 (60+) are elderly (live in a household of size 1 or 2 without children living with them)
    }
}

def foo():
    return 3

# proportion of all essential workers within each age group
prop_essential_per_age_group = np.array([
    0,   # 0-4
    0,   # 5-14
    .01,  # 15-19
    .08,  # 20-24
    .45,  # 25-44
    .25,  # 45-59
    .20, # 60-79
    0])  # 

prop_population_per_age_group = (np.array(population_per_age_group) / float(sum(population_per_age_group)))

def _essential_prop_per_age_group(prop_essential_total):
    return (prop_essential_per_age_group*prop_essential_total) / prop_population_per_age_group


def _essential_mobility(wtype):
    # site types: ['education', 'office', 'social', 'supermarket']
    if wtype in ['education', 0]:
        essential_mob_rate_per_type = [5.0, 0.0, 3.6, 0.27]
        essential_dur_mean_per_type = [5.0, 0.1, 0.64, 0.4]
        essential_variety_per_type = [1, 1, 10, 2]
    elif wtype in ['office',1]:
        essential_mob_rate_per_type = [0.0, 5.0, 3.6, 0.27]
        essential_dur_mean_per_type = [0.1, 5.0, 0.64, 0.4]
        essential_variety_per_type = [1, 1, 10, 2]
    elif wtype in ['social',2]:
        essential_mob_rate_per_type = [0.0, 0.0, 5.0, 0.27]
        essential_dur_mean_per_type = [0.1, 0.1, 5.0, 0.4]
        essential_variety_per_type = [1, 1, 10, 2]
    elif wtype in ['supermarket',3]:
        essential_mob_rate_per_type = [0.0, 0.0, 3.6,  5.0]
        essential_dur_mean_per_type = [0.1, 0.1, 0.64, 5.0]
        essential_variety_per_type = [1, 1, 10, 2]
    else:
        return None, None, None
    return essential_mob_rate_per_type, essential_dur_mean_per_type, essential_variety_per_type

def _worker_mobility():
    worker_mob_rate_per_types = [
        [5.0, 0.0, 3.6, 0.27],
        [0.0, 5.0, 3.6, 0.27],
        [0.0, 0.0, 5.0, 0.27],
        [0.0, 0.0, 3.6,  5.0]
    ]
    worker_dur_mean_per_types = [
        [5.0, 0.1, 0.64, 0.4],
        [0.1, 5.0, 0.64, 0.4],
        [0.1, 0.1, 5.0, 0.4],
        [0.1, 0.1, 0.64, 5.0]
    ]
    worker_variety_per_types = [
        [1, 1, 10, 2],
        [1, 1, 10, 2],
        [1, 1, 10, 2],
        [1, 1, 10, 2]
    ]
    return worker_mob_rate_per_types, worker_dur_mean_per_types, worker_variety_per_types
    
    
    
    
    
    
    
    
    
    
    
    