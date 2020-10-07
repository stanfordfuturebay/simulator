import numpy as np

'''
Settings for town generation
'''

town_name = 'Heinsberg'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path='lib/data/population/population_deu_2019-07-01.csv' # Population density file

sites_path='lib/data/queries/' # Directory containing OSM site query details
bbox = (51.0330, 51.0867, 6.0480, 6.1705) # Coordinate bounding box

# Population per age group in the region (matching the RKI age groups)
# Source for Germany: https://www.citypopulation.de/en/germany/
population_per_age_group = np.array([
    11508,  # 0-4
    24261,  # 5-14
    55436,  # 15-34
    91858,  # 35-59
    55703,  # 60-79
    15555]) # 80+

town_population = 41673 # Population of the central town of the region
region_population = population_per_age_group.sum()

# Daily testing capacity per 100k people
# Roughly 100k in total in Germany: https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/15_20.pdf?__blob=publicationFile
daily_tests_unscaled = int(100000 * town_population / 83000000)

# Information about household structure (set to None if not available)
# Source for Germany: https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/lrbev05.html 
household_info = {
    'size_dist' : [41.9, 33.8, 11.9, 9.1, 3.4], # distribution of household sizes (1-5 people)
    'soc_role' : {
    'children' : [1, 1, 3/20, 0, 0, 0], # age groups 0,1,2 can be children
    'parents' : [0, 0, 17/20, 1, 0, 0], # age groups 2,3 can be parents
    'elderly' : [0, 0, 0, 0, 1, 1] # age groups 4,5 are elderly
    }
}

