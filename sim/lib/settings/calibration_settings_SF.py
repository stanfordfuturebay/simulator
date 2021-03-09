import multiprocessing

'''
Default settings for model calibration
'''
TO_HOURS = 24.0

settings_data = {
    'verbose' : True,
    'use_households' : True,
    'data_start_date': '2020-03-04',
}

settings_simulation = {
    'n_init_samples': 20,  # initial random evaluations (use n_init_samples simulations results to initialize the model)
    'n_iterations': 100,  # iterations of BO (500 in original code)
    'simulation_roll_outs': 10, # roll-outs done in parallel per parameter setting (number of random repeats in each iteration)
    'cpu_count':  multiprocessing.cpu_count(), # cpus used for parallel computation
    'dynamic_tracing' : True,
}

# parameter bounds
settings_measures_params_bounds = {
    'p_stay_home':[0.0,1.0],
    'betas':[0.02,0.1],
}
settings_model_param_bounds = {
    'betas':[0.01,0.12],
}
settings_model_params_bounds = {
    'betas': {
        'education': [0.0, 0.2],
        'social': [0.0, 0.2],
        'office': [0.0, 0.2],
        'retail': [0.0, 0.2],
        'supermarket': [0.0, 0.2],
        'home': [0.0, 0.2],
    },
    'beta_household': [0.0, 0.2],
}

#settings_measures_param_bounds = {
#    'p_stay_home': [0.0, 1.0],
#    'betas': {
#        'education': [0.0, 0.2],
#        'social': [0.0, 0.2],
#        'office': [0.0, 0.2],
#        'supermarket': [0.0, 0.2],
#    },
#    'beta_household': [0.0, 0.2],
#}

# set testing parameters
settings_testing_params = {
    'testing_t_window': None,  # [set automatically in code]
    'testing_frequency': 1 * TO_HOURS,
    'test_reporting_lag': 2 * TO_HOURS,
    'tests_per_batch': None,  # [set automatically in code]
    'test_fpr': 0.0,
    'test_fnr': 0.0,
    'test_smart_delta': 3 * TO_HOURS,
    'test_smart_duration': 7 * TO_HOURS, 
    'test_smart_action': 'isolate',
    'test_smart_num_contacts': 10,
    'test_targets': 'isym',
    'test_queue_policy': 'fifo',
    'smart_tracing': None,
    'unlimited_tracing':False,
    'trace_friends_only':False,
    'trace_household_members':True
}

# BO acquisition function optimization (Knowledge gradient)
# default settings from botorch
settings_acqf = {
    'acqf_opt_num_fantasies': 64,
    'acqf_opt_num_restarts': 10,
    'acqf_opt_raw_samples': 256,
    'acqf_opt_batch_limit': 5,
    'acqf_opt_maxiter': 20,
}


# area codes
command_line_area_codes = {
    'US':{
        'SF':'San Francisco'
        }
}				

# optimized model parameters
beta_dummy = 0.5

beta_mult_csv = 'lib/data/beta_mult/beta_mult_sf.csv'

settings_optimized_town_params = {
    'US':{
        'SF':{
		    'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        }
    }
}

# lockdown dates
settings_lockdown_dates = {
    'US': {
        'start' : '03-17-2020',
        'end': '08-01-2020', # unknown
    }
}

