import multiprocessing

'''
Default settings for model calibration
'''
TO_HOURS = 24.0

settings_data = {
    'verbose' : True,
    'use_households' : True,
    'data_start_date': '2020-03-10',
}

settings_simulation = {
    'n_init_samples': 20,  # initial random evaluations
    'n_iterations': 500,  # iterations of BO
    'simulation_roll_outs': 40, # roll-outs done in parallel per parameter setting
    'cpu_count':  multiprocessing.cpu_count(), # cpus used for parallel computation
    'dynamic_tracing' : True,
}

# parameter bounds
settings_model_param_bounds = {
    'betas': {
        'education': [0.0, 1.5],
        'social': [0.0, 1.5],
        'bus_stop': [0.0, 1.5],
        'office': [0.0, 1.5],
        'supermarket': [0.0, 1.5],
    },
    'beta_household': [0.0, 1.5],
}

settings_measures_param_bounds = {
    'p_stay_home': [0.0, 1.0],
}

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
    'GER' : {
        'TU': 'LK Tübingen',
        'KL': 'SK Kaiserslautern',
        'RH': 'LK Rheingau-Taunus-Kreis',
        'HB': 'LK Heinsberg',
        'TR': 'LK Tirschenreuth'
    },
    'CH' : {
        'SZ': 'SZ',     # Canton Schwyz
        'TI': 'TI',     # Canton Ticino
        'LU': 'LU',     # Canton Lucerne
        'VD': 'VD',     # Canton Vaud
        'JU': 'JU',     # Canton Jura
    }
}				

# optimized model parameters
beta_dummy = 0.5

settings_optimized_town_params = {
    'GER': {
        'TU': { # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'KL': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'RH': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'TR': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
    },
    'CH': {
        'JU': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'TI': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'LU': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
        'VD': {  # dummy settings
            'betas': {
                'education': beta_dummy,
                'social': beta_dummy,
                'bus_stop': beta_dummy,
                'office': beta_dummy,
                'supermarket': beta_dummy,
            },
            'beta_household': beta_dummy,
        },
    }
}

# lockdown dates
settings_lockdown_dates = {
    'GER': {
        'start' : '03-23-2020',
        'end': '05-03-2020',
    },
    'CH': {
        'start': '03-16-2020',
        'end': '05-10-2020',
    },
}

