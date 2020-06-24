import pickle

#### Define standard testing parameters, same used for inference ####
def standard_testing(max_time):
    standard_testing_params = {
        'testing_t_window': [0.0, max_time],  # in hours
        'testing_frequency': 24.0,     # in hours
        'test_reporting_lag': 48.0,             # in hours (actual and self-report delay)
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

def save_summary(summary, filename):
    with open('summaries/' + filename, 'wb') as fp:
        pickle.dump(summary, fp)

def load_summary(filename):
    with open('summaries/' + filename, 'rb') as fp:
        summary = pickle.load(fp)
    return summary