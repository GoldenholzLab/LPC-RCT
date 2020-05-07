from seizure_diary_generation import recalculate_seizure_diary_time_scales
import scipy.stats as stats
import numpy as np


def select_time_scale(seizure_diary_time_scale):

    num_weeks_within_a_month = 4
    num_days_within_a_month  = 28

    if(seizure_diary_time_scale == 'daily'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_days_within_a_month
    
    elif(seizure_diary_time_scale == 'weekly'):

        num_finer_resolution_counts_within_one_coarser_resolution_count = num_weeks_within_a_month
    
    else:

        raise ValueError('The \'seizure_diary_time_scale\' parameter ' + seizure_diary_time_scale + ' is not included within the code.')

    [time_scaling_product, _] = \
         recalculate_seizure_diary_time_scales('zoom out',
                                               num_finer_resolution_counts_within_one_coarser_resolution_count)
    
    return time_scaling_product
    

def calculate_percent_changes(baseline_time_scale,
                              baseline_seizure_diaries,
                              maintenance_time_scale,
                              maintenance_seizure_diaries):

    baseline_seizure_frequencies    = np.mean(baseline_seizure_diaries,    1)
    maintenance_seizure_frequencies = np.mean(maintenance_seizure_diaries, 1)

    percent_changes = 1 - (maintenance_seizure_frequencies/baseline_seizure_frequencies)

    if(baseline_time_scale != maintenance_time_scale):

        baseline_time_scaling_product    = select_time_scale(   baseline_time_scale)
        maintenance_time_scaling_product = select_time_scale(maintenance_time_scale)

        # only makes sense so long as 'zoom out' is true
        percent_changes = 1 - (baseline_time_scaling_product/maintenance_time_scaling_product)*(1 - percent_changes)

    return percent_changes


def calculate_MPC_p_value(baseline_time_scale,
                          maintenance_time_scale,
                          placebo_arm_baseline_seizure_diaries,
                          placebo_arm_maintenance_seizure_diaries,
                          treatment_arm_baseline_seizure_diaries,
                          treatment_arm_maintenance_seizure_diaries):

    placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  placebo_arm_maintenance_seizure_diaries)

    treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  treatment_arm_maintenance_seizure_diaries)

    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, treatment_arm_percent_changes)

    return MPC_p_value

