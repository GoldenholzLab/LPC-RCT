from seizure_diary_generation import generate_NV_model_patient_pop_params
from seizure_diary_generation import generate_baseline_seizure_diaries
from seizure_diary_generation import generate_maintenance_seizure_diaries
from seizure_diary_generation import apply_percent_effects_to_seizure_diaries
from endpoint_functions import calculate_MPC_p_value
from endpoint_functions import calculate_percent_changes
import numpy as np


def generate_one_trial_seizure_diaries(one_or_two,
                                       num_placebo_arm_patients,
                                       num_drug_arm_patients,
                                       num_baseline_months,
                                       num_maintenance_months,
                                       baseline_time_scale,
                                       maintenance_time_scale,
                                       minimum_cumulative_baseline_seizure_count,
                                       placebo_percent_effect_mean_upper_bound,
                                       placebo_percent_effect_mean_lower_bound,
                                       placebo_percent_effect_std_dev,
                                       drug_efficacy_presence,
                                       drug_percent_effect_mean_upper_bound=None,
                                       drug_percent_effect_mean_lower_bound=None,
                                       drug_percent_effect_std_dev=None):

    [placebo_arm_NV_model_monthly_means,
     placebo_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_placebo_arm_patients,
                                              one_or_two)
    
    [treatment_arm_NV_model_monthly_means,
     treatment_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_drug_arm_patients,
                                              one_or_two)

    [placebo_arm_NV_model_monthly_means,
     placebo_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_placebo_arm_patients,
                                              one_or_two)
    
    [treatment_arm_NV_model_monthly_means,
     treatment_arm_NV_model_monthly_std_devs] = \
         generate_NV_model_patient_pop_params(num_drug_arm_patients,
                                              one_or_two)
    
    placebo_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                          placebo_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)
    
    treatment_arm_baseline_seizure_diaries = \
        generate_baseline_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                          treatment_arm_NV_model_monthly_std_devs,
                                          baseline_time_scale,
                                          num_baseline_months,
                                          minimum_cumulative_baseline_seizure_count)

    placebo_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(placebo_arm_NV_model_monthly_means,
                                             placebo_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)
    
    treatment_arm_maintenance_seizure_diaries = \
        generate_maintenance_seizure_diaries(treatment_arm_NV_model_monthly_means,
                                             treatment_arm_NV_model_monthly_std_devs,
                                             maintenance_time_scale,
                                             num_maintenance_months)

    placebo_percent_effect_mean = \
        np.random.uniform(placebo_percent_effect_mean_lower_bound,
                          placebo_percent_effect_mean_upper_bound)

    placebo_percent_effects = \
        np.random.normal(placebo_percent_effect_mean,
                         placebo_percent_effect_std_dev,
                         num_placebo_arm_patients)

    placebo_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(placebo_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    treatment_arm_maintenance_seizure_diaries = \
        apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                 placebo_percent_effects)
    
    if(drug_efficacy_presence == True):

        drug_percent_effect_mean = \
            np.random.uniform(drug_percent_effect_mean_lower_bound,
                              drug_percent_effect_mean_upper_bound)

        drug_percent_effects = \
            np.random.normal(drug_percent_effect_mean,
                             drug_percent_effect_std_dev,
                             num_drug_arm_patients)
        
        treatment_arm_maintenance_seizure_diaries = \
            apply_percent_effects_to_seizure_diaries(treatment_arm_maintenance_seizure_diaries,
                                                     drug_percent_effects)
    
    return [placebo_arm_baseline_seizure_diaries,
            placebo_arm_maintenance_seizure_diaries,
            treatment_arm_baseline_seizure_diaries,
            treatment_arm_maintenance_seizure_diaries]


def generate_true_drug_and_false_drug_trial_seizure_diaries(one_or_two,
                                                            num_placebo_arm_patients,
                                                            num_drug_arm_patients,
                                                            num_baseline_months,
                                                            num_maintenance_months,
                                                            baseline_time_scale,
                                                            maintenance_time_scale,
                                                            minimum_cumulative_baseline_seizure_count,
                                                            placebo_percent_effect_mean_upper_bound,
                                                            placebo_percent_effect_mean_lower_bound,
                                                            placebo_percent_effect_std_dev,
                                                            drug_percent_effect_mean_upper_bound,
                                                            drug_percent_effect_mean_lower_bound,
                                                            drug_percent_effect_std_dev):

    [true_drug_placebo_arm_baseline_seizure_diaries,
     true_drug_placebo_arm_maintenance_seizure_diaries,
     true_drug_treatment_arm_baseline_seizure_diaries,
     true_drug_treatment_arm_maintenance_seizure_diaries] = \
         generate_one_trial_seizure_diaries(one_or_two,
                                            num_placebo_arm_patients,
                                            num_drug_arm_patients,
                                            num_baseline_months,
                                            num_maintenance_months,
                                            baseline_time_scale,
                                            maintenance_time_scale,
                                            minimum_cumulative_baseline_seizure_count,
                                            placebo_percent_effect_mean_upper_bound,
                                            placebo_percent_effect_mean_lower_bound,
                                            placebo_percent_effect_std_dev,
                                            True,
                                            drug_percent_effect_mean_upper_bound,
                                            drug_percent_effect_mean_lower_bound,
                                            drug_percent_effect_std_dev)
        
    [false_drug_placebo_arm_baseline_seizure_diaries,
     false_drug_placebo_arm_maintenance_seizure_diaries,
     false_drug_treatment_arm_baseline_seizure_diaries,
     false_drug_treatment_arm_maintenance_seizure_diaries] = \
         generate_one_trial_seizure_diaries(one_or_two,
                                            num_placebo_arm_patients,
                                            num_drug_arm_patients,
                                            num_baseline_months,
                                            num_maintenance_months,
                                            baseline_time_scale,
                                            maintenance_time_scale,
                                            minimum_cumulative_baseline_seizure_count,
                                            placebo_percent_effect_mean_upper_bound,
                                            placebo_percent_effect_mean_lower_bound,
                                            placebo_percent_effect_std_dev,
                                            False)
    
    return [true_drug_placebo_arm_baseline_seizure_diaries,
            true_drug_placebo_arm_maintenance_seizure_diaries,
            true_drug_treatment_arm_baseline_seizure_diaries,
            true_drug_treatment_arm_maintenance_seizure_diaries,
            false_drug_placebo_arm_baseline_seizure_diaries,
            false_drug_placebo_arm_maintenance_seizure_diaries,
            false_drug_treatment_arm_baseline_seizure_diaries,
            false_drug_treatment_arm_maintenance_seizure_diaries]


def generate_true_and_false_drug_histograms(one_or_two,
                                            num_placebo_arm_patients,
                                            num_drug_arm_patients,
                                            num_baseline_months,
                                            num_maintenance_months,
                                            baseline_time_scale,
                                            maintenance_time_scale,
                                            minimum_cumulative_baseline_seizure_count,
                                            placebo_percent_effect_mean_upper_bound,
                                            placebo_percent_effect_mean_lower_bound,
                                            placebo_percent_effect_std_dev,
                                            drug_percent_effect_mean_upper_bound,
                                            drug_percent_effect_mean_lower_bound,
                                            drug_percent_effect_std_dev,
                                            num_bins,
                                            hist_range):

    [true_drug_placebo_arm_baseline_seizure_diaries,
     true_drug_placebo_arm_maintenance_seizure_diaries,
     true_drug_treatment_arm_baseline_seizure_diaries,
     true_drug_treatment_arm_maintenance_seizure_diaries,
     false_drug_placebo_arm_baseline_seizure_diaries,
     false_drug_placebo_arm_maintenance_seizure_diaries,
     false_drug_treatment_arm_baseline_seizure_diaries,
     false_drug_treatment_arm_maintenance_seizure_diaries] = \
         generate_true_drug_and_false_drug_trial_seizure_diaries(one_or_two,
                                                                 num_placebo_arm_patients,
                                                                 num_drug_arm_patients,
                                                                 num_baseline_months,
                                                                 num_maintenance_months,
                                                                 baseline_time_scale,
                                                                 maintenance_time_scale,
                                                                 minimum_cumulative_baseline_seizure_count,
                                                                 placebo_percent_effect_mean_upper_bound,
                                                                 placebo_percent_effect_mean_lower_bound,
                                                                 placebo_percent_effect_std_dev,
                                                                 drug_percent_effect_mean_upper_bound,
                                                                 drug_percent_effect_mean_lower_bound,
                                                                 drug_percent_effect_std_dev)
    
    true_drug_placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  true_drug_placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  true_drug_placebo_arm_maintenance_seizure_diaries)
    
    true_drug_treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  true_drug_treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  true_drug_treatment_arm_maintenance_seizure_diaries)
    
    false_drug_placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  false_drug_placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  false_drug_placebo_arm_maintenance_seizure_diaries)
    
    false_drug_treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  false_drug_treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  false_drug_treatment_arm_maintenance_seizure_diaries)

    [true_drug_placebo_arm_percent_change_hist, _] = \
        np.histogram(true_drug_placebo_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    [true_drug_treatment_arm_percent_change_hist, _] = \
        np.histogram(true_drug_treatment_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)
    
    [false_drug_placebo_arm_percent_change_hist, _] = \
        np.histogram(false_drug_placebo_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    [false_drug_treatment_arm_percent_change_hist, _] = \
        np.histogram(false_drug_treatment_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    return [true_drug_placebo_arm_percent_change_hist, 
            true_drug_treatment_arm_percent_change_hist,
            false_drug_placebo_arm_percent_change_hist,
            false_drug_treatment_arm_percent_change_hist]


def generate_true_and_false_drug_histograms_and_MPC_stat_significances(one_or_two,
                                                                       num_placebo_arm_patients,
                                                                       num_drug_arm_patients,
                                                                       num_baseline_months,
                                                                       num_maintenance_months,
                                                                       baseline_time_scale,
                                                                       maintenance_time_scale,
                                                                       minimum_cumulative_baseline_seizure_count,
                                                                       placebo_percent_effect_mean_upper_bound,
                                                                       placebo_percent_effect_mean_lower_bound,
                                                                       placebo_percent_effect_std_dev,
                                                                       drug_percent_effect_mean_upper_bound,
                                                                       drug_percent_effect_mean_lower_bound,
                                                                       drug_percent_effect_std_dev,
                                                                       num_bins,
                                                                       hist_range):

    [true_drug_placebo_arm_baseline_seizure_diaries,
     true_drug_placebo_arm_maintenance_seizure_diaries,
     true_drug_treatment_arm_baseline_seizure_diaries,
     true_drug_treatment_arm_maintenance_seizure_diaries,
     false_drug_placebo_arm_baseline_seizure_diaries,
     false_drug_placebo_arm_maintenance_seizure_diaries,
     false_drug_treatment_arm_baseline_seizure_diaries,
     false_drug_treatment_arm_maintenance_seizure_diaries] = \
         generate_true_drug_and_false_drug_trial_seizure_diaries(one_or_two,
                                                                 num_placebo_arm_patients,
                                                                 num_drug_arm_patients,
                                                                 num_baseline_months,
                                                                 num_maintenance_months,
                                                                 baseline_time_scale,
                                                                 maintenance_time_scale,
                                                                 minimum_cumulative_baseline_seizure_count,
                                                                 placebo_percent_effect_mean_upper_bound,
                                                                 placebo_percent_effect_mean_lower_bound,
                                                                 placebo_percent_effect_std_dev,
                                                                 drug_percent_effect_mean_upper_bound,
                                                                 drug_percent_effect_mean_lower_bound,
                                                                 drug_percent_effect_std_dev)
    
    MPC_true_drug_p_value = \
        calculate_MPC_p_value(baseline_time_scale,
                              maintenance_time_scale,
                              true_drug_placebo_arm_baseline_seizure_diaries,
                              true_drug_placebo_arm_maintenance_seizure_diaries,
                              true_drug_treatment_arm_baseline_seizure_diaries,
                              true_drug_treatment_arm_maintenance_seizure_diaries)

    MPC_false_drug_p_value = \
        calculate_MPC_p_value(baseline_time_scale,
                              maintenance_time_scale,
                              false_drug_placebo_arm_baseline_seizure_diaries,
                              false_drug_placebo_arm_maintenance_seizure_diaries,
                              false_drug_treatment_arm_baseline_seizure_diaries,
                              false_drug_treatment_arm_maintenance_seizure_diaries)

    MPC_true_drug_stat_significance  = MPC_true_drug_p_value  < 0.05
    MPC_false_drug_stat_significance = MPC_false_drug_p_value < 0.05

    true_drug_placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  true_drug_placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  true_drug_placebo_arm_maintenance_seizure_diaries)
    
    true_drug_treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  true_drug_treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  true_drug_treatment_arm_maintenance_seizure_diaries)
    
    false_drug_placebo_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  false_drug_placebo_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  false_drug_placebo_arm_maintenance_seizure_diaries)
    
    false_drug_treatment_arm_percent_changes = \
        calculate_percent_changes(baseline_time_scale,
                                  false_drug_treatment_arm_baseline_seizure_diaries,
                                  maintenance_time_scale,
                                  false_drug_treatment_arm_maintenance_seizure_diaries)

    [true_drug_placebo_arm_percent_change_hist, _] = \
        np.histogram(true_drug_placebo_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    [true_drug_treatment_arm_percent_change_hist, _] = \
        np.histogram(true_drug_treatment_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)
    
    [false_drug_placebo_arm_percent_change_hist, _] = \
        np.histogram(false_drug_placebo_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    [false_drug_treatment_arm_percent_change_hist, _] = \
        np.histogram(false_drug_treatment_arm_percent_changes, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)

    return [MPC_true_drug_stat_significance, 
            MPC_false_drug_stat_significance,
            true_drug_placebo_arm_percent_change_hist, 
            true_drug_treatment_arm_percent_change_hist,
            false_drug_placebo_arm_percent_change_hist,
            false_drug_treatment_arm_percent_change_hist]

