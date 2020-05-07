#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem-per-cpu=10G
#SBATCH -t 0-08:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

one_or_two='one'

num_patients_per_trial=306

placebo_percent_effect_mean_upper_bound=0.4
placebo_percent_effect_mean_lower_bound=0

drug_percent_effect_mean_upper_bound=0.25
drug_percent_effect_mean_lower_bound=0.2

placebo_percent_effect_std_dev=0.05
drug_percent_effect_std_dev=0.05

num_training_samples_per_classification=100000

training_data_file_name="200000_NV_model_one_weekly_regular_level_16_training_samples"

inputs[0]=$one_or_two
inputs[1]=$num_patients_per_trial
inputs[2]=$placebo_percent_effect_mean_upper_bound
inputs[3]=$placebo_percent_effect_mean_lower_bound
inputs[4]=$drug_percent_effect_mean_upper_bound
inputs[5]=$drug_percent_effect_mean_lower_bound
inputs[6]=$placebo_percent_effect_std_dev
inputs[7]=$drug_percent_effect_std_dev
inputs[8]=$num_training_samples_per_classification
inputs[9]=$training_data_file_name

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate working_env

srun -c 1 python -u python_scripts/generate_keras_training_data.py ${inputs[@]}
