#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem-per-cpu=10G
#SBATCH -t 0-00:25
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

training_samples_file_name=$1
validation_data_file_name=$2
testing_data_file_name=$3
model_storage_folder_name=$4
num_training_samples_per_classification=$5
dropout_rate=$6
CNNet_1D_num_filters=$7
CNNet_1D_kernel_size=$8
Leaky_ReLU_neg_slope=$9

model_type='intermediate'

training_inputs[0]=$training_samples_file_name
training_inputs[1]=$model_storage_folder_name
training_inputs[2]=$num_training_samples_per_classification
training_inputs[3]=$model_type
training_inputs[4]=$dropout_rate
training_inputs[5]=$CNNet_1D_num_filters
training_inputs[6]=$CNNet_1D_kernel_size
training_inputs[7]=$Leaky_ReLU_neg_slope

validation_inputs[0]=$model_storage_folder_name
validation_inputs[1]=$validation_data_file_name
validation_inputs[2]=$model_type
validation_inputs[3]=$dropout_rate
validation_inputs[4]=$CNNet_1D_num_filters
validation_inputs[5]=$CNNet_1D_kernel_size
validation_inputs[6]=$Leaky_ReLU_neg_slope

testing_inputs[0]=$model_storage_folder_name
testing_inputs[1]=$testing_data_file_name
testing_inputs[2]=$model_type
testing_inputs[3]=$dropout_rate
testing_inputs[4]=$CNNet_1D_num_filters
testing_inputs[5]=$CNNet_1D_kernel_size
testing_inputs[6]=$Leaky_ReLU_neg_slope

plotting_inputs[0]=$model_storage_folder_name
plotting_inputs[1]=$model_type
plotting_inputs[2]=$dropout_rate
plotting_inputs[3]=$CNNet_1D_num_filters
plotting_inputs[4]=$CNNet_1D_kernel_size
plotting_inputs[5]=$Leaky_ReLU_neg_slope

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate working_env

python python_scripts/train_keras_model.py ${training_inputs[@]}
python python_scripts/optimize_keras_model_thresholds.py ${validation_inputs[@]}
python python_scripts/test_keras_model.py ${testing_inputs[@]}
python python_scripts/plot_testing_set_results.py ${plotting_inputs[@]}