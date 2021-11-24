# DCGMM-E outliers: 3 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-E.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id outliers_dcgmm_e --epochs 80 \
--perform_variant_generation False \
--perform_unconditional_sampling False \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--dataset_name mnist --batch_size 100 \
--sampling_layer 6 --L4_wait 0.1 --L6_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 6 \
--no_avg_metrics scores blah \
--L2_K 25 --L4_K 25 --L6_K 49 --L2_convMode False --L4_convMode False \
--L7_input 6
# ---
chmod 777 ./tmp.bash
./tmp.bash

# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-E.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id outliers_dcgmm_e --epochs 0.05 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_inpainting False \
--perform_sampling True \
--cond_sampling_classes 0 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 6 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 1 --L6_sampling_S 1 \
--L2_sampling_divisor 5 --L4_sampling_divisor 5 --L6_sampling_divisor 5 \
--L1_target_layer 4 --L1_reconstruction_weight 0. --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--L3_target_layer 6 --L1_reconstruction_weight 0. --L3_sharpening_rate 0.01 --L3_sharpening_iterations 0 \
--dataset_name mnist --batch_size 100 \
--L2_K 25 --L4_K 25 --L6_K 49 --L2_convMode False --L4_convMode False \
--sampling_layer 7 --L2_wait 1 --L4_wait 1 --L6_wait 1 \
--L7_input 6  --nr_sampling_batches 10
#--cond_sampling_classes 2 4 \
# ---
chmod 777 ./tmp.bash
./tmp.bash

