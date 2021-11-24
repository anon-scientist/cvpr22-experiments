# DCGMM-B outliers: max-pooling 2 GMM
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-C.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 9426 --epochs 15 \
--model_type Stacked_GMM_Outliers \
--dataset_name mnist --batch_size 100 \
--L2_K 49 --L5_K 49 \
--L2_convMode False --L5_convMode True \
--sampling_layer 5 --L5_wait 0.1 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 5 \
--no_avg_metrics scores blah 
# ---
chmod 777 ./tmp.bash
./tmp.bash

# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-C.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 9426 --epochs 0.1 \
--load_task 1 \
--dataset_name mnist --batch_size 100 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_sampling True \
--cond_sampling_classes 0 \
--perform_variant_generation False \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 5 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 2 --L1_sharpening_rate 0.2 --L1_sharpening_iterations 0 --L1_reconstruction_weight 0 \
--L3_target_layer 5  --L3_sharpening_rate 0.1 --L3_sharpening_iterations 00 \
--L2_K 49 --L5_K 49 \
--L2_convMode False --L5_convMode True \
--L3_sampling_mode dense \
--sampling_layer 5 --L2_sampling_divisor 5 --L5_sampling_divisor 5 \
--L2_wait 1 --L5_wait 1 --nr_sampling_batches 10
# ---
chmod 777 ./tmp.bash
./tmp.bash




