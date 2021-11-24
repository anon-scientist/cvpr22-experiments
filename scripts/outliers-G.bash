# DCGMM-F outliers 4 GMMs pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-G.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 62314 --epochs 25 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--dataset_name mnist --batch_size 100 \
--sampling_layer 11 \
--L5_wait 0.1 --L8_wait 0.15 --L11_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 11 \
--no_avg_metrics scores blah \
--L2_K 25 --L5_K 25 --L8_K 25 --L11_K 49 \
--L2_convMode False --L5_convMode False --L8_convMode False --L11_convMode False
# ---
chmod 777 ./tmp.bash
./tmp.bash

# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-G.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 62314 --epochs 0.05 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_sampling True \
--cond_sampling_classes 0 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 11 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 4 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--dataset_name mnist --batch_size 100 \
--L2_K 25 --L5_K 25 --L8_K 25 --L11_K 49 \
--L2_wait 1 --L5_wait 1 --L8_wait 1 --L11_wait 1 \
--sampling_layer 11 --nr_sampling_batches 10 \ 
--L2_convMode False --L5_convMode False --L8_convMode False --L11_convMode False
# ---
chmod 777 ./tmp.bash
./tmp.bash



