# DCGMM-F outliers 4 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 0  --exp_id 62414 --epochs 80  \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--sampling_layer 8 --L4_wait 0.1 --L6_wait 0.2 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--no_avg_metrics scores blah \
--L2_K 25 --L4_K 25 --L6_K 25 --L8_K 49 \
--L2_convMode False --L4_convMode False --L6_convMode False --L8_convMode False \
--dataset_name mnist --batch_size 100 \
# ---
chmod 777 ./tmp.bash
./tmp.bash

# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-F.bash --T1 0  --T2 0 --exp_id 62414 --epochs 0.5 \
--load_task 1 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_sampling False \
--perform_inpainting True \
--cond_sampling_classes 0 1 2 3 4 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 8 \
--dataset_name mnist --batch_size 100 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 4 --L1_reconstruction_weight 0. --L1_sharpening_rate 0.02 --L1_sharpening_iterations 00 \
--L3_target_layer 6 --L3_reconstruction_weight 0. --L3_sharpening_rate 0.02 --L3_sharpening_iterations 00 \
--L2_K 25 --L4_K 25 --L6_K 25 --L8_K 49 \
--L2_convMode False --L4_convMode False --L6_convMode False --L8_convMode False \
--sampling_layer 9 --L2_wait 1 --L4_wait 1 --L6_wait 1 --L8_wait 1 --nr_sampling_batches 1 \
--variant_gmm_root 8
# ---
chmod 777 ./tmp.bash
./tmp.bash



