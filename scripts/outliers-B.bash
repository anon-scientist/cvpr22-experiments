# DCGMM-B outliers 2 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 0  --exp_id 12426 --epochs 80 \
--model_type Stacked_GMM_Outliers \
--dataset_name mnist --batch_size 100 \
--L2_K 49 --L4_K 49 \
--L2_convMode False --L4_convMode True \
--sampling_layer 4 --L4_wait 0.1 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah 
# ---
chmod 777 ./tmp.bash
./tmp.bash
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 0  --T2 0 --exp_id 12426 --epochs 0.1 \
--load_task 1 \
--dataset_name mnist --batch_size 100 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_inpainting True \
--perform_sampling False --cond_sampling_classes 1 2 3 4 5 6 7 8 9 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 --L4_sampling_I -1 --L4_sampling_S 2 \
--L1_target_layer 2 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--L2_sampling_divisor 10 --L4_sampling_divisor 10 \
--L2_K 49 --L4_K 49 \
--L2_convMode False --L4_convMode True \
--sampling_layer 4 --L2_wait 1 --L4_wait 1 --nr_sampling_batches 1 --variant_gmm_root 4
# ---
chmod 777 ./tmp.bash
./tmp.bash


