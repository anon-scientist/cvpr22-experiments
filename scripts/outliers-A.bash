# DCGMM-A outliers
# +++
python3 -m DCGMM.utils.executeExp bash/DCGMM-A.bash --T1 1 2 3 4 5 6 7 8 9  --exp_id 12429 --epochs 80 \
--model_type Stacked_GMM_Outliers \
--dataset_name mnist --batch_size 100 \
--L2_K 49  \
--sampling_layer 2 \
--outlier_detection_layer 2 \
--no_avg_metrics scores blah 
# ---
chmod 777 ./tmp.bash
#./tmp.bash

# +++
python3 -m DCGMM.utils.executeExp bash/DCGMM-A.bash --T1 1 2 3 4 5 6 7 8 9  --T2 0 --exp_id 12429 --epochs 0.5 \
--load_task 1 \
--dataset_name mnist --batch_size 100 \
--measuring_points 1 \
--model_type Stacked_GMM_Outliers \
--perform_variant_generation False \
--perform_sampling False \
--perform_inpainting True \
--cond_sampling_classes 0 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 2 \
--no_avg_metrics scores blah \
--L2_sampling_S 2 \
--L1_target_layer 2 --L1_reconstruction_weight 0.1 --L1_sharpening_rate 0.01 --L1_sharpening_iterations 0 \
--L2_sampling_divisor 10 \
--L2_K 49  \
--L2_wait 1  \
--sampling_layer 2 --nr_sampling_batches 1 --variant_gmm_root 2
# ---
chmod 777 ./tmp.bash
./tmp.bash

