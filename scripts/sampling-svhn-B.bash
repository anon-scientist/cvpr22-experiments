# SVHN sampling, one class

## training on class 0
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1  --exp_id svhn_b --epochs 120 \
--load_task 0  \
--save_All True \
--measuring_points 10 \
--model_type Stacked_GMM \
--dataset_name svhn_cropped --batch_size 100 \
--L1_patch_width 12 --L1_patch_height 12 --L1_stride_x 5 --L1_stride_y 5 \
--L2_K 64 --L4_K 100 --L2_convMode False \
--L3_patch_width 5 --L3_patch_height 5 --L3_stride_x 5 --L3_stride_y 5 \
--sampling_layer 4 --L4_wait 0.1 \
--no_avg_metrics scores blah \
--sampling_layer 4 --outlier_detection_layer 4
# ---
chmod 777 tmp.bash
#./tmp.bash

## sampling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 1  --T2 0 --exp_id svhn_b --epochs 0.1 \
--load_task 1 \
--measuring_points 1 \
--save_All False \
--model_type Stacked_GMM \
--perform_sampling True \
--perform_inpainting True \
--dataset_name svhn_cropped --batch_size 100 \
--L1_patch_width 12 --L1_patch_height 12 --L1_stride_x 5 --L1_stride_y 5 --L1_sharpening_rate 2 --L1_sharpening_iterations 300 --L1_target_layer 2 \
--L2_K 64 --L4_K 100 --L2_convMode False --L4_S 2 --L2_S 2 \
--L3_patch_width 5 --L3_patch_height 5 --L3_stride_x 5 --L3_stride_y 5 \
--sampling_layer 4 --L4_wait 0.1 \
--no_avg_metrics scores blah \
--sampling_layer 4 --outlier_detection_layer 4
# ---
chmod 777 tmp.bash
./tmp.bash

# visualize
python3 -m DCGMM.utils.vis --what mus --prefix results/svhn_b_sampling_ --channels 3  --out results/svhn_b_samples.png


