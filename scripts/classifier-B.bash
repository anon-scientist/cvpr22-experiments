# DCGMM-B outliers 2 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B-class.bash --T1 0 1 2 3 4 5 6 7 8 9  --exp_id 426 --epochs 15 \
--model_type Stacked_GMM_Outliers \
--dataset_name fashion_mnist --batch_size 100 \
--L2_K 121 --L4_K 16 \
--L2_convMode False --L4_convMode True --L5_input 2 --L5_epsC 0.1 \
--sampling_layer 4 --L4_wait 0.1 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah 
# ---
chmod 777 ./tmp.bash
./tmp.bash


