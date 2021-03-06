# DCGMM-B outliers 2 GMMs no pooling
# +++
python3  -m DCGMM.utils.executeExp bash/DCGMM-B.bash --T1 0 1 2 3 4 5 6 7 8 9  --exp_id 92426 --epochs 10 \
--model_type Stacked_GMM_Outliers \
--dataset_name fashion_mnist --batch_size 100 \
--L2_K 49 --L4_K 49 \
--L2_convMode True --L4_convMode True \
--sampling_layer 4 --L4_wait 0.1 \
--outlier_c 1 --outlier_eps 0.01 --outlier_detection_layer 4 \
--no_avg_metrics scores blah 
# ---
chmod 777 ./tmp.bash
./tmp.bash

python3 -m DCGMM.utils.vis --what mus ---prefix results/L2_ --out ./results/alphabet.png
