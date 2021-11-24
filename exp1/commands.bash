# +++
python3 spn28.py --exp_id outliers_dgcspn --epochs 15 --structure conv --train True --classes 1 2 3 4 5 6 7 8 9 \
  --outlier_classes 0  --base_K 16 --learning_rate 0.01 --batch_size 128 --fashion_mnist False
# ---
