# charades
dset_name=Charades
root=/root
data_root=${root}/data
video_path=${data_root}/videos/Charades
train_path=${data_root}/charades/metadata/charades_train.jsonl
eval_path=${data_root}/charades/metadata/charades_test.jsonl
test_path=${data_root}/charades/metadata/charades_test.jsonl
cache_path=${data_root}/charades/cache
dset_path=${data_root}/charades
clip_len=1
max_len=10
coef_1=10
coef_2=200

python3 train.py \
--dset_name ${dset_name}   \
--video_path ${video_path}   \
--train_path ${train_path}   \
--eval_path ${eval_path}   \
--test_path ${test_path}   \
--cache_path ${cache_path}   \
--dset_path ${dset_path}   \
--clip_len ${clip_len}   \
--max_len ${max_len}   \
--coef_1 ${coef_1} \
--coef_2 ${coef_2} \