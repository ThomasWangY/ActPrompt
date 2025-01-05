# tacos
dset_name=TaCoS
root=/root
data_root=${root}/data
video_path=${data_root}/videos/TaCoS
train_path=${data_root}/tacos/metadata/train.jsonl
eval_path=${data_root}/tacos/metadata/test.jsonl
test_path=${data_root}/tacos/metadata/test.jsonl
cache_path=${data_root}/tacos/cache
dset_path=${data_root}/tacos
clip_len=2
max_len=20
coef_1=5
coef_2=100

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