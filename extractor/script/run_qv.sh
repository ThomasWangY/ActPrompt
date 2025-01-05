# QVHighlights
dset_name=QVHighlights
root=/root
data_root=${root}/data
video_path=${data_root}/videos/QVHighlights
train_path=${data_root}/qvhighlights/metadata/qvhighlights_train.jsonl
eval_path=${data_root}/qvhighlights/metadata/qvhighlights_val.jsonl
test_path=${data_root}/qvhighlights/metadata/qvhighlights_test.jsonl
cache_path=${data_root}/qvhighlights/cache
dset_path=${data_root}/qvhighlights
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