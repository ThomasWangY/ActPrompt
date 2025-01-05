
export NCCL_SOCKET_IFNAME=ens32
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2


dset_type=mr
dset_name=charades
domain_name=null
model_id=univtg
exp_id=0
gpu_id=0
device=0
debug=false
seed=2018
local_rank=-1
eval_split_name=val
data_ratio=1.0
num_workers=16
no_pin_memory=false
bsz=32
n_epoch=50
max_es_cnt=200
lr=2e-5 # 
lr_drop=50
lr_gamma=0.1
lr_warmup=10.0
wd=1e-5 # 
grad_clip=0.1
span_loss_type=l1
b_loss_coef=10.0
g_loss_coef=1.0
eos_coef=0.1
f_loss_coef=10.0
s_loss_intra_coef=1.0
s_loss_inter_coef=0.2
main_metric=MR-full-R1@0.3-key
eval_mode=add
eval_bsz=4
eval_epoch=1
eval_init=true
save_interval=50
resume={}
resume_dir=null
resume_all=false
start_epoch=0
no_sort_results=false
max_before_nms=1000
max_after_nms=10
conf_thd=0.0
nms_thd=0.7
use_cache=1
max_q_l=75
max_v_l=75
clip_length=1.0
clip_len_list=null
max_windows=5
add_easy_negative=1
easy_negative_only=1
round_multiple=-1

train_path_list=null
eval_path_list=null
feat_root_list=null
no_norm_vfeat=false
no_norm_tfeat=false

v_feat_dim=2816
t_feat_dim=512
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
position_embedding=sine
n_input_proj=2
temperature=0.07
enc_layers=4
sub_enc_layers=2
dec_layers=2
dim_feedforward=1024
hidden_dim=1024
input_dropout=0.5
dropout=0.0
droppath=0.1
txt_drop_ratio=0
use_txt_pos=false
nheads=8
num_queries=10
pre_norm=false
set_cost_span=10
set_cost_giou=1
set_cost_class=4
saliency_margin=0.2
aux_loss=false
max_segment_num=20
max_frame_num=200
top_percent=0.02
qfvs_vid_feature=fps1
qfvs_txt_feature=query
qfvs_dense_shot=-1
qfvs_score_ensemble=-1
qfvs_score_gather=-1
qfvs_loss_gather=-1

######## data paths
root=/root
results_root=${root}/results
data_root=${root}/data
train_path=${data_root}/${dset_name}/metadata/charades_train.jsonl
eval_path=${data_root}/${dset_name}/metadata/charades_test.jsonl
feat_root=${data_root}/${dset_name}
eval_split_name=test

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"i3d"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_i3d)
  (( v_feat_dim += 1024 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"c3d"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_c3d)
  (( v_feat_dim += 500 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_clip)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/txt_clip
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

PYTHONPATH=$PYTHONPATH:. python3 univtg/train_mr.py \
--dset_type ${dset_type}   \
--results_root ${results_root}   \
--dset_name ${dset_name} \
--clip_length ${clip_length} \
--exp_id ${exp_id} \
--gpu_id ${gpu_id} \
--device ${device} \
--model_id ${model_id} \
--v_feat_types ${v_feat_types} \
--t_feat_type ${t_feat_type} \
--ctx_mode ${ctx_mode} \
--data_root ${data_root} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--eval_epoch ${eval_epoch} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--input_dropout ${input_dropout} \
--dropout ${dropout} \
--droppath ${droppath} \
--bsz ${bsz} \
--eval_bsz ${eval_bsz} \
--n_epoch ${n_epoch} \
--num_workers ${num_workers} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--lr_warmup ${lr_warmup} \
--wd ${wd} \
--use_cache ${use_cache} \
--enc_layers ${enc_layers} \
--main_metric ${main_metric} \
--nms_thd ${nms_thd} \
--easy_negative_only ${easy_negative_only} \
--max_before_nms ${max_before_nms} \
--b_loss_coef ${b_loss_coef} \
--g_loss_coef ${g_loss_coef} \
--eos_coef ${eos_coef} \
--f_loss_coef ${f_loss_coef} \
--s_loss_intra_coef ${s_loss_intra_coef}  \
--s_loss_inter_coef ${s_loss_inter_coef} \
--eval_mode ${eval_mode} \
--round_multiple ${round_multiple} \
--hidden_dim ${hidden_dim} \
--eval_init ${@:1}  \
--start_epoch ${start_epoch} \
# --resume ${resume}  \