dset_name=charades
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=0
device=0
bsz=32
lr=0.0002
lr_drop=60
wd=0.0001
n_epoch=100
clip_length=1
lw_saliency=20
max_v_l=-1
contrastive_align_loss_coef=0.02

######## data paths
data_root=/root/data
train_path=${data_root}/${dset_name}/metadata/charades_train.jsonl
eval_path=${data_root}/${dset_name}/metadata/charades_test.jsonl
eval_split_name=test

######## setup video+text features
feat_root=${data_root}/${dset_name}

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
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

#### training

PYTHONPATH=$PYTHONPATH:. python models/qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--device ${device} \
--data_root ${data_root} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--wd ${wd} \
--n_epoch ${n_epoch} \
--clip_length ${clip_length} \
--lw_saliency ${lw_saliency} \
--max_v_l ${max_v_l} \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
${@:1}