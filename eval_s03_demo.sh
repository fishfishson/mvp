EXP=$1
EPOCH=$2
PORT=$3
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT run/validate_3d.py \
    --cfg configs/chi3d/$EXP.yaml \
    --model_path output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-s03@$EPOCH
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT run/demo_3d.py \
    --cfg configs/demo/fight.yml \
    --model_path output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-fight@$EPOCH
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT run/demo_3d.py \
    --cfg configs/demo/crash.yml \
    --model_path output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-crash@$EPOCH
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT run/demo_3d.py \
    --cfg configs/demo/dance.yml \
    --model_path output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-dance@$EPOCH
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT run/demo_3d.py \
    --cfg configs/demo/511.yml \
    --model_path output/$EXP/chi3d/multi_person_posenet_50/$EXP/checkpoint_$EPOCH.pth.tar \
    --out exp-out/$EXP-511@$EPOCH