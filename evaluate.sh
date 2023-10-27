# Evaluate all results in the paper

GPU=0

## ModelNet40

### PointNeXt

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml --pretrained_path release_ckpt/modelnet40ply2048/pointnext-s/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 224

### PointNeXt (with normals)

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s_normal.yaml --pretrained_path release_ckpt/modelnet40ply2048/pointnext-s_normal/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 6589

### PointMLP

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointmlp.yaml --pretrained_path release_ckpt/modelnet40ply2048/pointmlp/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 7563

### PointMLP (with normals)

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointmlp_normal.yaml --pretrained_path release_ckpt/modelnet40ply2048/pointmlp_normal/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 5699

## ScanObjectNN (OBJ_BG)

### PointNeXt

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/scanobjectnn_objbg/pointnext-s.yaml --pretrained_path release_ckpt/scanobjectnn_objbg/pointnext-s/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 9886

### PointMLP

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/scanobjectnn_objbg/pointmlp.yaml --pretrained_path release_ckpt/scanobjectnn_objbg/pointmlp/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 9944

## ScanObjectNN (PB_T50_RS)

### PointNeXt

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/scanobjectnn_hardest/pointnext-s.yaml --pretrained_path release_ckpt/scanobjectnn_hardest/pointnext-s/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 1733

### PointMLP

CUDA_VISIBLE_DEVICES=$GPU python examples/classification/main.py --cfg cfgs/scanobjectnn_hardest/pointmlp.yaml --pretrained_path release_ckpt/scanobjectnn_hardest/pointmlp/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 2379

## ShapeNetPart

### PointNeXt

CUDA_VISIBLE_DEVICES=$GPU python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml --pretrained_path release_ckpt/shapenetpart/pointnext-s/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 4696

### PointNeXt (with normals)

CUDA_VISIBLE_DEVICES=$GPU python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s_normal.yaml --pretrained_path release_ckpt/shapenetpart/pointnext-s_normal/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 6109

### PointMLP

CUDA_VISIBLE_DEVICES=$GPU python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmlp.yaml --pretrained_path release_ckpt/shapenetpart/pointmlp/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 1932

### PointMLP (with normals)

CUDA_VISIBLE_DEVICES=$GPU python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointmlp_normal.yaml --pretrained_path release_ckpt/shapenetpart/pointmlp_normal/ckpt.pth --mode test_so3 --so3_mode random_rot --num_rot 1 --seed 4815


