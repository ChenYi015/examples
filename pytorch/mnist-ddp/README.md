# mnist-ddp

```bash
pip install -r requirements.txt
```

example usage:

```bash
torchrun \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --nnodes=2 \
    --nproc-per-node=4 \
    --node-rank=${RANK} \
    main.py \
    --epochs 10 \
    --backend nccl
```
