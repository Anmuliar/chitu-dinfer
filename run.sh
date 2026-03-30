./script/srun_multi_node.sh 1 1 \
    --master_port=22525 \
    -m chitu \
    serve.port=21111 \
    infer.tp_size=1 \
    infer.cache_type=paged \
    models=LLaDA2.0-mini \
    models.ckpt_dir=/data/nfs/LLaDA2.1-mini \
    infer.use_cuda_graph=True \
    infer.max_reqs=1 \
    infer.max_seq_len=256 \
    request.max_new_tokens=128 \
    infer.use_cuda_graph=False >& chitu_run.log