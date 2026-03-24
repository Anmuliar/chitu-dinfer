./script/srun_multi_node.sh 1 8 \
    --master_port=22525 \
    -m chitu \
    serve.port=21002 \
    infer.pp_size=1 \
    infer.tp_size=8 \
    infer.cache_type=paged \
    models=LLaDA2.0-mini \
    models.ckpt_dir=/data/nfs/LLaDA2.1-mini \
    infer.use_cuda_graph=True \
    infer.max_reqs=256 \
    infer.max_seq_len=2048 \
    request.max_new_tokens=1200 \
    infer.use_cuda_graph=True >& chitu_run.log