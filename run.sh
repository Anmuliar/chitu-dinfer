./script/srun_multi_node.sh 1 1 \
    --master_port=22525 \
    test/single_req_test.py \
    serve.port=21111 \
    infer.tp_size=1 \
    infer.cache_type=paged \
    models=LLaDA2.0-mini \
    models.ckpt_dir=/data/nfs/LLaDA2.1-mini \
    infer.use_cuda_graph=True \
    infer.attn_type=dllm \
    infer.max_reqs=1 \
    infer.max_seq_len=1200 \
    request.max_new_tokens=1024 \
    infer.use_cuda_graph=True >& chitu_run.log