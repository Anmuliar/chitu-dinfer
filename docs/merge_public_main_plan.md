# Merge public-main 到 model-llada 分支规划

## 一、冲突文件概览

共 **10 个文件**存在冲突：

| 文件 | 优先级 | 冲突类型 |
|------|--------|----------|
| `chitu/executor.py` | 高 | 大量新增功能 + 结构重构 |
| `chitu/task.py` | 高 | DLLM task 类型扩展 |
| `chitu/scheduler.py` | 高 | DLLM 调度支持 |
| `chitu/chitu_main.py` | 中 | 初始化流程变更 |
| `chitu/kv_cache/kv_cache.py` | 中 | cache_manager 重构 + DLLM 扩展 |
| `chitu/models/__init__.py` | 低 | model 注册 |
| `chitu/native_layout/marlin.py` | 低 | 量化相关 |
| `chitu/schemas/serve_config.py` | 低 | 配置项 |
| `chitu/serve/api_server.py` | 低 | API 相关 |
| `chitu/serve/common.py` | 低 | 服务相关 |

## 二、合并策略

### 总体原则
1. **优先采用 public-main 的代码结构**：public-main 有大量架构优化（如 Sampler 抽取、profile payload、embed_tokens_lm_head_tp 等）
2. **保留 DLLM 相关功能**：在 public-main 结构上叠加 DLLM 扩展
3. **保持 TaskType 扩展**：`is_prefill()` / `is_decode()` 函数和 `PREFILL_TYPES` / `DECODE_TYPES` 常量

### 逐文件策略

---

### 1. `chitu/executor.py` (最高优先级)

**public-main 变更**：
- 新增 `Sampler` 类抽取采样逻辑
- 新增 `embed_tokens_lm_head_tp_size` 支持
- 新增 profile payload 机制
- 移除部分旧依赖（`apply_frequency_penalty`, `DisaggregationMode`）
- KV hook 改用 `kv_cache` 而非 `cache_manager`

**model-llada 变更**：
- 新增 `prefill_dllm_step()` 和 `decode_dllm_step()` 方法
- 新增 `_prepare_blocks_for_decode_dllm()` 方法
- 新增 `_pending_dllm_block` 状态
- 新增 `_process_dllm_block_results()` 方法
- 修改 `step()` 分发逻辑支持 `TaskType.PrefillDLLM` / `TaskType.DecodeDLLM`
- 使用 `is_prefill()` / `is_decode()` 替代直接类型比较

**合并策略**：
```python
# 1. imports: 合并两边的新增 import
from chitu.sampling.sampler import Sampler  # from public-main
from chitu.dllm import TokenArray  # from model-llada
from chitu.task_type import is_prefill, is_decode  # from model-llada

# 2. Executor.__init__: 添加 public-main 的新属性 + model-llada 的 _pending_dllm_block
self.embed_tokens_lm_head_tp_size = int(args.infer.embed_tokens_lm_head_tp_size)  # public-main
self._pending_dllm_block = None  # model-llada

# 3. step(): 采用 public-main 结构 + 添加 DLLM 分支
if tasks.task_type == TaskType.Prefill:
    out = self.prefill_step(tasks)
elif tasks.task_type == TaskType.Decode:
    out = self.decode_step(tasks)
elif tasks.task_type == TaskType.PrefillDLLM:  # DLLM
    out = self.prefill_dllm_step(tasks)
elif tasks.task_type == TaskType.DecodeDLLM:  # DLLM
    out = self.decode_dllm_step(tasks)

# 4. 添加完整的 prefill_dllm_step() 和 decode_dllm_step() 方法
# 5. 保留 _process_dllm_block_results() 方法
```

---

### 2. `chitu/task.py`

**public-main 变更**：暂无（基于 divergence point）

**model-llada 变更**：
- Task 构造函数新增 `infermode` 和 `block_length` 参数
- 新增 `decoding_start`, `next_block`, `block_length` 属性
- `transition_to_decode()` 支持 `TaskType.DecodeDLLM`
- 使用 `is_prefill()` / `is_decode()` 替代类型比较

**合并策略**：采用 model-llada 的全部变更，同时保留 public-main 的任何新 import

---

### 3. `chitu/scheduler.py`

**public-main 变更**：可能有调度优化

**model-llada 变更**：
- 使用 `PREFILL_TYPES`, `DECODE_TYPES`, `ALL_ACTIVE_TYPES` 常量
- 使用 `is_prefill()` / `is_decode()` 函数
- 调度逻辑支持 `TaskType.PrefillDLLM` / `TaskType.DecodeDLLM`

**合并策略**：
- 采用 public-main 的调度框架
- 将 `TaskType.Prefill` 改为 `is_prefill(x)`
- 将 `TaskType.Decode` 改为 `is_decode(x)`
- 将类型集合改为 `ALL_ACTIVE_TYPES`

---

### 4. `chitu/chitu_main.py`

**public-main 变更**：初始化流程重构

**model-llada 变更**：DLLM 相关初始化

**合并策略**：采用 public-main 结构 + 保留 DLLM 初始化逻辑

---

### 5. `chitu/kv_cache/kv_cache.py` (原 `cache_manager.py`)

**public-main 变更**：
- 文件从 `chitu/cache_manager.py` 移动到 `chitu/kv_cache/kv_cache.py`
- 大量重构：新的 `KVCache` 类，provider 模式

**model-llada 变更**：
- `prepare_cache_decode_dllm()` 方法
- `finalize_cache_single_decode_dllm()` 方法

**合并策略**：
- 采用 public-main 的文件结构和 `KVCache` 类
- 在新类中添加 `prepare_cache_decode_dllm()` 和 `finalize_cache_single_decode_dllm()` 方法
- 注意：`PagedKVCacheManager` 已重构为 `PagedKVCache`

---

### 6. `chitu/models/__init__.py`

**合并策略**：合并两边的 model 注册，确保 `TransformerLLaDA2` 被注册

---

### 7. 其他文件

采用 public-main 版本，检查是否有 DLLM 相关内容需要保留

## 三、执行步骤

### Phase 1: 准备工作
```bash
git checkout model-llada
git pull origin model-llada
git fetch origin public-main
git checkout -b merge-public-main-to-llada
```

### Phase 2: 执行合并
```bash
git merge origin/public-main
```

### Phase 3: 逐个解决冲突

按以下顺序解决：

1. **task_type.py** - 确保扩展的 TaskType 定义存在
2. **task.py** - 合并 DLLM task 属性
3. **scheduler.py** - 使用 is_prefill/is_decode
4. **executor.py** - 最重要的文件，仔细合并
5. **kv_cache/kv_cache.py** - 添加 DLLM 方法
6. **models/__init__.py** - 注册 model
7. **其他文件** - 简单合并

### Phase 4: 验证
```bash
# 语法检查
python -m py_compile chitu/executor.py chitu/task.py chitu/scheduler.py

# 运行测试
python test/single_req_test.py models=LLaDA2.0-mini ...
```

## 四、关键代码片段

### task_type.py 扩展（确保存在）

```python
# chitu/task_type.py

class TaskType:
    Prefill = 0
    Decode = 1
    Special = 2
    PrefillDLLM = 3  # DLLM
    DecodeDLLM = 4   # DLLM

PREFILL_TYPES = {TaskType.Prefill, TaskType.PrefillDLLM}
DECODE_TYPES = {TaskType.Decode, TaskType.DecodeDLLM}
ALL_ACTIVE_TYPES = PREFILL_TYPES | DECODE_TYPES

def is_prefill(task_type: int) -> bool:
    return task_type in PREFILL_TYPES

def is_decode(task_type: int) -> bool:
    return task_type in DECODE_TYPES
```

### executor.py 关键合并点

```python
# imports 合并
from chitu.task_type import TaskType, is_prefill, is_decode
from chitu.sampling.sampler import Sampler  # public-main
from chitu.dllm import TokenArray  # model-llada

# step() 方法
def step(self, tasks: Optional[PackedTasksBase]):
    # ... existing code ...

    if tasks.task_type == TaskType.Prefill:
        out = self.prefill_step(tasks)
    elif tasks.task_type == TaskType.Decode:
        out = self.decode_step(tasks)
    elif tasks.task_type == TaskType.PrefillDLLM:
        out = self.prefill_dllm_step(tasks)
    elif tasks.task_type == TaskType.DecodeDLLM:
        out = self.decode_dllm_step(tasks)
    else:
        raise NotImplementedError

    if is_decode(tasks.task_type):
        self._lb_trigger()
        self._lb_sync()
```

## 五、注意事项

1. **public-main 中 cache_manager 重命名为 kv_cache**：需要更新所有 import
2. **Sampler 类**：public-main 抽取了采样逻辑，DLLM 可能不需要（有自己的 decoder）
3. **embed_tokens_lm_head_tp**：新特性，DLLM 可能需要适配
4. **profile payload**：新特性，与 DLLM 独立

## 六、回滚方案

如果合并后出现问题：
```bash
git merge --abort  # 合并过程中
# 或
git reset --hard model-llada  # 合并完成后
```
