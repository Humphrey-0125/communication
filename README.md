# BlindFugue minimal prototype

这是一个独立的最小实现，不依赖 `cabeza`。它只验证一条核心链路：

1. 多个 peer 独立回答同一个问题，各自生成私有 episode。
2. 每个 peer 根据自己的状态生成 `NeedState`。
3. requester 看不到其他 peer 的 note；后台 router 使用隐藏 note 自动选择 episode。
4. Delta Reader 读取选中的原始 episode，只返回与 Need 相关的增量证据。
5. peer 使用 Delta 修订答案，最后由一个聚合调用生成团队答案。

第一版固定执行一次 Need 通信轮，不包含长期循环、subscription 或训练。每个 peer
在独立检索阶段可以使用 `search`、`visit`、`google_scholar` 和
`code_interpreter`；路由、定向读取和最终汇总阶段不调用这些工具。
两个 peer 在不调用工具时一次运行通常需要 9 次较短的模型调用。

## 安装

```powershell
conda activate yhf
cd D:\2026\research\mas\project
python -m pip install -e .
```

复制并填写配置：

```powershell
Copy-Item .env.example .env
```

`.env` 中需要填写一个 OpenAI-compatible Chat Completions API：

```dotenv
BLINDFUGUE_API_KEY=...
BLINDFUGUE_BASE_URL=https://your-endpoint.example/v1
BLINDFUGUE_MODEL=your-model-name
BLINDFUGUE_ENABLE_THINKING=false
BLINDFUGUE_MAX_TOOL_ROUNDS=4

# search 和 google_scholar
SERPER_API_KEY=...

# visit；匿名 Jina Reader 受限时需要填写
JINA_API_KEY=...
```

本地 vLLM 不要求密钥时，可以使用 `BLINDFUGUE_API_KEY=EMPTY`。
SiliconFlow 上的 Qwen 推理模型建议关闭内部思考，因为这个原型要求模型直接返回结构化正文；
如果其他供应商不支持 `enable_thinking` 参数，删除这一配置即可。

## 运行

```powershell
python -m blindfugue "Identify the 2017 paper that introduced the Transformer architecture and name its first author."
```

或者：

```powershell
blindfugue "What evidence would distinguish correlation from causation in this claim?"
```

默认运行两个 peer，并启用全部四个工具。可以通过 `--peers 3` 修改数量，通过
`--tools none` 完全关闭工具。每个 peer 最多执行 4 轮工具调用；可用
`--max-tool-rounds N` 修改上限。

## 运行 BrowseComp 样例

下面的命令只运行前三条，并把每条完整结果保存为 JSONL：

```powershell
python -m blindfugue `
  --dataset data/bc/eval.jsonl `
  --limit 3 `
  --output outputs/bc_3.jsonl
```

数据集按顺序执行，结果在每条完成后立即写入。再次运行同一命令时，默认跳过输出文件中已经成功的样例。使用 `--no-resume` 可以覆盖输出并重新执行。

工具错误会作为观察结果返回给 peer，不会直接终止样例；数据集运行器也会隔离单条失败并继续处理后续样例。没有 `SERPER_API_KEY` 时，`search` 和
`google_scholar` 会报告未配置，BrowseComp 的事实检索能力仍然有限。

四个工具的职责如下：

- `search(query)`：通过 Serper 批量搜索网页。
- `visit(url, goal)`：通过 Jina Reader 获取网页正文；返回内容会截断以控制上下文。
- `google_scholar(query)`：通过 Serper 检索 Google Scholar。
- `code_interpreter(code)`：执行受限的短 Python 计算，不允许网络和文件操作。

## 离线测试

```powershell
python -m unittest discover -s tests -v
```

测试使用本地假模型，不访问外部 API。

## 信息可见性

- episode note：保存，但不展示给其他 peer；只有 router 自动计算时可见。
- episode raw content：保存，但只有 Delta Reader 在该 episode 被选中后读取。
- evidence delta：进入 requester 的修订上下文。
- 完整 episode 和 route：保留在返回结果中，供实验记录使用。
