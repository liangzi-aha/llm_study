import time
# mlx_lm 和 Hugging Face transformers 的核心区别在于：mlx_lm 是苹果为自家芯片打造的“特快专列”，而 transformers 是能跑在各类硬件上的“通用车辆”。下面我从几个维度为你详细拆解。
from mlx_lm import load, generate
import os

# 加载modelscope模型的路径，这里我放在了用户目录下的缓存文件夹里，当然你也可以放在其他地方，只要路径正确就行。
model_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct")

print("加载模型中...")
start = time.time()
model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
print(f"加载完成：{time.time()-start:.1f}秒")

prompt = "推荐一个好看的电影，并且给出理由。"
print(f"用户：{prompt}")

start = time.time()
# 最简参数，只用最基本的
response = generate(
    model, 
    tokenizer,
    prompt=prompt,
    max_tokens=200  # 先去掉temperature，用默认值
)
print(f"模型回复：{response}")
print(f"用时：{time.time()-start:.1f}秒")