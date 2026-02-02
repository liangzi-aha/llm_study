# llm_study
ai大模型学习

## 模型为 通义千问2.5-7B-Instruct
- 下载 pip install modelscope
- modelscope download --model Qwen/Qwen2.5-7B-Instruct

## 项目运行
# 1、python 版本
3.11

# 2、安装依赖
pip install -r requirements.txt

# 3、mac 进入虚拟环境
source .venv/bin/activate

# 4、运行项目
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# 5、安装新的依赖-记录到配置文件中
pip freeze > requirements.txt

# deactivate 退出虚拟环境




## 1、llm 大模型
用 Hugging Face Transformers 库调用大模型 （调用大模型的api 类似规范，所有大模型都遵循该规范，统一接口）
这是全球开源模型的标准调用方式，几乎所有主流开源模型（包括 Qwen、ChatGLM、LLaMA 等）都兼容 Hugging Face 的transformers库，也是脱离 ModelScope 后最推荐的方式 ——因为 ModelScope 下载的模型，本身就是 Hugging Face 标准格式（包含 config.json、tokenizer.json、权重文件等）

## 2、agent 框架，自主决策 + 工具使用 （理解一个代理人，调用llm进行处理）

## 3、RAG (Retrieval-Augmented Generation) - 检索增强生成  （知识检索 + 文本生成 减少 llm 的幻觉）