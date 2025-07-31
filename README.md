# Milvus NER检索系统

基于Milvus向量数据库的NER检索系统，支持多语种实体和句子的向量化存储与检索，并自动生成NER指令模板。

## 功能特性

- **多语种支持**: 支持德语、英语、西班牙语、法语、日语、韩语、俄语、中文等8种语言
- **双重数据库**: 分别存储实体和句子的向量化数据
- **智能检索**: 基于语义相似度的实体和句子检索
- **指令生成**: 自动生成完整的NER任务指令模板
- **高效存储**: 使用Milvus向量数据库进行高效的向量存储和检索

## 系统架构

```
milvus_ner_retrieval/
├── __init__.py                 # 包初始化
├── config.py                   # 配置文件
├── embedding_model.py          # LLM2Vec模型封装
├── milvus_client.py           # Milvus数据库客户端
├── data_processor.py          # 数据处理器
├── retrieval_engine.py        # 检索引擎
├── database_manager.py        # 数据库管理器
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖包列表
└── README.md                  # 说明文档
```

## 数据库设计

### 实体数据库 (entity_{language})
- `id`: 主键（自动生成）
- `entity_embedding`: 实体向量（4096维）
- `entity_text`: 实体文本
- `entity_type`: 实体类型（用于过滤）

### 句子数据库 (sentence_{language})
- `id`: 主键（自动生成）
- `sentence_embedding`: 句子向量（4096维）
- `sentence_text`: 句子文本
- `ner_labels`: NER标注结果（JSON格式）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据库初始化和数据导入

```bash
python -m milvus_ner_retrieval.main \
    --action setup \
    --entities-file extracted_entities_by_language.json \
    --sentences-file extracted_sentences_with_ner_by_language.json \
    --languages en de zh
```

### 2. 执行检索

```bash
python -m milvus_ner_retrieval.main \
    --action retrieve \
    --query "Barack Obama was born in Hawaii." \
    --language en \
    --output result.json
```

### 3. 查看数据库统计

```bash
python -m milvus_ner_retrieval.main \
    --action stats \
    --output stats.json
```

### 4. 清理数据库

```bash
python -m milvus_ner_retrieval.main \
    --action cleanup \
    --languages en de zh
```

## 编程接口使用

```python
from milvus_ner_retrieval.main import NERRetrievalSystem

# 初始化系统
system = NERRetrievalSystem()

# 设置数据库
system.setup_database(
    entities_file="extracted_entities_by_language.json",
    sentences_file="extracted_sentences_with_ner_by_language.json",
    languages=["en", "de", "zh"]
)

# 执行检索
result = system.retrieve(
    query="Barack Obama was born in Hawaii.",
    language="en"
)

# 打印生成的指令模板
print(result["instruction_template"])

# 关闭系统
system.close()
```

## 检索流程

1. **输入查询句子**: 用户输入待检索的句子和语种
2. **句子检索**: 从对应语种的句子数据库中检索出Top 5相似句子作为示例
3. **实体检索**: 从对应语种的实体数据库中按实体类型检索出每类Top 5相似实体
4. **指令生成**: 将检索到的实体和句子填充到指令模板中
5. **返回结果**: 返回完整的NER指令模板和检索统计信息

## 指令模板格式

生成的指令模板包含以下部分：
- 任务描述
- 8种实体类型及其示例
- 输入输出格式说明
- 相似句子示例
- 待处理的查询句子

## 配置说明

在 `config.py` 中可以配置：
- Milvus服务器连接信息
- 向量维度和索引参数
- 检索参数（Top-K值等）
- 支持的语种和实体类型
- LLM2Vec模型配置

## 注意事项

1. 确保Milvus服务器正在运行
2. LLM2Vec模型需要大量GPU内存（建议16GB+）
3. 首次加载模型可能需要较长时间
4. 大批量数据导入建议分批进行

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐）
- 16GB+ GPU内存
- Milvus 2.3.0+