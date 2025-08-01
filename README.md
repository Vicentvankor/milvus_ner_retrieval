# Milvus NER检索系统

基于Milvus向量数据库的NER检索系统，支持多语种实体和句子的向量化存储与检索，并自动生成NER指令模板。

## 功能特性

- **多语种支持**: 支持德语、英语、西班牙语、法语、日语、韩语、俄语、中文等8种语言
- **双重数据库**: 分别存储实体和句子的向量化数据
- **智能检索**: 基于语义相似度的实体和句子检索
- **指令生成**: 自动生成完整的NER任务指令模板
- **本地数据库**: 支持本地文件模式，无需启动Milvus服务器
- **高效存储**: 使用Milvus向量数据库进行高效的向量存储和检索

## 项目结构

```
milvus_ner_retrieval/
├── __init__.py                     # 包初始化
├── config.py                       # 配置文件（支持本地数据库模式）
├── main.py                         # 主程序入口
├── requirements.txt                # 依赖包列表
├── README.md                       # 项目文档
│
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── embedding_model.py         # LLM2Vec模型封装
│   └── retrieval_engine.py        # 检索引擎
│
├── database/                      # 数据库相关模块
│   ├── __init__.py
│   ├── milvus_client.py          # Milvus客户端（支持本地模式）
│   └── database_manager.py       # 数据库管理器
│
├── processors/                    # 数据处理模块
│   ├── __init__.py
│   ├── data_processor.py         # 数据处理器
│   └── test_set_processor.py     # 测试集处理器
│
├── examples/                      # 示例脚本
│   ├── __init__.py
│   ├── simple_local_demo.py      # 本地数据库演示
│   ├── simple_test_runner.py     # 简单测试运行器
│   ├── batch_test_processor.py   # 批量测试处理器
│   └── run_complete_pipeline.py  # 完整管道运行器
│
└── utils/                         # 工具函数
    ├── __init__.py
    └── quick_test.py             # 快速测试脚本
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

## 快速开始

### 方法1：本地数据库模式（推荐）
无需启动Milvus服务器，直接使用本地文件：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行本地演示
python examples/simple_local_demo.py
```

### 方法2：快速测试
```bash
# 运行快速测试检查系统状态
python utils/quick_test.py
```

### 方法3：命令行使用
```bash
# 本地数据库模式
python -m milvus_ner_retrieval.main \
    --action setup \
    --entities-file extracted_entities_by_language.json \
    --sentences-file extracted_sentences_with_ner_by_language.json \
    --local-db-path ./my_milvus_db \
    --languages en de zh

# 执行检索
python -m milvus_ner_retrieval.main \
    --action retrieve \
    --query "Barack Obama was born in Hawaii." \
    --language en \
    --local-db-path ./my_milvus_db
```

## 使用示例

### 编程接口使用

```python
from milvus_ner_retrieval.processors.test_set_processor import TestSetProcessor

# 1. 初始化（本地数据库模式）
processor = TestSetProcessor(local_db_path="./my_local_db")

# 2. 存储数据
processor.setup_database(
    entities_file="extracted_entities_by_language.json",
    sentences_file="extracted_sentences_with_ner_by_language.json",
    languages=["en"]
)

# 3. 处理测试输入
test_input = "It is located in the San'a' Governorate."
instruction = processor.process_test_input(test_input, "en")

print("生成的instruction:")
print(instruction)

processor.close()
```

### 获取详细检索信息

```python
# 获取包含检索统计的详细结果
detailed_result = processor.create_instruction_for_single_input(test_input, "en")

print(f"找到相似句子: {len(detailed_result['similar_sentences'])} 个")
print(f"找到相关实体: {detailed_result['retrieval_statistics']['total_entities_found']} 个")

# 查看召回的内容
print("\n召回的示例句子:")
for sent in detailed_result["similar_sentences"][:3]:
    print(f"- {sent['sentence_text']}")
    print(f"  相似度: {sent['score']:.4f}")

print("\n召回的实体:")
for entity_type, entities in detailed_result["entity_results"].items():
    if entities:
        print(f"{entity_type}: {[e['entity_text'] for e in entities[:3]]}")
```

### 批量处理

```bash
# 批量处理测试文件
python examples/batch_test_processor.py \
    --action setup \
    --entities-file extracted_entities_by_language.json \
    --sentences-file extracted_sentences_with_ner_by_language.json

python examples/batch_test_processor.py \
    --action process \
    --test-file your_test_file.json \
    --output-file processed_results.json \
    --language en
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

### 数据库模式
```python
MILVUS_CONFIG = {
    "mode": "local",  # "local" 或 "remote"
    "local_db_path": "./milvus_data",  # 本地数据库路径
    "host": "localhost",  # 远程模式使用
    "port": "19530",     # 远程模式使用
}
```

### 检索参数
```python
RETRIEVAL_CONFIG = {
    "top_k_entities": 5,  # 每个实体类型返回的top K个实体
    "top_k_sentences": 5,  # 返回的top K个示例句子
}
```

### 支持的实体类型和语种
- **实体类型**: PERSON, LOCATION, PRODUCT, FACILITY, ART, GROUP, MISCELLANEOUS, SCIENCE ENTITY
- **支持语种**: de, en, es, fr, ja, ko, ru, zh

## 生成结果格式

生成的instruction包含以下部分：

```
Please list all named entities of the following entity types in the input sentence
- PERSON: 
  e.g. Barack Obama, Donald Trump, Angela Merkel
- LOCATION: 
  e.g. San'a' Governorate, New York City, Berlin
- PRODUCT: 
  e.g. iPhone, Windows 10
...

Here are some examples:
Input: It is located in the Abyan Governorate .
Output: {"LOCATION": ["Abyan Governorate"]}
Input: Barack Obama was born in Hawaii.
Output: {"PERSON": ["Barack Obama"], "LOCATION": ["Hawaii"]}

You should output your results in the format {"type": ["entity"]} as a JSON.
Input: It is located in the San'a' Governorate.
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 确保有足够GPU内存（推荐16GB+）
   - 或在config.py中设置 `"device": "cpu"`

2. **数据文件不存在**
   - 确保有 `extracted_entities_by_language.json`
   - 确保有 `extracted_sentences_with_ner_by_language.json`

3. **Milvus连接失败（远程模式）**
   - 确保Milvus服务器正在运行
   - Docker启动: `docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest`

4. **检索结果为空**
   - 检查语种设置是否正确
   - 确认数据库已正确初始化

### 性能优化

- **本地模式**: 推荐用于开发和测试，无需网络连接
- **GPU加速**: 使用CUDA可显著提升模型推理速度
- **批量处理**: 大量数据建议使用批量处理接口

## 系统要求

- **Python**: 3.8+
- **GPU**: CUDA支持的GPU（推荐，可选）
- **内存**: 16GB+ GPU内存（使用GPU时）
- **存储**: 足够空间存储模型和数据库文件
- **依赖**: 见 requirements.txt

## 许可证

本项目基于开源许可证发布，详见项目根目录的LICENSE文件。