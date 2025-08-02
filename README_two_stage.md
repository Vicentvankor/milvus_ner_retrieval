# NER检索系统 - 两阶段架构

## 系统概述

本系统采用两阶段架构设计，将NER检索系统分为**数据准备阶段**和**检索服务阶段**，实现了数据处理与服务提供的解耦，提高了系统的可维护性和扩展性。

## 架构设计

### 阶段一：数据准备阶段 (`stage1_data_preparation.py`)
负责原始数据的加载、处理、向量化和存储到Milvus数据库。

**主要功能：**
- 📁 加载和解析JSON格式的实体和句子数据
- 🔄 数据预处理和清洗
- 🧮 生成文本向量嵌入
- 🗄️ 创建和管理Milvus数据库集合
- 💾 批量存储数据到向量数据库
- 📊 提供数据统计信息

**输入：**
- `extracted_entities_by_language.json` - 按语言分类的实体数据
- `extracted_sentences_with_ner_by_language.json` - 按语言分类的句子和NER标签数据

**输出：**
- Milvus向量数据库中的实体和句子集合
- 数据统计报告

### 阶段二：检索服务阶段 (`stage2_retrieval_service.py`)
负责处理用户查询，执行向量检索，生成NER指令模板。

**主要功能：**
- 🔍 接收用户查询（句子+语言）
- 🎯 执行相似句子和实体的向量检索
- 📝 生成完整的NER指令模板
- 🚀 支持单次查询和批量查询
- 📈 提供服务统计信息
- 💬 支持交互式查询模式

**输入：**
- 用户查询句子
- 目标语言代码

**输出：**
- 完整的NER指令模板
- 检索统计信息

## 配置文件 (`config.py`)

配置文件采用分阶段设计，为每个阶段提供独立的配置参数：

### 阶段一配置
```python
STAGE1_DATA_CONFIG = {
    "entities_file": "data/extracted_entities_by_language.json",
    "sentences_file": "data/extracted_sentences_with_ner_by_language.json",
    "target_languages": ["en", "zh", "ko"],
    "batch_size_entities": 100,
    "batch_size_sentences": 50,
    "processing_delay": 0.1
}

STAGE1_MILVUS_CONFIG = {
    "mode": "local",  # 或 "remote"
    "host": "localhost",
    "port": 19530
}

STAGE1_DATABASE_CONFIG = {
    "entity_db_prefix": "ner_entities_",
    "sentence_db_prefix": "ner_sentences_",
    "vector_dim": 768,
    "index_type": "IVF_FLAT",
    "metric_type": "L2"
}
```

### 阶段二配置
```python
STAGE2_RETRIEVAL_CONFIG = {
    "top_k_entities": 10,
    "top_k_sentences": 5,
    "similarity_threshold": 0.7,
    "max_instruction_length": 4000
}

STAGE2_OUTPUT_CONFIG = {
    "include_statistics": True,
    "include_similarity_scores": False,
    "format_type": "template"
}
```

## 使用方法

### 环境准备

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **启动Milvus服务**（如果使用本地模式）：
```bash
# 使用Docker启动Milvus
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

### 阶段一：数据准备

运行数据准备脚本，将实体和句子数据导入向量数据库：

```bash
# 基本运行
python stage1_data_preparation.py

# 自定义配置
python stage1_data_preparation.py \
    --entities-file data/custom_entities.json \
    --sentences-file data/custom_sentences.json \
    --vector-dim 768 \
    --batch-size 1000
```

**参数说明**：
- `--entities-file`: 实体数据文件路径
- `--sentences-file`: 句子数据文件路径
- `--vector-dim`: 向量维度
- `--batch-size`: 批处理大小
- `--index-type`: 索引类型（IVF_FLAT, HNSW等）
- `--metric-type`: 距离度量类型（L2, IP, COSINE）

### 阶段二：检索服务

#### 1. 单次查询
```bash
python stage2_retrieval_service.py \
    --query "Apple Inc. is a technology company." \
    --language en
```

#### 2. 批量查询
```bash
python stage2_retrieval_service.py \
    --batch-file queries.json \
    --output-file results.json
```

#### 3. 处理RetriAll数据
```bash
# 使用默认配置处理所有RetriAll数据
python process_retriall_data.py

# 或者使用stage2脚本处理特定目录
python stage2_retrieval_service.py \
    --jsonl-dirs data/RetriAll_data/test data/RetriAll_data/train \
    --input-field input \
    --output-dir data/RetriAll_output
```

#### 4. 交互模式
```bash
python stage2_retrieval_service.py --interactive
```

**参数说明**：
- `--query`: 单次查询文本
- `--language`: 查询语言（en, zh, de, es, fr, ja, ko, ru）
- `--batch-file`: 批量查询文件（JSON格式）
- `--jsonl-dirs`: JSONL文件目录列表
- `--input-field`: JSONL文件中的输入字段名（默认：input）
- `--output-dir`: 输出目录（默认：data/RetriAll_output）
- `--interactive`: 启动交互模式
- `--top-k-entities`: 每个实体类型返回的数量
- `--top-k-sentences`: 返回的相似句子数量

### 4. 系统测试
```bash
# 运行完整系统测试
python test_system.py
```

## 项目结构

```
milvus_ner_retrieval/
├── config.py                      # 配置文件
├── stage1_data_preparation.py     # 阶段1：数据准备
├── stage2_retrieval_service.py    # 阶段2：检索服务
├── process_retriall_data.py       # RetriAll数据处理脚本
├── main.py                        # 主入口（兼容旧版本）
├── core/                          # 核心组件
│   ├── embedding_model.py         # 嵌入模型
│   └── retrieval_engine.py        # 检索引擎
├── database/                      # 数据库组件
│   ├── milvus_client.py          # Milvus客户端
│   └── database_manager.py       # 数据库管理器
├── processors/                    # 数据处理器
│   ├── data_processor.py         # 数据处理器
│   └── test_set_processor.py     # 测试集处理器
├── utils/                         # 工具函数
│   └── quick_test.py             # 快速测试工具
├── data/                          # 数据目录
│   ├── extracted_entities_by_language.json
│   ├── extracted_sentences_with_ner_by_language.json
│   └── RetriAll_data/            # RetriAll数据集
└── requirements.txt               # 依赖包列表
```

## 测试和验证

### 快速测试
```bash
# 使用便捷脚本处理RetriAll数据
python process_retriall_data.py
```

## 数据格式

### 输入数据格式

**实体数据 (`extracted_entities_by_language.json`):**
```json
{
  "en": {
    "PERSON": ["John Smith", "Mary Johnson"],
    "ORGANIZATION": ["Apple Inc.", "Google"],
    "LOCATION": ["New York", "California"]
  },
  "zh": {
    "PERSON": ["张三", "李四"],
    "ORGANIZATION": ["苹果公司", "谷歌"],
    "LOCATION": ["北京", "上海"]
  }
}
```

**句子数据 (`extracted_sentences_with_ner_by_language.json`):**
```json
{
  "en": [
    {
      "sentence": "Apple Inc. is located in California.",
      "ner_labels": "B-ORG I-ORG O O O B-LOC",
      "entities": {
        "ORGANIZATION": ["Apple Inc."],
        "LOCATION": ["California"]
      }
    }
  ]
}
```

### 输出格式

**NER指令模板:**
```
Based on the following examples and entity types, perform Named Entity Recognition on the given text.

Entity Types:
- PERSON: Names of people
- ORGANIZATION: Names of companies, institutions
- LOCATION: Names of places, locations

Similar Examples:
1. "Apple Inc. is located in California." → B-ORG I-ORG O O O B-LOC
2. "John works at Google in New York." → B-PER O O B-ORG O B-LOC I-LOC

Related Entities:
PERSON: John Smith, Mary Johnson
ORGANIZATION: Apple Inc., Google, Microsoft
LOCATION: New York, California, Texas

Please analyze the following text and provide NER labels:
Text: [USER_INPUT]
```

## 日志系统

系统提供详细的日志记录，包括：

- 📊 **数据处理日志**: 数据加载、处理、存储的详细信息
- 🔍 **检索日志**: 查询处理、向量检索、结果生成的过程
- ⚠️ **错误日志**: 异常情况和错误处理信息
- 📈 **性能日志**: 处理时间、数据量统计

日志文件：
- `stage1_data_preparation.log` - 数据准备阶段日志
- `stage2_retrieval_service.log` - 检索服务阶段日志
- `test_system.log` - 系统测试日志

## 性能优化

### 数据准备阶段优化
- **批量处理**: 支持可配置的批处理大小
- **并行处理**: 多语言数据并行处理
- **内存管理**: 分批加载大型数据集
- **处理延迟**: 可配置的批处理间隔

### 检索服务阶段优化
- **向量索引**: 使用高效的向量索引算法
- **缓存机制**: 嵌入模型结果缓存
- **批量查询**: 支持批量处理多个查询
- **结果过滤**: 基于相似度阈值的结果过滤

## 故障排除

### 常见问题

1. **Milvus连接失败**
   - 检查Milvus服务是否启动
   - 验证连接配置（host、port）
   - 确认网络连接正常

2. **数据文件不存在**
   - 检查数据文件路径是否正确
   - 确认文件格式是否为UTF-8编码的JSON

3. **内存不足**
   - 减小批处理大小
   - 增加处理延迟时间
   - 检查系统可用内存

4. **向量维度不匹配**
   - 确认嵌入模型配置正确
   - 检查数据库集合的向量维度设置

### 调试模式

启用详细日志：
```python
# 在config.py中设置
LOGGING_CONFIG = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
```

## 扩展功能

### 添加新语言支持
1. 在`config.py`中的`SUPPORTED_LANGUAGES`添加新语言代码
2. 准备对应语言的数据文件
3. 重新运行数据准备阶段

### 自定义实体类型
1. 在`config.py`中的`ENTITY_TYPES`添加新实体类型
2. 更新指令模板中的实体类型描述
3. 重新处理数据

### 集成其他嵌入模型
1. 在`models/embedding_model.py`中添加新模型支持
2. 更新`config.py`中的模型配置
3. 重新生成向量嵌入

## 系统要求

- Python 3.8+
- Milvus 2.0+
- 至少4GB可用内存
- 支持CUDA的GPU（可选，用于加速嵌入生成）

## 许可证

本项目采用MIT许可证，详见LICENSE文件。