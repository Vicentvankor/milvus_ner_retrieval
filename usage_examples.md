# 🎯 使用示例

这里提供了完整的使用示例，展示如何使用NER检索系统完成您的流程：**存储 -> 输入测试集input -> 召回 -> 填充instruction**

## 快速开始

### 1. 运行您的具体例子

```bash
cd milvus_ner_retrieval
python simple_test_runner.py
# 选择 1 - 单个测试样例
```

这将处理您提到的测试输入：
- **输入**: "It is located in the San'a' Governorate."
- **语种**: en
- **流程**: 存储 → 召回 → 填充instruction

### 2. 完整命令行方式

```bash
# 完整管道运行
python run_complete_pipeline.py \
    --entities-file ../extracted_entities_by_language.json \
    --sentences-file ../extracted_sentences_with_ner_by_language.json \
    --test-input "It is located in the San'a' Governorate." \
    --language en \
    --output result.json
```

### 3. 批量处理测试文件

```bash
# 设置数据库
python batch_test_processor.py \
    --action setup \
    --entities-file ../extracted_entities_by_language.json \
    --sentences-file ../extracted_sentences_with_ner_by_language.json

# 处理测试文件
python batch_test_processor.py \
    --action process \
    --test-file your_test_file.json \
    --output-file processed_results.json \
    --language en
```

## 编程接口使用

### 基本使用

```python
from test_set_processor import TestSetProcessor

# 1. 初始化
processor = TestSetProcessor()

# 2. 存储数据
processor.setup_database(
    entities_file="../extracted_entities_by_language.json",
    sentences_file="../extracted_sentences_with_ner_by_language.json",
    languages=["en"]
)

# 3. 处理测试输入
test_input = "It is located in the San'a' Governorate."
instruction = processor.process_test_input(test_input, "en")

print("生成的instruction:")
print(instruction)

processor.close()
```

### 获取详细信息

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

```python
# 批量处理多个测试输入
test_inputs = [
    {"input": "It is located in the San'a' Governorate.", "language": "en"},
    {"input": "Barack Obama was born in Hawaii.", "language": "en"}
]

results = processor.process_test_inputs_batch(test_inputs, "en")

for result in results:
    print(f"输入: {result['original_input']}")
    print(f"Instruction: {result['generated_instruction'][:100]}...")
```

## 生成结果格式

生成的instruction将包含以下部分：

```
Please list all named entities of the following entity types in the input sentence
- PERSON: 
  e.g. Barack Obama, 
       Donald Trump, 
       Angela Merkel
- LOCATION: 
  e.g. San'a' Governorate,
       New York City,
       Berlin
- PRODUCT: 
  e.g. iPhone,
       Windows 10
...

Here are some examples:
Input: It is located in the Abyan Governorate .
Output: {"LOCATION": ["Abyan Governorate"]}
Input: Barack Obama was born in Hawaii.
Output: {"PERSON": ["Barack Obama"], "LOCATION": ["Hawaii"]}

You should output your results in the format {"type": ["entity"]} as a JSON.
Input: It is located in the San'a' Governorate.
```

## 配置选项

在 `config.py` 中可以调整：

- `top_k_entities`: 每个实体类型返回的数量 (默认: 5)
- `top_k_sentences`: 返回的示例句子数量 (默认: 5)
- `vector_dim`: 向量维度 (默认: 4096)
- `supported_languages`: 支持的语种列表

## 故障排除

1. **Milvus连接失败**: 确保Docker中的Milvus服务正在运行
2. **模型加载失败**: 确保有足够GPU内存或使用CPU模式
3. **数据文件不存在**: 确保运行了数据提取脚本生成JSON文件
4. **检索结果为空**: 检查语种设置和数据库是否正确初始化

## 文件说明

- `simple_test_runner.py`: 运行您的具体例子
- `test_set_processor.py`: 核心处理器类
- `batch_test_processor.py`: 批量处理工具
- `run_complete_pipeline.py`: 完整管道运行器
- `demo.py`: 完整演示脚本

选择适合您需求的工具开始使用！