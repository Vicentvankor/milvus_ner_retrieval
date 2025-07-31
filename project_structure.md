# 项目结构说明

## 📁 重新组织后的文件结构

```
milvus_ner_retrieval/
├── __init__.py                     # 包初始化
├── config.py                       # 系统配置（支持本地数据库模式）
├── main.py                         # 主程序入口和系统类
├── requirements.txt                # 依赖包列表
├── README.md                       # 详细文档
├── usage_examples.md               # 使用示例
├── example_usage.py                # 使用示例代码
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
│   ├── simple_test_runner.py     # 简单测试运行器
│   ├── batch_test_processor.py   # 批量测试处理器
│   ├── run_complete_pipeline.py  # 完整管道运行器
│   └── demo.py                   # 演示脚本
│
└── utils/                         # 工具函数
    ├── __init__.py
    └── quick_test.py             # 快速测试脚本
```

## 🔧 主要改进

### 1. 层级结构优化
- **core/**: 核心功能（模型、检索引擎）
- **database/**: 数据库相关（客户端、管理器）
- **processors/**: 数据处理（处理器、测试集处理）
- **examples/**: 示例和演示脚本
- **utils/**: 工具函数

### 2. 本地数据库支持
- 修改 `config.py` 支持本地模式：
  ```python
  MILVUS_CONFIG = {
      "mode": "local",  # "local" 或 "remote"
      "local_db_path": "./milvus_data",  # 本地数据库路径
  }
  ```

- 修改 `MilvusClient` 支持本地文件：
  ```python
  connections.connect(
      alias=self.connection_alias,
      uri=f"file://{self.local_db_path}"
  )
  ```

### 3. 错误修复
- 移除代码中的 `</code_edit>` 错误标签
- 修复所有导入路径以适应新结构
- 统一使用相对导入

## 🚀 使用方法

### 本地数据库模式
```python
from milvus_ner_retrieval.main import NERRetrievalSystem

# 使用本地数据库
system = NERRetrievalSystem(local_db_path="./my_local_db")
```

### 命令行使用
```bash
# 使用本地数据库路径
python -m milvus_ner_retrieval.main \
    --action setup \
    --entities-file ../extracted_entities_by_language.json \
    --sentences-file ../extracted_sentences_with_ner_by_language.json \
    --local-db-path ./my_milvus_db
```

## 📝 导入路径更新

所有文件的导入路径已更新为相对导入：
- `from .config import MILVUS_CONFIG`
- `from .database.milvus_client import MilvusClient`
- `from .core.embedding_model import EmbeddingModel`
- 等等...

这样的结构更加清晰、模块化，便于维护和扩展。