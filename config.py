"""
配置文件
定义系统的各种配置参数
支持两阶段架构：数据准备阶段 + 检索服务阶段
"""

# ==================== 阶段一：数据准备阶段配置 ====================

# 数据准备阶段 - 输入数据配置
STAGE1_DATA_CONFIG = {
    "entities_file": "./data/extracted_entities_by_language.json",  # 实体数据文件路径
    "sentences_file": "./data/extracted_sentences_with_ner_by_language.json",  # 句子数据文件路径
    "target_languages": ["de", "en", "es", "fr", "ja", "ko", "ru", "zh"],  # 要处理的语言列表
    "batch_size_entities": 1000,  # 实体数据批量处理大小
    "batch_size_sentences": 500,  # 句子数据批量处理大小
    "processing_delay": 0.1,  # 批次间延迟（秒），避免过载
}

# 数据准备阶段 - Milvus数据库配置
STAGE1_MILVUS_CONFIG = {
    "mode": "local",  # "local" 或 "remote"
    "local_db_path": "./milvus_data",  # 本地数据库路径
    "host": "localhost",  # 远程模式使用
    "port": "19530",     # 远程模式使用
    "connection_timeout": 30,  # 连接超时时间（秒）
}

# 数据准备阶段 - 数据库结构配置
STAGE1_DATABASE_CONFIG = {
    "entity_db_prefix": "entity_",  # 实体数据库前缀
    "sentence_db_prefix": "sentence_",  # 句子数据库前缀
    "vector_dim": 4096,  # LLM2Vec模型的向量维度
    "index_type": "IVF_FLAT",  # 索引类型
    "metric_type": "COSINE",  # 相似度计算方式
    "nlist": 1024,  # 索引参数
    "enable_auto_flush": True,  # 自动刷新
    "auto_flush_interval": 1,  # 自动刷新间隔（秒）
}

# 数据准备阶段 - 嵌入模型配置
STAGE1_MODEL_CONFIG = {
    "model_name": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    "supervised_model": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised",
    "pooling_mode": "mean",
    "max_length": 512,
    "device": "cuda",  # 或 "cpu"
    "batch_size": 32,  # 嵌入生成批次大小
    "enable_cache": True,  # 启用嵌入缓存
}

# ==================== 阶段二：检索服务阶段配置 ====================

# 第二阶段：检索服务配置
STAGE2_CONFIG = {
    # 输入配置
    'input': {
        # JSONL文件输入路径
        'jsonl_directories': [
            'data/RetriAll_data/test',
            'data/RetriAll_data/train',
            'data/RetriAll_data/dev'
        ],
        'input_field': 'input',  # JSONL文件中包含句子的字段名
        'query_text': '',  # 单次查询的文本
        'batch_queries': [],  # 批量查询的文本列表
        'language': 'en',  # 查询语言
    },
    
    # 输出配置
    'output': {
        'output_directory': 'data/RetriAll_output',  # 输出目录
        'format': 'jsonl',  # 输出格式：jsonl, json, text
        'include_scores': True,  # 是否包含相似度分数
        'include_metadata': True,  # 是否包含元数据
        'preserve_original_fields': True,  # 是否保留原始字段
    }
}

# 检索服务阶段 - 数据库连接配置
STAGE2_MILVUS_CONFIG = {
    "mode": "local",  # "local" 或 "remote"
    "local_db_path": "./milvus_data",  # 本地数据库路径
    "host": "localhost",  # 远程模式使用
    "port": "19530",     # 远程模式使用
    "connection_timeout": 10,  # 连接超时时间（秒）
    "connection_pool_size": 5,  # 连接池大小
}

# 检索服务阶段 - 检索参数配置
STAGE2_RETRIEVAL_CONFIG = {
    "top_k_entities": 5,  # 每个实体类型返回的top K个实体
    "top_k_sentences": 5,  # 返回的top K个示例句子
    "max_entities_per_type": 5,  # 每个实体类型在指令中显示的最大数量
    "max_examples_in_instruction": 5,  # 指令中显示的最大示例数量
    "search_params": {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    },
    "similarity_threshold": 0.0,  # 相似度阈值，低于此值的结果将被过滤
}

# 检索服务阶段 - 嵌入模型配置
STAGE2_MODEL_CONFIG = {
    "model_name": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    "supervised_model": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised",
    "pooling_mode": "mean",
    "max_length": 512,
    "device": "cuda",  # 或 "cpu"
    "enable_cache": True,  # 启用嵌入缓存
    "cache_size": 1000,  # 缓存大小
}

# 检索服务阶段 - 输出格式配置
STAGE2_OUTPUT_CONFIG = {
    "include_similarity_scores": True,  # 是否包含相似度分数
    "include_statistics": True,  # 是否包含统计信息
    "output_format": "json",  # 输出格式：json, text
    "save_results": False,  # 是否保存结果到文件
    "results_dir": "./results",  # 结果保存目录
}

# ==================== 通用配置 ====================

# 实体类型配置
ENTITY_TYPES = [
    "PERSON",
    "LOCATION", 
    "PRODUCT",
    "FACILITY",
    "ART",
    "GROUP",
    "MISCELLANEOUS",
    "SCIENCE ENTITY"
]

# 支持的语种
SUPPORTED_LANGUAGES = [
    "de", "en", "es", "fr", "ja", "ko", "ru", "zh"
]

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",  # 日志级别：DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "enable_file_logging": True,  # 是否启用文件日志
    "log_file": "./logs/ner_retrieval.log",  # 日志文件路径
    "max_file_size": 10 * 1024 * 1024,  # 最大文件大小（字节）
    "backup_count": 5,  # 备份文件数量
}

# Instruction模板配置
INSTRUCTION_TEMPLATE = """Please list all named entities of the following entity types in the input sentence
{entity_types}
Here are some examples:
{examples}
You should output your results in the format {{"type": ["entity"]}} as a JSON.
Input: """

# 实体类型格式化模板
ENTITY_TYPE_FORMAT = "- {entity_type}: \n  e.g. {examples}"

# ==================== 兼容性配置（向后兼容） ====================

# 为了保持向后兼容，保留原有的配置变量名
MILVUS_CONFIG = STAGE1_MILVUS_CONFIG
DATABASE_CONFIG = STAGE1_DATABASE_CONFIG
RETRIEVAL_CONFIG = STAGE2_RETRIEVAL_CONFIG
MODEL_CONFIG = STAGE1_MODEL_CONFIG