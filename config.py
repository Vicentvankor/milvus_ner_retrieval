"""
配置文件
定义系统的各种配置参数
"""

# Milvus连接配置 - 支持本地数据库文件
MILVUS_CONFIG = {
    "mode": "local",  # "local" 或 "remote"
    "local_db_path": "./milvus_data",  # 本地数据库路径
    "host": "localhost",  # 远程模式使用
    "port": "19530",     # 远程模式使用
}

# 数据库配置
DATABASE_CONFIG = {
    "entity_db_prefix": "entity_",  # 实体数据库前缀
    "sentence_db_prefix": "sentence_",  # 句子数据库前缀
    "vector_dim": 4096,  # LLM2Vec模型的向量维度
    "index_type": "IVF_FLAT",  # 索引类型
    "metric_type": "COSINE",  # 相似度计算方式
    "nlist": 1024,  # 索引参数
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k_entities": 5,  # 每个实体类型返回的top K个实体
    "top_k_sentences": 5,  # 返回的top K个示例句子
    "search_params": {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
}

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

# 模型配置
MODEL_CONFIG = {
    "model_name": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    "supervised_model": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised",
    "pooling_mode": "mean",
    "max_length": 512,
    "device": "cuda",  # 或 "cpu"
}

# Instruction模板
INSTRUCTION_TEMPLATE = """Please list all named entities of the following entity types in the input sentence
{entity_types}
Here are some examples:
{examples}
You should output your results in the format {{"type": ["entity"]}} as a JSON.
Input: """

# 实体类型格式化模板
ENTITY_TYPE_FORMAT = "- {entity_type}: \n  e.g. {examples}"