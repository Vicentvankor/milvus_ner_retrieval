"""
embedding模型封装
负责文本的向量化处理
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
from typing import List, Union, Tuple
import logging

from .config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    LLM2Vec embedding模型封装类
    提供文本向量化功能
    """
    
    def __init__(self, model_config: dict = None):
        """
        初始化embedding模型
        
        Args:
            model_config: 模型配置，如果为None则使用默认配置
        """
        self.config = model_config or MODEL_CONFIG
        self.model = None
        self.tokenizer = None
        self.l2v = None
        self._load_model()
    
    def _load_model(self):
        """
        加载LLM2Vec模型
        """
        try:
            logger.info("开始加载LLM2Vec模型...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"]
            )
            
            # 加载配置
            config = AutoConfig.from_pretrained(
                self.config["model_name"], 
                trust_remote_code=True
            )
            
            # 加载基础模型
            self.model = AutoModel.from_pretrained(
                self.config["model_name"],
                trust_remote_code=True,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.config["device"] if torch.cuda.is_available() else "cpu",
            )
            
            # 加载MNTP LoRA权重
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config["model_name"],
            )
            self.model = self.model.merge_and_unload()
            
            # 加载监督学习LoRA权重
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.config["supervised_model"]
            )
            
            # 初始化LLM2Vec包装器
            self.l2v = LLM2Vec(
                self.model, 
                self.tokenizer, 
                pooling_mode=self.config["pooling_mode"], 
                max_length=self.config["max_length"]
            )
            
            logger.info("LLM2Vec模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def encode_queries(self, queries: List[Union[str, Tuple[str, str]]]) -> torch.Tensor:
        """
        编码查询文本（带指令）
        
        Args:
            queries: 查询列表，可以是字符串或(instruction, query)元组
            
        Returns:
            torch.Tensor: 查询向量
        """
        if not self.l2v:
            raise RuntimeError("模型未正确加载")
        
        try:
            # 如果查询是字符串，转换为元组格式
            formatted_queries = []
            for query in queries:
                if isinstance(query, str):
                    # 使用默认指令
                    instruction = "Given a text, retrieve relevant named entities:"
                    formatted_queries.append([instruction, query])
                else:
                    formatted_queries.append(query)
            
            return self.l2v.encode(formatted_queries)
            
        except Exception as e:
            logger.error(f"查询编码失败: {e}")
            raise
    
    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """
        编码文档文本（不需要指令）
        
        Args:
            documents: 文档列表
            
        Returns:
            torch.Tensor: 文档向量
        """
        if not self.l2v:
            raise RuntimeError("模型未正确加载")
        
        try:
            return self.l2v.encode(documents)
            
        except Exception as e:
            logger.error(f"文档编码失败: {e}")
            raise
    
    def compute_similarity(self, query_embeddings: torch.Tensor, 
                          doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算查询和文档之间的余弦相似度
        
        Args:
            query_embeddings: 查询向量
            doc_embeddings: 文档向量
            
        Returns:
            torch.Tensor: 相似度矩阵
        """
        try:
            # 归一化向量
            query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
            
            # 计算余弦相似度
            similarity = torch.mm(query_norm, doc_norm.transpose(0, 1))
            
            return similarity
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            raise
    
    def get_vector_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度
        """
        return self.config.get("vector_dim", 4096)