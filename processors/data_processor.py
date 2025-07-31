"""
数据处理器
负责处理和准备向量数据库的数据
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from ..config import ENTITY_TYPES, SUPPORTED_LANGUAGES
from ..core.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理器类
    负责处理实体和句子数据，生成向量嵌入
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化数据处理器
        
        Args:
            embedding_model: 嵌入模型实例
        """
        self.embedding_model = embedding_model
    
    def process_entities_data(self, entities_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理实体数据文件，生成向量嵌入
        
        Args:
            entities_file: 实体数据文件路径 (extracted_entities_by_language.json)
            
        Returns:
            Dict: 按语种分组的实体数据，包含向量嵌入
        """
        try:
            # 加载实体数据
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            
            processed_data = {}
            
            for language, entity_types in entities_data.items():
                if language not in SUPPORTED_LANGUAGES:
                    logger.warning(f"不支持的语种: {language}")
                    continue
                
                language_entities = []
                
                # 处理每个实体类型
                for entity_type, entities in entity_types.items():
                    if entity_type not in ENTITY_TYPES:
                        logger.warning(f"不支持的实体类型: {entity_type}")
                        continue
                    
                    if not entities:  # 跳过空的实体列表
                        continue
                    
                    logger.info(f"处理 {language} 语种的 {entity_type} 类型实体，共 {len(entities)} 个")
                    
                    # 批量生成向量嵌入
                    embeddings = self.embedding_model.encode_documents(entities)
                    
                    # 转换为numpy数组
                    embeddings_np = embeddings.detach().cpu().numpy()
                    
                    # 为每个实体创建数据记录
                    for i, entity_text in enumerate(entities):
                        language_entities.append({
                            "entity_embedding": embeddings_np[i],
                            "entity_text": entity_text,
                            "entity_type": entity_type
                        })
                
                processed_data[language] = language_entities
                logger.info(f"完成 {language} 语种实体处理，共 {len(language_entities)} 个实体")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理实体数据失败: {e}")
            raise
    
    def process_sentences_data(self, sentences_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        处理句子数据文件，生成向量嵌入
        
        Args:
            sentences_file: 句子数据文件路径 (extracted_sentences_with_ner_by_language.json)
            
        Returns:
            Dict: 按语种分组的句子数据，包含向量嵌入
        """
        try:
            # 加载句子数据
            with open(sentences_file, 'r', encoding='utf-8') as f:
                sentences_data = json.load(f)
            
            processed_data = {}
            
            for language, sentences in sentences_data.items():
                if language not in SUPPORTED_LANGUAGES:
                    logger.warning(f"不支持的语种: {language}")
                    continue
                
                if not sentences:  # 跳过空的句子列表
                    continue
                
                logger.info(f"处理 {language} 语种的句子，共 {len(sentences)} 个")
                
                # 提取句子文本
                sentence_texts = [item["sentence"] for item in sentences]
                
                # 批量生成向量嵌入
                embeddings = self.embedding_model.encode_documents(sentence_texts)
                
                # 转换为numpy数组
                embeddings_np = embeddings.detach().cpu().numpy()
                
                # 为每个句子创建数据记录
                language_sentences = []
                for i, item in enumerate(sentences):
                    language_sentences.append({
                        "sentence_embedding": embeddings_np[i],
                        "sentence_text": item["sentence"],
                        "ner_labels": item["ner_labels"]
                    })
                
                processed_data[language] = language_sentences
                logger.info(f"完成 {language} 语种句子处理，共 {len(language_sentences)} 个句子")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"处理句子数据失败: {e}")
            raise
    
    def batch_process_entities(self, entities_list: List[str], entity_type: str) -> List[Dict[str, Any]]:
        """
        批量处理实体列表
        
        Args:
            entities_list: 实体文本列表
            entity_type: 实体类型
            
        Returns:
            List[Dict]: 处理后的实体数据
        """
        try:
            if not entities_list:
                return []
            
            # 生成向量嵌入
            embeddings = self.embedding_model.encode_documents(entities_list)
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # 创建数据记录
            processed_entities = []
            for i, entity_text in enumerate(entities_list):
                processed_entities.append({
                    "entity_embedding": embeddings_np[i],
                    "entity_text": entity_text,
                    "entity_type": entity_type
                })
            
            return processed_entities
            
        except Exception as e:
            logger.error(f"批量处理实体失败: {e}")
            raise
    
    def batch_process_sentences(self, sentences_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量处理句子列表
        
        Args:
            sentences_list: 句子数据列表，每个元素包含sentence和ner_labels
            
        Returns:
            List[Dict]: 处理后的句子数据
        """
        try:
            if not sentences_list:
                return []
            
            # 提取句子文本
            sentence_texts = [item["sentence"] for item in sentences_list]
            
            # 生成向量嵌入
            embeddings = self.embedding_model.encode_documents(sentence_texts)
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # 创建数据记录
            processed_sentences = []
            for i, item in enumerate(sentences_list):
                processed_sentences.append({
                    "sentence_embedding": embeddings_np[i],
                    "sentence_text": item["sentence"],
                    "ner_labels": item["ner_labels"]
                })
            
            return processed_sentences
            
        except Exception as e:
            logger.error(f"批量处理句子失败: {e}")
            raise
    
    def validate_entity_data(self, entity_data: Dict[str, Any]) -> bool:
        """
        验证实体数据格式
        
        Args:
            entity_data: 实体数据
            
        Returns:
            bool: 验证结果
        """
        required_fields = ["entity_embedding", "entity_text", "entity_type"]
        
        for field in required_fields:
            if field not in entity_data:
                logger.error(f"实体数据缺少必需字段: {field}")
                return False
        
        # 验证向量维度
        if len(entity_data["entity_embedding"]) != self.embedding_model.get_vector_dimension():
            logger.error("实体向量维度不匹配")
            return False
        
        # 验证实体类型
        if entity_data["entity_type"] not in ENTITY_TYPES:
            logger.warning(f"未知实体类型: {entity_data['entity_type']}")
        
        return True
    
    def validate_sentence_data(self, sentence_data: Dict[str, Any]) -> bool:
        """
        验证句子数据格式
        
        Args:
            sentence_data: 句子数据
            
        Returns:
            bool: 验证结果
        """
        required_fields = ["sentence_embedding", "sentence_text", "ner_labels"]
        
        for field in required_fields:
            if field not in sentence_data:
                logger.error(f"句子数据缺少必需字段: {field}")
                return False
        
        # 验证向量维度
        if len(sentence_data["sentence_embedding"]) != self.embedding_model.get_vector_dimension():
            logger.error("句子向量维度不匹配")
            return False
        
        # 验证NER标签格式
        try:
            json.loads(sentence_data["ner_labels"])
        except json.JSONDecodeError:
            logger.error("NER标签JSON格式无效")
            return False
        
        return True