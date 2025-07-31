"""
检索引擎
负责实体和句子的检索，以及指令模板的生成
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

from ..config import ENTITY_TYPES, INSTRUCTION_TEMPLATE, ENTITY_TYPE_FORMAT, RETRIEVAL_CONFIG
from ..database.milvus_client import MilvusClient
from .embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    检索引擎类
    负责执行检索任务和生成指令模板
    """
    
    def __init__(self, milvus_client: MilvusClient, embedding_model: EmbeddingModel):
        """
        初始化检索引擎
        
        Args:
            milvus_client: Milvus客户端实例
            embedding_model: 嵌入模型实例
        """
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model
    
    def retrieve_similar_sentences(self, query: str, language: str, 
                                 top_k: int = None) -> List[Dict[str, Any]]:
        """
        检索相似句子
        
        Args:
            query: 查询句子
            language: 语种
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似句子列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode_documents([query])
            query_vector = query_embedding.detach().cpu().numpy()[0]
            
            # 构建集合名称
            collection_name = f"sentence_{language}"
            
            # 执行检索
            results = self.milvus_client.search_sentences(
                collection_name=collection_name,
                query_embedding=query_vector,
                top_k=top_k or RETRIEVAL_CONFIG["top_k_sentences"]
            )
            
            logger.info(f"检索到 {len(results)} 个相似句子")
            return results
            
        except Exception as e:
            logger.error(f"检索相似句子失败: {e}")
            raise
    
    def retrieve_entities_by_type(self, query: str, language: str, 
                                entity_type: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        按类型检索相似实体
        
        Args:
            query: 查询文本
            language: 语种
            entity_type: 实体类型
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相似实体列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode_documents([query])
            query_vector = query_embedding.detach().cpu().numpy()[0]
            
            # 构建集合名称
            collection_name = f"entity_{language}"
            
            # 执行检索
            results = self.milvus_client.search_entities(
                collection_name=collection_name,
                query_embedding=query_vector,
                entity_type=entity_type,
                top_k=top_k or RETRIEVAL_CONFIG["top_k_entities"]
            )
            
            logger.info(f"检索到 {len(results)} 个 {entity_type} 类型的实体")
            return results
            
        except Exception as e:
            logger.error(f"检索实体失败: {e}")
            raise
    
    def retrieve_all_entity_types(self, query: str, language: str, 
                                top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        检索所有实体类型的相似实体
        
        Args:
            query: 查询文本
            language: 语种
            top_k: 每个类型返回的结果数量
            
        Returns:
            Dict: 按实体类型分组的检索结果
        """
        try:
            all_results = {}
            k = top_k or RETRIEVAL_CONFIG["top_k_entities"]
            
            for entity_type in ENTITY_TYPES:
                try:
                    results = self.retrieve_entities_by_type(
                        query=query,
                        language=language,
                        entity_type=entity_type,
                        top_k=k
                    )
                    all_results[entity_type] = results
                except Exception as e:
                    logger.warning(f"检索 {entity_type} 类型实体失败: {e}")
                    all_results[entity_type] = []
            
            return all_results
            
        except Exception as e:
            logger.error(f"检索所有实体类型失败: {e}")
            raise
    
    def generate_instruction_template(self, query: str, language: str, 
                                    top_k_entities: int = None, 
                                    top_k_sentences: int = None) -> str:
        """
        生成完整的指令模板
        
        Args:
            query: 查询句子
            language: 语种
            top_k_entities: 每个实体类型返回的实体数量
            top_k_sentences: 返回的示例句子数量
            
        Returns:
            str: 完整的指令模板
        """
        try:
            # 检索相似句子
            similar_sentences = self.retrieve_similar_sentences(
                query=query,
                language=language,
                top_k=top_k_sentences
            )
            
            # 检索各类型实体
            entity_results = self.retrieve_all_entity_types(
                query=query,
                language=language,
                top_k=top_k_entities
            )
            
            # 生成实体类型格式化文本
            entity_type_texts = []
            for entity_type in ENTITY_TYPES:
                entities = entity_results.get(entity_type, [])
                entity_examples = [item["entity_text"] for item in entities[:5]]  # 最多5个示例
                
                if entity_examples:
                    examples_text = ", \n       ".join(entity_examples)
                    entity_type_text = ENTITY_TYPE_FORMAT.format(
                        entity_type=entity_type,
                        examples=examples_text
                    )
                else:
                    entity_type_text = f"- {entity_type}: \n  e.g. (no examples available)"
                
                entity_type_texts.append(entity_type_text)
            
            entity_types_formatted = "\n".join(entity_type_texts)
            
            # 生成示例文本
            examples_text = ""
            for i, sentence_info in enumerate(similar_sentences[:5]):  # 最多5个示例
                sentence = sentence_info["sentence_text"]
                ner_labels = sentence_info["ner_labels"]
                
                examples_text += f"Input: {sentence}\n"
                examples_text += f"Output: {ner_labels}\n"
                
                if i < len(similar_sentences) - 1 and i < 4:  # 不是最后一个且没超过5个
                    examples_text += "\n"
            
            # 填充模板
            instruction = INSTRUCTION_TEMPLATE.format(
                entity_types=entity_types_formatted,
                examples=examples_text
            )
            
            return instruction + query
            
        except Exception as e:
            logger.error(f"生成指令模板失败: {e}")
            raise
    
    def retrieve_and_format(self, query: str, language: str) -> Dict[str, Any]:
        """
        执行完整的检索和格式化流程
        
        Args:
            query: 查询句子
            language: 语种
            
        Returns:
            Dict: 包含检索结果和格式化指令的完整结果
        """
        try:
            logger.info(f"开始检索流程，查询: {query}, 语种: {language}")
            
            # 检索相似句子
            similar_sentences = self.retrieve_similar_sentences(query, language)
            
            # 检索各类型实体
            entity_results = self.retrieve_all_entity_types(query, language)
            
            # 生成指令模板
            instruction = self.generate_instruction_template(query, language)
            
            # 统计信息
            total_entities = sum(len(entities) for entities in entity_results.values())
            
            result = {
                "query": query,
                "language": language,
                "similar_sentences": similar_sentences,
                "entity_results": entity_results,
                "instruction_template": instruction,
                "statistics": {
                    "total_similar_sentences": len(similar_sentences),
                    "total_entities_found": total_entities,
                    "entity_types_found": len([k for k, v in entity_results.items() if v])
                }
            }
            
            logger.info(f"检索完成，找到 {len(similar_sentences)} 个相似句子，{total_entities} 个实体")
            
            return result
            
        except Exception as e:
            logger.error(f"检索和格式化流程失败: {e}")
            raise
    
    def batch_retrieve(self, queries: List[str], language: str) -> List[Dict[str, Any]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            language: 语种
            
        Returns:
            List[Dict]: 批量检索结果
        """
        try:
            results = []
            
            for i, query in enumerate(queries):
                logger.info(f"处理查询 {i+1}/{len(queries)}: {query}")
                
                try:
                    result = self.retrieve_and_format(query, language)
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理查询失败: {e}")
                    results.append({
                        "query": query,
                        "language": language,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"批量检索失败: {e}")
            raise