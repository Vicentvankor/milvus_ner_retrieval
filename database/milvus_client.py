"""
Milvus数据库客户端
负责与Milvus向量数据库的交互，支持本地和远程模式
"""

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import os

from ..config import MILVUS_CONFIG, DATABASE_CONFIG, RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus数据库客户端类
    封装Milvus数据库的连接和操作，支持本地文件和远程服务器两种模式
    """
    
    def __init__(self, host: str = None, port: str = None, local_db_path: str = None):
        """
        初始化Milvus客户端
        
        Args:
            host: Milvus服务器地址（远程模式）
            port: Milvus服务器端口（远程模式）
            local_db_path: 本地数据库路径（本地模式）
        """
        self.mode = MILVUS_CONFIG.get("mode", "local")
        self.host = host or MILVUS_CONFIG["host"]
        self.port = port or MILVUS_CONFIG["port"]
        self.local_db_path = local_db_path or MILVUS_CONFIG["local_db_path"]
        self.connection_alias = "default"
        self._connect()
    
    def _connect(self):
        """
        连接到Milvus数据库
        """
        try:
            if self.mode == "local":
                # 本地文件模式
                os.makedirs(self.local_db_path, exist_ok=True)
                connections.connect(
                    alias=self.connection_alias,
                    uri=f"file://{self.local_db_path}"
                )
                logger.info(f"成功连接到本地Milvus数据库: {self.local_db_path}")
            else:
                # 远程服务器模式
                connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port
                )
                logger.info(f"成功连接到远程Milvus服务器 {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def create_entity_collection(self, language: str) -> str:
        """
        创建实体集合
        
        Args:
            language: 语种代码
            
        Returns:
            str: 集合名称
        """
        collection_name = f"{DATABASE_CONFIG['entity_db_prefix']}{language}"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            logger.info(f"实体集合 {collection_name} 已存在")
            return collection_name
        
        try:
            # 定义字段
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True
                ),
                FieldSchema(
                    name="entity_embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=DATABASE_CONFIG["vector_dim"]
                ),
                FieldSchema(
                    name="entity_text",
                    dtype=DataType.VARCHAR,
                    max_length=1000
                ),
                FieldSchema(
                    name="entity_type",
                    dtype=DataType.VARCHAR,
                    max_length=100
                )
            ]
            
            # 创建集合schema
            schema = CollectionSchema(
                fields=fields,
                description=f"实体向量数据库 - {language}"
            )
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 创建索引
            index_params = {
                "metric_type": DATABASE_CONFIG["metric_type"],
                "index_type": DATABASE_CONFIG["index_type"],
                "params": {"nlist": DATABASE_CONFIG["nlist"]}
            }
            
            collection.create_index(
                field_name="entity_embedding",
                index_params=index_params
            )
            
            logger.info(f"成功创建实体集合: {collection_name}")
            return collection_name
            
        except Exception as e:
            logger.error(f"创建实体集合失败: {e}")
            raise
    
    def create_sentence_collection(self, language: str) -> str:
        """
        创建句子集合
        
        Args:
            language: 语种代码
            
        Returns:
            str: 集合名称
        """
        collection_name = f"{DATABASE_CONFIG['sentence_db_prefix']}{language}"
        
        # 检查集合是否已存在
        if utility.has_collection(collection_name):
            logger.info(f"句子集合 {collection_name} 已存在")
            return collection_name
        
        try:
            # 定义字段
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True
                ),
                FieldSchema(
                    name="sentence_embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=DATABASE_CONFIG["vector_dim"]
                ),
                FieldSchema(
                    name="sentence_text",
                    dtype=DataType.VARCHAR,
                    max_length=2000
                ),
                FieldSchema(
                    name="ner_labels",
                    dtype=DataType.VARCHAR,
                    max_length=5000
                )
            ]
            
            # 创建集合schema
            schema = CollectionSchema(
                fields=fields,
                description=f"句子向量数据库 - {language}"
            )
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 创建索引
            index_params = {
                "metric_type": DATABASE_CONFIG["metric_type"],
                "index_type": DATABASE_CONFIG["index_type"],
                "params": {"nlist": DATABASE_CONFIG["nlist"]}
            }
            
            collection.create_index(
                field_name="sentence_embedding",
                index_params=index_params
            )
            
            logger.info(f"成功创建句子集合: {collection_name}")
            return collection_name
            
        except Exception as e:
            logger.error(f"创建句子集合失败: {e}")
            raise
    
    def insert_entities(self, collection_name: str, entities_data: List[Dict[str, Any]]):
        """
        插入实体数据
        
        Args:
            collection_name: 集合名称
            entities_data: 实体数据列表，每个元素包含entity_embedding, entity_text, entity_type
        """
        try:
            collection = Collection(collection_name)
            
            # 准备数据
            entity_embeddings = [data["entity_embedding"] for data in entities_data]
            entity_texts = [data["entity_text"] for data in entities_data]
            entity_types = [data["entity_type"] for data in entities_data]
            
            # 插入数据
            collection.insert([
                entity_embeddings,
                entity_texts,
                entity_types
            ])
            
            # 刷新数据
            collection.flush()
            
            logger.info(f"成功插入 {len(entities_data)} 条实体数据到 {collection_name}")
            
        except Exception as e:
            logger.error(f"插入实体数据失败: {e}")
            raise
    
    def insert_sentences(self, collection_name: str, sentences_data: List[Dict[str, Any]]):
        """
        插入句子数据
        
        Args:
            collection_name: 集合名称
            sentences_data: 句子数据列表，每个元素包含sentence_embedding, sentence_text, ner_labels
        """
        try:
            collection = Collection(collection_name)
            
            # 准备数据
            sentence_embeddings = [data["sentence_embedding"] for data in sentences_data]
            sentence_texts = [data["sentence_text"] for data in sentences_data]
            ner_labels = [data["ner_labels"] for data in sentences_data]
            
            # 插入数据
            collection.insert([
                sentence_embeddings,
                sentence_texts,
                ner_labels
            ])
            
            # 刷新数据
            collection.flush()
            
            logger.info(f"成功插入 {len(sentences_data)} 条句子数据到 {collection_name}")
            
        except Exception as e:
            logger.error(f"插入句子数据失败: {e}")
            raise
    
    def search_entities(self, collection_name: str, query_embedding: np.ndarray, 
                       entity_type: str = None, top_k: int = None) -> List[Dict[str, Any]]:
        """
        搜索相似实体
        
        Args:
            collection_name: 集合名称
            query_embedding: 查询向量
            entity_type: 实体类型过滤条件
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 构建搜索参数
            search_params = RETRIEVAL_CONFIG["search_params"]
            limit = top_k or RETRIEVAL_CONFIG["top_k_entities"]
            
            # 构建过滤表达式
            expr = None
            if entity_type:
                expr = f'entity_type == "{entity_type}"'
            
            # 执行搜索
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="entity_embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["entity_text", "entity_type"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "entity_text": hit.entity.get("entity_text"),
                        "entity_type": hit.entity.get("entity_type"),
                        "score": hit.score,
                        "distance": hit.distance
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            raise
    
    def search_sentences(self, collection_name: str, query_embedding: np.ndarray, 
                        top_k: int = None) -> List[Dict[str, Any]]:
        """
        搜索相似句子
        
        Args:
            collection_name: 集合名称
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 构建搜索参数
            search_params = RETRIEVAL_CONFIG["search_params"]
            limit = top_k or RETRIEVAL_CONFIG["top_k_sentences"]
            
            # 执行搜索
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="sentence_embedding",
                param=search_params,
                limit=limit,
                output_fields=["sentence_text", "ner_labels"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "sentence_text": hit.entity.get("sentence_text"),
                        "ner_labels": hit.entity.get("ner_labels"),
                        "score": hit.score,
                        "distance": hit.distance
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索句子失败: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict: 集合信息
        """
        try:
            if not utility.has_collection(collection_name):
                return {"exists": False}
            
            collection = Collection(collection_name)
            
            return {
                "exists": True,
                "name": collection_name,
                "description": collection.description,
                "num_entities": collection.num_entities,
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            raise
    
    def drop_collection(self, collection_name: str):
        """
        删除集合
        
        Args:
            collection_name: 集合名称
        """
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"成功删除集合: {collection_name}")
            else:
                logger.warning(f"集合不存在: {collection_name}")
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise
    
    def close_connection(self):
        """
        关闭连接
        """
        try:
            connections.disconnect(alias=self.connection_alias)
            logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开连接失败: {e}")