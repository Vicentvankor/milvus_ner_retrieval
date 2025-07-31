"""
数据库管理器
负责数据库的初始化、数据导入和管理
"""

import logging
from typing import Dict, List, Any
import time

from ..config import SUPPORTED_LANGUAGES
from .milvus_client import MilvusClient
from ..core.embedding_model import EmbeddingModel
from ..processors.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    数据库管理器类
    负责管理Milvus数据库的创建、初始化和数据导入
    """
    
    def __init__(self, milvus_client: MilvusClient, embedding_model: EmbeddingModel):
        """
        初始化数据库管理器
        
        Args:
            milvus_client: Milvus客户端实例
            embedding_model: 嵌入模型实例
        """
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model
        self.data_processor = DataProcessor(embedding_model)
    
    def initialize_all_collections(self, languages: List[str] = None):
        """
        初始化所有语种的集合
        
        Args:
            languages: 要初始化的语种列表，如果为None则初始化所有支持的语种
        """
        try:
            target_languages = languages or SUPPORTED_LANGUAGES
            
            logger.info(f"开始初始化 {len(target_languages)} 个语种的集合...")
            
            for language in target_languages:
                logger.info(f"初始化 {language} 语种的集合...")
                
                # 创建实体集合
                entity_collection = self.milvus_client.create_entity_collection(language)
                logger.info(f"实体集合创建完成: {entity_collection}")
                
                # 创建句子集合
                sentence_collection = self.milvus_client.create_sentence_collection(language)
                logger.info(f"句子集合创建完成: {sentence_collection}")
            
            logger.info("所有集合初始化完成")
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise
    
    def import_entities_data(self, entities_file: str, languages: List[str] = None):
        """
        导入实体数据
        
        Args:
            entities_file: 实体数据文件路径
            languages: 要导入的语种列表，如果为None则导入所有语种
        """
        try:
            logger.info(f"开始导入实体数据: {entities_file}")
            
            # 处理实体数据
            processed_data = self.data_processor.process_entities_data(entities_file)
            
            target_languages = languages or list(processed_data.keys())
            
            for language in target_languages:
                if language not in processed_data:
                    logger.warning(f"语种 {language} 的实体数据不存在，跳过")
                    continue
                
                language_data = processed_data[language]
                if not language_data:
                    logger.warning(f"语种 {language} 的实体数据为空，跳过")
                    continue
                
                logger.info(f"开始导入 {language} 语种的实体数据，共 {len(language_data)} 条")
                
                # 分批导入数据
                batch_size = 1000
                collection_name = f"entity_{language}"
                
                for i in range(0, len(language_data), batch_size):
                    batch_data = language_data[i:i + batch_size]
                    
                    try:
                        self.milvus_client.insert_entities(collection_name, batch_data)
                        logger.info(f"导入批次 {i//batch_size + 1}: {len(batch_data)} 条记录")
                        
                        # 短暂延迟，避免过载
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"导入批次失败: {e}")
                        raise
                
                logger.info(f"{language} 语种实体数据导入完成")
            
            logger.info("所有实体数据导入完成")
            
        except Exception as e:
            logger.error(f"导入实体数据失败: {e}")
            raise
    
    def import_sentences_data(self, sentences_file: str, languages: List[str] = None):
        """
        导入句子数据
        
        Args:
            sentences_file: 句子数据文件路径
            languages: 要导入的语种列表，如果为None则导入所有语种
        """
        try:
            logger.info(f"开始导入句子数据: {sentences_file}")
            
            # 处理句子数据
            processed_data = self.data_processor.process_sentences_data(sentences_file)
            
            target_languages = languages or list(processed_data.keys())
            
            for language in target_languages:
                if language not in processed_data:
                    logger.warning(f"语种 {language} 的句子数据不存在，跳过")
                    continue
                
                language_data = processed_data[language]
                if not language_data:
                    logger.warning(f"语种 {language} 的句子数据为空，跳过")
                    continue
                
                logger.info(f"开始导入 {language} 语种的句子数据，共 {len(language_data)} 条")
                
                # 分批导入数据
                batch_size = 500
                collection_name = f"sentence_{language}"
                
                for i in range(0, len(language_data), batch_size):
                    batch_data = language_data[i:i + batch_size]
                    
                    try:
                        self.milvus_client.insert_sentences(collection_name, batch_data)
                        logger.info(f"导入批次 {i//batch_size + 1}: {len(batch_data)} 条记录")
                        
                        # 短暂延迟，避免过载
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"导入批次失败: {e}")
                        raise
                
                logger.info(f"{language} 语种句子数据导入完成")
            
            logger.info("所有句子数据导入完成")
            
        except Exception as e:
            logger.error(f"导入句子数据失败: {e}")
            raise
    
    def setup_database(self, entities_file: str, sentences_file: str, 
                      languages: List[str] = None):
        """
        完整的数据库设置流程
        
        Args:
            entities_file: 实体数据文件路径
            sentences_file: 句子数据文件路径
            languages: 要设置的语种列表，如果为None则设置所有语种
        """
        try:
            logger.info("开始完整数据库设置流程...")
            
            # 1. 初始化集合
            self.initialize_all_collections(languages)
            
            # 2. 导入实体数据
            self.import_entities_data(entities_file, languages)
            
            # 3. 导入句子数据
            self.import_sentences_data(sentences_file, languages)
            
            logger.info("数据库设置完成")
            
        except Exception as e:
            logger.error(f"数据库设置失败: {e}")
            raise
    
    def get_database_statistics(self, languages: List[str] = None) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Args:
            languages: 要统计的语种列表，如果为None则统计所有语种
            
        Returns:
            Dict: 统计信息
        """
        try:
            target_languages = languages or SUPPORTED_LANGUAGES
            statistics = {
                "languages": {},
                "total_entities": 0,
                "total_sentences": 0
            }
            
            for language in target_languages:
                entity_collection = f"entity_{language}"
                sentence_collection = f"sentence_{language}"
                
                # 获取实体集合信息
                entity_info = self.milvus_client.get_collection_info(entity_collection)
                
                # 获取句子集合信息
                sentence_info = self.milvus_client.get_collection_info(sentence_collection)
                
                if entity_info["exists"] or sentence_info["exists"]:
                    language_stats = {
                        "entity_collection": entity_info,
                        "sentence_collection": sentence_info,
                        "entity_count": entity_info.get("num_entities", 0) if entity_info["exists"] else 0,
                        "sentence_count": sentence_info.get("num_entities", 0) if sentence_info["exists"] else 0
                    }
                    
                    statistics["languages"][language] = language_stats
                    statistics["total_entities"] += language_stats["entity_count"]
                    statistics["total_sentences"] += language_stats["sentence_count"]
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取数据库统计信息失败: {e}")
            raise
    
    def cleanup_collections(self, languages: List[str] = None):
        """
        清理集合
        
        Args:
            languages: 要清理的语种列表，如果为None则清理所有语种
        """
        try:
            target_languages = languages or SUPPORTED_LANGUAGES
            
            logger.info(f"开始清理 {len(target_languages)} 个语种的集合...")
            
            for language in target_languages:
                entity_collection = f"entity_{language}"
                sentence_collection = f"sentence_{language}"
                
                # 删除实体集合
                self.milvus_client.drop_collection(entity_collection)
                
                # 删除句子集合
                self.milvus_client.drop_collection(sentence_collection)
                
                logger.info(f"{language} 语种的集合已清理")
            
            logger.info("集合清理完成")
            
        except Exception as e:
            logger.error(f"清理集合失败: {e}")
            raise