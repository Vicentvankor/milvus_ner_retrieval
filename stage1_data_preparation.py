#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段一：数据准备阶段
负责数据加载、处理、向量化和存储到Milvus数据库

主要功能：
1. 加载和解析原始JSON数据文件
2. 使用LLM2Vec模型生成文本向量嵌入
3. 创建Milvus数据库集合和索引
4. 批量存储实体和句子数据到向量数据库
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    STAGE1_DATA_CONFIG, 
    STAGE1_MILVUS_CONFIG, 
    STAGE1_DATABASE_CONFIG,
    STAGE1_MODEL_CONFIG,
    LOGGING_CONFIG,
    ENTITY_TYPES,
    SUPPORTED_LANGUAGES
)
from database.milvus_client import MilvusClient
from core.embedding_model import EmbeddingModel
from processors.data_processor import DataProcessor


def setup_logging():
    """
    设置日志配置
    """
    # 创建日志目录
    log_dir = Path(LOGGING_CONFIG["log_file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(
                LOGGING_CONFIG["log_file"], 
                encoding='utf-8'
            ) if LOGGING_CONFIG["enable_file_logging"] else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class Stage1DataPreparation:
    """
    阶段一：数据准备处理器
    负责完整的数据准备流程
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        初始化数据准备处理器
        
        Args:
            config_override: 配置覆盖参数
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 合并配置
        self.data_config = {**STAGE1_DATA_CONFIG}
        self.milvus_config = {**STAGE1_MILVUS_CONFIG}
        self.database_config = {**STAGE1_DATABASE_CONFIG}
        self.model_config = {**STAGE1_MODEL_CONFIG}
        
        if config_override:
            self.data_config.update(config_override.get('data', {}))
            self.milvus_config.update(config_override.get('milvus', {}))
            self.database_config.update(config_override.get('database', {}))
            self.model_config.update(config_override.get('model', {}))
        
        self.logger.info("🚀 初始化阶段一：数据准备处理器")
        self.logger.info(f"📁 实体数据文件: {self.data_config['entities_file']}")
        self.logger.info(f"📁 句子数据文件: {self.data_config['sentences_file']}")
        self.logger.info(f"🌍 目标语言: {self.data_config['target_languages']}")
        
        # 初始化组件
        self.milvus_client = None
        self.embedding_model = None
        self.data_processor = None
    
    def initialize_components(self):
        """
        初始化核心组件
        """
        try:
            self.logger.info("🔧 初始化核心组件...")
            
            # 1. 初始化Milvus客户端
            self.logger.info("📊 初始化Milvus数据库客户端...")
            self.milvus_client = MilvusClient(
                mode=self.milvus_config["mode"],
                local_db_path=self.milvus_config.get("local_db_path"),
                host=self.milvus_config.get("host"),
                port=self.milvus_config.get("port")
            )
            self.logger.info("✅ Milvus客户端初始化完成")
            
            # 2. 初始化嵌入模型
            self.logger.info("🤖 初始化LLM2Vec嵌入模型...")
            self.embedding_model = EmbeddingModel(
                model_name=self.model_config["model_name"],
                supervised_model=self.model_config.get("supervised_model"),
                pooling_mode=self.model_config["pooling_mode"],
                max_length=self.model_config["max_length"],
                device=self.model_config["device"]
            )
            self.logger.info("✅ 嵌入模型初始化完成")
            
            # 3. 初始化数据处理器
            self.logger.info("⚙️ 初始化数据处理器...")
            self.data_processor = DataProcessor(self.embedding_model)
            self.logger.info("✅ 数据处理器初始化完成")
            
            self.logger.info("🎉 所有核心组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    def create_database_collections(self):
        """
        创建数据库集合和索引
        """
        try:
            self.logger.info("🏗️ 开始创建数据库集合...")
            
            target_languages = self.data_config["target_languages"]
            
            for language in target_languages:
                self.logger.info(f"📚 创建 {language} 语言的数据库集合...")
                
                # 创建实体集合
                entity_collection = self.milvus_client.create_entity_collection(
                    language=language,
                    vector_dim=self.database_config["vector_dim"],
                    index_type=self.database_config["index_type"],
                    metric_type=self.database_config["metric_type"]
                )
                self.logger.info(f"✅ 实体集合创建完成: {entity_collection}")
                
                # 创建句子集合
                sentence_collection = self.milvus_client.create_sentence_collection(
                    language=language,
                    vector_dim=self.database_config["vector_dim"],
                    index_type=self.database_config["index_type"],
                    metric_type=self.database_config["metric_type"]
                )
                self.logger.info(f"✅ 句子集合创建完成: {sentence_collection}")
            
            self.logger.info("🎉 所有数据库集合创建完成")
            
        except Exception as e:
            self.logger.error(f"❌ 数据库集合创建失败: {e}")
            raise
    
    def process_and_store_entities(self):
        """
        处理和存储实体数据
        """
        try:
            self.logger.info("🔄 开始处理实体数据...")
            
            entities_file = self.data_config["entities_file"]
            if not os.path.exists(entities_file):
                raise FileNotFoundError(f"实体数据文件不存在: {entities_file}")
            
            # 处理实体数据
            self.logger.info(f"📖 加载实体数据文件: {entities_file}")
            processed_data = self.data_processor.process_entities_data(entities_file)
            
            target_languages = self.data_config["target_languages"]
            batch_size = self.data_config["batch_size_entities"]
            processing_delay = self.data_config["processing_delay"]
            
            for language in target_languages:
                if language not in processed_data:
                    self.logger.warning(f"⚠️ 语言 {language} 的实体数据不存在，跳过")
                    continue
                
                language_data = processed_data[language]
                if not language_data:
                    self.logger.warning(f"⚠️ 语言 {language} 的实体数据为空，跳过")
                    continue
                
                self.logger.info(f"💾 开始存储 {language} 语言的实体数据，共 {len(language_data)} 条")
                
                # 分批存储数据
                collection_name = f"{self.database_config['entity_db_prefix']}{language}"
                total_batches = (len(language_data) + batch_size - 1) // batch_size
                
                for i in range(0, len(language_data), batch_size):
                    batch_data = language_data[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    try:
                        self.milvus_client.insert_entities(collection_name, batch_data)
                        self.logger.info(f"✅ 批次 {batch_num}/{total_batches}: 存储 {len(batch_data)} 条实体记录")
                        
                        # 添加处理延迟
                        if processing_delay > 0:
                            time.sleep(processing_delay)
                        
                    except Exception as e:
                        self.logger.error(f"❌ 批次 {batch_num} 存储失败: {e}")
                        raise
                
                self.logger.info(f"🎉 {language} 语言实体数据存储完成")
            
            self.logger.info("🎉 所有实体数据处理和存储完成")
            
        except Exception as e:
            self.logger.error(f"❌ 实体数据处理失败: {e}")
            raise
    
    def process_and_store_sentences(self):
        """
        处理和存储句子数据
        """
        try:
            self.logger.info("🔄 开始处理句子数据...")
            
            sentences_file = self.data_config["sentences_file"]
            if not os.path.exists(sentences_file):
                raise FileNotFoundError(f"句子数据文件不存在: {sentences_file}")
            
            # 处理句子数据
            self.logger.info(f"📖 加载句子数据文件: {sentences_file}")
            processed_data = self.data_processor.process_sentences_data(sentences_file)
            
            target_languages = self.data_config["target_languages"]
            batch_size = self.data_config["batch_size_sentences"]
            processing_delay = self.data_config["processing_delay"]
            
            for language in target_languages:
                if language not in processed_data:
                    self.logger.warning(f"⚠️ 语言 {language} 的句子数据不存在，跳过")
                    continue
                
                language_data = processed_data[language]
                if not language_data:
                    self.logger.warning(f"⚠️ 语言 {language} 的句子数据为空，跳过")
                    continue
                
                self.logger.info(f"💾 开始存储 {language} 语言的句子数据，共 {len(language_data)} 条")
                
                # 分批存储数据
                collection_name = f"{self.database_config['sentence_db_prefix']}{language}"
                total_batches = (len(language_data) + batch_size - 1) // batch_size
                
                for i in range(0, len(language_data), batch_size):
                    batch_data = language_data[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    try:
                        self.milvus_client.insert_sentences(collection_name, batch_data)
                        self.logger.info(f"✅ 批次 {batch_num}/{total_batches}: 存储 {len(batch_data)} 条句子记录")
                        
                        # 添加处理延迟
                        if processing_delay > 0:
                            time.sleep(processing_delay)
                        
                    except Exception as e:
                        self.logger.error(f"❌ 批次 {batch_num} 存储失败: {e}")
                        raise
                
                self.logger.info(f"🎉 {language} 语言句子数据存储完成")
            
            self.logger.info("🎉 所有句子数据处理和存储完成")
            
        except Exception as e:
            self.logger.error(f"❌ 句子数据处理失败: {e}")
            raise
    
    def get_database_statistics(self):
        """
        获取数据库统计信息
        """
        try:
            self.logger.info("📊 获取数据库统计信息...")
            
            statistics = {
                "total_entities": 0,
                "total_sentences": 0,
                "languages": {},
                "entity_types": ENTITY_TYPES,
                "supported_languages": SUPPORTED_LANGUAGES
            }
            
            target_languages = self.data_config["target_languages"]
            
            for language in target_languages:
                try:
                    # 获取实体集合统计
                    entity_collection = f"{self.database_config['entity_db_prefix']}{language}"
                    entity_count = self.milvus_client.get_collection_count(entity_collection)
                    
                    # 获取句子集合统计
                    sentence_collection = f"{self.database_config['sentence_db_prefix']}{language}"
                    sentence_count = self.milvus_client.get_collection_count(sentence_collection)
                    
                    statistics["languages"][language] = {
                        "entities": entity_count,
                        "sentences": sentence_count
                    }
                    
                    statistics["total_entities"] += entity_count
                    statistics["total_sentences"] += sentence_count
                    
                    self.logger.info(f"📈 {language}: 实体 {entity_count} 条，句子 {sentence_count} 条")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 获取 {language} 统计信息失败: {e}")
                    statistics["languages"][language] = {
                        "entities": 0,
                        "sentences": 0,
                        "error": str(e)
                    }
            
            self.logger.info(f"📊 总计：实体 {statistics['total_entities']} 条，句子 {statistics['total_sentences']} 条")
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ 获取统计信息失败: {e}")
            raise
    
    def run_complete_preparation(self):
        """
        运行完整的数据准备流程
        """
        try:
            start_time = time.time()
            self.logger.info("🚀 开始完整的数据准备流程...")
            
            # 1. 初始化组件
            self.initialize_components()
            
            # 创建数据库集合
            self.create_database_collections()
            
            # 3. 处理和存储实体数据
            self.process_and_store_entities()
            
            # 4. 处理和存储句子数据
            self.process_and_store_sentences()
            
            # 5. 获取统计信息
            statistics = self.get_database_statistics()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info("🎉 数据准备流程完成！")
            self.logger.info(f"⏱️ 总耗时: {total_time:.2f} 秒")
            self.logger.info(f"📊 处理结果: {statistics['total_entities']} 个实体，{statistics['total_sentences']} 个句子")
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ 数据准备流程失败: {e}")
            raise
        finally:
            # 清理资源
            if self.milvus_client:
                self.milvus_client.close()
                self.logger.info("🔒 Milvus连接已关闭")


def main():
    """
    命令行主程序
    """
    parser = argparse.ArgumentParser(description="阶段一：数据准备处理器")
    parser.add_argument("--entities-file", type=str, help="实体数据文件路径")
    parser.add_argument("--sentences-file", type=str, help="句子数据文件路径")
    parser.add_argument("--languages", nargs="+", help="目标语言列表")
    parser.add_argument("--local-db-path", type=str, help="本地数据库路径")
    parser.add_argument("--batch-size-entities", type=int, help="实体批处理大小")
    parser.add_argument("--batch-size-sentences", type=int, help="句子批处理大小")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.log_level:
        LOGGING_CONFIG["level"] = args.log_level
    
    # 设置日志
    logger = setup_logging()
    
    try:
        # 构建配置覆盖
        config_override = {}
        
        if args.entities_file or args.sentences_file or args.languages:
            config_override["data"] = {}
            if args.entities_file:
                config_override["data"]["entities_file"] = args.entities_file
            if args.sentences_file:
                config_override["data"]["sentences_file"] = args.sentences_file
            if args.languages:
                config_override["data"]["target_languages"] = args.languages
            if args.batch_size_entities:
                config_override["data"]["batch_size_entities"] = args.batch_size_entities
            if args.batch_size_sentences:
                config_override["data"]["batch_size_sentences"] = args.batch_size_sentences
        
        if args.local_db_path:
            config_override["milvus"] = {"local_db_path": args.local_db_path}
        
        if args.device:
            config_override["model"] = {"device": args.device}
        
        # 创建并运行数据准备处理器
        processor = Stage1DataPreparation(config_override)
        statistics = processor.run_complete_preparation()
        
        logger.info("✅ 数据准备阶段完成")
        logger.info(f"📊 最终统计: {json.dumps(statistics, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()