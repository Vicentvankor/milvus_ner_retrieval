#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二：检索服务阶段
负责查询处理、向量检索、指令生成和结果返回

主要功能：
1. 接收用户输入的句子和语言参数
2. 执行语义相似度检索（句子和实体）
3. 生成完整的NER指令模板
4. 返回格式化的检索结果
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    STAGE2_MILVUS_CONFIG,
    STAGE2_RETRIEVAL_CONFIG,
    STAGE2_MODEL_CONFIG,
    STAGE2_OUTPUT_CONFIG,
    LOGGING_CONFIG,
    ENTITY_TYPES,
    SUPPORTED_LANGUAGES,
    INSTRUCTION_TEMPLATE,
    ENTITY_TYPE_FORMAT
)
from database.milvus_client import MilvusClient
from core.embedding_model import EmbeddingModel
from core.retrieval_engine import RetrievalEngine


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


class Stage2RetrievalService:
    """
    阶段二：检索服务处理器
    负责完整的检索和指令生成流程
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        初始化检索服务处理器
        
        Args:
            config_override: 配置覆盖参数
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 合并配置
        self.milvus_config = {**STAGE2_MILVUS_CONFIG}
        self.retrieval_config = {**STAGE2_RETRIEVAL_CONFIG}
        self.model_config = {**STAGE2_MODEL_CONFIG}
        self.output_config = {**STAGE2_OUTPUT_CONFIG}
        
        if config_override:
            self.milvus_config.update(config_override.get('milvus', {}))
            self.retrieval_config.update(config_override.get('retrieval', {}))
            self.model_config.update(config_override.get('model', {}))
            self.output_config.update(config_override.get('output', {}))
        
        self.logger.info("🚀 初始化阶段二：检索服务处理器")
        self.logger.info(f"🔍 检索配置: Top-K实体={self.retrieval_config['top_k_entities']}, Top-K句子={self.retrieval_config['top_k_sentences']}")
        self.logger.info(f"📊 数据库模式: {self.milvus_config['mode']}")
        
        # 初始化组件
        self.milvus_client = None
        self.embedding_model = None
        self.retrieval_engine = None
    
    def initialize_components(self):
        """
        初始化核心组件
        """
        try:
            self.logger.info("🔧 初始化检索服务组件...")
            
            # 1. 初始化Milvus客户端
            self.logger.info("📊 连接Milvus数据库...")
            self.milvus_client = MilvusClient(
                mode=self.milvus_config["mode"],
                local_db_path=self.milvus_config.get("local_db_path"),
                host=self.milvus_config.get("host"),
                port=self.milvus_config.get("port")
            )
            self.logger.info("✅ Milvus数据库连接成功")
            
            # 2. 初始化嵌入模型
            self.logger.info("🤖 加载LLM2Vec嵌入模型...")
            self.embedding_model = EmbeddingModel(
                model_name=self.model_config["model_name"],
                supervised_model=self.model_config.get("supervised_model"),
                pooling_mode=self.model_config["pooling_mode"],
                max_length=self.model_config["max_length"],
                device=self.model_config["device"]
            )
            self.logger.info("✅ 嵌入模型加载完成")
            
            # 3. 初始化检索引擎
            self.logger.info("⚙️ 初始化检索引擎...")
            self.retrieval_engine = RetrievalEngine(
                milvus_client=self.milvus_client,
                embedding_model=self.embedding_model
            )
            self.logger.info("✅ 检索引擎初始化完成")
            
            self.logger.info("🎉 所有检索服务组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {e}")
            raise
    
    def validate_input(self, query: str, language: str) -> bool:
        """
        验证输入参数
        
        Args:
            query: 查询句子
            language: 语言代码
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 验证查询句子
            if not query or not query.strip():
                self.logger.error("❌ 查询句子不能为空")
                return False
            
            # 验证语言代码
            if language not in SUPPORTED_LANGUAGES:
                self.logger.error(f"❌ 不支持的语言: {language}，支持的语言: {SUPPORTED_LANGUAGES}")
                return False
            
            # 验证句子长度
            if len(query.strip()) > self.model_config["max_length"]:
                self.logger.warning(f"⚠️ 查询句子长度超过限制 ({self.model_config['max_length']})，将被截断")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 输入验证失败: {e}")
            return False
    
    def retrieve_similar_sentences(self, query: str, language: str) -> List[Dict[str, Any]]:
        """
        检索相似句子
        
        Args:
            query: 查询句子
            language: 语言代码
            
        Returns:
            List[Dict]: 相似句子列表
        """
        try:
            self.logger.info(f"🔍 检索相似句子 (语言: {language}, Top-K: {self.retrieval_config['top_k_sentences']})")
            
            start_time = time.time()
            
            similar_sentences = self.retrieval_engine.retrieve_similar_sentences(
                query=query,
                language=language,
                top_k=self.retrieval_config["top_k_sentences"]
            )
            
            retrieval_time = time.time() - start_time
            
            # 过滤低相似度结果
            threshold = self.retrieval_config["similarity_threshold"]
            if threshold > 0:
                similar_sentences = [
                    sentence for sentence in similar_sentences 
                    if sentence.get("score", 0) >= threshold
                ]
            
            self.logger.info(f"✅ 检索到 {len(similar_sentences)} 个相似句子 (耗时: {retrieval_time:.3f}s)")
            
            return similar_sentences
            
        except Exception as e:
            self.logger.error(f"❌ 相似句子检索失败: {e}")
            raise
    
    def retrieve_entities_by_types(self, query: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        按类型检索相关实体
        
        Args:
            query: 查询句子
            language: 语言代码
            
        Returns:
            Dict: 按实体类型分组的检索结果
        """
        try:
            self.logger.info(f"🔍 检索相关实体 (语言: {language}, 实体类型: {len(ENTITY_TYPES)})")
            
            start_time = time.time()
            
            entity_results = self.retrieval_engine.retrieve_all_entity_types(
                query=query,
                language=language,
                top_k=self.retrieval_config["top_k_entities"]
            )
            
            retrieval_time = time.time() - start_time
            
            # 过滤低相似度结果
            threshold = self.retrieval_config["similarity_threshold"]
            if threshold > 0:
                for entity_type in entity_results:
                    entity_results[entity_type] = [
                        entity for entity in entity_results[entity_type]
                        if entity.get("score", 0) >= threshold
                    ]
            
            # 统计检索结果
            total_entities = sum(len(entities) for entities in entity_results.values())
            found_types = len([k for k, v in entity_results.items() if v])
            
            self.logger.info(f"✅ 检索到 {total_entities} 个实体，覆盖 {found_types} 个类型 (耗时: {retrieval_time:.3f}s)")
            
            return entity_results
            
        except Exception as e:
            self.logger.error(f"❌ 实体检索失败: {e}")
            raise
    
    def generate_instruction_template(self, query: str, similar_sentences: List[Dict], 
                                    entity_results: Dict[str, List[Dict]]) -> str:
        """
        生成完整的指令模板
        
        Args:
            query: 查询句子
            similar_sentences: 相似句子列表
            entity_results: 实体检索结果
            
        Returns:
            str: 完整的指令模板
        """
        try:
            self.logger.info("📝 生成NER指令模板...")
            
            # 1. 生成实体类型格式化文本
            entity_type_texts = []
            max_entities_per_type = self.retrieval_config["max_entities_per_type"]
            
            for entity_type in ENTITY_TYPES:
                entities = entity_results.get(entity_type, [])
                entity_examples = [
                    item["entity_text"] for item in entities[:max_entities_per_type]
                ]
                
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
            
            # 2. 生成示例文本
            examples_text = ""
            max_examples = self.retrieval_config["max_examples_in_instruction"]
            
            for i, sentence_info in enumerate(similar_sentences[:max_examples]):
                sentence = sentence_info["sentence_text"]
                ner_labels = sentence_info["ner_labels"]
                
                examples_text += f"Input: {sentence}\n"
                examples_text += f"Output: {ner_labels}\n"
                
                if i < len(similar_sentences) - 1 and i < max_examples - 1:
                    examples_text += "\n"
            
            # 3. 填充模板
            instruction = INSTRUCTION_TEMPLATE.format(
                entity_types=entity_types_formatted,
                examples=examples_text
            )
            
            # 4. 添加查询句子
            full_instruction = instruction + query
            
            self.logger.info("✅ 指令模板生成完成")
            
            return full_instruction
            
        except Exception as e:
            self.logger.error(f"❌ 指令模板生成失败: {e}")
            raise
    
    def process_single_query(self, query: str, language: str) -> Dict[str, Any]:
        """
        处理单个查询请求
        
        Args:
            query: 查询句子
            language: 语言代码
            
        Returns:
            Dict: 完整的检索结果
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"🔍 开始处理查询: '{query}' (语言: {language})")
            
            # 1. 验证输入
            if not self.validate_input(query, language):
                raise ValueError("输入验证失败")
            
            # 2. 检索相似句子
            similar_sentences = self.retrieve_similar_sentences(query, language)
            
            # 3. 检索相关实体
            entity_results = self.retrieve_entities_by_types(query, language)
            
            # 4. 生成指令模板
            instruction_template = self.generate_instruction_template(
                query, similar_sentences, entity_results
            )
            
            # 5. 构建结果
            processing_time = time.time() - start_time
            
            result = {
                "query": query,
                "language": language,
                "instruction_template": instruction_template,
                "processing_time": processing_time
            }
            
            # 根据配置添加详细信息
            if self.output_config["include_similarity_scores"]:
                result["similar_sentences"] = similar_sentences
                result["entity_results"] = entity_results
            
            if self.output_config["include_statistics"]:
                total_entities = sum(len(entities) for entities in entity_results.values())
                result["statistics"] = {
                    "total_similar_sentences": len(similar_sentences),
                    "total_entities_found": total_entities,
                    "entity_types_found": len([k for k, v in entity_results.items() if v]),
                    "processing_time": processing_time
                }
            
            self.logger.info(f"✅ 查询处理完成 (耗时: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 查询处理失败: {e}")
            raise
    
    def process_jsonl_files(self, jsonl_directories: List[str], input_field: str = 'input', 
                           output_directory: str = 'data/RetriAll_output') -> Dict[str, Any]:
        """
        处理JSONL文件，为每个句子生成新的指令
        
        Args:
            jsonl_directories: JSONL文件目录列表
            input_field: 输入字段名
            output_directory: 输出目录
            
        Returns:
            Dict: 处理统计信息
        """
        try:
            self.logger.info("📂 开始处理JSONL文件...")
            
            # 创建输出目录
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            processing_stats = {
                'files_processed': 0,
                'total_entries': 0,
                'successful_entries': 0,
                'failed_entries': 0,
                'output_files': []
            }
            
            # 处理每个目录
            for directory in jsonl_directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    self.logger.warning(f"⚠️ 目录不存在: {directory}")
                    continue
                
                # 处理目录中的所有JSONL文件
                for jsonl_file in dir_path.glob("*.jsonl"):
                    self.logger.info(f"📄 处理文件: {jsonl_file}")
                    
                    # 确定语言
                    language = self._extract_language_from_filename(jsonl_file.name)
                    
                    # 创建输出文件路径
                    output_file = output_path / f"enhanced_{jsonl_file.name}"
                    
                    # 处理单个JSONL文件
                    file_stats = self._process_single_jsonl_file(
                        jsonl_file, output_file, input_field, language
                    )
                    
                    # 更新统计信息
                    processing_stats['files_processed'] += 1
                    processing_stats['total_entries'] += file_stats['total_entries']
                    processing_stats['successful_entries'] += file_stats['successful_entries']
                    processing_stats['failed_entries'] += file_stats['failed_entries']
                    processing_stats['output_files'].append(str(output_file))
            
            self.logger.info(f"🎉 JSONL文件处理完成: {processing_stats}")
            return processing_stats
            
        except Exception as e:
            self.logger.error(f"❌ 处理JSONL文件时出错: {e}")
            raise
    
    def _extract_language_from_filename(self, filename: str) -> str:
        """从文件名中提取语言代码"""
        language_map = {
            'en': 'en', 'de': 'de', 'es': 'es', 'fr': 'fr',
            'ja': 'ja', 'ko': 'ko', 'ru': 'ru', 'zh': 'zh'
        }
        
        for lang_code in language_map:
            if f"_{lang_code}." in filename or f"_{lang_code}_" in filename:
                return lang_code
        
        return 'en'  # 默认语言
    
    def _process_single_jsonl_file(self, input_file: Path, output_file: Path, 
                                  input_field: str, language: str) -> Dict[str, int]:
        """
        处理单个JSONL文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            input_field: 输入字段名
            language: 语言代码
            
        Returns:
            Dict: 文件处理统计信息
        """
        stats = {
            'total_entries': 0,
            'successful_entries': 0,
            'failed_entries': 0
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8') as infile, \
                 open(output_file, 'w', encoding='utf-8') as outfile:
                
                for line_num, line in enumerate(infile, 1):
                    try:
                        # 解析JSON行
                        data = json.loads(line.strip())
                        stats['total_entries'] += 1
                        
                        # 获取输入文本
                        if input_field not in data:
                            self.logger.warning(f"⚠️ 文件 {input_file} 第 {line_num} 行缺少字段 '{input_field}'")
                            stats['failed_entries'] += 1
                            continue
                        
                        query_text = data[input_field]
                        
                        # 生成新的指令
                        result = self.process_single_query(query_text, language)
                        
                        # 创建输出数据
                        output_data = data.copy() if self.output_config.get('preserve_original_fields', True) else {}
                        
                        # 添加生成的指令
                        output_data['enhanced_instruction'] = result.get('instruction_template', '')
                        
                        # 可选：添加检索信息
                        if self.output_config.get('include_metadata', True):
                            output_data['retrieval_metadata'] = {
                                'retrieved_entities': result.get('entity_results', {}),
                                'retrieved_sentences': result.get('similar_sentences', []),
                                'language': language,
                                'processing_time': result.get('processing_time', 0)
                            }
                        
                        # 可选：添加相似度分数
                        if self.output_config.get('include_similarity_scores', False):
                            output_data['similarity_scores'] = {
                                'entity_scores': {k: [item.get('score', 0) for item in v] 
                                                for k, v in result.get('entity_results', {}).items()},
                                'sentence_scores': [item.get('score', 0) 
                                                  for item in result.get('similar_sentences', [])]
                            }
                        
                        # 写入输出文件
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        stats['successful_entries'] += 1
                        
                        # 定期记录进度
                        if line_num % 100 == 0:
                            self.logger.info(f"📊 已处理 {line_num} 行，成功: {stats['successful_entries']}, 失败: {stats['failed_entries']}")
                    
                    except json.JSONDecodeError as e:
                        self.logger.error(f"❌ 文件 {input_file} 第 {line_num} 行JSON解析错误: {e}")
                        stats['failed_entries'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ 处理文件 {input_file} 第 {line_num} 行时出错: {e}")
                        stats['failed_entries'] += 1
            
            self.logger.info(f"✅ 文件 {input_file} 处理完成: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ 处理文件 {input_file} 时出错: {e}")
            raise
    
    def process_batch_queries(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量处理查询请求
        
        Args:
            queries: 查询列表，每个元素包含 'query' 和 'language'
            
        Returns:
            List[Dict]: 批量检索结果
        """
        try:
            self.logger.info(f"🔄 开始批量处理 {len(queries)} 个查询...")
            
            results = []
            start_time = time.time()
            
            for i, query_info in enumerate(queries):
                query = query_info.get("query", "")
                language = query_info.get("language", "en")
                
                self.logger.info(f"📝 处理查询 {i+1}/{len(queries)}: '{query}' ({language})")
                
                try:
                    result = self.process_single_query(query, language)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"❌ 查询 {i+1} 处理失败: {e}")
                    results.append({
                        "query": query,
                        "language": language,
                        "error": str(e),
                        "processing_time": 0
                    })
            
            total_time = time.time() - start_time
            
            self.logger.info(f"🎉 批量处理完成，共处理 {len(queries)} 个查询 (总耗时: {total_time:.3f}s)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 批量处理失败: {e}")
            raise
    
    def save_results(self, results: Union[Dict, List[Dict]], output_file: str):
        """
        保存结果到文件
        
        Args:
            results: 检索结果
            output_file: 输出文件路径
        """
        try:
            self.logger.info(f"💾 保存结果到文件: {output_file}")
            
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ 结果保存完成")
            
        except Exception as e:
            self.logger.error(f"❌ 结果保存失败: {e}")
            raise
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        获取服务统计信息
        
        Returns:
            Dict: 服务统计信息
        """
        try:
            self.logger.info("📊 获取服务统计信息...")
            
            statistics = {
                "service_status": "running",
                "supported_languages": SUPPORTED_LANGUAGES,
                "entity_types": ENTITY_TYPES,
                "configuration": {
                    "top_k_entities": self.retrieval_config["top_k_entities"],
                    "top_k_sentences": self.retrieval_config["top_k_sentences"],
                    "similarity_threshold": self.retrieval_config["similarity_threshold"],
                    "database_mode": self.milvus_config["mode"]
                }
            }
            
            # 获取数据库统计
            try:
                database_stats = {}
                for language in SUPPORTED_LANGUAGES:
                    try:
                        entity_collection = f"entity_{language}"
                        sentence_collection = f"sentence_{language}"
                        
                        entity_count = self.milvus_client.get_collection_count(entity_collection)
                        sentence_count = self.milvus_client.get_collection_count(sentence_collection)
                        
                        database_stats[language] = {
                            "entities": entity_count,
                            "sentences": sentence_count
                        }
                        
                    except Exception:
                        database_stats[language] = {
                            "entities": 0,
                            "sentences": 0,
                            "status": "not_available"
                        }
                
                statistics["database_statistics"] = database_stats
                
            except Exception as e:
                self.logger.warning(f"⚠️ 获取数据库统计失败: {e}")
                statistics["database_statistics"] = {"error": str(e)}
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ 获取服务统计失败: {e}")
            raise


def main():
    """
    命令行主程序
    """
    parser = argparse.ArgumentParser(description="阶段二：检索服务处理器")
    parser.add_argument("--query", type=str, help="查询句子")
    parser.add_argument("--language", type=str, default="en", 
                       choices=SUPPORTED_LANGUAGES, help="语言代码")
    parser.add_argument("--batch-file", type=str, help="批量查询文件路径 (JSON格式)")
    parser.add_argument("--jsonl-dirs", nargs="+", help="JSONL文件目录列表")
    parser.add_argument("--input-field", type=str, default="input", help="JSONL文件中的输入字段名")
    parser.add_argument("--output-file", type=str, help="结果输出文件路径")
    parser.add_argument("--output-dir", type=str, default="data/RetriAll_output", help="JSONL处理结果输出目录")
    parser.add_argument("--local-db-path", type=str, help="本地数据库路径")
    parser.add_argument("--top-k-entities", type=int, help="每个实体类型返回的数量")
    parser.add_argument("--top-k-sentences", type=int, help="返回的示例句子数量")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--stats", action="store_true", help="显示服务统计信息")
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
        
        if args.local_db_path:
            config_override["milvus"] = {"local_db_path": args.local_db_path}
        
        if args.top_k_entities or args.top_k_sentences:
            config_override["retrieval"] = {}
            if args.top_k_entities:
                config_override["retrieval"]["top_k_entities"] = args.top_k_entities
            if args.top_k_sentences:
                config_override["retrieval"]["top_k_sentences"] = args.top_k_sentences
        
        if args.device:
            config_override["model"] = {"device": args.device}
        
        if args.output_file or args.output_dir:
            config_override["output"] = {"save_results": True}
            if args.output_dir:
                config_override["output"]["output_directory"] = args.output_dir
        
        # 创建检索服务处理器
        service = Stage2RetrievalService(config_override)
        service.initialize_components()
        
        # 处理不同的操作模式
        if args.stats:
            # 显示服务统计信息
            statistics = service.get_service_statistics()
            logger.info("📊 服务统计信息:")
            print(json.dumps(statistics, indent=2, ensure_ascii=False))
            
        elif args.jsonl_dirs:
            # JSONL文件处理模式
            logger.info(f"📂 处理JSONL目录: {args.jsonl_dirs}")
            
            results = service.process_jsonl_files(
                jsonl_directories=args.jsonl_dirs,
                input_field=args.input_field,
                output_directory=args.output_dir
            )
            
            logger.info(f"🎉 JSONL处理完成: {results}")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
        elif args.batch_file:
            # 批量处理模式
            logger.info(f"📂 加载批量查询文件: {args.batch_file}")
            
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            results = service.process_batch_queries(queries)
            
            if args.output_file:
                service.save_results(results, args.output_file)
            else:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            
        elif args.query:
            # 单个查询模式
            result = service.process_single_query(args.query, args.language)
            
            if args.output_file:
                service.save_results(result, args.output_file)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
        else:
            # 交互模式
            logger.info("🎯 进入交互模式，输入 'quit' 退出")
            
            while True:
                try:
                    query = input("\n请输入查询句子: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    language = input(f"请输入语言代码 ({'/'.join(SUPPORTED_LANGUAGES)}) [默认: en]: ").strip()
                    if not language:
                        language = "en"
                    
                    result = service.process_single_query(query, language)
                    
                    print("\n" + "="*80)
                    print("🎯 生成的NER指令:")
                    print("="*80)
                    print(result["instruction_template"])
                    print("="*80)
                    
                    if "statistics" in result:
                        stats = result["statistics"]
                        print(f"📊 统计信息: 相似句子 {stats['total_similar_sentences']} 个, "
                              f"相关实体 {stats['total_entities_found']} 个, "
                              f"耗时 {stats['processing_time']:.3f}s")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"❌ 处理失败: {e}")
        
        logger.info("✅ 检索服务完成")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        sys.exit(1)
    
    finally:
        # 清理资源
        try:
            if 'service' in locals() and service.milvus_client:
                service.milvus_client.close()
                logger.info("🔒 Milvus连接已关闭")
        except:
            pass


if __name__ == "__main__":
    main()