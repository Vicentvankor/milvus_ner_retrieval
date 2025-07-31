"""
主程序入口
提供完整的NER检索系统功能
"""

import logging
import argparse
import json
from typing import List, Optional

from .config import MILVUS_CONFIG, SUPPORTED_LANGUAGES
from .database.milvus_client import MilvusClient
from .core.embedding_model import EmbeddingModel
from .database.database_manager import DatabaseManager
from .core.retrieval_engine import RetrievalEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NERRetrievalSystem:
    """
    NER检索系统主类
    整合所有功能模块
    """
    
    def __init__(self, milvus_host: str = None, milvus_port: str = None, local_db_path: str = None):
        """
        初始化NER检索系统
        
        Args:
            milvus_host: Milvus服务器地址（远程模式）
            milvus_port: Milvus服务器端口（远程模式）
            local_db_path: 本地数据库路径（本地模式）
        """
        logger.info("初始化NER检索系统...")
        
        # 初始化组件
        self.milvus_client = MilvusClient(milvus_host, milvus_port, local_db_path)
        self.embedding_model = EmbeddingModel()
        self.database_manager = DatabaseManager(self.milvus_client, self.embedding_model)
        self.retrieval_engine = RetrievalEngine(self.milvus_client, self.embedding_model)
        
        logger.info("NER检索系统初始化完成")
    
    def setup_database(self, entities_file: str, sentences_file: str, 
                      languages: List[str] = None):
        """
        设置数据库
        
        Args:
            entities_file: 实体数据文件路径
            sentences_file: 句子数据文件路径
            languages: 要设置的语种列表
        """
        logger.info("开始设置数据库...")
        self.database_manager.setup_database(entities_file, sentences_file, languages)
        logger.info("数据库设置完成")
    
    def retrieve(self, query: str, language: str) -> dict:
        """
        执行检索
        
        Args:
            query: 查询句子
            language: 语种
            
        Returns:
            dict: 检索结果
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语种: {language}")
        
        return self.retrieval_engine.retrieve_and_format(query, language)
    
    def batch_retrieve(self, queries: List[str], language: str) -> List[dict]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            language: 语种
            
        Returns:
            List[dict]: 批量检索结果
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的语种: {language}")
        
        return self.retrieval_engine.batch_retrieve(queries, language)
    
    def get_statistics(self) -> dict:
        """
        获取数据库统计信息
        
        Returns:
            dict: 统计信息
        """
        return self.database_manager.get_database_statistics()
    
    def cleanup(self, languages: List[str] = None):
        """
        清理数据库
        
        Args:
            languages: 要清理的语种列表
        """
        self.database_manager.cleanup_collections(languages)
    
    def close(self):
        """
        关闭系统
        """
        self.milvus_client.close_connection()
        logger.info("NER检索系统已关闭")


def main():
    """
    命令行主程序
    """
    parser = argparse.ArgumentParser(description="NER检索系统")
    parser.add_argument("--action", choices=["setup", "retrieve", "stats", "cleanup"], 
                       required=True, help="要执行的操作")
    parser.add_argument("--entities-file", help="实体数据文件路径")
    parser.add_argument("--sentences-file", help="句子数据文件路径")
    parser.add_argument("--query", help="查询句子")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, help="语种")
    parser.add_argument("--languages", nargs="+", choices=SUPPORTED_LANGUAGES, help="语种列表")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--local-db-path", help="本地数据库路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化系统
        system = NERRetrievalSystem(local_db_path=args.local_db_path)
        
        if args.action == "setup":
            if not args.entities_file or not args.sentences_file:
                raise ValueError("setup操作需要指定--entities-file和--sentences-file")
            
            system.setup_database(args.entities_file, args.sentences_file, args.languages)
            print("数据库设置完成")
        
        elif args.action == "retrieve":
            if not args.query or not args.language:
                raise ValueError("retrieve操作需要指定--query和--language")
            
            result = system.retrieve(args.query, args.language)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"检索结果已保存到: {args.output}")
            else:
                print("检索结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.action == "stats":
            stats = system.get_statistics()
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                print(f"统计信息已保存到: {args.output}")
            else:
                print("数据库统计信息:")
                print(json.dumps(stats, ensure_ascii=False, indent=2))
        
        elif args.action == "cleanup":
            system.cleanup(args.languages)
            print("数据库清理完成")
        
        # 关闭系统
        system.close()
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()