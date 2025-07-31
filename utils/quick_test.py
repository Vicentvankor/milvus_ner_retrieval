#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
用于验证NER检索系统的基本功能
"""

import os
import sys
import json
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SUPPORTED_LANGUAGES, ENTITY_TYPES
from milvus_client import MilvusClient
from embedding_model import EmbeddingModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_milvus_connection():
    """测试Milvus连接"""
    try:
        logger.info("测试Milvus连接...")
        client = MilvusClient()
        logger.info("✅ Milvus连接成功")
        client.close_connection()
        return True
    except Exception as e:
        logger.error(f"❌ Milvus连接失败: {e}")
        return False


def test_embedding_model():
    """测试embedding模型加载"""
    try:
        logger.info("测试embedding模型加载...")
        logger.warning("注意: 首次加载模型可能需要较长时间...")
        
        # 创建模型实例
        model = EmbeddingModel()
        
        # 测试编码
        test_docs = ["This is a test sentence.", "这是一个测试句子。"]
        embeddings = model.encode_documents(test_docs)
        
        logger.info(f"✅ 模型加载成功，向量维度: {embeddings.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return False


def check_data_files():
    """检查数据文件是否存在"""
    files_to_check = [
        "../extracted_entities_by_language.json",
        "../extracted_sentences_with_ner_by_language.json"
    ]
    
    results = {}
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[file_path] = {
                        "exists": True,
                        "languages": list(data.keys()),
                        "total_items": sum(len(v) if isinstance(v, list) else 
                                         sum(len(vv) for vv in v.values()) if isinstance(v, dict) else 0 
                                         for v in data.values())
                    }
                logger.info(f"✅ 数据文件 {file_path} 存在")
            except Exception as e:
                results[file_path] = {"exists": True, "error": str(e)}
                logger.warning(f"⚠️  数据文件 {file_path} 存在但读取失败: {e}")
        else:
            results[file_path] = {"exists": False}
            logger.warning(f"⚠️  数据文件 {file_path} 不存在")
    
    return results


def main():
    """主测试函数"""
    print("=" * 60)
    print("🔍 NER检索系统快速测试")
    print("=" * 60)
    
    # 1. 检查配置
    print(f"\n📋 系统配置:")
    print(f"  支持语种: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"  实体类型: {', '.join(ENTITY_TYPES)}")
    
    # 2. 检查数据文件
    print(f"\n📁 检查数据文件...")
    data_files = check_data_files()
    
    for file_path, info in data_files.items():
        if info["exists"] and "error" not in info:
            print(f"  {file_path}:")
            print(f"    语种: {', '.join(info['languages'])}")
            print(f"    数据量: {info['total_items']}")
    
    # 3. 测试Milvus连接
    print(f"\n🗄️  测试Milvus连接...")
    milvus_ok = test_milvus_connection()
    
    # 4. 测试模型加载（可选，因为比较耗时）
    test_model = input("\n🤖 是否测试embedding模型加载？(y/N): ").lower().strip() == 'y'
    model_ok = True
    if test_model:
        model_ok = test_embedding_model()
    else:
        print("⏭️  跳过模型测试")
    
    # 5. 总结
    print(f"\n📊 测试总结:")
    print(f"  Milvus连接: {'✅' if milvus_ok else '❌'}")
    print(f"  模型加载: {'✅' if model_ok else '❌' if test_model else '⏭️'}")
    
    all_data_ok = all(info["exists"] and "error" not in info for info in data_files.values())
    print(f"  数据文件: {'✅' if all_data_ok else '⚠️'}")
    
    # 6. 使用建议
    print(f"\n💡 下一步操作建议:")
    
    if not milvus_ok:
        print("  1. 请确保Milvus服务器正在运行")
        print("     Docker启动: docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest")
    
    if not all_data_ok:
        print("  2. 请确保数据文件存在:")
        print("     - extracted_entities_by_language.json")
        print("     - extracted_sentences_with_ner_by_language.json")
    
    if milvus_ok and all_data_ok:
        print("  ✨ 系统准备就绪！可以开始使用:")
        print("     python -m milvus_ner_retrieval.main --action setup --entities-file ../extracted_entities_by_language.json --sentences-file ../extracted_sentences_with_ner_by_language.json")
        print("     python -m milvus_ner_retrieval.main --action retrieve --query 'Barack Obama was born in Hawaii.' --language en")
    
    print("=" * 60)


if __name__ == "__main__":
    main()