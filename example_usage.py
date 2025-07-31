"""
使用示例
演示如何使用NER检索系统
"""

import json
import logging
from .main import NERRetrievalSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    主函数 - 演示系统使用流程
    """
    try:
        logger.info("开始NER检索系统演示...")
        
        # 1. 初始化系统
        logger.info("初始化系统...")
        system = NERRetrievalSystem()
        
        # 2. 设置数据库（如果还未设置）
        logger.info("设置数据库...")
        try:
            system.setup_database(
                entities_file="../extracted_entities_by_language.json",
                sentences_file="../extracted_sentences_with_ner_by_language.json",
                languages=["en", "de", "zh"]  # 演示用，只设置3种语言
            )
            logger.info("数据库设置完成")
        except Exception as e:
            logger.warning(f"数据库设置失败（可能已存在）: {e}")
        
        # 3. 查看数据库统计信息
        logger.info("获取数据库统计信息...")
        stats = system.get_statistics()
        print("\n=== 数据库统计信息 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        
        # 4. 执行检索示例
        test_queries = [
            ("Barack Obama was born in Hawaii.", "en"),
            ("Angela Merkel ist eine deutsche Politikerin.", "de"),
            ("北京是中国的首都。", "zh")
        ]
        
        for query, language in test_queries:
            logger.info(f"检索查询: {query} ({language})")
            
            # 执行检索
            result = system.retrieve(query, language)
            
            print(f"\n=== 检索结果: {query} ===")
            print(f"语种: {language}")
            print(f"找到相似句子数: {result['statistics']['total_similar_sentences']}")
            print(f"找到实体数: {result['statistics']['total_entities_found']}")
            print(f"涉及实体类型数: {result['statistics']['entity_types_found']}")
            
            # 显示生成的指令模板
            print("\n生成的指令模板:")
            print("-" * 80)
            print(result["instruction_template"])
            print("-" * 80)
            
            # 显示检索到的相似句子
            print("\n检索到的相似句子:")
            for i, sent in enumerate(result["similar_sentences"][:3]):  # 只显示前3个
                print(f"{i+1}. {sent['sentence_text']}")
                print(f"   NER: {sent['ner_labels']}")
                print(f"   相似度: {sent['score']:.4f}")
            
            # 显示各类型实体
            print("\n检索到的实体（每类型显示前3个）:")
            for entity_type, entities in result["entity_results"].items():
                if entities:
                    print(f"{entity_type}:")
                    for i, entity in enumerate(entities[:3]):
                        print(f"  {i+1}. {entity['entity_text']} (相似度: {entity['score']:.4f})")
            
            print("\n" + "="*100 + "\n")
        
        # 5. 批量检索示例
        logger.info("执行批量检索...")
        batch_queries = [
            "The Eiffel Tower is located in Paris.",
            "Google is a technology company.",
            "Einstein developed the theory of relativity."
        ]
        
        batch_results = system.batch_retrieve(batch_queries, "en")
        
        print("\n=== 批量检索结果 ===")
        for i, result in enumerate(batch_results):
            if "error" not in result:
                print(f"查询 {i+1}: {result['query']}")
                print(f"  相似句子数: {result['statistics']['total_similar_sentences']}")
                print(f"  实体数: {result['statistics']['total_entities_found']}")
            else:
                print(f"查询 {i+1}: {result['query']} - 错误: {result['error']}")
        
        # 6. 关闭系统
        system.close()
        logger.info("演示完成")
        
    except Exception as e:
        logger.error(f"演示执行失败: {e}")
        raise


if __name__ == "__main__":
    main()