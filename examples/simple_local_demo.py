#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的本地数据库演示
专门展示如何使用本地Milvus数据库进行NER检索
"""

import sys
import os
import logging

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.test_set_processor import TestSetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    本地数据库演示主函数
    """
    print("🏠 本地数据库NER检索系统演示")
    print("=" * 50)
    
    # 您的测试输入
    test_input = "It is located in the San'a' Governorate."
    language = "en"
    local_db_path = "./local_milvus_db"
    
    print(f"📝 测试输入: {test_input}")
    print(f"🌍 语种: {language}")
    print(f"🏠 本地数据库路径: {local_db_path}")
    
    try:
        # 步骤1: 初始化本地数据库处理器
        print("\n🔧 步骤1: 初始化本地数据库处理器")
        processor = TestSetProcessor(local_db_path=local_db_path)
        print("✅ 本地数据库处理器初始化完成")
        
        # 步骤2: 存储数据到本地数据库
        entities_file = "../extracted_entities_by_language.json"
        sentences_file = "../extracted_sentences_with_ner_by_language.json"
        
        print("\n📊 步骤2: 存储数据到本地数据库")
        
        if not os.path.exists(entities_file):
            print(f"❌ 实体文件不存在: {entities_file}")
            print("请确保运行了数据提取脚本")
            return
        
        if not os.path.exists(sentences_file):
            print(f"❌ 句子文件不存在: {sentences_file}")
            print("请确保运行了数据提取脚本")
            return
        
        # 只设置英语（演示用）
        processor.setup_database(entities_file, sentences_file, [language])
        print("✅ 数据存储到本地数据库完成")
        
        # 步骤3: 处理测试输入
        print(f"\n🔍 步骤3: 处理测试输入")
        print("3.1 从本地句子数据库召回top5示例句子...")
        print("3.2 从本地实体数据库召回各实体类型的top5实体...")
        
        # 获取详细检索结果
        detailed_result = processor.create_instruction_for_single_input(test_input, language)
        
        # 显示召回的内容
        print("\n📝 召回的示例句子:")
        for i, sent in enumerate(detailed_result["similar_sentences"][:3], 1):
            print(f"  {i}. {sent['sentence_text'][:80]}...")
            print(f"     相似度: {sent['score']:.4f}")
        
        print("\n🏷️ 召回的实体（各类型Top 3）:")
        for entity_type, entities in detailed_result["entity_results"].items():
            if entities:
                print(f"\n  {entity_type}:")
                for j, entity in enumerate(entities[:3], 1):
                    print(f"    {j}. {entity['entity_text']} (相似度: {entity['score']:.4f})")
        
        # 步骤4: 显示生成的指令模板
        print(f"\n📋 步骤4: 生成的完整Instruction")
        instruction = detailed_result["generated_instruction"]
        
        print("\n" + "="*80)
        print("🎯 最终生成的完整Instruction:")
        print("="*80)
        print(instruction)
        print("="*80)
        
        # 保存结果
        result_file = "local_db_result.json"
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "local_db_path": local_db_path,
                "test_input": test_input,
                "language": language,
                "generated_instruction": instruction,
                "statistics": detailed_result["retrieval_statistics"]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {result_file}")
        
        processor.close()
        print("\n🎉 本地数据库演示完成!")
        
        print(f"\n💡 优势:")
        print(f"  ✅ 无需启动Milvus服务器")
        print(f"  ✅ 数据存储在本地文件: {local_db_path}")
        print(f"  ✅ 不依赖网络连接")
        print(f"  ✅ 快速启动和测试")
        
    except Exception as e:
        logger.error(f"本地数据库演示失败: {e}")
        raise


if __name__ == "__main__":
    main()