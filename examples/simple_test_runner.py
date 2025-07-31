#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试运行器
专门处理您提到的流程：存储 -> 输入测试集input -> 召回 -> 填充instruction
"""

import json
import logging
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_set_processor import TestSetProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_your_pipeline():
    """
    运行您指定的完整流程：
    1. 存储数据到向量数据库
    2. 输入测试句子
    3. 召回相似实体和句子
    4. 填充instruction模板
    """
    
    print("🚀 执行您的测试流程")
    print("=" * 60)
    
    # 您的测试输入
    test_input = "It is located in the San'a' Governorate."
    language = "en"
    
    print(f"📝 测试输入: {test_input}")
    print(f"🌍 语种: {language}")
    
    try:
        # 步骤1: 初始化处理器
        print("\n🔧 步骤1: 初始化系统")
        processor = TestSetProcessor()
        
        # 步骤2: 存储数据（确保数据文件存在）
        entities_file = "../extracted_entities_by_language.json"
        sentences_file = "../extracted_sentences_with_ner_by_language.json"
        
        print("\n📊 步骤2: 存储数据到向量数据库")
        
        if not os.path.exists(entities_file):
            print(f"❌ 实体文件不存在: {entities_file}")
            print("请确保已运行数据提取脚本生成此文件")
            return
        
        if not os.path.exists(sentences_file):
            print(f"❌ 句子文件不存在: {sentences_file}")
            print("请确保已运行数据提取脚本生成此文件")
            return
        
        # 只设置英语（加快演示速度）
        processor.setup_database(entities_file, sentences_file, [language])
        print("✅ 数据存储完成")
        
        # 步骤3: 输入测试集的input进行召回
        print(f"\n🔍 步骤3: 召回相似内容")
        print("3.1 从句子数据库召回top5示例句子...")
        print("3.2 从实体数据库召回各实体类型的top5实体...")
        
        # 获取详细检索结果
        detailed_result = processor.create_instruction_for_single_input(test_input, language)
        
        # 显示召回的内容
        print("\n📝 召回的示例句子 (Top 5):")
        for i, sent in enumerate(detailed_result["similar_sentences"], 1):
            print(f"  {i}. {sent['sentence_text']}")
            print(f"     相似度: {sent['score']:.4f}")
            print(f"     NER标签: {sent['ner_labels']}")
            print()
        
        print("🏷️ 召回的实体 (各类型Top 5):")
        for entity_type, entities in detailed_result["entity_results"].items():
            if entities:
                print(f"\n  {entity_type}:")
                for j, entity in enumerate(entities, 1):
                    print(f"    {j}. {entity['entity_text']} (相似度: {entity['score']:.4f})")
        
        # 步骤4: 填充instruction
        print(f"\n📋 步骤4: 填充instruction模板")
        instruction = detailed_result["generated_instruction"]
        
        print("\n" + "="*80)
        print("🎯 最终生成的完整Instruction:")
        print("="*80)
        print(instruction)
        print("="*80)
        
        # 保存结果
        result_file = "pipeline_result.json"
        result_data = {
            "test_input": test_input,
            "language": language,
            "generated_instruction": instruction,
            "retrieval_details": {
                "similar_sentences_count": len(detailed_result["similar_sentences"]),
                "total_entities_found": detailed_result["retrieval_statistics"]["total_entities_found"],
                "entity_types_found": detailed_result["retrieval_statistics"]["entity_types_found"]
            },
            "similar_sentences": detailed_result["similar_sentences"],
            "entity_results": detailed_result["entity_results"]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {result_file}")
        
        processor.close()
        print("\n🎉 流程执行完成!")
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise


def batch_process_test_cases():
    """批量处理多个测试样例"""
    print("\n🔄 批量处理示例")
    print("-" * 40)
    
    # 多个测试用例
    test_cases = [
        {"input": "It is located in the San'a' Governorate.", "language": "en"},
        {"input": "Barack Obama was born in Hawaii.", "language": "en"},
        {"input": "Google was founded by Larry Page.", "language": "en"}
    ]
    
    try:
        processor = TestSetProcessor()
        
        # 设置数据库
        entities_file = "../extracted_entities_by_language.json"
        sentences_file = "../extracted_sentences_with_ner_by_language.json"
        
        if os.path.exists(entities_file) and os.path.exists(sentences_file):
            processor.setup_database(entities_file, sentences_file, ["en"])
            
            results = []
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n处理测试用例 {i}: {test_case['input']}")
                
                instruction = processor.process_test_input(
                    test_case["input"], 
                    test_case["language"]
                )
                
                results.append({
                    "test_case": test_case,
                    "generated_instruction": instruction
                })
            
            # 保存批量结果
            with open("batch_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print("✅ 批量处理完成，结果保存到: batch_results.json")
            
        processor.close()
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 单个测试样例 (您提到的具体例子)")
    print("2. 批量测试样例")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        run_your_pipeline()
    elif choice == "2":
        batch_process_test_cases()
    else:
        print("无效选择，运行默认单个测试...")
        run_your_pipeline()