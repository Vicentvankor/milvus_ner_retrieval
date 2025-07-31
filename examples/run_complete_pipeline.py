#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整管道运行脚本
执行完整的存储->检索->填充instruction流程
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from test_set_processor import TestSetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_pipeline(entities_file: str, sentences_file: str, 
                         test_input: str, language: str = "en",
                         languages_to_setup: list = None,
                         output_file: str = None):
    """
    运行完整的管道流程
    
    Args:
        entities_file: 实体数据文件
        sentences_file: 句子数据文件
        test_input: 测试输入句子
        language: 测试输入的语种
        languages_to_setup: 要设置的语种列表
        output_file: 输出文件路径
    """
    print("🚀 启动完整管道流程")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 步骤1: 初始化系统
        print("\n🔧 步骤1: 初始化系统")
        processor = TestSetProcessor()
        print("✅ 系统初始化完成")
        
        # 步骤2: 存储数据到向量数据库
        print("\n📊 步骤2: 存储数据到向量数据库")
        print(f"   实体文件: {entities_file}")
        print(f"   句子文件: {sentences_file}")
        print(f"   目标语种: {languages_to_setup or '全部'}")
        
        storage_start = time.time()
        processor.setup_database(entities_file, sentences_file, languages_to_setup)
        storage_time = time.time() - storage_start
        print(f"✅ 数据存储完成 (耗时: {storage_time:.1f}秒)")
        
        # 步骤3: 输入测试句子进行检索
        print(f"\n🔍 步骤3: 检索和生成instruction")
        print(f"   测试输入: {test_input}")
        print(f"   语种: {language}")
        
        retrieval_start = time.time()
        
        # 获取详细的检索信息
        detailed_result = processor.create_instruction_for_single_input(test_input, language)
        
        retrieval_time = time.time() - retrieval_start
        print(f"✅ 检索完成 (耗时: {retrieval_time:.1f}秒)")
        
        # 步骤4: 显示结果
        print(f"\n📋 步骤4: 检索结果分析")
        stats = detailed_result["retrieval_statistics"]
        print(f"   检索到的相似句子: {stats['total_similar_sentences']} 个")
        print(f"   检索到的相关实体: {stats['total_entities_found']} 个")
        print(f"   涉及的实体类型: {stats['entity_types_found']} 种")
        
        print(f"\n   📝 相似句子示例 (前3个):")
        for i, sent in enumerate(detailed_result["similar_sentences"][:3], 1):
            print(f"   {i}. {sent['sentence_text'][:80]}...")
            print(f"      相似度: {sent['score']:.4f}")
            print(f"      NER标签: {sent['ner_labels']}")
        
        print(f"\n   🏷️ 检索到的实体示例 (每类前2个):")
        for entity_type, entities in detailed_result["entity_results"].items():
            if entities:
                print(f"   {entity_type}:")
                for entity in entities[:2]:
                    print(f"     • {entity['entity_text']} (相似度: {entity['score']:.4f})")
        
        # 步骤5: 显示生成的instruction
        print(f"\n📄 步骤5: 生成的完整Instruction")
        print("=" * 80)
        print(detailed_result["generated_instruction"])
        print("=" * 80)
        
        # 步骤6: 保存结果
        total_time = time.time() - start_time
        
        final_result = {
            "pipeline_info": {
                "entities_file": entities_file,
                "sentences_file": sentences_file,
                "test_input": test_input,
                "language": language,
                "languages_setup": languages_to_setup,
                "total_time": total_time,
                "storage_time": storage_time,
                "retrieval_time": retrieval_time
            },
            "result": detailed_result
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 完整结果已保存到: {output_file}")
        
        print(f"\n⏱️ 总耗时: {total_time:.1f}秒")
        print(f"   存储阶段: {storage_time:.1f}秒 ({storage_time/total_time*100:.1f}%)")
        print(f"   检索阶段: {retrieval_time:.1f}秒 ({retrieval_time/total_time*100:.1f}%)")
        
        processor.close()
        
        print("\n🎉 完整管道流程执行完成!")
        
        return final_result
        
    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        raise
    finally:
        print("\n" + "=" * 60)


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="完整管道运行器")
    parser.add_argument("--entities-file", required=True, help="实体数据文件路径")
    parser.add_argument("--sentences-file", required=True, help="句子数据文件路径")
    parser.add_argument("--test-input", required=True, help="测试输入句子")
    parser.add_argument("--language", default="en", help="测试输入的语种")
    parser.add_argument("--languages", nargs="+", help="要设置的语种列表")
    parser.add_argument("--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    try:
        result = run_complete_pipeline(
            entities_file=args.entities_file,
            sentences_file=args.sentences_file,
            test_input=args.test_input,
            language=args.language,
            languages_to_setup=args.languages,
            output_file=args.output
        )
        
        print("\n📋 快速使用示例:")
        print("python run_complete_pipeline.py \\")
        print("    --entities-file ../extracted_entities_by_language.json \\")
        print("    --sentences-file ../extracted_sentences_with_ner_by_language.json \\")
        print("    --test-input \"It is located in the San'a' Governorate.\" \\")
        print("    --language en \\")
        print("    --output result.json")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()