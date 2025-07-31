#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试处理器
专门用于批量处理测试集文件
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from test_set_processor import TestSetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_test_dataset(processor: TestSetProcessor, test_file: str, 
                        output_file: str, language: str = "en",
                        input_field: str = "input", language_field: str = "language"):
    """
    处理测试数据集
    
    Args:
        processor: 测试集处理器
        test_file: 测试文件路径
        output_file: 输出文件路径
        language: 默认语种
        input_field: 输入字段名
        language_field: 语种字段名
    """
    logger.info(f"开始处理测试数据集: {test_file}")
    
    # 加载测试数据
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    # 读取测试文件
    test_data = []
    file_path = Path(test_file)
    
    if file_path.suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
            if isinstance(file_content, list):
                test_data = file_content
            else:
                test_data = [file_content]
    
    logger.info(f"加载了 {len(test_data)} 个测试样本")
    
    # 处理每个测试样本
    processed_results = []
    
    for i, item in enumerate(test_data):
        logger.info(f"处理进度: {i+1}/{len(test_data)}")
        
        try:
            # 获取输入文本和语种
            test_input = item.get(input_field, "")
            test_language = item.get(language_field, language)
            
            if not test_input:
                logger.warning(f"样本 {i+1} 缺少输入字段: {input_field}")
                continue
            
            # 生成instruction
            instruction = processor.process_test_input(test_input, test_language)
            
            # 创建结果记录
            result = {
                "sample_id": i,
                "original_input": test_input,
                "language": test_language,
                "generated_instruction": instruction
            }
            
            # 保留原始数据的其他字段
            for key, value in item.items():
                if key not in result:
                    result[f"original_{key}"] = value
            
            processed_results.append(result)
            
        except Exception as e:
            logger.error(f"处理样本 {i+1} 失败: {e}")
            error_result = {
                "sample_id": i,
                "original_input": item.get(input_field, ""),
                "language": item.get(language_field, language),
                "generated_instruction": None,
                "error": str(e)
            }
            processed_results.append(error_result)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成，结果保存到: {output_file}")
    
    # 统计信息
    successful = sum(1 for r in processed_results if r.get("generated_instruction") is not None)
    failed = len(processed_results) - successful
    
    print(f"\n📊 处理统计:")
    print(f"   总样本数: {len(test_data)}")
    print(f"   成功处理: {successful}")
    print(f"   处理失败: {failed}")
    print(f"   成功率: {successful/len(test_data)*100:.1f}%")
    
    return processed_results


def create_sample_test_file():
    """创建示例测试文件"""
    sample_data = [
        {
            "input": "It is located in the San'a' Governorate.",
            "language": "en",
            "expected_output": {"LOCATION": ["San'a' Governorate"]}
        },
        {
            "input": "Barack Obama was born in Hawaii and served as President.",
            "language": "en",
            "expected_output": {"PERSON": ["Barack Obama"], "LOCATION": ["Hawaii"]}
        },
        {
            "input": "Google was founded by Larry Page and Sergey Brin.",
            "language": "en",
            "expected_output": {"GROUP": ["Google"], "PERSON": ["Larry Page", "Sergey Brin"]}
        },
        {
            "input": "The Eiffel Tower is located in Paris, France.",
            "language": "en",
            "expected_output": {"FACILITY": ["Eiffel Tower"], "LOCATION": ["Paris", "France"]}
        },
        {
            "input": "Berlin ist die Hauptstadt von Deutschland.",
            "language": "de",
            "expected_output": {"LOCATION": ["Berlin", "Deutschland"]}
        }
    ]
    
    sample_file = "sample_test_data.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 示例测试文件已创建: {sample_file}")
    return sample_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量测试集处理器")
    parser.add_argument("--action", choices=["setup", "process", "demo"], 
                       required=True, help="要执行的操作")
    parser.add_argument("--entities-file", help="实体数据文件路径")
    parser.add_argument("--sentences-file", help="句子数据文件路径")
    parser.add_argument("--test-file", help="测试数据文件路径")
    parser.add_argument("--output-file", help="输出文件路径")
    parser.add_argument("--language", default="en", help="默认语种")
    parser.add_argument("--languages", nargs="+", help="要设置的语种列表")
    parser.add_argument("--input-field", default="input", help="输入字段名")
    parser.add_argument("--language-field", default="language", help="语种字段名")
    
    args = parser.parse_args()
    
    try:
        processor = TestSetProcessor()
        
        if args.action == "setup":
            # 设置数据库
            if not args.entities_file or not args.sentences_file:
                raise ValueError("setup操作需要指定--entities-file和--sentences-file")
            
            print("🔧 设置数据库...")
            processor.setup_database(
                args.entities_file, 
                args.sentences_file, 
                args.languages
            )
            print("✅ 数据库设置完成")
        
        elif args.action == "process":
            # 处理测试文件
            if not args.test_file or not args.output_file:
                raise ValueError("process操作需要指定--test-file和--output-file")
            
            print("🧪 处理测试文件...")
            process_test_dataset(
                processor=processor,
                test_file=args.test_file,
                output_file=args.output_file,
                language=args.language,
                input_field=args.input_field,
                language_field=args.language_field
            )
        
        elif args.action == "demo":
            # 演示模式
            print("🎭 演示模式")
            
            # 创建示例测试文件
            sample_file = create_sample_test_file()
            
            # 设置数据库
            entities_file = "../extracted_entities_by_language.json"
            sentences_file = "../extracted_sentences_with_ner_by_language.json"
            
            if os.path.exists(entities_file) and os.path.exists(sentences_file):
                print("🔧 设置数据库...")
                processor.setup_database(entities_file, sentences_file, ["en", "de"])
                print("✅ 数据库设置完成")
                
                # 处理示例文件
                output_file = "sample_test_results.json"
                print("🧪 处理示例测试文件...")
                process_test_dataset(
                    processor=processor,
                    test_file=sample_file,
                    output_file=output_file,
                    language="en"
                )
                
                print(f"\n📁 生成的文件:")
                print(f"   - {sample_file} (示例测试数据)")
                print(f"   - {output_file} (处理结果)")
            else:
                print("❌ 数据文件不存在，跳过数据库设置")
        
        processor.close()
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()