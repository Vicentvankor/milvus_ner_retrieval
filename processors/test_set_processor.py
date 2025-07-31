#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集处理器
先存储数据，后处理测试集input，召回相似内容并填充instruction
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..main import NERRetrievalSystem
from ..config import ENTITY_TYPES

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSetProcessor:
    """
    测试集处理器
    负责存储数据和处理测试集
    """
    
    def __init__(self, milvus_host: str = None, milvus_port: str = None, local_db_path: str = None):
        """
        初始化处理器
        
        Args:
            milvus_host: Milvus服务器地址
            milvus_port: Milvus服务器端口
            local_db_path: 本地数据库路径
        """
        self.system = NERRetrievalSystem(milvus_host, milvus_port, local_db_path)
        logger.info("测试集处理器初始化完成")
    
    def setup_database(self, entities_file: str, sentences_file: str, 
                      languages: List[str] = None):
        """
        设置数据库 - 存储阶段
        
        Args:
            entities_file: 实体数据文件路径
            sentences_file: 句子数据文件路径
            languages: 要设置的语种列表
        """
        logger.info("开始数据库存储阶段...")
        
        # 检查文件是否存在
        if not os.path.exists(entities_file):
            raise FileNotFoundError(f"实体数据文件不存在: {entities_file}")
        
        if not os.path.exists(sentences_file):
            raise FileNotFoundError(f"句子数据文件不存在: {sentences_file}")
        
        # 设置数据库
        self.system.setup_database(entities_file, sentences_file, languages)
        logger.info("数据库存储完成")
    
    def process_test_input(self, test_input: str, language: str) -> str:
        """
        处理单个测试输入，生成填充后的instruction
        
        Args:
            test_input: 测试输入句子
            language: 语种
            
        Returns:
            str: 填充后的完整instruction
        """
        logger.info(f"处理测试输入: {test_input} ({language})")
        
        # 执行检索
        result = self.system.retrieve(test_input, language)
        
        # 生成instruction（已经在retrieve中完成）
        instruction = result["instruction_template"]
        
        # 替换最后的Input部分为实际的测试输入
        if instruction.endswith("Input: "):
            instruction = instruction + test_input
        else:
            # 如果格式不符合预期，手动添加
            instruction = instruction.rstrip() + f"\nInput: {test_input}"
        
        return instruction
    
    def process_test_file(self, test_file: str, output_file: str, 
                         input_field: str = "input", language_field: str = "language"):
        """
        处理测试文件，为每个input生成instruction
        
        Args:
            test_file: 测试文件路径（JSON/JSONL格式）
            output_file: 输出文件路径
            input_field: 输入字段名
            language_field: 语种字段名
        """
        logger.info(f"处理测试文件: {test_file}")
        
        # 读取测试文件
        test_data = self._load_test_file(test_file)
        
        # 处理每个测试样本
        processed_data = []
        total = len(test_data)
        
        for i, item in enumerate(test_data, 1):
            logger.info(f"处理进度: {i}/{total}")
            
            try:
                # 获取输入和语种
                test_input = item.get(input_field, "")
                language = item.get(language_field, "en")  # 默认英语
                
                if not test_input:
                    logger.warning(f"第{i}个样本缺少输入字段: {input_field}")
                    continue
                
                # 生成instruction
                instruction = self.process_test_input(test_input, language)
                
                # 创建输出项
                output_item = item.copy()  # 保留原有字段
                output_item["generated_instruction"] = instruction
                output_item["original_input"] = test_input
                output_item["language"] = language
                
                processed_data.append(output_item)
                
            except Exception as e:
                logger.error(f"处理第{i}个样本失败: {e}")
                # 添加错误记录
                error_item = item.copy()
                error_item["error"] = str(e)
                error_item["generated_instruction"] = None
                processed_data.append(error_item)
        
        # 保存结果
        self._save_processed_data(processed_data, output_file)
        logger.info(f"处理完成，结果保存到: {output_file}")
        
        return processed_data
    
    def process_test_inputs_batch(self, test_inputs: List[Dict[str, str]], 
                                 language: str = "en") -> List[Dict[str, Any]]:
        """
        批量处理测试输入
        
        Args:
            test_inputs: 测试输入列表，每个元素包含input字段
            language: 语种
            
        Returns:
            List[Dict]: 处理结果列表
        """
        logger.info(f"批量处理 {len(test_inputs)} 个测试输入")
        
        results = []
        
        for i, item in enumerate(test_inputs, 1):
            logger.info(f"批量处理进度: {i}/{len(test_inputs)}")
            
            try:
                test_input = item.get("input", "")
                if not test_input:
                    continue
                
                instruction = self.process_test_input(test_input, language)
                
                result = {
                    "original_input": test_input,
                    "language": language,
                    "generated_instruction": instruction,
                    "index": i - 1
                }
                
                # 保留原有的其他字段
                for key, value in item.items():
                    if key not in result:
                        result[key] = value
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"批量处理第{i}个输入失败: {e}")
                results.append({
                    "original_input": item.get("input", ""),
                    "language": language,
                    "generated_instruction": None,
                    "error": str(e),
                    "index": i - 1
                })
        
        return results
    
    def create_instruction_for_single_input(self, test_input: str, language: str) -> Dict[str, Any]:
        """
        为单个输入创建完整的instruction信息
        
        Args:
            test_input: 测试输入
            language: 语种
            
        Returns:
            Dict: 完整的instruction信息
        """
        # 执行检索获取详细信息
        result = self.system.retrieve(test_input, language)
        
        # 生成最终的instruction
        instruction = result["instruction_template"]
        if not instruction.endswith(test_input):
            instruction = instruction.rstrip()
            if not instruction.endswith("Input: "):
                instruction += "\nInput: "
            instruction += test_input
        
        return {
            "test_input": test_input,
            "language": language,
            "generated_instruction": instruction,
            "retrieval_statistics": result["statistics"],
            "similar_sentences": result["similar_sentences"],
            "entity_results": result["entity_results"],
            "raw_instruction_template": result["instruction_template"]
        }
    
    def _load_test_file(self, test_file: str) -> List[Dict]:
        """
        加载测试文件
        
        Args:
            test_file: 测试文件路径
            
        Returns:
            List[Dict]: 测试数据列表
        """
        file_path = Path(test_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        data = []
        
        if file_path.suffix == '.jsonl':
            # JSONL格式
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"JSONL文件第{line_num}行解析失败: {e}")
        
        elif file_path.suffix == '.json':
            # JSON格式
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data = file_data
                elif isinstance(file_data, dict):
                    # 如果是字典，尝试提取列表
                    for key, value in file_data.items():
                        if isinstance(value, list):
                            data = value
                            break
                    if not data:
                        data = [file_data]  # 单个对象
        
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"从 {test_file} 加载了 {len(data)} 个测试样本")
        return data
    
    def _save_processed_data(self, data: List[Dict], output_file: str):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.jsonl':
            # 保存为JSONL格式
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            # 保存为JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """关闭系统"""
        self.system.close()


def main():
    """
    主函数 - 演示完整流程
    """
    print("🔄 测试集处理完整流程演示")
    print("=" * 60)
    
    try:
        # 初始化处理器
        processor = TestSetProcessor()
        
        # 阶段1: 存储数据
        print("\n📊 阶段1: 存储数据到向量数据库")
        entities_file = "../extracted_entities_by_language.json"
        sentences_file = "../extracted_sentences_with_ner_by_language.json"
        
        # 检查数据文件
        if not os.path.exists(entities_file):
            print(f"❌ 实体数据文件不存在: {entities_file}")
            return
        
        if not os.path.exists(sentences_file):
            print(f"❌ 句子数据文件不存在: {sentences_file}")
            return
        
        # 设置数据库（演示用，只使用部分语种）
        demo_languages = ["en", "de"]
        processor.setup_database(entities_file, sentences_file, demo_languages)
        print("✅ 数据存储完成")
        
        # 阶段2: 处理测试输入
        print("\n🧪 阶段2: 处理测试集输入")
        
        # 示例测试输入
        test_cases = [
            {"input": "It is located in the San'a' Governorate.", "language": "en"},
            {"input": "Barack Obama was born in Hawaii.", "language": "en"},
            {"input": "Berlin ist die Hauptstadt von Deutschland.", "language": "de"},
        ]
        
        print("处理示例测试输入:")
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 测试样本 {i}:")
            print(f"   输入: {test_case['input']}")
            print(f"   语种: {test_case['language']}")
            
            # 生成instruction
            instruction = processor.process_test_input(
                test_case["input"], 
                test_case["language"]
            )
            
            result = {
                "test_case": test_case,
                "generated_instruction": instruction
            }
            results.append(result)
            
            print("   ✅ Instruction生成完成")
        
        # 保存示例结果
        example_output = "test_instructions_example.json"
        with open(example_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 示例结果已保存到: {example_output}")
        
        # 显示第一个生成的instruction
        print("\n📋 第一个生成的Instruction示例:")
        print("=" * 80)
        print(results[0]["generated_instruction"])
        print("=" * 80)
        
        # 关闭系统
        processor.close()
        
        print("\n🎉 完整流程演示完成!")
        print("\n📖 使用方法:")
        print("   1. 确保Milvus服务运行")
        print("   2. 准备实体和句子数据文件")
        print("   3. 调用 processor.setup_database() 存储数据")
        print("   4. 调用 processor.process_test_input() 处理测试输入")
        print("   5. 或使用 processor.process_test_file() 批量处理文件")
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise


if __name__ == "__main__":
    main()