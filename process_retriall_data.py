#!/usr/bin/env python3
"""
RetriAll数据处理脚本
用于处理RetriAll_data目录下的JSONL文件，为每个句子生成增强的NER指令
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage2_retrieval_service import Stage2RetrievalService
from config import STAGE2_CONFIG, LOGGING_CONFIG

def setup_logging():
    """设置日志配置"""
    # 确保日志目录存在
    log_dir = Path(LOGGING_CONFIG["log_file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.handlers.RotatingFileHandler(
                LOGGING_CONFIG["log_file"],
                maxBytes=LOGGING_CONFIG["max_file_size"],
                backupCount=LOGGING_CONFIG["backup_count"],
                encoding='utf-8'
            ) if LOGGING_CONFIG["enable_file_logging"] else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🚀 开始处理RetriAll数据...")
    
    try:
        # 创建检索服务处理器
        service = Stage2RetrievalService()
        service.initialize_components()
        
        # 获取配置
        jsonl_directories = STAGE2_CONFIG['input']['jsonl_directories']
        input_field = STAGE2_CONFIG['input']['input_field']
        output_directory = STAGE2_CONFIG['output']['output_directory']
        
        logger.info(f"📂 输入目录: {jsonl_directories}")
        logger.info(f"📝 输入字段: {input_field}")
        logger.info(f"📁 输出目录: {output_directory}")
        
        # 处理JSONL文件
        results = service.process_jsonl_files(
            jsonl_directories=jsonl_directories,
            input_field=input_field,
            output_directory=output_directory
        )
        
        # 输出处理结果
        logger.info("🎉 RetriAll数据处理完成!")
        logger.info(f"📊 处理统计: {json.dumps(results, indent=2, ensure_ascii=False)}")
        
        # 保存处理报告
        report_file = Path(output_directory) / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 处理报告已保存到: {report_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 处理RetriAll数据时出错: {e}")
        raise

if __name__ == "__main__":
    main()