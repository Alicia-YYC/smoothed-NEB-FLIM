#!/usr/bin/env python3
"""
实验运行脚本
用于快速执行非参数贝叶斯FLIM分析实验
"""

import os
import sys
import time
from paper_experiment_reproduction_fixed import main

def run_experiments():
    """运行所有实验"""
    print("=" * 60)
    print("非参数贝叶斯FLIM分析实验")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 运行主实验
        main()
        
        end_time = time.time()
        print(f"\n实验完成！总耗时: {end_time - start_time:.2f} 秒")
        print("结果已保存到: paper_experiment_results_fixed.png")
        
    except Exception as e:
        print(f"实验运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_experiments()
