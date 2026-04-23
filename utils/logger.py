from collections import defaultdict
import pandas as pd
import os
import torch
from typing import Dict, Any, List, Tuple, Set, Union, Optional
import argparse


class MetricsTracker:
    """
    记录并保存训练指标到不同的sheet中，支持索引字段(prefix)和指标值覆盖。
    """
    def __init__(self, filename: str, args=None):
        # sheet_name -> (prefix_values) -> {metric_name: value}
        # 使用嵌套字典而非列表，支持根据前缀索引覆盖数据
        self.data = defaultdict(dict)
        self.filename = filename
        if args is not None:
            self.save_args(args)
        # 存储每个sheet需要的prefix字段名
        self.sheet_prefixes = {}
        
    def set_sheet_prefix(self, sheet_name: str, prefix_keys: List[str]):
        """
        为指定sheet设置必须的索引字段(prefix)
        
        Args:
            sheet_name: sheet名称
            prefix_keys: 该sheet必须的索引字段列表
        """
        self.sheet_prefixes[sheet_name] = prefix_keys
        # 确保该sheet存在于data中
        if sheet_name not in self.data:
            self.data[sheet_name] = {}
    
    def track(self, metrics: Dict[str, Any], sheet_name: str, prefix_values: Dict[str, Any]):
        """
        记录指标值到指定sheet，根据prefix_values确定记录位置，支持覆盖
        
        Args:
            metrics: 指标名称和值的字典
            sheet_name: 要记录到的sheet名称
            prefix_values: 索引字段及其值的字典，用于确定记录位置
        """
        # 检查sheet是否存在
        if sheet_name not in self.sheet_prefixes:
            raise KeyError(f"Sheet '{sheet_name}' 未设置，请先使用set_sheet_prefix进行设置")
        
        # 检查所有必须的prefix字段是否都提供了
        required_prefixes = self.sheet_prefixes[sheet_name]
        for prefix in required_prefixes:
            if prefix not in prefix_values:
                raise ValueError(f"缺少必需的索引字段 '{prefix}' 用于sheet '{sheet_name}'")
        
        # 构建prefix值的元组作为key
        prefix_key = tuple(prefix_values[prefix] for prefix in required_prefixes)
        
        # 如果该prefix_key不存在，创建一个新的字典
        if prefix_key not in self.data[sheet_name]:
            self.data[sheet_name][prefix_key] = {}
            
            # 将prefix值添加到数据中
            for i, prefix in enumerate(required_prefixes):
                self.data[sheet_name][prefix_key][prefix] = prefix_values[prefix]
        
        # 更新或添加metrics值，但不覆盖已有的其他指标
        if metrics is not None:
            # 将新的指标合并到现有数据中，而不是完全替换
            for name, value in metrics.items():
                # 如果值是tensor，转换为Python标量
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.tolist()
                # 直接覆盖或新增该指标
                self.data[sheet_name][prefix_key][name] = value
    
    def save_args(self, args: argparse.Namespace, sheet_name: str = "args"):
        """
        保存命令行参数到指定sheet，使用args.time作为值的列名
        
        Args:
            args: 命令行参数对象
            sheet_name: 要保存到的sheet名称，默认为"args"
        """
        # 确保sheet存在
        if sheet_name not in self.data:
            self.data[sheet_name] = {}
        
        # 获取时间戳作为列名
        time_col = getattr(args, 'time', 'value')
        
        # 将args转换为字典
        args_dict = vars(args)
        
        # 将每个参数保存为一行，但跳过time参数本身
        for arg_name, arg_value in args_dict.items():
            if arg_name == 'time':
                continue
                
            # 处理特殊类型，如列表、元组等
            if isinstance(arg_value, (list, tuple)):
                arg_value = str(arg_value)
            elif isinstance(arg_value, bool):
                arg_value = "True" if arg_value else "False"
            
            # 使用参数名作为key
            if (arg_name,) not in self.data[sheet_name]:
                self.data[sheet_name][(arg_name,)] = {"param": arg_name}
            
            # 将值保存在以time命名的列中
            self.data[sheet_name][(arg_name,)][time_col] = str(arg_value)
    
    def save(self):
        """
        将记录的指标保存为Excel文件，每个sheet对应一个表
        
        Args:
            filename: 保存的文件名，应包含路径和.xlsx扩展名
        """
            
        with pd.ExcelWriter(self.filename) as writer:
            for sheet_name, sheet_data in self.data.items():
                # 如果sheet没有数据，跳过
                if not sheet_data:
                    continue
                    
                # 将嵌套字典转换为DataFrame
                rows = []
                for _, metric_dict in sheet_data.items():
                    rows.append(metric_dict)
                
                if rows:
                    df = pd.DataFrame(rows)
                    
                    # 确保prefix列在最前面
                    if sheet_name in self.sheet_prefixes:
                        prefix_cols = self.sheet_prefixes[sheet_name]
                        # 重新排列列，使prefix列在前
                        all_cols = [col for col in prefix_cols if col in df.columns] + [
                            col for col in df.columns if col not in prefix_cols
                        ]
                        # 确保列存在
                        df = df[all_cols]
                    
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
    def get_sheet_data(self, sheet_name: str) -> pd.DataFrame:
        """
        获取指定sheet的数据作为DataFrame
        
        Args:
            sheet_name: sheet名称
            
        Returns:
            包含该sheet所有记录数据的DataFrame
        """
        if sheet_name not in self.data:
            raise KeyError(f"Sheet '{sheet_name}' 不存在")
            
        # 将嵌套字典转换为DataFrame
        rows = []
        for _, metric_dict in self.data[sheet_name].items():
            rows.append(metric_dict)
            
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        
        # 确保prefix列在最前面
        if sheet_name in self.sheet_prefixes:
            prefix_cols = self.sheet_prefixes[sheet_name]
            # 重新排列列，使prefix列在前
            all_cols = [col for col in prefix_cols if col in df.columns] + [
                col for col in df.columns if col not in prefix_cols
            ]
            # 确保列存在
            df = df[all_cols]
            
        return df
        
    def get_latest(self, sheet_name: str, metric_name: str, prefix_filters: Optional[Dict[str, Any]] = None) -> Any:
        """
        获取指定sheet和指标的最新值，可通过prefix_filters过滤
        
        Args:
            sheet_name: sheet名称
            metric_name: 指标名称
            prefix_filters: 可选，用于筛选特定prefix值的字典
            
        Returns:
            指标的最新值，如果不存在则返回None
        """
        if sheet_name not in self.data:
            return None
            
        # 获取该sheet的所有数据
        df = self.get_sheet_data(sheet_name)
        if df.empty or metric_name not in df.columns:
            return None
            
        # 应用prefix过滤
        if prefix_filters:
            for prefix, value in prefix_filters.items():
                if prefix in df.columns:
                    df = df[df[prefix] == value]
                    
        if df.empty:
            return None
            
        # 返回最后一行的值
        return df[metric_name].iloc[-1]

    def load(self, filename: str):
        """
        从Excel文件加载指标数据，保留不同time列的数据

        Args:
            filename: 包含路径的Excel文件名，应包含.xlsx扩展名
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件 '{filename}' 不存在")

        # 读取Excel文件
        xls = pd.ExcelFile(filename)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                continue

            # 获取sheet的索引字段（如果有）
            prefix_cols = self.sheet_prefixes.get(sheet_name, [])

            # 区分处理args sheet和其他sheet
            if sheet_name == "args" or not prefix_cols:
                # 处理args sheet或没有索引字段的sheet
                if sheet_name not in self.data:
                    self.data[sheet_name] = {}
                
                # 对于args sheet，使用param列作为key
                if "param" in df.columns:
                    for _, row in df.iterrows():
                        param_name = row["param"]
                        row_dict = row.to_dict()
                        
                        # 使用参数名作为key
                        if (param_name,) not in self.data[sheet_name]:
                            self.data[sheet_name][(param_name,)] = {"param": param_name}
                        
                        # 将所有其他列（时间戳列）添加到该参数行
                        for col, value in row_dict.items():
                            if col != "param":
                                self.data[sheet_name][(param_name,)][col] = value
                else:
                    # 对于没有param列的非索引sheet，使用行号作为key
                    for idx, row in df.iterrows():
                        row_dict = row.to_dict()
                        self.data[sheet_name][(idx,)] = row_dict
            else:
                # 有索引字段的sheet: 使用索引字段构建数据
                for _, row in df.iterrows():
                    prefix_values = {col: row[col] for col in prefix_cols if col in df.columns}
                    metrics = {col: row[col] for col in df.columns if col not in prefix_cols}
                    
                    # 确保所有必需的索引字段都存在
                    if len(prefix_values) == len(prefix_cols):
                        self.track(metrics, sheet_name, prefix_values)
                    else:
                        print(f"警告: 在sheet '{sheet_name}' 中发现缺少索引字段的行，已跳过")

    def clear_data(self):
        """
        清除所有记录的数据, 维持数据结构
        """
        for sheet_name in self.data:
            self.data[sheet_name].clear()


# 使用示例
if __name__ == "__main__":
    tracker = MetricsTracker()
    
    # 设置训练sheet，需要epoch和phase作为索引
    tracker.set_sheet_prefix("training", ["epoch", "phase"])
    
    # 记录训练数据
    for epoch in range(5):
        # 训练阶段
        tracker.track(
            {"loss": 0.5 - epoch * 0.1, "accuracy": 0.8 + epoch * 0.04}, 
            "training", 
            {"epoch": epoch, "phase": "train"}
        )
        
        # 验证阶段
        tracker.track(
            {"loss": 0.6 - epoch * 0.1, "accuracy": 0.75 + epoch * 0.05}, 
            "training", 
            {"epoch": epoch, "phase": "val"}
        )
    
    # 保存到Excel
    tracker.save("metrics.xlsx")
    
    # 获取最新的验证准确率
    latest_val_acc = tracker.get_latest("training", "accuracy", {"phase": "val"})
    print(f"最新验证准确率: {latest_val_acc}")