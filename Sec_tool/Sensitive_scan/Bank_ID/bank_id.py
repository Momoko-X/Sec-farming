#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银行卡号检测模块
-----------------------------------------
功能：
1. 检测银行卡号长度和数字格式
2. Luhn 校验算法验证
3. BIN 码匹配（前6位）
4. 输出检测结果到 result.txt
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==========================================================
# 数据加载：解析 TypeScript 文件
# ==========================================================
def load_banks(banks_file: Path) -> Dict[str, str]:
    """加载银行代码和名称对照表"""
    bank_dict = {}
    try:
        content = banks_file.read_text(encoding="utf-8", errors="ignore")
        # 使用正则表达式提取 { code: 'XXX', name: 'YYY' }
        pattern = r"\{\s*code:\s*['\"]([^'\"]+)['\"],\s*name:\s*['\"]([^'\"]+)['\"]\s*\}"
        matches = re.findall(pattern, content)
        for code, name in matches:
            bank_dict[code] = name
        print(f"[信息] 已加载 {len(bank_dict)} 个银行信息")
        return bank_dict
    except Exception as e:
        print(f"[错误] 加载银行信息失败：{e}")
        return {}


def load_bins(bin_file: Path) -> List[Dict]:
    """加载 BIN 码列表"""
    bin_list = []
    try:
        content = bin_file.read_text(encoding="utf-8", errors="ignore")
        # 使用正则表达式提取 { bin: 'XXX', bank: 'YYY', type: ZZZ, len: N }
        # 注意：type 可能是 DC, CC, SCC, PC 等常量
        pattern = r"\{\s*bin:\s*['\"]([^'\"]+)['\"],\s*bank:\s*['\"]([^'\"]+)['\"],\s*type:\s*([A-Z]+),\s*len:\s*(\d+)\s*\}"
        matches = re.findall(pattern, content)
        for bin_code, bank, card_type, length in matches:
            bin_list.append({
                "bin": bin_code,
                "bank": bank,
                "type": card_type,
                "len": int(length)
            })
        print(f"[信息] 已加载 {len(bin_list)} 个 BIN 码")
        return bin_list
    except Exception as e:
        print(f"[错误] 加载 BIN 码失败：{e}")
        return []


# ==========================================================
# Luhn 校验算法
# ==========================================================
def luhn_check(card_number: str) -> bool:
    """
    Luhn 算法校验银行卡号
    
    算法步骤：
    1. 从右到左，对偶数位数字乘以2
    2. 如果乘积是两位数，将各位数字相加（即减去9）
    3. 将所有数字相加
    4. 如果总和能被10整除，则卡号有效
    """
    # 移除空格和横杠
    card_number = re.sub(r"[\s-]", "", card_number)
    
    if not card_number.isdigit():
        return False
    
    # 从右到左处理
    total = 0
    reverse_digits = card_number[::-1]
    
    for i, digit in enumerate(reverse_digits):
        num = int(digit)
        if i % 2 == 1:  # 偶数位（从右数第2, 4, 6...位）
            num *= 2
            if num > 9:
                num -= 9  # 等价于各位数字相加
        total += num
    
    return total % 10 == 0


# ==========================================================
# BIN 匹配逻辑
# ==========================================================
def find_bin_match(card_number: str, bin_list: List[Dict]) -> Optional[Dict]:
    """
    查找匹配的 BIN 码
    使用最长匹配原则：优先匹配更长的 BIN 码（支持最长10位BIN码）
    """
    # 移除空格和横杠
    card_number = re.sub(r"[\s-]", "", card_number)
    
    if not card_number.isdigit():
        return None
    
    # 按 BIN 长度降序排序，优先匹配更长的（最长10位）
    #优先匹配最长 BIN 可以保证匹配结果最精确。因此先排序再匹配。
    sorted_bins = sorted(bin_list, key=lambda x: len(x["bin"]), reverse=True)
    
    # 尝试匹配（最长BIN码可达10位）
    for bin_info in sorted_bins:
        bin_code = bin_info["bin"]
        if card_number.startswith(bin_code):
            # 检查长度是否匹配
            if len(card_number) == bin_info["len"]:
                return bin_info
    
    return None


# ==========================================================
# 银行卡类型映射
# ==========================================================
TYPE_MAP = {
    "DC": "借记卡",
    "CC": "信用卡",
    "SCC": "准贷记卡",
    "PC": "预付卡"
}


# ==========================================================
# 主检测逻辑
# ==========================================================
def check_bank_card(card_number: str, bin_list: List[Dict], bank_dict: Dict[str, str]) -> Dict:
    """
    检测银行卡号
    
    返回：
    {
        "card_number": 卡号,
        "is_valid": 是否合法,
        "card_type": 卡类型,
        "bank": 银行名称,
        "reason": 原因说明
    }
    """
    result = {
        "card_number": card_number,
        "is_valid": False,
        "card_type": "",
        "bank": "",
        "reason": ""
    }
    
    # 清理卡号
    cleaned = re.sub(r"[\s-]", "", card_number)
    
    # 1. 长度和数字格式检查
    if not cleaned.isdigit():
        result["reason"] = "格式错误：包含非数字字符"
        return result
    
    if len(cleaned) < 13 or len(cleaned) > 19:
        result["reason"] = f"长度错误：{len(cleaned)} 位（应为 13-19 位）"
        return result
    
    # 2. Luhn 校验
    if not luhn_check(cleaned):
        result["reason"] = "Luhn 校验失败"
        return result
    
    # 3. BIN 检查
    bin_match = find_bin_match(cleaned, bin_list)
    if not bin_match:
        result["reason"] = "BIN 码无效：前6位未匹配到合法BIN码"
        return result
    
    # 检查长度是否匹配
    if len(cleaned) != bin_match["len"]:
        result["reason"] = f"长度不匹配：BIN码要求 {bin_match['len']} 位，实际 {len(cleaned)} 位"
        return result
    
    # 获取银行名称
    bank_code = bin_match["bank"]
    bank_name = bank_dict.get(bank_code, bank_code)
    
    # 获取卡类型
    card_type_code = bin_match["type"]
    card_type_name = TYPE_MAP.get(card_type_code, card_type_code)
    
    result.update({
        "is_valid": True,
        "card_type": card_type_name,
        "bank": bank_name,
        "reason": "合法"
    })
    
    return result


# ==========================================================
# 主执行逻辑
# ==========================================================
def main():
    """主函数"""
    # 文件路径
    base_dir = Path(__file__).parent
    banks_file = base_dir / "src" / "banks.ts"
    bin_file = base_dir / "src" / "bin.ts"
    input_file = base_dir / "bank_id.txt"
    output_file = base_dir / "result.txt"
    
    # 检查输入文件
    if not input_file.exists():
        print(f"[错误] 输入文件不存在：{input_file}")
        print(f"[提示] 请创建 {input_file} 文件，每行一个银行卡号")
        sys.exit(1)
    
    # 加载数据
    print("[信息] 正在加载银行和 BIN 码数据...")
    bank_dict = load_banks(banks_file)
    bin_list = load_bins(bin_file)
    
    if not bank_dict or not bin_list:
        print("[错误] 数据加载失败，程序中止")
        sys.exit(1)
    
    # 读取银行卡号
    print(f"[信息] 正在读取银行卡号：{input_file}")
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        card_numbers = [line.strip() for line in f if line.strip()]
    
    if not card_numbers:
        print("[错误] 输入文件为空")
        sys.exit(1)
    
    print(f"[信息] 共读取 {len(card_numbers)} 个银行卡号")
    
    # 检测每个卡号
    results = []
    for card_number in card_numbers:
        result = check_bank_card(card_number, bin_list, bank_dict)
        results.append(result)
        # 调试输出
        print(f"检测: {card_number[:6]}**** -> {'合法' if result['is_valid'] else '不合法'}: {result['reason']}")
    
    # 写入结果
    print(f"[信息] 正在写入结果：{output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("卡号\t合法\t卡类型\t银行\n")
        for r in results:
            f.write(
                f"{r['card_number']}\t"
                f"{'是' if r['is_valid'] else '否'}\t"
                f"{r['card_type']}\t"
                f"{r['bank']}\n"
            )
    
    # 统计
    valid_count = sum(1 for r in results if r['is_valid'])
    print(f"[完成] 检测完成！")
    print(f"    - 总计：{len(results)} 条")
    print(f"    - 合法：{valid_count} 条")
    print(f"    - 不合法：{len(results) - valid_count} 条")
    print(f"    - 结果已保存至：{output_file}")


if __name__ == "__main__":
    main()

