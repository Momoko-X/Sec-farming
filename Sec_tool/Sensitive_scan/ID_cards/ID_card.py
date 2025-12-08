#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
身份证号检测模块（增强版）
-----------------------------------------
功能：
1. 自动识别 region_codes 文件格式（.xlsx / .xls / .csv / .txt）
2. 自动清理身份证号空格、横杠等
3. 区码无效即判为不合法
4. 输出 result.txt，每条记录都有检测结果
5. 可命令行运行，也可作为模块使用
"""

import sys
import re
import pandas as pd
from datetime import datetime

# ==========================================================
# 行政区码表加载
# ==========================================================
def load_region_codes(region_file_path):
    region_dict = {}
    try:
        # 自动识别 Excel
        if region_file_path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(region_file_path, header=None, dtype=str)
            print(f"[信息] 已识别为 Excel 格式，加载中：{region_file_path}")
        else:
            encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "big5"]
            for enc in encodings:
                try:
                    df = pd.read_csv(region_file_path, header=None, dtype=str, encoding=enc)
                    print(f"[信息] 已识别为 CSV/TXT 格式，使用编码：{enc}")
                    break
                except Exception:
                    continue
        if df is None or df.empty:
            raise ValueError("文件为空或无法解析")

        count = 0
        for _, row in df.iterrows():
            line = " ".join(str(v) for v in row if pd.notna(v)).strip()
            match = re.match(r"^\s*(\d{6})[\s,，;、\t ]+(.+)$", line)
            if match:
                code, name = match.groups()
                region_dict[code] = name.strip()
                count += 1

        print(f"[信息] 行政区码表加载完成：{count} 条有效记录")
        return region_dict
    except Exception as e:
        print(f"[错误] 加载行政区码表失败：{e}")
        return {}

# ==========================================================
# 身份证号检测逻辑
# ==========================================================
def calc_check_digit(id17):
    """计算身份证校验位"""
    weight_factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_map = "10X98765432"
    s = sum(int(a) * b for a, b in zip(id17, weight_factors))
    return check_map[s % 11]

def check_id_card(id_number, region_dict):
    """单条身份证号检测"""
    result = {
        "id": id_number,
        "is_valid": False,
        "region": "",
        "birthday": "",
        "gender": "",
        "reason": "",
    }

    # 清理空格、横杠、不可见字符
    id_number = re.sub(r"[\s-]", "", id_number).upper()

    if not re.fullmatch(r"\d{15}|\d{17}[\dX]", id_number):
        result["reason"] = "格式错误"
        return result

    # 若是15位，转18位
    if len(id_number) == 15:
        id_number = id_number[:6] + "19" + id_number[6:]
        id_number += calc_check_digit(id_number)

    # 区码
    region_code = id_number[:6]
    province_code = id_number[:2] + "0000"
    city_code = id_number[:4] + "00"
    region_name = (
        region_dict.get(region_code)
        or region_dict.get(city_code)
        or region_dict.get(province_code)
    )

    # 区码无效判为不合法
    if not region_name:
        result["reason"] = "行政区码无效"
        result["region"] = "未知地区"
        return result

    # 出生日期校验
    try:
        birth = id_number[6:14]
        birth_date = datetime.strptime(birth, "%Y%m%d")
        now = datetime.now()
        age = now.year - birth_date.year
        if not (0 <= age <= 120):
            result["reason"] = "出生日期不合法"
            return result
        birthday = birth_date.strftime("%Y-%m-%d")
    except Exception:
        result["reason"] = "出生日期格式错误"
        return result

    # 校验位
    if calc_check_digit(id_number[:-1]) != id_number[-1]:
        result["reason"] = "校验位错误"
        return result

    # 性别
    gender_flag = int(id_number[16])
    gender = "男" if gender_flag % 2 == 1 else "女"

    result.update({
        "is_valid": True,
        "region": region_name,
        "birthday": birthday,
        "gender": gender,
        "reason": "合法",
    })
    return result

# ==========================================================
# 主执行逻辑
# ==========================================================
def main(id_file_path, region_file_path="region_codes.xlsx", output_file="result.txt"):
    region_dict = load_region_codes(region_file_path)
    if not region_dict:
        print("[错误] 未能加载任何行政区数据，程序中止。")
        return

    # 读取身份证号
    with open(id_file_path, "r", encoding="utf-8", errors="ignore") as f:
        ids = [line.strip() for line in f if line.strip()]

    results = []
    for idn in ids:
        res = check_id_card(idn, region_dict)
        results.append(res)
        # 调试输出每条结果
        print(res)

    # 写入结果
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("身份证号\t是否合法\t地区\t生日\t性别\t说明\n")
        for r in results:
            f.write(
                f"{r['id']}\t"
                f"{'是' if r['is_valid'] else '否'}\t"
                f"{r['region']}\t"
                f"{r['birthday']}\t"
                f"{r['gender']}\t"
                f"{r['reason']}\n"
            )

    print(f"[完成] 检测完成，共 {len(results)} 条结果，已输出至 {output_file}")

# ==========================================================
# 命令行入口
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 ID_card.py id.txt")
        sys.exit(1)

    id_file = sys.argv[1]
    region_file = "region_codes.xlsx"
    main(id_file, region_file)
