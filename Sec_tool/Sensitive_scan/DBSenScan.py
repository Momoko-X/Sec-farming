import re
import sqlite3
from sentence_transformers import SentenceTransformer, util

# ======== 1. 正则 + 校验算法 ========
re_idcard = re.compile(r"\b\d{17}[0-9Xx]\b")
re_phone = re.compile(r"1[3-9]\d{9}")
re_bank = re.compile(r"\b\d{13,19}\b")

# 身份证校验位
def validate_china_id(id):
    if len(id) != 18:
        return False
    factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    checkmap = "10X98765432"
    s = sum(int(id[i]) * factors[i] for i in range(17))
    return checkmap[s % 11] == id[-1].upper()

# Luhn 校验（银行卡）
def luhn(card):
    digits = [int(d) for d in card]
    checksum = 0
    odd = True
    for d in digits[::-1]:
        if odd:
            checksum += d
        else:
            d = d * 2
            checksum += d - 9 if d > 9 else d
        odd = not odd
    return checksum % 10 == 0


# ======== 2. 字段名语义模型 ========
model = SentenceTransformer("shibing624/text2vec-base-chinese")
sensitive_names = [
    "身份证号", "手机号", "银行卡号", "客户姓名", "住址", "设备号", "邮箱"
]
sens_emb = model.encode(sensitive_names, convert_to_tensor=True)

def column_name_score(col):
    emb = model.encode(col, convert_to_tensor=True)
    sim = util.cos_sim(emb, sens_emb).max().item()
    return sim
#计算数据库字段名与敏感信息字段名的语义相似度，返回一个 0-1 之间的分数。

# ======== 3. 字段值分析 ========
def analyze_value(v):
    if not isinstance(v, str):
        v = str(v)

    results = []

    # 手机号
    m = re_phone.search(v)
    if m:
        results.append(("phone", 1.0))

    # 身份证
    m = re_idcard.search(v)
    if m and validate_china_id(m.group()):
        results.append(("idcard", 1.0))

    # 银行卡
    m = re_bank.search(v)
    if m and luhn(m.group()):
        results.append(("bankcard", 1.0))

    return results


# ======== 4. Demo：数据库扫描 ========
def scan_table(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table} LIMIT 20")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] #提取全部列名

    print(f"扫描表：{table}\n")

    for idx, col in enumerate(cols):
        name_conf = column_name_score(col)
        values = [str(r[idx]) for r in rows if r[idx] is not None]

        match_results = []
        for v in values:
            match_results += analyze_value(v)

        # 字段值置信度
        if match_results:
            value_type = match_results[0][0]
            value_conf = 1.0
        else:
            value_type = None
            value_conf = 0.0

        final_score = 0.6 * value_conf + 0.4 * name_conf

        print(f"字段：{col}")
        print(f"  字段名语义置信度：{name_conf:.3f}")
        print(f"  字段值识别：{value_type}")
        print(f"  最终敏感评分：{final_score:.3f}\n")


# ======== 5. main ========
if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")#创建一个内存数据库（in-memory database）。
    cur = conn.cursor()

    # 创建样例表
    cur.execute("""CREATE TABLE users (
        id INT,
        id_no TEXT,
        phone TEXT,
        remark TEXT
    )""")

    cur.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", [
        (1, "310101199001018888", "13812345678", "测试用户1"),
        (2, "not-idcard", "not-phone", "银行卡：6228480031452339821"),
    ])
    conn.commit()

    scan_table(conn, "users")
