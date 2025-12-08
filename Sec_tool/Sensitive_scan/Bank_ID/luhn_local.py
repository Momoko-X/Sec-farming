def luhn_check(card_number: str) -> bool:
    """
    本地版 Luhn 校验
    :param card_number: 字符串格式的卡号
    :return: True=校验通过, False=校验失败
    """
# 这段代码实现了 Luhn 算法的核心步骤：
# 反转卡号，从右到左处理
# 对偶数位（从右数第 2、4、6... 位）乘以 2
# 如果乘积大于 9，减去 9
# 累加所有数字
# 最后检查总和是否能被 10 整除（在函数最后一行：return total % 10 == 0）
# 这是银行卡号校验的标准算法，用于检测卡号输入错误。

    # 必须是数字
    if not card_number.isdigit():
        return False

    total = 0
    reverse_digits = card_number[::-1]

    for i, ch in enumerate(reverse_digits):
        digit = int(ch)

        # 从右起（索引0）开始，每隔一位 *2
        if i % 2 == 1:
            digit = digit * 2
            if digit > 9:
                digit -= 9

        total += digit

    # 结果必须 mod10 == 0
    return total % 10 == 0


# 示例测试
if __name__ == "__main__":
    tests = [
        "6222021111111111",
        "6216611234567890",
        "49927398716",     # 标准示例（合法）
        "49927398717",     # 标准示例（不合法）
        "1234567812345670",
        "86011110001764441"
    ]

    for t in tests:
        print(t, "→", "合法" if luhn_check(t) else "不合法")
