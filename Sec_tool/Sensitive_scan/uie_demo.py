from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json

class UIETester:
    def __init__(self):
        print("[-] 正在加载 ModelScope UIE 模型 (第一次运行需要下载约 400MB+ 模型)...")
        # 加载通用信息抽取 pipeline
        # model 参数指定为你选中的 iic/nlp_structbert_siamese-uie_chinese-base
        self.uie_pipeline = pipeline(
            task=Tasks.information_extraction, 
            model='iic/nlp_structbert_siamese-uie_chinese-base'
        )
        print("[+] 模型加载完成！")

    def scan(self, text, targets):
        """
        灵活扫描：通过 targets 定义你想找什么，而不是模型定死找什么
        """
        # 定义 Schema (提取目标)
        # 格式：{'标签名': None} - 这是 ModelScope UIE 的正确格式
        # 例子: ['姓名', '身份证'] -> {'姓名': None, '身份证': None}
        schema = {target: None for target in targets}
        
        # 运行模型
        try:
            result = self.uie_pipeline(input=text, schema=schema)
            return self._parse_result(result)
        except Exception as e:
            print(f"检测出错: {e}")
            return []

    def _parse_result(self, raw_result):
        """
        将复杂的 UIE 输出结构清洗为产品易读的格式
        """
        findings = []
        # raw_result 的结构是：{'标签名': [{'text': '...', 'probs': 0.99, ...}, ...], ...}
        for label, items in raw_result.items():
            if items:  # 如果该标签有提取结果
                for item in items:
                    findings.append({
                        "text": item['text'],
                        "biz_tag": label,                    # 标签名（如姓名、身份证）
                        "confidence": item.get('probs', 0.0), # 置信度
                        "span": item.get('span', []),        # 位置信息（如果有）
                        "source": "UIE_Model"
                    })
        return findings

# ============================
# 产品经理验证环节
# ============================

if __name__ == "__main__":
    tester = UIETester()

    # 场景一：常规敏感信息（和 HanLP 能力重叠，但看置信度）
    text_1 = "嫌疑人张三（身份证号：110101199001011234）居住在北京市朝阳区。"
    targets_1 = ['姓名', '身份证号', '详细地址']

    # 场景二：业务特定敏感信息（这是 UIE 的杀手锏，HanLP 做不到）
    # 假设我们要检测合同或法律文书中的角色
    text_2 = "本合同由甲方（阿里云计算有限公司）与乙方（致远协创软件）签署，涉及金额为500万元人民币。"
    targets_2 = ['甲方', '乙方', '合同金额']

    print("\n" + "="*50)
    print("   UIE 模型 (Zero-Shot) 灵活性验证")
    print("="*50)

    # 测试场景 1
    print(f"\n[测试 1] 原文: {text_1}")
    print(f" -> 定义提取目标: {targets_1}")
    res1 = tester.scan(text_1, targets_1)
    for r in res1:
        # UIE 的结构里，biz_tag 就是我们在 target 里定义的 key
        # 注意：UIE 基础版可能直接返回 key 为类型，这里做简单适配打印
        print(f"    [命中] {r['text']} (置信度: {r['confidence']:.4f})")

    # 测试场景 2
    print(f"\n[测试 2] 原文: {text_2}")
    print(f" -> 定义提取目标: {targets_2}")
    res2 = tester.scan(text_2, targets_2)
    for r in res2:
        # 这里的 key 应该是我们定义的 '甲方', '乙方' 等
        # 在 ModelScope 输出结构中，schema 的 key 是分类，value 是提取结果
        # 简单打印
         print(f"    [命中] {r['text']} (置信度: {r['confidence']:.4f})")
    
    # 补充说明：
    # 上面的解析代码 _parse_result 做了简化。
    # 实际 UIE 返回结构可能是：{'敏感实体': [{'text': '张三', 'type': '姓名', ...}, {'text': '110...', 'type': '身份证号'}]}
    # 这完全取决于 schema 怎么写。