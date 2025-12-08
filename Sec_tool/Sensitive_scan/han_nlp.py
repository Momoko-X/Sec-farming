import hanlp
import re

class HybridScanner:
    def __init__(self):
        print("[-] 正在加载 HanLP 多任务流水线...")
        try:
            # 加载 NLP 模型
            self.pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        except Exception as e:
            self.pipeline = hanlp.load('CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH')
        
        # 定义正则规则 (弥补模型的短板)
        self.regex_rules = [
            {
                "name": "手机号", 
                "tag": "PHONE", 
                "pattern": re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
            },
            {
                "name": "电子邮箱", 
                "tag": "EMAIL", 
                "pattern": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
            }
        ]
        print("[+] 混合引擎加载完成！(模型 + 正则)")

    def scan(self, text):
        findings = []

        # ==========================
        # 步骤 1: 正则扫描 (处理强规则)
        # ==========================
        for rule in self.regex_rules:
            for match in rule['pattern'].finditer(text):
                findings.append({
                    "text": match.group(),
                    "biz_tag": rule['name'],
                    "model_tag": "REGEX",
                    "start": match.start(), # 记录位置
                    "end": match.end(),
                    "source": "Regex_Engine"
                })

        # ==========================
        # 步骤 2: 模型扫描 (处理语义实体)
        # ==========================
        try:
            doc = self.pipeline(text)
            
            # 统一处理 HanLP 输出（可能是字典或 Document 对象）
            ner_data = None
            
            # 方法1: 尝试作为字典访问
            if isinstance(doc, dict):
                # 寻找 NER 结果 - 尝试多种可能的键名
                possible_keys = [k for k in doc.keys() if 'ner' in k.lower()]
                if possible_keys:
                    ner_data = doc[possible_keys[0]]
                else:
                    # 尝试常见的键名
                    for key in ['ner', 'NER', 'ner/con', 'ner/msra', 'ner/ontonotes']:
                        if key in doc:
                            ner_data = doc[key]
                            break
            else:
                # 方法2: 尝试作为对象访问（Document 对象通常支持字典式访问）
                try:
                    # 尝试字典式访问
                    for key in ['ner', 'ner/con', 'ner/msra', 'ner/ontonotes']:
                        try:
                            if key in doc:
                                ner_data = doc[key]
                                break
                        except (KeyError, TypeError):
                            continue
                    
                    # 如果还是没找到，尝试属性访问
                    if ner_data is None and hasattr(doc, 'ner'):
                        ner_data = doc.ner
                except Exception:
                    pass
            
            if ner_data:
                # 处理不同的输出格式
                flat_entities = []
                
                # 情况1: 嵌套列表 [[[实体]]] - 分句后的格式
                if isinstance(ner_data, list) and len(ner_data) > 0:
                    if isinstance(ner_data[0], list) and len(ner_data[0]) > 0:
                        if isinstance(ner_data[0][0], list):
                            # 三层嵌套：[[[实体]]]
                            flat_entities = [ent for sent in ner_data for ent in sent]
                        else:
                            # 两层嵌套：[[实体]] 或 已经是 [实体]
                            # 检查是否是实体格式 [text, tag] 或 (text, tag)
                            if len(ner_data[0]) >= 2:
                                flat_entities = ner_data
                            else:
                                flat_entities = ner_data
                    else:
                        flat_entities = ner_data
                else:
                    flat_entities = ner_data if isinstance(ner_data, list) else []

                # 处理实体
                for entity in flat_entities:
                    # 处理不同的实体格式
                    text_val = None
                    tag_val = None
                    
                    if isinstance(entity, (list, tuple)) and len(entity) >= 2:
                        # 格式1: ['文本', '标签'] 或 ('文本', '标签')
                        text_val = entity[0]
                        tag_val = entity[1]
                    elif isinstance(entity, dict):
                        # 格式2: {'text': '...', 'label': '...'} 或其他字典格式
                        text_val = entity.get('text') or entity.get('word') or entity.get('entity')
                        tag_val = entity.get('label') or entity.get('tag') or entity.get('type')
                    
                    if text_val and tag_val:
                        mapped_tag = self._map_tag(tag_val)
                        if mapped_tag:
                            findings.append({
                                "text": text_val,
                                "biz_tag": mapped_tag,
                                "model_tag": tag_val,
                                "source": "HanLP_AI"
                            })
                        # else:
                        #     # 调试：显示未映射的标签（帮助排查问题）
                        #     print(f"  [调试] 未映射的实体: '{text_val}' -> 标签: '{tag_val}'")
        except Exception as e:
            # 如果模型处理出错，不影响正则结果
            print(f"[警告] AI 模型处理出错: {e}")
            import traceback
            traceback.print_exc()

        return findings

    def _map_tag(self, tag):
        """
        标签映射：将模型输出的各种标签格式映射为业务标签
        """
        if not tag:
            return None
        
        tag = str(tag).upper().strip()
        
        # 处理带前缀的标签，如 "B-NT", "I-NT", "S-ORG" 等
        if '-' in tag:
            tag = tag.split('-')[-1]  # 取最后一部分
        
        # 人名标签
        if tag in ['NR', 'PERSON', 'PER', 'PERS', 'NAME']:
            return "中文姓名"
        
        # 地名标签
        if tag in ['NS', 'LOC', 'GPE', 'LOCATION', 'PLACE']:
            return "地址/地名"
        
        # 机构/公司标签
        if tag in ['NT', 'ORG', 'ORGANIZATION', 'COMPANY', 'CORP', 'CORPORATION']:
            return "公司/机构名"
        
        return None

# ============================
# 验证入口
# ============================

if __name__ == "__main__":
    scanner = HybridScanner()

    test_cases = [
        "张三和李四在北京市海淀区百度大厦开会。", # 典型的人名、地名、机构
        "我的名字叫高德。",                 # 歧义测试：高德是公司还是地图还是人名？
        "请把文件发给马云。",               # 名人测试
        "联系电话是13800138000。",          # 负向测试：HanLP不应该识别手机号（正则的事）
        "致远协创软件有限公司中标了。", 
        "华安基金公司。", 
    ]
    
    # # 临时调试：查看 HanLP 原始输出（针对机构名测试用例）
    # print("\n[DEBUG] 查看 HanLP 对机构名的识别结果...")
    # test_text = "上海华安基金管理有限公司"
    # doc = scanner.pipeline(test_text)
    # print(f"doc type: {type(doc)}")
    # 
    # # 尝试获取所有可能的 NER 数据
    # ner_data = None
    # if isinstance(doc, dict):
    #     print(f"doc keys: {list(doc.keys())}")
    #     for key in doc.keys():
    #         if 'ner' in key.lower():
    #             print(f"\nNER key '{key}':")
    #             print(f"  type: {type(doc[key])}")
    #             print(f"  value: {doc[key]}")
    #             if not ner_data:
    #                 ner_data = doc[key]
    # else:
    #     # 尝试字典式访问
    #     for key in ['ner', 'ner/con', 'ner/msra', 'ner/ontonotes']:
    #         try:
    #             if key in doc:
    #                 print(f"\n找到 NER key: {key}")
    #                 print(f"  value: {doc[key]}")
    #                 if not ner_data:
    #                     ner_data = doc[key]
    #                 break
    #         except:
    #             pass
    # 
    # # 打印所有识别到的实体（包括未映射的）
    # if ner_data:
    #     print(f"\n所有识别到的实体（原始格式）:")
    #     if isinstance(ner_data, list):
    #         for i, item in enumerate(ner_data):
    #             print(f"  [{i}] {item} (type: {type(item)})")
    #     else:
    #         print(f"  {ner_data}")
    # print("\n" + "="*50)

    print("\n" + "="*50)
    print("   混合敏感信息检测 (AI + Regex)")
    print("="*50)

    for text in test_cases:
        print(f"\n原文: {text}")
        results = scanner.scan(text)
        
        if not results:
            print("  -> [安全] 未发现敏感实体")
        else:
            for res in results:
                # 打印出是谁发现的 (Source)
                print(f"  -> [命中] {res['biz_tag']}: {res['text']} (引擎: {res['source']})")