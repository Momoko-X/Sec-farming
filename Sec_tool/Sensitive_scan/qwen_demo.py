import os
import json
from openai import OpenAI
from typing import List, Dict, Optional

class QianwenSensitiveDetector:
    """
    使用千问 API 进行敏感信息检测
    支持检测：手机号、身份证号、姓名、银行卡号、邮箱、地址等
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化千问客户端
        :param api_key: API Key，如果不提供则从环境变量或默认值读取
        :param base_url: API 基础 URL
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY", "sk-4552c3f158e24650b65e82b9b16c2ae1"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-plus"  # 可以使用 qwen-plus, qwen-turbo 等
        
    def detect(self, text: str, target_types: Optional[List[str]] = None) -> List[Dict]:
        """
        检测文本中的敏感信息
        :param text: 待检测的文本
        :param target_types: 要检测的敏感信息类型列表，如果为 None 则检测所有常见类型
        :return: 检测结果列表，每个结果包含 type, value, position 等信息
        """
        if not text:
            return []
        
        # 默认检测类型
        if target_types is None:
            target_types = ['手机号', '身份证号', '姓名', '银行卡号', '邮箱', '详细地址', 'IP地址']
        
        # 构建 prompt
        prompt = self._build_prompt(text, target_types)
        
        try:
            # 调用千问 API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一个专业的敏感信息检测助手。你需要从文本中准确识别并提取敏感信息，并以 JSON 格式返回结果。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                temperature=0.1,  # 降低温度以提高准确性，数值越低越准确
            )
            
            # 解析返回结果
            response_text = completion.choices[0].message.content
            return self._parse_response(response_text, text)
            
        except Exception as e:
            print(f"检测出错: {e}")
            return []
    
    def _build_prompt(self, text: str, target_types: List[str]) -> str:
        """
        构建检测 prompt
        """
        types_str = '、'.join(target_types)
        prompt = f"""请从以下文本中检测并提取所有敏感信息。注意：信息可能隐藏在自然语言中，没有明确的标签，格式可能不标准（包含空格、分隔符等）。

需要检测的类型：{types_str}

检测规则（请仔细识别，即使格式不标准也要提取）：
1. 手机号：11位数字，通常以1开头。即使中间有空格或分隔符也要识别（如：138 0013 8000、138-0013-8000、13800138000）
2. 身份证号：18位或15位数字，可能包含X。即使有空格也要识别（如：110101 19900101 1234、110101199001011234、11010119900101123X）
3. 姓名：中文姓名，通常2-4个汉字。需要从上下文中识别，即使没有"姓名"、"叫"等明确标签
4. 银行卡号：16-19位连续数字。即使有分隔符也要识别（如：6222-8888-9999-1234、6222888899991234）
5. 邮箱：包含@符号的邮箱地址，格式为xxx@xxx.xxx（如：test@example.com、user.name@company.com.cn）
6. 详细地址：包含省市区街道的完整地址信息，即使没有"地址"、"住址"等标签也要识别
7. IP地址：IPv4格式，四组数字用点分隔（如：192.168.1.1）

重要提示：
- 信息可能分散在文本的不同位置
- 数字可能包含空格、横线等分隔符，需要识别并提取纯数字
- 姓名可能出现在"叫"、"是"、"联系人"等词之后，也可能直接出现
- 地址可能包含"在"、"住"、"位于"等词，也可能直接出现
- 不要被其他数字（如日期、编号、金额等）干扰

请以 JSON 数组格式返回结果，每个结果包含以下字段：
- type: 敏感信息类型（如：手机号、身份证号等）
- value: 检测到的具体值（去除空格和分隔符，保留原始格式中的纯值）
- start: 在原文中的起始位置（字符索引，从0开始）
- end: 在原文中的结束位置（字符索引，不包含）

如果未检测到任何敏感信息，返回空数组 []。

文本内容：
{text}

请直接返回 JSON 格式，不要包含其他说明文字："""
        return prompt
    
    def _parse_response(self, response_text: str, original_text: str) -> List[Dict]:
        """
        解析千问返回的结果
        """
        findings = []
        
        try:
            # 尝试提取 JSON 部分（可能包含 markdown 代码块）
            response_text = response_text.strip()
            
            # 如果包含代码块标记，提取其中的 JSON
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            
            # 解析 JSON
            results = json.loads(response_text)
            
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'type' in item and 'value' in item:
                        findings.append({
                            'type': item['type'],
                            'value': item['value'],
                            'start': item.get('start', -1),
                            'end': item.get('end', -1),
                            'source': 'Qianwen_API'
                        })
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始响应: {response_text}")
        except Exception as e:
            print(f"解析响应时出错: {e}")
        
        return findings
    
    def format_output(self, findings: List[Dict]) -> str:
        """
        格式化输出检测结果
        """
        if not findings:
            return "未检测到敏感信息"
        
        output_lines = []
        output_lines.append(f"检测到 {len(findings)} 条敏感信息：\n")
        
        for i, finding in enumerate(findings, 1):
            output_lines.append(f"{i}. [{finding['type']}] {finding['value']}")
            if finding['start'] >= 0 and finding['end'] >= 0:
                output_lines.append(f"   位置: {finding['start']}-{finding['end']}")
        
        return '\n'.join(output_lines)


# ============================
# 测试和演示
# ============================

if __name__ == "__main__":
    print("="*60)
    print("  千问 API 敏感信息检测演示")
    print("="*60)
    
    # 初始化检测器
    detector = QianwenSensitiveDetector()
    
    # 测试案例 1：信息隐藏在对话中，格式不标准
    print("\n[测试案例 1] 对话中的敏感信息（格式不标准）")
    text1 = "昨天跟王小明聊了会，他说他的号码是135 1234 5678，让我记一下。还有他住址是北京市海淀区中关村大街1号，回头寄个快递过去。对了，他那个证件号码是110101199001011234，记得帮他办手续。"
    print(f"原文: {text1}\n")
    
    results1 = detector.detect(text1)
    print(detector.format_output(results1))
    
    # 测试案例 2：信息分散，有干扰数字
    print("\n" + "-"*60)
    print("[测试案例 2] 信息分散，包含干扰数字")
    text2 = "会议记录：2024年3月15日，参会人员包括张伟、李娜。会议编号20240315001。会后需要联系张伟，他的联系方式是15987654321。另外，李娜的工号是2024，她的邮箱是lina.work@company.com.cn，记得发会议纪要给她。项目预算500万元，合同编号CT2024001。"
    print(f"原文: {text2}\n")
    
    results2 = detector.detect(text2)
    print(detector.format_output(results2))
    
    # 测试案例 3：银行卡号隐藏在转账信息中
    print("\n" + "-"*60)
    print("[测试案例 3] 转账信息中的银行卡号（无明确标签）")
    text3 = "财务通知：本月工资已发放，请查收。收款账户尾号1234，开户行工商银行。如有疑问，请联系财务部赵敏，电话18600001234，或发邮件至finance@company.com。"
    print(f"原文: {text3}\n")
    
    results3 = detector.detect(text3)
    print(detector.format_output(results3))
    
    # 测试案例 4：身份证号在非正式文本中
    print("\n" + "-"*60)
    print("[测试案例 4] 非正式文本中的身份证信息")
    text4 = "帮朋友办个事，他叫陈建国，91年1月1号生的，身份证是32010119910101123X，老家江苏南京的。现在在深圳工作，住南山区科技园那边。他手机我记在备忘录里了，是13712345678。"
    print(f"原文: {text4}\n")
    
    results4 = detector.detect(text4)
    print(detector.format_output(results4))
    
    # 测试案例 5：混合格式，有分隔符和空格
    print("\n" + "-"*60)
    print("[测试案例 5] 混合格式，包含分隔符和空格")
    text5 = "客户信息：姓名-刘德华，联系电话：1 3 9 0 1 2 3 4 5 6 7，邮箱地址：andy.lau@email.com，居住地址：香港特别行政区九龙尖沙咀弥敦道100号。银行账号：6222-8888-9999-1234，开户日期2020年。"
    print(f"原文: {text5}\n")
    
    results5 = detector.detect(text5)
    print(detector.format_output(results5))
    
    # 测试案例 6：极简文本，信息高度隐蔽
    print("\n" + "-"*60)
    print("[测试案例 6] 极简文本，信息高度隐蔽")
    text6 = "周杰伦，19800118，台北，0912345678，jay@jvr.com.tw，台北市信义区信义路五段7号"
    print(f"原文: {text6}\n")
    
    results6 = detector.detect(text6)
    print(detector.format_output(results6))
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)

