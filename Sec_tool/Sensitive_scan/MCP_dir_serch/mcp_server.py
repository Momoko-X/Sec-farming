#!/usr/bin/env python3
"""
MCP Server for Directory Scanner
将 dir_serch.py 封装为 MCP (Model Context Protocol) 工具
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# 导入 dir_serch 的核心功能
from dir_serch import (
    DEFAULT_WORDLIST,
    normalize_target,
    load_wordlist,
    load_targets,
    scan_target,
    save_json,
    save_csv,
)


class MCPServer:
    """MCP 服务器实现，通过 stdio 进行 JSON-RPC 通信"""
    
    def __init__(self):
        self.request_id = None
    
    def send_response(self, result: Any = None, error: Optional[Dict] = None):
        """发送 JSON-RPC 响应"""
        response = {
            "jsonrpc": "2.0",
            "id": self.request_id,
        }
        if error:
            response["error"] = error
        else:
            response["result"] = result
        
        print(json.dumps(response), flush=True)
    
    def handle_request(self, request: Dict):
        """处理 JSON-RPC 请求"""
        self.request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        # 如果是通知（没有 id），不发送响应
        is_notification = "id" not in request
        
        try:
            if method == "initialize":
                self.handle_initialize(params)
            elif method == "tools/list":
                if not is_notification:
                    self.handle_tools_list()
            elif method == "tools/call":
                if not is_notification:
                    self.handle_tools_call(params)
            elif method == "notifications/initialized":
                # 初始化通知，不需要响应
                pass
            else:
                if not is_notification:
                    self.send_response(error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    })
        except Exception as e:
            if not is_notification:
                self.send_response(error={
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                })
    
    def handle_initialize(self, params: Dict):
        """处理初始化请求"""
        self.send_response({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "directory-scanner",
                "version": "1.0.0"
            }
        })
    
    def handle_tools_list(self):
        """返回可用工具列表"""
        self.send_response({
            "tools": [
                {
                    "name": "scan_directory",
                    "description": "扫描目标网站的敏感路径和文件，检测目录遍历、敏感文件泄露等安全问题",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "目标 URL 或域名（例如：example.com 或 https://example.com）"
                            },
                            "targets_file": {
                                "type": "string",
                                "description": "目标列表文件路径（每行一个目标，可选）"
                            },
                            "wordlist": {
                                "type": "string",
                                "description": "路径字典文件路径（每行一个路径，可选，不提供则使用默认字典）"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "请求超时时间（秒，默认 8）",
                                "default": 8
                            },
                            "follow_redirects": {
                                "type": "boolean",
                                "description": "是否跟随 HTTP 重定向（默认 true）",
                                "default": True
                            },
                            "save_all": {
                                "type": "boolean",
                                "description": "是否保存所有请求结果（包括 404，默认 false）",
                                "default": False
                            },
                            "user_agent": {
                                "type": "string",
                                "description": "自定义 User-Agent（可选）",
                                "default": "DirScanSync/1.0"
                            },
                            "output_json": {
                                "type": "string",
                                "description": "JSON 输出文件路径（可选，不提供则不保存文件）"
                            },
                            "output_csv": {
                                "type": "string",
                                "description": "CSV 输出文件路径（可选）"
                            },
                            "output_html": {
                                "type": "string",
                                "description": "HTML 报告输出文件路径（可选，美化可视化报告）"
                            }
                        },
                        "required": ["target"]
                    }
                }
            ]
        })
    
    def handle_tools_call(self, params: Dict):
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "scan_directory":
            result = self.scan_directory(arguments)
            self.send_response({
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, ensure_ascii=False)
                    }
                ]
            })
        else:
            self.send_response(error={
                "code": -32602,
                "message": f"Unknown tool: {tool_name}"
            })
    
    def scan_directory(self, args: Dict) -> Dict[str, Any]:
        """执行目录扫描"""
        # 收集目标
        targets = []
        
        if args.get("target"):
            targets.append(normalize_target(args["target"]))
        
        if args.get("targets_file"):
            targets_file = Path(args["targets_file"])
            if targets_file.exists():
                targets.extend(load_targets(targets_file))
        
        if not targets:
            return {
                "error": "No targets provided. Please provide 'target' or 'targets_file'.",
                "results": []
            }
        
        # 加载路径字典
        if args.get("wordlist"):
            wordlist_path = Path(args["wordlist"])
            paths = load_wordlist(wordlist_path)
        else:
            paths = DEFAULT_WORDLIST.copy()
        
        # 准备请求头
        headers = {"User-Agent": args.get("user_agent", "DirScanSync/1.0")}
        
        # 扫描参数
        timeout = args.get("timeout", 8)
        follow_redirects = args.get("follow_redirects", True)
        save_all = args.get("save_all", False)
        
        # 执行扫描
        all_results = []
        scan_summary = []
        total_keyword_hits = 0
        
        for target in targets:
            results = scan_target(
                target=target,
                paths=paths,
                timeout=timeout,
                follow_redirects=follow_redirects,
                save_all=save_all,
                headers=headers
            )
            all_results.extend(results)
            # 统计关键词命中
            for r in results:
                if r.get("keyword_hits"):
                    total_keyword_hits += 1
            scan_summary.append({
                "target": target,
                "findings_count": len(results)
            })
        
        # 保存文件（如果指定）
        output_info = {}
        if args.get("output_json"):
            json_path = Path(args["output_json"])
            save_json(json_path, all_results)
            output_info["json_file"] = str(json_path)
        
        if args.get("output_csv"):
            csv_path = Path(args["output_csv"])
            save_csv(csv_path, all_results)
            output_info["csv_file"] = str(csv_path)

        # HTML 美化报告（可选）
        if args.get("output_html"):
            html_path = Path(args["output_html"])
            # 选择第一个目标用于标题展示（多目标时可扩展为多页）
            title_target = targets[0] if targets else ""
            # 构造摘要
            summary = {
                "target": title_target,
                "total_records": len(all_results),
                "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "total_keyword_hits": total_keyword_hits
            }
            self._render_html_report(all_results, summary, html_path)
            output_info["html_file"] = str(html_path)
        
        # 返回结果
        return {
            "success": True,
            "summary": {
                "targets_scanned": len(targets),
                "total_findings": len(all_results),
                "targets": scan_summary,
                "keyword_hit_records": total_keyword_hits
            },
            "results": all_results,
            "output_files": output_info
        }

    def _render_html_report(self, results: List[Dict[str, Any]], summary: Dict[str, Any], output_path: Path) -> None:
        """将扫描结果渲染为美化的 HTML 报告"""
        # 统计状态码分布
        status_counts: Dict[Any, int] = {}
        for r in results:
            status = r.get("status")
            status_counts[status] = status_counts.get(status, 0) + 1

        def html_escape(s: str) -> str:
            return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        rows: List[str] = []
        for idx, r in enumerate(results, 1):
            ok = r.get("ok")
            row_class = "ok" if ok else "fail"
            kh = r.get("keyword_hits") or []
            if kh:
                row_class += " keyword"
            snippet = html_escape(r.get("snippet") or "")
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            rows.append(f"""
            <tr class="{row_class}">
              <td>{idx}</td>
              <td>{html_escape(str(r.get('target', '')))}</td>
              <td>{html_escape(str(r.get('path', '')))}</td>
              <td><a href="{html_escape(str(r.get('url','')))}" target="_blank">{html_escape(str(r.get('url','')))}</a></td>
              <td>{html_escape(str(r.get('status')))}</td>
              <td>{html_escape(str(r.get('length')))}</td>
              <td>{html_escape(", ".join(kh) if kh else "-")}</td>
              <td><details><summary>查看</summary><pre>{snippet}</pre></details></td>
            </tr>
            """)

        status_items = "".join(
            f"<li><strong>{html_escape(str(code))}</strong>: {count} 条</li>"
            for code, count in sorted(status_counts.items(), key=lambda x: (x[0] is None, x[0]))
        ) or "<li>无</li>"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>目录扫描报告 - {html_escape(summary.get('target',''))}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',Arial,sans-serif; background: #f6f7fb; color:#1f2937; margin:0; }}
    header {{ background: linear-gradient(135deg,#2563eb,#1e40af); color:#fff; padding:28px 32px; }}
    header h1 {{ margin:0 0 6px 0; font-size:24px; }}
    header p {{ margin:4px 0; opacity:.95; }}
    main {{ padding: 24px 28px; }}
    .card {{ background:#fff; border-radius:12px; box-shadow: 0 10px 30px rgba(30,64,175,.12); padding:18px 18px; margin-bottom:22px; border:1px solid #e5e7eb; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
    .tile {{ background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:12px 14px; }}
    .tile h3 {{ margin:0; font-size:13px; color:#475569; }}
    .tile p {{ margin:6px 0 0 0; font-size:20px; font-weight:700; color:#0f172a; }}
    .tablewrap {{ overflow:auto; border-radius:10px; border:1px solid #e5e7eb; }}
    table {{ border-collapse:collapse; width:100%; min-width:1080px; }}
    th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #f1f5f9; font-size:14px; }}
    th {{ background:#f1f5f9; color:#0f172a; position:sticky; top:0; z-index:1; }}
    tr:nth-child(even) {{ background:#fbfdff; }}
    tr.ok {{ border-left:4px solid #22c55e; }}
    tr.fail {{ border-left:4px solid #ef4444; }}
    tr.keyword {{ background: rgba(244,114,182,.12); }}
    tr.keyword td {{ background: rgba(244,114,182,.06); }}
    details summary {{ cursor:pointer; color:#2563eb; }}
    pre {{ white-space: pre-wrap; background:#0f172a; color:#e2e8f0; padding:10px 12px; border-radius:8px; font-size:13px; }}
    ul {{ margin: 8px 0 0 18px; padding:0; }}
    .muted {{ color:#64748b; font-size:12px; }}
  </style>
</head>
<body>
  <header>
    <h1>目录扫描报告</h1>
    <p>目标：{html_escape(summary.get('target',''))}</p>
    <p class="muted">报告生成时间：{html_escape(summary.get('generated_at',''))}</p>
  </header>
  <main>
    <section class="card">
      <div class="grid">
        <div class="tile"><h3>记录总数</h3><p>{summary.get('total_records', 0)}</p></div>
        <div class="tile"><h3>关键词命中记录</h3><p>{summary.get('total_keyword_hits', 0)}</p></div>
        <div class="tile"><h3>HTTP 状态分布</h3><ul>{status_items}</ul></div>
      </div>
    </section>
    <section class="card">
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>目标</th>
              <th>路径</th>
              <th>URL</th>
              <th>状态码</th>
              <th>响应长度</th>
              <th>关键词命中</th>
              <th>内容片段</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows) if rows else "<tr><td colspan='8' style='text-align:center;padding:26px;'>无数据</td></tr>"}
          </tbody>
        </table>
      </div>
    </section>
    <p class="muted">本报告仅用于授权测试。请确保遵循所有相关法律与合规要求。</p>
  </main>
</body>
</html>
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")


def main():
    """主函数：从 stdio 读取 JSON-RPC 请求并处理"""
    server = MCPServer()
    
    # 从标准输入读取 JSON-RPC 请求
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            server.handle_request(request)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()

