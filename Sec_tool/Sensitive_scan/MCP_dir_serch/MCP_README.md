# MCP Directory Scanner 使用说明

## 概述

`mcp_server.py` 将 `dir_serch.py` 的目录扫描功能封装为 MCP (Model Context Protocol) 工具，可以通过 MCP 协议调用目录扫描功能。

## 安装依赖

```bash
pip install -r requirements.txt
```

## MCP 工具说明

### 工具名称
`scan_directory`

### 功能描述
扫描目标网站的敏感路径和文件，检测目录遍历、敏感文件泄露等安全问题。

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `target` | string | 是 | - | 目标 URL 或域名（例如：`example.com` 或 `https://example.com`） |
| `targets_file` | string | 否 | - | 目标列表文件路径（每行一个目标） |
| `wordlist` | string | 否 | - | 路径字典文件路径（每行一个路径，不提供则使用默认字典） |
| `timeout` | integer | 否 | 8 | 请求超时时间（秒） |
| `follow_redirects` | boolean | 否 | true | 是否跟随 HTTP 重定向 |
| `save_all` | boolean | 否 | false | 是否保存所有请求结果（包括 404） |
| `user_agent` | string | 否 | "DirScanSync/1.0" | 自定义 User-Agent |
| `output_json` | string | 否 | - | JSON 输出文件路径（可选） |
| `output_csv` | string | 否 | - | CSV 输出文件路径（可选） |

### 返回值

返回一个包含以下字段的字典：

- `success`: 布尔值，表示扫描是否成功
- `summary`: 扫描摘要
  - `targets_scanned`: 扫描的目标数量
  - `total_findings`: 发现的总记录数
  - `targets`: 每个目标的扫描结果统计
- `results`: 详细的扫描结果列表（每个结果包含 URL、状态码、响应长度、关键词命中等信息）
- `output_files`: 保存的文件路径（如果指定了输出文件）

## 配置 MCP 服务器

### 方式 1：使用配置文件

编辑 `mcp_config.json`，确保路径正确：

```json
{
  "mcpServers": {
    "directory-scanner": {
      "command": "python3",
      "args": [
        "/绝对路径/Sec_tool/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

### 方式 2：在 Cursor/Claude Desktop 中配置

在 Cursor 或 Claude Desktop 的 MCP 配置文件中添加：

```json
{
  "mcpServers": {
    "directory-scanner": {
      "command": "python3",
      "args": [
        "/Users/a58/Desktop/Script_tools/Sec-farming/Sec_tool/mcp_server.py"
      ]
    }
  }
}
```

## 使用示例

### 示例 1：基本扫描

```json
{
  "method": "tools/call",
  "params": {
    "name": "scan_directory",
    "arguments": {
      "target": "example.com"
    }
  }
}
```

### 示例 2：完整参数扫描

```json
{
  "method": "tools/call",
  "params": {
    "name": "scan_directory",
    "arguments": {
      "target": "https://example.com",
      "wordlist": "/path/to/wordlist.txt",
      "timeout": 10,
      "follow_redirects": true,
      "save_all": false,
      "user_agent": "MyScanner/1.0",
      "output_json": "/path/to/results.json",
      "output_csv": "/path/to/results.csv"
    }
  }
}
```

### 示例 3：批量目标扫描

```json
{
  "method": "tools/call",
  "params": {
    "name": "scan_directory",
    "arguments": {
      "targets_file": "/path/to/targets.txt",
      "wordlist": "/path/to/wordlist.txt",
      "output_json": "/path/to/results.json"
    }
  }
}
```

## 测试 MCP 服务器

### 手动测试

可以通过标准输入/输出测试服务器：

```bash
# 测试初始化
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python3 mcp_server.py

# 测试工具列表
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python3 mcp_server.py

# 测试工具调用
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"scan_directory","arguments":{"target":"127.0.0.1:8888"}}}' | python3 mcp_server.py
```

## 注意事项

1. **授权使用**：仅在拥有足够授权的前提下对目标进行测试
2. **路径配置**：确保 `mcp_config.json` 中的路径是绝对路径
3. **Python 版本**：需要 Python 3.7+
4. **依赖**：确保已安装 `requests` 库

## 故障排查

### 问题：MCP 服务器无法启动
- 检查 Python 路径是否正确
- 检查 `mcp_server.py` 文件是否有执行权限
- 检查依赖是否已安装

### 问题：工具调用失败
- 检查目标 URL 是否可访问
- 检查网络连接
- 查看错误信息中的详细错误描述

### 问题：文件保存失败
- 检查输出文件路径是否有写权限
- 确保目录存在

## 与原始命令行工具的区别

| 特性 | 命令行工具 | MCP 工具 |
|------|-----------|---------|
| 调用方式 | 命令行参数 | JSON-RPC 协议 |
| 输出方式 | 文件 + 控制台 | JSON 响应 + 可选文件 |
| 集成方式 | 独立运行 | 可集成到 AI 助手 |
| 参数传递 | 命令行参数 | JSON 对象 |

## 开发说明

MCP 服务器通过 stdio 进行 JSON-RPC 通信：
1. 从标准输入读取 JSON-RPC 请求
2. 解析请求并调用相应的方法
3. 将结果通过标准输出返回 JSON-RPC 响应

核心功能直接复用 `dir_serch.py` 中的函数，确保功能一致性。

