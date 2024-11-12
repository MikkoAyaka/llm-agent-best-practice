# LLM Agent Best Practice：大语言模型智能体最佳实践

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Issues](https://img.shields.io/github/issues/mikkoayaka/llm-agent-best-practice)](https://github.com/your-username/llm-agent-best-practice/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/mikkoayaka/llm-agent-best-practice)](https://github.com/your-username/llm-agent-best-practice/graphs/contributors)

---

`llm-agent-best-practice` 是一个轻量级且最简化的 LLM 智能体 **应用模板**
，旨在为开发者提供一套最佳实践。通过减少繁杂的配置和代码编写量，使开发者能够更加专注于功能开发和业务实现。

本项目基于 [Llama-Index](https://github.com/jerryjliu/llama_index) 设计，支持 **M-RAG（Mixed Retrieval-Augmented
Generation）**。该设计允许在智能体中使用不同的数据存储方式，包括文档、向量、图、关系型数据等，以满足各种数据查询和生成需求。

## ✨ 特性

- **轻量化设计**：仅保留核心结构，便于开发者快速上手并实现关键功能。
- **最佳实践架构**：采用清晰的代码结构，确保代码简洁易懂，便于维护和扩展。
- **多类型数据支持**：通过 M-RAG 实现对多种数据格式（文档、向量、图、关系型数据）的混合处理与存储，提升模型的检索和生成能力。
- **基于 Llama-Index**：高效的索引和检索功能，支持对多种数据格式的快速查询。
- **模块化扩展**：灵活的模块化设计，便于自定义扩展功能，以适应不同的业务需求。

## 🚀 快速开始

### 环境要求

- Python 版本：3.12 及以上
- 依赖包：项目的依赖包可以在 `requirements.txt` 中找到。

### 安装步骤

1. 克隆项目到本地：

   ```bash
   git clone https://github.com/mikkoayaka/llm-agent-best-practice.git
   cd llm-agent-best-practice
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 运行示例：

   ```bash
   cd llm_agent_best_practice
   python main.py
   ```

   该项目附带了一个简单的示例脚本 `main.py`，可以帮助你快速上手并了解如何使用智能体进行查询和生成。

### 配置说明

`llm-agent-best-practice` 使用 `.env` 文件来配置一些必需的密钥和 API
信息。可以参考该文件创建和设置自己的 `.env.development` 和 `.env.production` 文件。

## 📚 项目结构

```plaintext
llm-agent-best-practice/
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖文件
├── .env                        # 环境变量示例文件
└── llm_agent_best_practice/    # 项目代码
    ├── agent/                  # 智能体模块
    ├── config/                 # 系统配置模块
    ├── prompt/                 # 提示词模块
    ├── repository/             # 数据仓库模块
    ├── util/                   # 封装工具模块
    └── main.py                 # 主入口
```

## 🛠️ 功能说明

### 数据加载

支持文档、向量、图、关系型数据的加载，能够轻松扩展其他数据源，实现数据的灵活管理。

### 数据索引

基于 Llama-Index 创建高效的数据索引，能够迅速处理大量数据，并支持不同的数据格式。

### 检索与生成

通过 M-RAG 设计的多模态检索生成机制，使智能体能够在多种数据格式中进行检索，并生成高质量的内容。

## 📜 开源协议

本项目遵循 MIT 开源协议，详细内容请参见 [MIT License](https://mit-license.org/)。

## 🤝 贡献

我们欢迎开发者和贡献者共同完善 `llm-agent-best-practice` 项目！您可以通过以下方式进行贡献：

1. Fork 本仓库，并创建自己的分支。
2. 提交 Pull Request，描述您的更改。
3. 在 Issue 中提出建议或反馈问题。

---

在 GitHub 上点击 Star 来支持这个项目吧！如果你觉得这个项目对你有帮助，欢迎分享给更多的开发者。