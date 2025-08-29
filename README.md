# AI Debate Arena

AI Debate Arena 是一个多智能体辩论系统。系统中包含 Pro（正方）、Con（反方）、Judge（裁判）、Host（主持人）四类角色，能够围绕某个议题进行多轮交替辩论，并由裁判给出最终裁决。  
本项目基于 [Streamlit](https://streamlit.io/) 构建前端界面，结合自定义的 `agents` 和 `graph` 流程管理，支持实时增量渲染和可视化评分。

## 项目说明
- 这是一个演示性的项目，仅供学习和研究使用。

## 功能特点

- 多轮 Pro vs Con 辩论，由 Judge 最终裁决
- 支持增量渲染：每个角色发言后即时更新
- 角色区分显示（正方/反方/裁判/主持人）
- 裁决可视化：胜者提示、双方分数卡片、交互式雷达图（argument/evidence/rebuttal/clarity）、裁判理由
- LLM 配置持久化：支持保存到 `settings.json`，无需每次重新输入
- 可选配置检查：快速验证 API Key、URL、Model 是否可用
- 历史对局管理：保存所有运行结果到 `runs/`，支持打开和下载
- 支持运行中实时状态提示（例如“正方正在回答”、“裁判正在评价”）

## 文件结构
```
.
├── ui_app.py # Streamlit UI，入口文件
├── graph.py # 构建辩论流程图（StateGraph），管理 Pro/Con/Judge 节点
├── agents.py # 定义和构建 Pro/Con/Judge 智能体
├── llm.py # 封装 LLM 接口（对话消息、调用方法）
├── config.py # 配置常量，包含 RUNS_DIR 路径等
├── runs/ # 保存所有对局运行结果的 JSON 文件
├── settings.json # 持久化的 LLM 配置（运行后生成）
└── README.md # 使用说明
```

## 环境依赖

- Python 3.9+
- 主要依赖：
  - `streamlit`
  - `plotly`
  - `langgraph`
  - `matplotlib`
  - `openai`

安装依赖：

```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，可直接安装核心依赖：

```
pip install streamlit plotly matplotlib langgraph openai
```

## 运行方法

在项目根目录下执行：

```
streamlit run ui_app.py
```

浏览器会自动打开一个本地页面（默认地址为 [http://localhost:8501](http://localhost:8501)）。

也可以通过命令行方式运行，不依赖 Streamlit UI：  

```bash
python run.py --topic "是否应当在所有中学强制开设编程必修课？" --round 3
```

其中 `--topic` 参数代表辩论题目，是必须的；`--round` 代表最大轮数，默认为 3。

注意，通过命令行方式运行时，大模型的配置信息需要在 `config.py` 中填写。此外，辩论内容会在整个辩论结束之后才会显示输出在命令行中。

## 使用说明
1.配置智能体
- 在左侧栏填写 Pro、Con、Judge 三个 Agent 的 API Key、Base URL、Model 名称和 Temperature。
- 可以点击 Check Settings 对配置进行验证（推荐但不强制）。
- 检查通过的配置会自动保存到 settings.json，下次启动会自动预填。
  
2.新建对局
- 在左侧填写辩论主题（Topic）和回合数（Rounds）。
- 点击 Run Debate 启动辩论。
- 页面中会逐步显示 Pro/Con 的发言，并在最后由 Judge 给出裁决。

3.裁决展示
- 最终裁决包括：
    - 胜者（Winner）
    - 双方总分（Pro vs Con）
    - 交互式雷达图（可单独显示 Pro 或 Con 的得分）
    - 裁判给出的理由（Rationale）

4.历史对局
- 每次运行后会自动保存 JSON 文件到 runs/ 目录。
- 在左侧 Previous Runs 中可选择并打开已保存的对局，或下载结果。

5.维护功能
- Clear UI State：清空当前 UI 状态
- Delete all runs：删除 runs/ 下所有保存的对局文件

## 文件保存说明
- 每次运行的最终状态会保存到 runs/<topic>-<随机id>.json。
- 文件中包含完整的 transcript（所有发言）、最终 verdict（裁判评分与结论）。
- 打开历史文件时，系统会重新渲染对应的对局内容和裁决结果。
