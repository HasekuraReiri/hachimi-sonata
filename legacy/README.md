# legacy — 上游旧脚本存档

这些文件来自重构前的版本，仅作参考，**不再维护**，也不建议直接运行：

| 文件 | 说明 |
| --- | --- |
| `main - 副本.py` / `main2.py` / `main2 - 副本.py` | 早期逐音符合成版本 |
| `main新.py` / `main简版.py` | 批量 / 简化实验版本 |
| `midiparse.py` | 早期 MIDI 信息打印 / 音频预览脚本 |
| `pitchshift-pyworld.py` | 早期 WORLD 合成实验 |
| `deepseek_python_20260713_89a43e.py` | 历史实验代码 |

新代码请使用：

- 命令行：项目根目录 `main.py`
- 核心库：`hachimi/`（`pipeline.py`、`render.py`、`analysis.py` 等）
- 网页：`python main.py web`
- 启动器：`python launcher.py` / `start.bat`

根目录仍保留 `character_markov.py`、`pitchshift.py`、`reverb.py`、`animate.py`
作为**兼容 shim**，它们会把导入转发到 `hachimi` 包中的新实现。
