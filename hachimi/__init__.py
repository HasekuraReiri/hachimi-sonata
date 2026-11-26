"""Hachimi Sonata · 哈基米音律工坊。

歌词生成模型统一入口：

    from hachimi import create_model, MarkovModel, TransformerModel

    # 两种模型共用同一接口，可互换
    model = create_model("transformer", allowed_classes=["哈基米"])
    model = create_model("markov", allowed_classes=["曼波only"])

    model.generate(length=50)   # 生成 50 个字符
    model.sample("哈基米")      # 采样下一个字符

完整渲染流程（推荐）：

    from hachimi.pipeline import render_midi
    result = render_midi("song.mid")

模块地图
--------
- :mod:`hachimi.score`     MusicXML / MXL 读取（速度地图 / 力度 / 奏法 / 连音线）
- :mod:`hachimi.analysis`  轨道/乐器、调性、和弦、结构、乐句与动机分析
- :mod:`hachimi.arrangement` 每段马尔科夫比例 / 随机种子 / 音源配比
- :mod:`hachimi.lyrics`    和弦一致 + 动机复用 + 每段编曲的歌词生成
- :mod:`hachimi.synthesis` WORLD 合成（音源压缩 / 去齿音 / 中段拉伸 / 频段与可听度）
- :mod:`hachimi.render`    高性能渲染（单位缓存 + 多进程 + 力度与渐强渐弱）
- :mod:`hachimi.mixing`    自动混音（真实摆位 / 相位对齐 / 自动电平 / 母带动态压缩）
- :mod:`hachimi.report`    乐器输出清单（输出 / 未输出 / 音频文件名）
- :mod:`hachimi.qc`        不良音符检查（过高 / 过低 / 音量过低）与专用 MIDI
- :mod:`hachimi.params`    工程参数包（保存 / 读取 / 复用旧工程设置）
- :mod:`hachimi.subtitles` SRT / LRC / JSON / Markdown / HTML 实时字幕
- :mod:`hachimi.video`      绿幕字幕视频（单条包含全部字幕，宽 1920，每次最多 1 行）
- :mod:`hachimi.pipeline`  端到端流程
- :mod:`hachimi.server`    FastAPI 网页后端
- :mod:`hachimi.browser`   Edge / 默认浏览器自动打开与就绪轮询
- :mod:`hachimi.launcher`  tkinter 启动器
"""

__version__ = "0.22.0"

from hachimi.tokenization import Tokenizer, train_bpe
from hachimi.base import LyricsModel, load_songs
from hachimi.markov import MarkovModel
from hachimi.transformer import TransformerModel
from hachimi.models import create_model, load_model

__all__ = [
    "Tokenizer",
    "train_bpe",
    "LyricsModel",
    "load_songs",
    "MarkovModel",
    "TransformerModel",
    "create_model",
    "load_model",
    "__version__",
]
