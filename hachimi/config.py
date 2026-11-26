"""全局配置常量与运行参数。

本模块同时提供：

* 路径 / 字符集 / 采样 / 合成 / 混响等常量；
* :class:`RenderOptions` —— 一次性描述一次渲染的所有可选参数，
  供 CLI、网页后端与启动器共用；
* 语音模式（哈基米 / 曼波 / 自动混合）的定义。

所有路径都基于项目根目录解析，因此无论从哪个工作目录启动都能正常工作。
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "Hachimi Sonata"
APP_NAME_CN = "哈基米音律工坊"
APP_SLUG = "hachimi-sonata"
APP_VERSION = "0.22.0"
APP_DOMAIN = os.environ.get("HACHIMI_PUBLIC_HOST", "hachimi-sonata.localhost")

SAMPLE_PATH = str(PROJECT_ROOT / "assets" / "samples") + os.sep
SOURCE_DIR = str(PROJECT_ROOT / "音源") + os.sep
SOURCE_HACHIMI_DIR = str(PROJECT_ROOT / "音源" / "哈基米") + os.sep
SOURCE_DD_DIR = str(PROJECT_ROOT / "音源" / "叮咚鸡") + os.sep
SOURCE_MB_DIR = str(PROJECT_ROOT / "音源" / "曼巴") + os.sep
SOURCE_MANBO_DIR = str(PROJECT_ROOT / "音源" / "曼波") + os.sep
HACHIMI_SET2_DIR = str(PROJECT_ROOT / "音源" / "哈基米第二版") + os.sep
HACHIMI_SET2_FALLBACK_DIR = str(PROJECT_ROOT / "补充" / "哈基米音源") + os.sep
HACHIMI_SET2_DIRS = [HACHIMI_SET2_DIR, HACHIMI_SET2_FALLBACK_DIR]
HACHIMI_SET2_CHARS = "哈基米摸南北绿豆阿西噶呀库那路"
EXTRA_SAMPLE_DIRS = [
    str(PROJECT_ROOT / "补充" / "叮咚鸡") + os.sep,
    str(PROJECT_ROOT / "补充" / "曼巴") + os.sep,
]
SAMPLE_SOURCE_DIRS = [
    SOURCE_DIR,
    SOURCE_HACHIMI_DIR,
    SOURCE_DD_DIR,
    SOURCE_MB_DIR,
    SOURCE_MANBO_DIR,
    HACHIMI_SET2_DIR,
]
MIDI_GLOB = str(PROJECT_ROOT / "assets" / "midi" / "**" / "*.mid")
LOCAL_MIDI_GLOB = str(PROJECT_ROOT / "midi" / "**" / "*.mid")
MUSIC_GLOB = str(PROJECT_ROOT / "assets" / "midi" / "**" / "*")
LOCAL_MUSIC_GLOB = str(PROJECT_ROOT / "midi" / "**" / "*")
TEST_MUSIC_GLOB = str(PROJECT_ROOT / "测试" / "**" / "*")
MIDI_EXTS = (".mid", ".midi", ".kar", ".rmi")
SCORE_EXTS = (".xml", ".musicxml", ".mxl")
MUSIC_EXTS = MIDI_EXTS + SCORE_EXTS
OUTPUT_DIR = str(PROJECT_ROOT / "output") + os.sep
MODEL_DIR = str(PROJECT_ROOT / "models" / "character_markov") + os.sep
DATA_PATH = str(PROJECT_ROOT / "data" / "corpus.json")
ANIMATION_DIR = str(PROJECT_ROOT / "assets" / "animation") + os.sep
CACHE_DIR = str(PROJECT_ROOT / "cache") + os.sep
UPLOAD_DIR = str(PROJECT_ROOT / "uploads") + os.sep
WEB_DIR = str(PROJECT_ROOT / "web") + os.sep
TEMP_DIR = str(PROJECT_ROOT / "tmp") + os.sep

# 记录系统原临时目录，仅用于清理旧版遗留的 hachimi_* 文件
SYSTEM_TEMP_DIR = os.environ.get("TEMP") or os.environ.get("TMP") or ""


def _probe_writable(path: Path) -> bool:
    """实际写入一个探针文件，确认目录可写（不是只看 ACL）。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".hachimi_write_{os.getpid()}_{time.time_ns()}"
        with open(probe, "wb") as fh:
            fh.write(b"ok")
        try:
            probe.unlink()
        except OSError:
            # 某些受保护目录允许写入但不允许删除；写入成功即可。
            pass
        return True
    except OSError:
        return False


def _fallback_roots() -> list[Path]:
    """可写回退目录，优先项目内 _runtime（同盘、便于找到）。"""
    roots: list[Path] = [PROJECT_ROOT / "_runtime"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / APP_SLUG)
    if SYSTEM_TEMP_DIR:
        roots.append(Path(SYSTEM_TEMP_DIR) / APP_SLUG)
    try:
        roots.append(Path.home() / APP_SLUG)
    except OSError:
        pass
    return roots


#: 记录发生回退的目录：原目录 -> 实际目录（供网页 / 日志提示）
DIR_FALLBACKS: dict[str, str] = {}


def ensure_writable_dir(preferred: str | Path, name: str | None = None) -> str:
    """返回可写目录（带结尾分隔符）。

    安装目录位于只读盘、Low integrity 进程、或 output/uploads 被单独
    限制时，回退到项目 ``_runtime`` / LocalAppData / 系统临时目录。
    这样网页仍能上传与渲染，而不是在 ``mkdir`` 处直接失败。
    """
    path = Path(preferred)
    if _probe_writable(path):
        return str(path) + os.sep

    folder = name or path.name or "data"
    for root in _fallback_roots():
        candidate = root / folder
        if _probe_writable(candidate):
            DIR_FALLBACKS[str(path)] = str(candidate)
            logger.warning("目录不可写，改用：%s（原：%s）", candidate, path)
            return str(candidate) + os.sep

    logger.error("找不到可写目录，仍使用：%s；请检查磁盘 / 目录权限", path)
    return str(path) + os.sep


# 运行期目录在此统一解析，后续模块拿到的就是实际可写路径。
OUTPUT_DIR = ensure_writable_dir(OUTPUT_DIR, "output")
UPLOAD_DIR = ensure_writable_dir(UPLOAD_DIR, "uploads")
TEMP_DIR = ensure_writable_dir(TEMP_DIR, "tmp")
CACHE_DIR = ensure_writable_dir(CACHE_DIR, "cache")
for _path in (os.path.join(CACHE_DIR, "cuda"), os.path.join(CACHE_DIR, "torch")):
    try:
        Path(_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# 让 tempfile / ffmpeg / PyTorch / CUDA / numba 的临时/缓存文件都落到可写目录
os.environ["TMP"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMPDIR"] = TEMP_DIR
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_DIR, "torch"))
os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(CACHE_DIR, "cuda"))
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(256 * 1024 * 1024))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(CACHE_DIR, "triton"))
try:
    tempfile.tempdir = TEMP_DIR
except Exception:
    pass

STEMS_DIRNAME = "stems"
PREMIX_DIRNAME = "stems_premix"
BAD_NOTES_DIRNAME = "bad_notes"
ZIP_SUFFIX = "_package.zip"
PARAMS_SCHEMA = "hachimi-sonata.project-params"
PARAMS_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# 其余常量
# ---------------------------------------------------------------------------

ALPHABET = {'哈基米': '哈基米摸南北绿豆阿西噶呀库那路', '哈基米only': '哈基米', '曼波': '曼波欧马吉利哇夏达不耶', '曼波only': '曼波', '叮咚鸡': '叮咚鸡大狗叫楼上下来搞核酸的份伞健出示证身酸雨带康手码核机', '曼巴': '曼巴欧嘛吉利'}
VOICE_MODES = {'auto': ['哈基米', '叮咚鸡', '曼巴', '曼波only'], '哈基米': ['哈基米'], '哈基米only': ['哈基米only'], '曼波': ['曼波'], '曼波only': ['曼波only'], '叮咚鸡': ['叮咚鸡'], '曼巴': ['曼巴']}
VOICE_LABELS = {'auto': '自动混合（哈基米主 + 叮咚鸡/曼巴/曼波辅）', '哈基米': '哈基米（完整词汇）', '哈基米only': '哈基米 only', '曼波': '曼波（完整词汇）', '曼波only': '曼波 only', '叮咚鸡': '叮咚鸡', '曼巴': '曼巴'}
DEFAULT_VOICE = 'auto'
SAMPLE_SET = 1
SAMPLE_SETS = [{'value': 1, 'label': '第一套（音源/哈基米）'}, {'value': 2, 'label': '第二套（音源/哈基米第二版）'}]
SAMPLE_SET_VALUES = [1, 2]
AVAILABLE_CLASSES = ['哈基米', '哈基米only', '曼波', '曼波only', '叮咚鸡', '曼巴', 'auto']
VOICE_SAMPLE_VARIANTS = {'叮咚鸡': {'鸡': ['鸡', '鸡2'], '叫': ['叫', '叫2']}, '曼巴': {'曼': ['曼1', '曼2', '曼3'], '巴': ['巴1', '巴2', '巴3']}}
BASS_MAX_PITCH = 60
BASS_GAIN = 0.35
SAMPLE_ALIASES = {'欧': '摸', '马': '那', '吉': '基', '利': '米', '哇': '呀', '夏': '哈', '达': '那', '不': '波', '耶': '呀', '巴': '巴1'}
SAMPLE_FALLBACK = '哈'
CONS_FRAMES = {'哈': 3, '基': 8, '米': 8, '摸': 8, '南': 8, '北': 8, '绿': 8, '豆': 8, '阿': 8, '西': 30, '噶': 8, '呀': 8, '库': 8, '那': 8, '路': 8, '曼': 6, '波': 8, '欧': 6, '马': 8, '吉': 4, '利': 4, '哇': 8, '夏': 8, '达': 8, '不': 8, '耶': 8, '叮': 6, '咚': 6, '鸡': 6, '大': 4, '狗': 6, '叫': 8, '楼': 4, '上': 3, '下': 3, '来': 4, '搞': 4, '核': 4, '酸': 8, '的': 3, '份': 6, '伞': 6, '健': 4, '出': 4, '示': 4, '证': 6, '身': 6, '雨': 6, '带': 4, '康': 4, '手': 4, '码': 6, '机': 6, '嘛': 6, '巴': 6}
MODEL_KIND = 'markov'
MODEL_KINDS = ['markov', 'transformer']
MODEL_LABELS = {'markov': '马尔科夫链（快、稳定）', 'transformer': 'Transformer（更连贯）'}
SYNTAX_WEIGHT = 0.3
DEFAULT_MARKOV_RATIO = 0.7
DEFAULT_SOURCE_WEIGHTS = {'哈基米': 0.76, '叮咚鸡': 0.16, '曼巴': 0.05, '曼波only': 0.03}
DEFAULT_VIOLIN_GAIN = 1.5
AUTO_MIX = True
AUTO_MIX_MAX_ADJUST_DB = 6.0
SYNTAX_LONG_NOTE_SECONDS = 0.9
SYNTAX_PHRASE_GAP = 0.9
SYNTAX_MAX_PHRASE_SLOTS = 16
AUDIO_FORMAT = 'mp3'
AUDIO_FORMATS = ['mp3', 'wav']
MP3_SUBTYPE = 'MPEG_LAYER_III'
AUTO_DOWNLOAD = 'all'
AUTO_DOWNLOAD_MODES = ['all', 'zip', 'off']
AUTO_DOWNLOAD_INTERVAL_MS = 900
GPU_MODE = 'auto'
GPU_MEMORY_MB = 2048
GPU_MIN_NOTES = 120
GPU_MAX_FRACTION = 0.25
TRANSFORMER = {'block_size': 32, 'n_embd': 64, 'n_head': 4, 'n_layer': 3, 'steps': 1500, 'batch_size': 32, 'lr': 0.001, 'temperature': 1.0, 'seed': 1337}
DEFAULT_TRACK_CLASSES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
DEFAULT_OCTAVE_SHIFT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]
DEFAULT_VOLUME_FACTOR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
SR = 44100
FRAME_PERIOD = 5.0
SIMULTANEOUS_THRESHOLD = 0.05
START_TIME = 0.0
END_TIME_BUFFER = 2.0
AP_SCALE = 0.05
SP_SCALE = 1.0
MAX_RENDER_SECONDS = None
LOW_CUT_FREQ = 80
TRACK_LOW_CUT_FREQ = 24
STEREO_SPREAD = 0.15
MULTITAP_DECAY = 0.4
MULTITAP_DELAY_MS = 50
MULTITAP_TAPS = 5
HF_DAMPING_CUTOFF = 10000
SAMPLE_TARGET_DB = -18.0
SAMPLE_COMPRESSOR_THRESHOLD_DB = -28.0
SAMPLE_COMPRESSOR_RATIO = 8.0
SAMPLE_COMPRESSOR_ATTACK_MS = 4.0
SAMPLE_COMPRESSOR_RELEASE_MS = 90.0
#: 混音总线压缩：比旧版 1.8:1 更积极，但仍保留乐句起伏，避免流行音乐式压扁。
MIX_COMPRESSOR_THRESHOLD_DB = -24.0
MIX_COMPRESSOR_RATIO = 3.4
MIX_COMPRESSOR_ATTACK_MS = 14.0
MIX_COMPRESSOR_RELEASE_MS = 220.0
MIX_COMPRESSOR_KNEE_DB = 8.0
MIX_COMPRESSOR_MAX_MAKEUP_DB = 10.0
#: 弱段向上压缩：只抬 -36dB 以下的乐句 / 弱音，最多 +4.5dB。
MIX_UPWARD_THRESHOLD_DB = -36.0
MIX_UPWARD_RATIO = 1.7
MIX_UPWARD_MAX_DB = 4.5
MIX_UPWARD_ATTACK_MS = 40.0
MIX_UPWARD_RELEASE_MS = 320.0
UNIT_TARGET_DB = -20.0
#: 母带目标：RMS -14dBFS、峰值上限 -1dBFS；保留动态，不追流行乐式极限响度。
MASTER_TARGET_DB = -14.0
STEM_TARGET_DB = -16.0
MASTER_CEILING_DB = -1.0
SIBILANCE_RATIO_THRESHOLD = 0.45
SIBILANCE_PERIODICITY_THRESHOLD = 0.6
SIBILANCE_MAX_REDUCTION_DB = 6.0
SIBILANCE_UNIT_REDUCTION_DB = 1.8
VOICELESS_MAX_REDUCTION_DB = 2.5
#: 所有辅音的统一衰减；爆破音字（基 / 库 / 叮 / 咚 等）开头再额外大幅衰减。
#: 2026-11 调整：爆破音起始瞬态从 -12dB 进一步加强到约 -18dB（含 3dB 通用辅音），
#: 并加长爆破段 / 回落后，让“基 / 库”等字更柔、不刺耳。
CONSONANT_ATTENUATION_DB = 3.0
PLOSIVE_ATTENUATION_DB = 13.0
PLOSIVE_BURST_MS = 26.0
PLOSIVE_RAMP_MS = 36.0
#: 判定“发声困难”的实用人声音域（含轨道八度移调后的实际发声音高）。
SINGABLE_MIN_PITCH = 36   # C2
SINGABLE_MAX_PITCH = 84   # C6
#: 不良音符：音高偏离轨道中位数超过该值也视为过高 / 过低（约两个八度）。
BAD_NOTE_RELATIVE_PITCH_SPAN = 24
#: 低音量判定：相对轨道中位电平、绝对电平与力度三重判据。
BAD_NOTE_QUIET_REL_DB = -15.0
BAD_NOTE_EXPRESSIVE_REL_DB = -26.0
BAD_NOTE_QUIET_ABS_DB = -46.0
BAD_NOTE_QUIET_VELOCITY = 28
CHORD_ONSET_TOL = 0.06
CHORD_MERGE_GAP = 0.4
CHORD_MAX_WINDOW = 2.0
CHORD_MIN_NOTES = 2
DECOMPOSED_MERGE = True
DECOMPOSED_GAP = 0.38
SECTION_BEATS = 32
SECTION_SIMILARITY = 0.86
MOTIF_MIN_NOTES = 4
MOTIF_MAX_NOTES = 8
MOTIF_MAX_RESULTS = 24
MOTIF_MIN_OCCURRENCES = 2
ANALYSIS_MAX_NOTES = 30000
RENDER_CACHE_BYTES = 201326592
RENDER_UNIT_BYTES = 268435456
RENDER_DURATION_QUANTUM = 0.02
RENDER_MAX_UNITS = 1400
RENDER_MIN_PARALLEL_UNITS = 24
RENDER_DEFAULT_WORKERS = 0
RENDER_MAX_WORKERS = 8
RENDER_NOTE_FADE = 0.004
RENDER_NOTE_FADE_IN = 0.0015
RENDER_NOTE_FADE_OUT = 0.004
RENDER_NOTE_MAX_SECONDS = 10.0
RENDER_TAIL_SECONDS = 3.0
#: 弱化音头：每个音起音用更长的软起音包络，削掉“啪”的瞬态，
#: 但保留辅音主体（14ms 仍能听清字头）。连奏后的下一个音用更长的 30ms。
ATTACK_SOFTEN_MS = 14.0
#: 爆破音字（基 / 库 / 叮 / 咚 / 巴 / 波等）额外使用更长的软起音：
#: 普通音 14ms，爆破音 22ms；连奏进入音仍优先 30ms。
PLOSIVE_ATTACK_SOFTEN_MS = 22.0
ATTACK_SOFTEN_CURVE = 0.65
#: 连奏：同一轨连续两个音间隔不超过该值时，把前一个音尾延伸进下一个音，
#: 形成真实乐器换弓 / 换气时的交叉过渡。
LEGATO_MAX_GAP = 0.16
LEGATO_OVERLAP_SECONDS = 0.035
LEGATO_MAX_EXTEND = 0.14
LEGATO_ATTACK_MS = 30.0
VELOCITY_GAMMA = 1.6
VELOCITY_MIN = 0.02
EXPRESSION_MAX_GAP = 0.4
EXPRESSION_MIN_DELTA = 3.0
NORMALIZE_MODE = 'peak'
NORMALIZE_PEAK = 0.97
#: 绿幕字幕视频：单条视频包含全部字幕，每次最多 1 行，宽 1920。
WRITE_SUBTITLE_VIDEOS = True
SUBTITLE_VIDEO_WIDTH = 1920
SUBTITLE_VIDEO_FPS = 30
#: 字号单位“号”：8 号默认按 1 号≈10px（1920 宽）换算 ≈ 80px。
SUBTITLE_VIDEO_FONT_FAMILY = 'SimSun'
SUBTITLE_VIDEO_FONT_SIZE = 8.0
SUBTITLE_VIDEO_FONT_UNIT_PX = 10.0
SUBTITLE_VIDEO_TEXT_COLOR = '#FFFFFF'
#: 轻微黑边：约 2px，让白字在纯绿幕上仍有清晰轮廓。
SUBTITLE_VIDEO_TEXT_STROKE_WIDTH = 2
SUBTITLE_VIDEO_TEXT_STROKE_COLOR = '#000000'
SUBTITLE_VIDEO_MAX_LINES = 1
SUBTITLE_VIDEO_LINE_SPACING = 1.25
SUBTITLE_VIDEO_PADDING_RATIO = 0.04
SUBTITLE_VIDEO_GREEN = '#00FF00'
#: 单行字幕每行最多字数；20 个全角字在默认 8 号（80px）下仍能放进 1920。
SUBTITLE_VIDEO_MAX_CHARS = 20
SUBTITLE_VIDEO_CRF = 16
HOST = '127.0.0.1'
PORT = 8787
MAX_UPLOAD_MB = 64
JOB_TTL_SECONDS = 21600
GM_PROGRAMS = ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano', 'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet', 'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer', 'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar harmonics', 'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2', 'SynthStrings 1', 'SynthStrings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn', 'Brass Section', 'SynthBrass 1', 'SynthBrass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina', 'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)', 'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag pipe', 'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot']
GM_DRUM_KIT = 'Drum Kit'


def public_url(port: int | None = None, host: str | None = None) -> str:
    bind = host or HOST
    if bind in ("127.0.0.1", "localhost"):
        display = APP_DOMAIN
    elif bind in ("0.0.0.0", "::"):
        display = "127.0.0.1"
    else:
        display = bind
    return f"http://{display}:{port or PORT}"


def local_url(port: int | None = None, host: str | None = None) -> str:
    bind = host or HOST
    if bind in ("0.0.0.0", "::", "127.0.0.1", "localhost"):
        display = "localhost"
    else:
        display = bind
    return f"http://{display}:{port or PORT}"




def instrument_name(program: int, is_drum: bool = False) -> str:
    if is_drum:
        return GM_DRUM_KIT
    if 0 <= program < len(GM_PROGRAMS):
        return GM_PROGRAMS[program]
    return f"Program {program}"


@dataclass
class RenderOptions:
    """一次渲染的全部可调参数（可 JSON 序列化）。"""

    # 歌词
    model_kind: str = MODEL_KIND
    voice: str = DEFAULT_VOICE
    sample_set: int = SAMPLE_SET
    seed: int = 1337
    syntax_weight: float = SYNTAX_WEIGHT
    section_plans: list[dict[str, Any]] = field(default_factory=list)

    # 轨道
    tracks: list[dict[str, Any]] = field(default_factory=list)

    # 渲染
    velocity_gamma: float = VELOCITY_GAMMA
    workers: int = RENDER_DEFAULT_WORKERS
    duration_quantum: float = RENDER_DURATION_QUANTUM
    max_units: int = RENDER_MAX_UNITS
    normalize: str = NORMALIZE_MODE
    auto_mix: bool = AUTO_MIX
    start_time: float = START_TIME
    end_time: float | None = None
    max_seconds: float | None = MAX_RENDER_SECONDS
    tail_seconds: float = RENDER_TAIL_SECONDS
    audio_format: str = AUDIO_FORMAT
    gpu: str = GPU_MODE
    gpu_memory_mb: int = GPU_MEMORY_MB

    # 音乐分析
    decomposed_merge: bool = DECOMPOSED_MERGE
    chord_onset_tol: float = CHORD_ONSET_TOL
    chord_merge_gap: float = CHORD_MERGE_GAP

    # 输出
    write_stems: bool = True
    #: 混音前的干声统一导出为 FLAC，并额外压成一个 ZIP。
    write_premix: bool = True
    #: 每轨只包含“不良音符”的 MIDI（过高 / 过低 / 实际音量过低）。
    write_bad_note_midis: bool = True
    write_zip: bool = True
    write_animation: bool = False
    #: 生成一条包含全部字幕的绿幕字幕视频（每次最多 1 行）
    write_subtitle_videos: bool = WRITE_SUBTITLE_VIDEOS
    subtitle_video_width: int = SUBTITLE_VIDEO_WIDTH
    subtitle_video_fps: int = SUBTITLE_VIDEO_FPS
    subtitle_video_font_family: str = SUBTITLE_VIDEO_FONT_FAMILY
    subtitle_video_font_size: float = SUBTITLE_VIDEO_FONT_SIZE
    subtitle_video_text_color: str = SUBTITLE_VIDEO_TEXT_COLOR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RenderOptions":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    def track_options(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for item in self.tracks:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            out[idx] = item
        return out


_TEMP_CLEANED = False


def ensure_dirs() -> None:
    """创建运行期需要的目录；单个目录失败不应阻止服务启动。"""
    for path in (OUTPUT_DIR, MODEL_DIR, CACHE_DIR, UPLOAD_DIR, TEMP_DIR):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("无法创建目录 %s：%s", path, exc)
    global _TEMP_CLEANED
    if not _TEMP_CLEANED:
        _TEMP_CLEANED = True
        try:
            cleanup_legacy_temp()
        except Exception:
            pass


def cleanup_legacy_temp(max_age_seconds: int = 7 * 24 * 3600) -> int:
    """清理旧版本遗留在系统临时目录 / 项目 tmp 中的 hachimi_* 文件。"""
    now = time.time()
    removed = 0
    folders = {TEMP_DIR, SYSTEM_TEMP_DIR}
    for folder in folders:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if not name.startswith("hachimi_"):
                continue
            path = os.path.join(folder, name)
            try:
                if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass
    return removed
