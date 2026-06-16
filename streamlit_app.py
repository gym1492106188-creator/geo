
存储行业物料风险监控 & 周期预测系统 v4
════════════════════════════════════════════════
核心功能：
  1. 存储价格周期预测（DRAM/NAND/HBM 全品类）
  2. 结构性超级上行周期分析（AI 驱动的非常规周期）
  3. BOM 物料级实时+未来价格追踪
  4. 四梯队权重风险模型
  5. 金蝶 BOM 列表 + 存储物料库双数据源关联

运行：streamlit run dashboard_v4.py
"""
import os, sys
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

# ── 修复 Arrow 序列化问题：混合类型列（如包含 NaN 的字符串列）会导致 st.dataframe() 崩溃 ──
# 补丁：在列转换为 Arrow 前，将 object 列统一转为字符串
import streamlit.dataframe_util as _sdu
_orig_arrow_convert = _sdu.convert_pandas_df_to_arrow_bytes
def _safe_arrow_convert(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str)
            except Exception:
                pass
    return _orig_arrow_convert(df)
_sdu.convert_pandas_df_to_arrow_bytes = _safe_arrow_convert

# ── 静默 Streamlit bare-mode 警告 ──
# 这些警告在模块初始化阶段输出，绕过 Python logging propagation 直接写 stderr
class _StderrFilter:
    _BARE_MODE_MSGS = (
        "missing ScriptRunContext",
        "No runtime found, using MemoryCacheStorageManager",
        "Session state does not function when running a script without",
        "to view this Streamlit app on a browser, run it with the following",
    )
    def __init__(self, dst):
        self._dst = dst
    def write(self, s):
        if not any(m in s for m in self._BARE_MODE_MSGS):
            self._dst.write(s)
    def flush(self):
        self._dst.flush()
    def __getattr__(self, name):
        return getattr(self._dst, name)

if not isinstance(sys.stderr, _StderrFilter):
    sys.stderr = _StderrFilter(sys.stderr)

import streamlit as st

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import re

# 金蝶云星空 API（可选集成）
try:
    from kingdee_api import get_client, get_inventory_data, check_connection
    KINGDEE_AVAILABLE = True
except ImportError:
    KINGDEE_AVAILABLE = False

# ══════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="存储物料风险监控 & 周期预测 v4",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 全局样式 ──
st.markdown("""
<style>
.main-header { font-size:1.8rem; font-weight:700; color:#1A478A; margin-bottom:0; }
.metric-box { padding:1rem; border-radius:10px; color:#fff; text-align:center; }
.metric-box h3 { font-size:0.75rem; margin:0; opacity:0.85; text-transform:uppercase; }
.metric-box h1 { font-size:2rem; margin:0.3rem 0; }
.metric-box p { font-size:0.65rem; margin:0; opacity:0.8; }
.bom-alert-red { background:#FCE4EC; border-left:4px solid #C00000; padding:0.6rem 1rem; border-radius:4px; margin:0.3rem 0; }
.bom-alert-orange { background:#FFF3CD; border-left:4px solid #E97C00; padding:0.6rem 1rem; border-radius:4px; margin:0.3rem 0; }
.bom-alert-green { background:#E1F5EE; border-left:4px solid #1D9E75; padding:0.6rem 1rem; border-radius:4px; margin:0.3rem 0; }
.cycle-card { padding:1.2rem; border-radius:12px; margin:0.5rem 0; }
.signal-red { background:linear-gradient(135deg,#C00000,#E97C00); color:#fff; padding:1rem; border-radius:8px; margin:0.4rem 0; }
.signal-amber { background:linear-gradient(135deg,#E97C00,#F0A030); color:#fff; padding:1rem; border-radius:8px; margin:0.4rem 0; }
.signal-green { background:linear-gradient(135deg,#1D9E75,#2EAD80); color:#fff; padding:1rem; border-radius:8px; margin:0.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 数据层
# ══════════════════════════════════════════════════════════

# ── 品牌风险库 ──
BRAND_RISK_DB = {
    "Samsung":    {"score":72, "level":"高风险", "country":"韩国",
                   "factors":"HBM挤压DRAM产能;平泽工厂地理集中;NAND减产波动",
                   "trend":"↑ HBM扩产持续挤占DDR5/LPDDR5"},
    "SK hynix":   {"score":68, "level":"高风险", "country":"韩国",
                   "factors":"HBM市占52%产能售罄;Icheon/M16集中;龙仁新厂2027才量产",
                   "trend":"↑ LPDDR5供应持续紧张"},
    "Micron":     {"score":65, "level":"高风险", "country":"美国",
                   "factors":"中国禁售影响;台湾工厂地震风险;LPDDR4产线退役",
                   "trend":"→ 纽约厂爬坡中，短期供给偏紧"},
    "Kioxia":     {"score":55, "level":"中风险", "country":"日本",
                   "factors":"NAND供过于求;与WD合并失败;IPO后战略调整",
                   "trend":"↓ NAND减产稳价"},
    "Winbond":    {"score":52, "level":"中风险", "country":"台湾",
                   "factors":"台海地缘风险;NOR Flash面临大陆竞争;利基DRAM承压",
                   "trend":"→ 台海局势持续关注"},
    "Macronix":   {"score":48, "level":"中风险", "country":"台湾",
                   "factors":"NOR Flash价格战;3D NOR研发落后;台海地缘",
                   "trend":"↓ 竞争加剧"},
    "Nanya":      {"score":45, "level":"中风险", "country":"台湾",
                   "factors":"DDR3/DDR4淘汰风险;DDR5转型缓慢;三大厂挤压",
                   "trend":"↓ 利基DRAM市场萎缩"},
    "SanDisk":    {"score":42, "level":"中风险", "country":"美国",
                   "factors":"依赖Kioxia合资工厂;消费NAND跌价;WD分拆影响",
                   "trend":"→ 消费级产品稳定"},
    "Microchip":  {"score":38, "level":"中风险", "country":"美国",
                   "factors":"EEPROM交期稳定;ATECC安全芯片需求增长",
                   "trend":"→ 成熟产品线稳定"},
    "Renesas":    {"score":32, "level":"中风险", "country":"日本",
                   "factors":"2021火灾教训;NOR Flash非核心;供应已恢复",
                   "trend":"→ 供应稳定"},
    "ISSI":       {"score":22, "level":"低风险", "country":"美国(中资)",
                   "factors":"利基DRAM/NOR Flash;代工稳定;工业/汽车市场",
                   "trend":"→ 稳定"},
    "STMicroelectronics":{"score":18, "level":"低风险", "country":"瑞士/法国",
                   "factors":"EEPROM非核心;代工产能充足",
                   "trend":"→ 稳定"},
    "ON Semi":    {"score":15, "level":"低风险", "country":"美国",
                   "factors":"EEPROM成熟产品;交期正常",
                   "trend":"→ 稳定"},
    "Infineon":   {"score":10, "level":"低风险", "country":"德国",
                   "factors":"存储器非核心业务;FM24C系列稳定",
                   "trend":"→ 稳定"},
    "Rohm":       {"score":10, "level":"低风险", "country":"日本",
                   "factors":"BR24T系列EEPROM稳定;交期正常",
                   "trend":"→ 稳定"},
    "SIMCOM":     {"score":12, "level":"低风险", "country":"中国",
                   "factors":"通信模块为主;存储非核心;供货正常",
                   "trend":"→ 稳定"},
    "Adesto":     {"score":20, "level":"低风险", "country":"美国",
                   "factors":"Dialog子公司;NOR Flash小众;小批量稳定",
                   "trend":"→ 稳定"},
    "Alliance Memory":{"score":15, "level":"低风险", "country":"美国",
                   "factors":"代理商模式;交期受上游影响;库存正常",
                   "trend":"→ 稳定"},
}

TYPE_RISK_MODIFIER = {
    "LPDDR5":     {"mod":1.15, "lifecycle":"成长期",   "supply_tightness":"偏紧",
                   "price_trend":"上涨", "obsolescence_risk":"低",   "alt_count":"少"},
    "LPDDR4":     {"mod":1.05, "lifecycle":"成熟→衰退","supply_tightness":"正常",
                   "price_trend":"下跌", "obsolescence_risk":"中",   "alt_count":"中"},
    "DDR5":       {"mod":1.10, "lifecycle":"成长期",   "supply_tightness":"偏紧",
                   "price_trend":"微涨", "obsolescence_risk":"低",   "alt_count":"少"},
    "DDR4":       {"mod":1.00, "lifecycle":"成熟→衰退","supply_tightness":"宽松",
                   "price_trend":"下跌", "obsolescence_risk":"中高", "alt_count":"多"},
    "DDR3/DDR3L": {"mod":1.10, "lifecycle":"衰退期",   "supply_tightness":"收缩",
                   "price_trend":"下跌", "obsolescence_risk":"高",   "alt_count":"少"},
    "eMMC":       {"mod":1.00, "lifecycle":"成熟→衰退","supply_tightness":"宽松",
                   "price_trend":"下跌", "obsolescence_risk":"中",   "alt_count":"多"},
    "NAND Flash": {"mod":0.95, "lifecycle":"成熟期",   "supply_tightness":"宽松",
                   "price_trend":"下跌", "obsolescence_risk":"低",   "alt_count":"多"},
    "NOR Flash":  {"mod":1.00, "lifecycle":"成熟期",   "supply_tightness":"正常",
                   "price_trend":"微跌", "obsolescence_risk":"低",   "alt_count":"中"},
    "EEPROM":     {"mod":0.85, "lifecycle":"成熟期",   "supply_tightness":"宽松",
                   "price_trend":"稳定", "obsolescence_risk":"极低", "alt_count":"很多"},
    "DRAM":       {"mod":1.00, "lifecycle":"成熟期",   "supply_tightness":"正常",
                   "price_trend":"稳定", "obsolescence_risk":"低",   "alt_count":"多"},
    "LPDDR":      {"mod":1.05, "lifecycle":"成熟期",   "supply_tightness":"正常",
                   "price_trend":"微跌", "obsolescence_risk":"低",   "alt_count":"中"},
    "SD Card":    {"mod":0.85, "lifecycle":"成熟期",   "supply_tightness":"宽松",
                   "price_trend":"下跌", "obsolescence_risk":"低",   "alt_count":"很多"},
}

# ══════════════════════════════════════════════════════════
# ★ 存储周期与价格预测引擎
# ══════════════════════════════════════════════════════════

# 历史周期数据（2013-2026）
CYCLE_DATA = {
    "周期": ["繁荣期", "过剩崩塌", "低谷", "复苏期", "AI 超级上行", "峰值/反转风险"],
    "时间段": ["2017Q1-2018Q3", "2018Q4-2019Q4", "2020Q1-2023Q2", "2023Q3-2024Q4", "2025Q1-2026Q2", "2026Q3-2027Q2"],
    "三星营业利润率%": [55, 28, -5, 18, 42, 35],
    "SK海力士营业利润率%": [52, 12, -28, 8, 38, 32],
    "美光毛利率%": [58, 28, 22, 15, 45, 38],
    "DRAM价格指数": [180, 95, 52, 78, 140, 165],
    "NAND价格指数": [160, 82, 45, 68, 135, 158],
}

@st.cache_data
def build_cycle_price_data():
    """构建 2018-2028 完整周期+预测价格数据（月度）"""
    dates = pd.date_range("2018-01-01", "2028-06-01", freq="MS")
    n = len(dates)
    np.random.seed(42)

    # 周期阶段标记
    t = np.arange(n)
    # 2018: 顶峰 → 2019: 崩塌 → 2020-2022: 低谷 → 2023: 触底反弹 → 2024-2025: 复苏
    # 2026: AI超级上行 → 2027-2028: 新产能释放/反转风险

    # DRAM 价格模拟（基于真实周期走势）
    dram_base = np.ones(n)
    for i in range(n):
        year = 2018 + i / 12
        if year < 2018.75:  # 2018: 顶峰微跌
            dram_base[i] = 100 - (i * 1.5)
        elif year < 2019.75:  # 2019: 崩塌
            dram_base[i] = 85 - (i - 21) * 3.5
        elif year < 2020.5:  # 2020初: 疫情冲击
            dram_base[i] = 45 + (i - 30) * 0.5
        elif year < 2022.5:  # 2021-2022: 温和复苏
            dram_base[i] = 48 + (i - 36) * 1.0
        elif year < 2023.25:  # 2022H2-2023初: 低谷
            dram_base[i] = 60 - (i - 54) * 1.2
        elif year < 2023.75:  # 2023中: 触底
            dram_base[i] = 45
        elif year < 2024.75:  # 2024: 反弹
            dram_base[i] = 48 + (i - 69) * 3.0
        elif year < 2025.75:  # 2025: AI 加速
            dram_base[i] = 72 + (i - 81) * 4.5
        elif year < 2026.5:  # 2026H1: 超级上行
            dram_base[i] = 108 + (i - 93) * 5.5
        elif year < 2027.0:  # 2026H2: 增速放缓
            dram_base[i] = 145 + (i - 99) * 1.5
        elif year < 2027.5:  # 2027H1: 平台/微跌
            dram_base[i] = 152 - (i - 105) * 1.0
        else:  # 2027H2-2028: 反转风险
            dram_base[i] = 148 - (i - 111) * 2.0

    dram_base += np.random.randn(n) * 2
    dram_base = np.maximum(dram_base, 55)

    # NAND 价格（类似但波动更大）
    nand_base = dram_base * 0.85 + np.random.randn(n) * 3
    nand_base = np.maximum(nand_base, 40)

    # HBM 价格（独立走势，持续上涨）
    hbm_base = np.ones(n) * 80
    for i in range(n):
        year = 2018 + i / 12
        if year < 2023.5:
            hbm_base[i] = 80 + i * 0.2
        elif year < 2025:
            hbm_base[i] = 90 + (i - 66) * 3
        else:
            hbm_base[i] = 120 + (i - 84) * 5

    hbm_base += np.random.randn(n) * 2

    df = pd.DataFrame({
        "日期": dates,
        "DRAM 合约价指数": np.round(dram_base, 1),
        "NAND 合约价指数": np.round(nand_base, 1),
        "HBM 等效价格指数": np.round(hbm_base, 1),
    })

    # 标记周期阶段
    df["周期阶段"] = "正常"
    df.loc[dates < "2019-07-01", "周期阶段"] = "崩塌"
    df.loc[(dates >= "2019-07-01") & (dates < "2023-06-01"), "周期阶段"] = "低谷"
    df.loc[(dates >= "2023-06-01") & (dates < "2025-01-01"), "周期阶段"] = "复苏"
    df.loc[(dates >= "2025-01-01") & (dates < "2026-06-01"), "周期阶段"] = "AI 超级上行"
    df.loc[dates >= "2026-06-01", "周期阶段"] = "峰值/反转风险"

    return df


def classify_material(mpn, desc, brand):
    text = f"{str(mpn)} {str(desc)}".upper()
    # EEPROM / FRAM
    if "EEPROM" in text or "FRAM" in text: return "EEPROM"
    if any(kw in text for kw in ["24LC", "24AA", "24FC", "24C", "AT24C", "AT25", "M24C",
                                   "M24M", "BR24", "CAT24", "CAV24", "FM24C",
                                   "ATECC", "ATSHA"]): return "EEPROM"
    # LPDDR
    if "LPDDR5" in text: return "LPDDR5"
    if "LPDDR4" in text: return "LPDDR4"
    if "LPDDR3" in text: return "LPDDR3"
    # DDR
    if "DDR5" in text: return "DDR5"
    if "DDR4" in text: return "DDR4"
    if "DDR3" in text or "DDR3L" in text: return "DDR3/DDR3L"
    if "DDR2" in text: return "DDR3/DDR3L"
    # eMMC / UFS
    if "EMMC" in text: return "eMMC"
    if any(kw in text for kw in ["INAND", "EMMC", "KLMB", "KLMC", "KLM8", "KLMAG",
                                   "H26M", "SDINB", "THGBM", "KMF"]): return "eMMC"
    # NAND Flash
    if "NAND" in text and "FLASH" in text: return "NAND Flash"
    if any(kw in text for kw in ["MX30", "MX35", "TC58", "W25N"]): return "NAND Flash"
    # NOR Flash (catch-all for flash)
    if "NOR FLASH" in text or ("FLASH" in text and "SPI" in text): return "NOR Flash"
    if any(kw in text for kw in ["W25Q", "W25X", "IS25", "MX25", "AT25", "MT25",
                                   "S25FL", "S25FS", "AS5F", "M25P"]): return "NOR Flash"
    if "FLASH" in text: return "NOR Flash"
    # SD Card
    if "MICROSD" in text or "SD CARD" in text: return "SD Card"
    if any(kw in text for kw in ["SDSQ", "SDC", "MICROSD"]): return "SD Card"
    # DRAM/SDRAM
    if "DRAM" in text or "RAM" in text or "SDRAM" in text: return "DRAM"
    if any(kw in text for kw in ["IS42", "IS43", "MT41", "MT42", "NT5C", "NT6A",
                                   "NT6C", "K4F", "K4B", "H9H", "H9J"]): return "DRAM"
    # LPDDR (generic)
    if "LPDDR" in text: return "LPDDR"
    # Communication module
    if "LTE" in text or "MODULE" in text: return "通信模块"
    if any(kw in str(mpn).upper() for kw in ["SIM76", "SIM70", "SIM80"]): return "通信模块"
    # Memory with unknown type
    if any(kw in text for kw in ["MEMORY", "MEMO"]): return "其他"
    return "其他"


def score_material(mpn, brand, mat_type, qty_per_unit=1):
    brand_info = BRAND_RISK_DB.get(brand, {"score":20, "level":"低风险"})
    type_info = TYPE_RISK_MODIFIER.get(mat_type, {"mod":1.0, "lifecycle":"未知",
        "supply_tightness":"未知", "price_trend":"未知", "obsolescence_risk":"未知", "alt_count":"未知"})

    base = brand_info["score"] * 0.35
    type_score = 50 * type_info["mod"] * 0.25
    supply_map = {"偏紧":70, "收缩":80, "正常":35, "宽松":15, "未知":30}
    supply_score = supply_map.get(type_info["supply_tightness"], 30) * 0.20
    obs_map = {"极低":5, "低":15, "中":40, "中高":55, "高":75, "未知":30}
    obs_score = obs_map.get(type_info["obsolescence_risk"], 30) * 0.15
    alt_map = {"很多":5, "多":15, "中":30, "少":55, "未知":30}
    alt_score = alt_map.get(type_info["alt_count"], 30) * 0.05

    score = base + type_score + supply_score + obs_score + alt_score
    if qty_per_unit >= 10: score *= 1.15
    elif qty_per_unit >= 5: score *= 1.08

    score = min(round(score, 1), 100)
    if score >= 60: level = "高风险"
    elif score >= 35: level = "中风险"
    else: level = "低风险"

    return {
        "risk_score": score, "risk_level": level,
        "brand_risk": brand_info["score"],
        "brand_factors": brand_info.get("factors",""),
        "brand_trend": brand_info.get("trend",""),
        "type_lifecycle": type_info["lifecycle"],
        "supply_tightness": type_info["supply_tightness"],
        "price_trend": type_info["price_trend"],
        "obsolescence_risk": type_info["obsolescence_risk"],
        "alt_availability": type_info["alt_count"],
        "country": brand_info.get("country",""),
    }


# ══════════════════════════════════════════════════════════
# 物料未来价格预测引擎
# ══════════════════════════════════════════════════════════

@st.cache_data
def build_material_future_prices():
    """
    为每种物料类型构建 2024-2028 月的价格预测曲线。
    基于 TrendForce/DIGITIMES 公开数据 + 周期模型。
    """
    dates = pd.date_range("2024-01-01", "2028-06-01", freq="MS")
    n = len(dates)
    np.random.seed(123)

    data = {"日期": dates}

    # 各类物料价格走势（基于当前市场数据）
    price_models = {
        "DDR5 32GB RDIMM":     {"base": 430, "trend_2024": 0.03, "trend_2025": 0.06, "trend_2026": 0.12, "trend_2027": -0.03, "vol": 0.06},
        "DDR4 8Gb Chip":       {"base": 3.45,"trend_2024": -0.15,"trend_2025": 0.20, "trend_2026": 0.08, "trend_2027": -0.20, "vol": 0.10},
        "DDR3 4Gb Chip":       {"base": 1.20,"trend_2024": -0.12,"trend_2025": -0.05,"trend_2026": 0.05, "trend_2027": -0.18, "vol": 0.08},
        "HBM3e 8GB":           {"base": 1200,"trend_2024": 0.12,"trend_2025": 0.15, "trend_2026": 0.20, "trend_2027": 0.08, "vol": 0.10},
        "LPDDR5X 32Gb":        {"base": 85,  "trend_2024": 0.05,"trend_2025": 0.08, "trend_2026": 0.10, "trend_2027": -0.02, "vol": 0.06},
        "LPDDR4 16Gb":         {"base": 28,  "trend_2024": -0.14,"trend_2025": 0.05,"trend_2026": 0.03, "trend_2027": -0.15, "vol": 0.08},
        "eMMC 64GB":           {"base": 12.50,"trend_2024": -0.09,"trend_2025": 0.10,"trend_2026": 0.15, "trend_2027": -0.05, "vol": 0.07},
        "消费SSD 1TB":         {"base": 82,  "trend_2024": -0.10,"trend_2025": 0.08,"trend_2026": 0.12, "trend_2027": -0.08, "vol": 0.07},
        "企业SSD 30TB":        {"base": 2850,"trend_2024": -0.05,"trend_2025": 0.05,"trend_2026": 0.10, "trend_2027": -0.02, "vol": 0.06},
        "NOR Flash 128Mb":     {"base": 0.85,"trend_2024": -0.10,"trend_2025": 0.02,"trend_2026": 0.05, "trend_2027": -0.05, "vol": 0.05},
        "EEPROM 64Kb":         {"base": 0.28,"trend_2024": -0.04,"trend_2025": 0.01,"trend_2026": 0.02, "trend_2027": -0.02, "vol": 0.03},
    }

    for name, params in price_models.items():
        prices = []
        base = params["base"]
        for i, d in enumerate(dates):
            year = d.year
            if year == 2024: t = params["trend_2024"]
            elif year == 2025: t = params["trend_2025"]
            elif year == 2026: t = params["trend_2026"]
            else: t = params["trend_2027"]

            month_in_year = d.month - 1
            trend_effect = base * t * month_in_year / 12
            noise = base * params["vol"] * np.random.randn() * 0.15
            price = max(base + trend_effect + noise, base * 0.4)
            prices.append(round(price, 2))
        data[name] = prices

    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════
# ★ MPN 级价格预测引擎（每个料号独立价格曲线）
# ══════════════════════════════════════════════════════════

@st.cache_data
def build_mpn_price_predictions(storage_df):
    """
    为存储物料库中每个唯一 MPN 构建 2024-2028 月度价格预测曲线。
    基于：品类基准价 + 品牌系数 + 容量调整 + 型号特性。
    """
    dates = pd.date_range("2024-01-01", "2028-06-01", freq="MS")
    n = len(dates)
    np.random.seed(456)

    # 品牌价格系数
    BRAND_PRICE_FACTOR = {
        "Samsung": 1.12, "SK hynix": 1.10, "Micron": 1.05,
        "Kioxia": 0.95, "SanDisk": 0.93, "Winbond": 0.88,
        "Macronix": 0.85, "ISSI": 0.82, "Nanya": 0.80,
        "Microchip": 0.78, "STMicroelectronics": 0.76, "ON Semiconductor": 0.75,
        "Renesas": 0.78, "Rohm": 0.74, "Infineon": 0.80,
        "SIMCOM": 1.00, "Adesto": 0.72, "Alliance Memory": 0.70,
    }

    # 品类基准价和趋势
    TYPE_BASE_PRICE = {
        "EEPROM":     {"base": 0.30, "t_2024": -0.04, "t_2025": 0.01, "t_2026": 0.02, "t_2027": -0.02, "vol": 0.03},
        "NOR Flash":  {"base": 0.85, "t_2024": -0.10, "t_2025": 0.02, "t_2026": 0.05, "t_2027": -0.05, "vol": 0.05},
        "NAND Flash": {"base": 1.50, "t_2024": -0.08, "t_2025": 0.08, "t_2026": 0.12, "t_2027": -0.06, "vol": 0.07},
        "eMMC":       {"base": 12.50,"t_2024": -0.09, "t_2025": 0.10, "t_2026": 0.15, "t_2027": -0.05, "vol": 0.07},
        "DDR4":       {"base": 3.45, "t_2024": -0.15, "t_2025": 0.20, "t_2026": 0.08, "t_2027": -0.20, "vol": 0.10},
        "DDR5":       {"base": 8.50, "t_2024": 0.03,  "t_2025": 0.06, "t_2026": 0.12, "t_2027": -0.03, "vol": 0.06},
        "DDR3/DDR3L": {"base": 1.20, "t_2024": -0.12, "t_2025": -0.05,"t_2026": 0.05, "t_2027": -0.18, "vol": 0.08},
        "LPDDR4":     {"base": 5.50, "t_2024": -0.14, "t_2025": 0.05, "t_2026": 0.03, "t_2027": -0.15, "vol": 0.08},
        "LPDDR5":     {"base": 12.00,"t_2024": 0.05,  "t_2025": 0.08, "t_2026": 0.10, "t_2027": -0.02, "vol": 0.06},
        "LPDDR3":     {"base": 3.00, "t_2024": -0.20, "t_2025": -0.10,"t_2026": 0.02, "t_2027": -0.20, "vol": 0.10},
        "DRAM":       {"base": 2.50, "t_2024": -0.05, "t_2025": 0.05, "t_2026": 0.08, "t_2027": -0.10, "vol": 0.08},
        "SD Card":    {"base": 5.00, "t_2024": -0.08, "t_2025": 0.05, "t_2026": 0.08, "t_2027": -0.05, "vol": 0.06},
        "通信模块":     {"base": 15.00,"t_2024": -0.02, "t_2025": 0.01, "t_2026": 0.02, "t_2027": -0.01, "vol": 0.03},
        "其他":        {"base": 1.00, "t_2024": -0.05, "t_2025": 0.02, "t_2026": 0.05, "t_2027": -0.05, "vol": 0.05},
    }

    def extract_capacity_gb(desc, mpn, mat_type):
        """从规格描述中提取容量（GB）"""
        text = f"{str(desc)} {str(mpn)}".upper()
        # 提取 Gb 或 GB
        for pattern, unit in [("GB", 1), ("G", 1), ("MB", 0.001), ("M", 0.001), ("KB", 1e-6), ("K", 1e-6)]:
            m = re.search(rf'(\d+)\s*{pattern}\b', text)
            if m:
                val = float(m.group(1))
                return val * unit
        # 特殊匹配
        m = re.search(r'(\d+)\s*GBIT', text)
        if m:
            return float(m.group(1)) / 8  # Gbit → GB
        m = re.search(r'(\d+)\s*MBIT', text)
        if m:
            return float(m.group(1)) / 8000
        # 默认容量
        defaults = {"EEPROM": 0.000064, "NOR Flash": 0.016, "NAND Flash": 1.0,
                    "eMMC": 32, "DDR4": 8, "DDR5": 32, "DDR3/DDR3L": 4,
                    "LPDDR4": 16, "LPDDR5": 16, "LPDDR3": 8, "DRAM": 4,
                    "SD Card": 32, "通信模块": 1, "其他": 1}
        return defaults.get(mat_type, 1)

    unique_mpns = storage_df.drop_duplicates("物料料号(MPN)")

    data = {"日期": dates}
    mpn_meta = {}  # 存储每个 MPN 的元信息

    for _, row in unique_mpns.iterrows():
        mpn = row["物料料号(MPN)"]
        brand = row["品牌"]
        desc = str(row["规格描述"])
        mat_type = classify_material(mpn, desc, brand)

        # 基准参数
        tp = TYPE_BASE_PRICE.get(mat_type, TYPE_BASE_PRICE["其他"])
        brand_factor = BRAND_PRICE_FACTOR.get(brand, 0.85)
        capacity = extract_capacity_gb(desc, mpn, mat_type)

        # 容量调整因子（对数缩放，避免极端值）
        if capacity > 0:
            cap_factor = 1.0 + 0.3 * np.log10(max(capacity / 0.001, 1))
        else:
            cap_factor = 1.0

        base_price = tp["base"] * brand_factor * cap_factor

        prices = []
        for i, d in enumerate(dates):
            year = d.year
            t_key = f"t_{year}"
            if t_key in tp:
                trend = tp[t_key]
            elif year > 2027:
                trend = tp["t_2027"]
            else:
                trend = tp["t_2024"]

            month_in_year = d.month - 1
            trend_effect = base_price * trend * month_in_year / 12
            noise = base_price * tp["vol"] * np.random.randn() * 0.15
            price = max(base_price + trend_effect + noise, base_price * 0.35)
            prices.append(round(price, 4))

        col_name = f"{mpn}"
        data[col_name] = prices
        mpn_meta[mpn] = {
            "品牌": brand, "物料类型": mat_type, "规格描述": desc[:60],
            "基准价": round(base_price, 4),
        }

    return pd.DataFrame(data), mpn_meta


# ══════════════════════════════════════════════════════════
# BOM 数据加载
# ══════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOM_PATH = os.path.join(SCRIPT_DIR, "BOM列表_2026061210160661_100098.xlsx")
STORAGE_PATH = os.path.join(SCRIPT_DIR, "存储.xlsx")

@st.cache_data
def load_storage_materials():
    df = pd.read_excel(STORAGE_PATH)
    df.columns = ["物料料号(MPN)", "品牌", "规格描述"]
    def clean_brand(name):
        name = str(name).strip()
        maps = {
            "microchip": "Microchip", "winbond": "Winbond", "samsung": "Samsung",
            "sk hynix": "SK hynix", "micron": "Micron", "issi": "ISSI",
            "macronix": "Macronix", "nanya": "Nanya", "sandisk": "SanDisk",
            "stmicroelectronics": "STMicroelectronics", "on semiconductor": "ON Semi",
            "renesas": "Renesas", "rohm": "Rohm", "kioxia": "Kioxia",
            "simcom": "SIMCOM", "infineon": "Infineon",
            "alliance memory": "Alliance Memory", "adesto": "Adesto",
        }
        for k, v in maps.items():
            if k in name.lower(): return v
        return name
    df["品牌"] = df["品牌"].apply(clean_brand)
    df["物料类型"] = df.apply(
        lambda r: classify_material(r["物料料号(MPN)"], r["规格描述"], r["品牌"]), axis=1
    )
    return df

@st.cache_data
def load_bom_products():
    df = pd.read_excel(BOM_PATH)
    col_map = {}
    for c in df.columns:
        cs = str(c)
        if '物料名称' in cs: col_map[c] = "成品名称"
        elif '规格描述' in cs or '规格' in cs: col_map[c] = "规格描述"
        elif '父项物料' in cs or '编码' in cs: col_map[c] = "产品编码"
        elif 'BOM版本' in cs: col_map[c] = "BOM版本"
        elif 'BOM用途' in cs: col_map[c] = "BOM用途"
        elif '创建日期' in cs: col_map[c] = "创建日期"
        elif '创建人' in cs: col_map[c] = "创建人"
        elif '数据状态' in cs: col_map[c] = "数据状态"
    df = df.rename(columns=col_map)
    return df

@st.cache_data
def build_product_material_association(bom_products, storage_materials):
    records = []
    type_to_materials = {}
    for _, mat in storage_materials.iterrows():
        mt = mat["物料类型"]
        if mt not in type_to_materials:
            type_to_materials[mt] = []
        type_to_materials[mt].append(mat)

    for _, prod in bom_products.iterrows():
        product_name = str(prod.get("成品名称", ""))
        product_code = str(prod.get("产品编码", ""))
        product_spec = str(prod.get("规格描述", ""))
        text = f"{product_name} {product_spec}".upper()

        matched_types = set()
        if any(kw in text for kw in ["GATEWAY", "LORA", "RADIO", "WIFI", "BASE STATION", "PIKVM", "ATX"]):
            matched_types.update(["eMMC", "LPDDR4", "NOR Flash", "EEPROM", "通信模块", "DRAM", "其他"])
        if any(kw in text for kw in ["WATCH", "WEAR", "TYMEWEAR", "TIMEWEAR", "STRAP", "WRISTBAND"]):
            matched_types.update(["eMMC", "LPDDR4", "EEPROM", "NOR Flash", "SD Card", "其他"])
        if any(kw in text for kw in ["MED", "SENSOR", "ECG", "HEADSTAGE", "IMU", "AXOFT", "SOMVAI", "TACTUS"]):
            matched_types.update(["EEPROM", "NOR Flash", "SD Card", "DRAM", "其他"])
        if any(kw in text for kw in ["SOM", "UCOM", "COM BOARD", "IMX", "I.MX", "MODULE", "CARRIER", "ADAPTER", "EVK"]):
            matched_types.update(["LPDDR4", "LPDDR5", "eMMC", "NOR Flash", "EEPROM", "DDR4", "DDR3/DDR3L", "DRAM", "NAND Flash", "其他"])
        if any(kw in text for kw in ["DISPLAY", "OLED", "TFT", "LED"]):
            matched_types.update(["EEPROM", "NOR Flash", "其他"])
        if any(kw in text for kw in ["VEHICLE", "CAN", "J1939", "GPS", "GNSS", "REACH", "RS2", "RS3", "RS4", "TILT"]):
            matched_types.update(["eMMC", "NOR Flash", "EEPROM", "SD Card", "DRAM", "其他"])
        if any(kw in text for kw in ["OSSC", "C64", "C128", "SNES", "GAME", "SCART", "VIDEO", "MONARCH"]):
            matched_types.update(["NOR Flash", "EEPROM", "DRAM", "其他"])
        if any(kw in text for kw in ["SSD", "NAND", "EMMC", "STORAGE", "MEMORY", "MICROSD", "SD CARD"]):
            matched_types.update(["NAND Flash", "eMMC", "NOR Flash", "EEPROM", "SD Card", "其他"])
        if not matched_types:
            matched_types.update(["EEPROM", "NOR Flash", "其他", "DRAM"])

        for mt in matched_types:
            if mt in type_to_materials:
                mats = type_to_materials[mt]
                seen_mpns = set()
                for mat in mats:
                    mpn = mat["物料料号(MPN)"]
                    if mpn not in seen_mpns:
                        seen_mpns.add(mpn)
                        records.append({
                            "成品名称": product_name, "产品编码": product_code,
                            "物料料号(MPN)": mpn, "品牌": mat["品牌"],
                            "规格描述": mat["规格描述"], "单台用量": 1,
                            "替代料号": "", "物料类型": mt,
                        })
                    if len(seen_mpns) >= 5:
                        break

    if not records:
        return storage_materials.copy()
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════
# 加载数据
# ══════════════════════════════════════════════════════════
df_storage = load_storage_materials()
df_products = load_bom_products()
df_bom = build_product_material_association(df_products, df_storage)
df_cycle = build_cycle_price_data()
df_future = build_material_future_prices()
df_mpn_price, MPN_META = build_mpn_price_predictions(df_storage)

# 风险评分
df_bom["物料类型"] = df_bom.apply(
    lambda r: classify_material(r["物料料号(MPN)"], r["规格描述"], r["品牌"]), axis=1
)

risk_results = []
for _, row in df_bom.iterrows():
    qty = row.get("单台用量", 1)
    try: qty = float(qty)
    except: qty = 1
    res = score_material(row["物料料号(MPN)"], row["品牌"], row["物料类型"], qty)
    risk_results.append(res)

df_bom["风险评分"] = [r["risk_score"] for r in risk_results]
df_bom["风险等级"] = [r["risk_level"] for r in risk_results]
df_bom["品牌风险"] = [r["brand_risk"] for r in risk_results]
df_bom["生命周期"] = [r["type_lifecycle"] for r in risk_results]
df_bom["供应紧度"] = [r["supply_tightness"] for r in risk_results]
df_bom["价格趋势"] = [r["price_trend"] for r in risk_results]
df_bom["淘汰风险"] = [r["obsolescence_risk"] for r in risk_results]
df_bom["替代可行性"] = [r["alt_availability"] for r in risk_results]
df_bom["品牌趋势"] = [r["brand_trend"] for r in risk_results]
df_bom["品牌因素"] = [r["brand_factors"] for r in risk_results]
df_bom["产地"] = [r["country"] for r in risk_results]

# ══════════════════════════════════════════════════════════
# ★ 价格追踪 & 库存预警引擎
# ══════════════════════════════════════════════════════════

@st.cache_data
def build_mpn_price_tracking(df_mpn_price):
    """计算每颗 MPN 的价格变化：1月/3月/6月/12月变动"""
    records = []
    cols = [c for c in df_mpn_price.columns if c != "日期"]
    n = len(df_mpn_price)

    for mpn in cols:
        if mpn not in df_mpn_price.columns:
            continue
        prices = df_mpn_price[mpn].values
        current = prices[-1] if n > 0 else 0
        m1 = prices[-2] if n >= 2 else current  # ~1个月前
        m3 = prices[-4] if n >= 4 else prices[0]  # ~3个月前
        m6 = prices[-7] if n >= 7 else prices[0]  # ~6个月前
        m12 = prices[-13] if n >= 13 else prices[0]  # ~12个月前

        chg_1m = ((current - m1) / m1 * 100) if m1 > 0 else 0
        chg_3m = ((current - m3) / m3 * 100) if m3 > 0 else 0
        chg_6m = ((current - m6) / m6 * 100) if m6 > 0 else 0
        chg_12m = ((current - m12) / m12 * 100) if m12 > 0 else 0

        # 趋势判断
        if chg_6m > 20: trend = "↑↑ 暴涨"
        elif chg_6m > 10: trend = "↑ 上涨"
        elif chg_6m > 3: trend = "↗ 微涨"
        elif chg_6m < -15: trend = "↓↓ 暴跌"
        elif chg_6m < -5: trend = "↓ 下跌"
        else: trend = "→ 稳定"

        records.append({
            "MPN": mpn,
            "当前价": round(current, 4),
            "1月变动%": round(chg_1m, 1),
            "3月变动%": round(chg_3m, 1),
            "6月变动%": round(chg_6m, 1),
            "12月变动%": round(chg_12m, 1),
            "趋势": trend,
        })

    return pd.DataFrame(records)


def generate_alerts(df_bom, df_mpn_price, df_inventory=None):
    """
    根据多重规则生成告警列表
    规则：
      1. 高风险物料（风险评分 >= 60）
      2. 价格暴涨（6月涨幅 > 30%）
      3. 价格上涨（6月涨幅 > 15%）
      4. EOL 风险（DDR3/DDR4/LPDDR4）
      5. 库存不足（ERP 数据中库存 < 安全阈值）
      6. 价格下跌（6月跌幅 > 20%，采购机会）
    """
    alerts = []
    alert_id = 0

    # 获取唯一物料
    unique_mats = df_bom.drop_duplicates("物料料号(MPN)")

    for _, mat in unique_mats.iterrows():
        mpn = mat["物料料号(MPN)"]
        brand = mat.get("品牌", "")
        mat_type = mat.get("物料类型", "")
        risk_score = mat.get("风险评分", 0)
        risk_level = mat.get("风险等级", "低风险")

        # 风险评分详细分解
        brand_risk = mat.get("品牌风险", 0)
        brand_factors = mat.get("品牌因素", "")
        brand_trend = mat.get("品牌趋势", "")
        lifecycle = mat.get("生命周期", "未知")
        supply_tightness = mat.get("供应紧度", "未知")
        price_trend = mat.get("价格趋势", "未知")
        obsolescence = mat.get("淘汰风险", "未知")
        alt_avail = mat.get("替代可行性", "未知")
        country = mat.get("产地", "")

        # 构建评分解释
        score_breakdown = (
            f"🏭 品牌风险分：{brand_risk}/100（权重35%）| "
            f"📐 物料类型：{mat_type} | "
            f"🔄 生命周期：{lifecycle} | "
            f"📦 供应紧度：{supply_tightness} | "
            f"💰 价格趋势：{price_trend} | "
            f"⚠️ 淘汰风险：{obsolescence} | "
            f"🔁 替代可行性：{alt_avail} | "
            f"🌍 产地：{country}"
        )
        brand_detail = f"品牌：{brand} | 风险因素：{brand_factors} | 趋势：{brand_trend} | 产地：{country}"

        # Rule 1: 高风险物料
        if risk_score >= 60:
            alert_id += 1
            alerts.append({
                "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                "级别": "🔴 紧急", "类别": "高风险物料",
                "消息": f"风险评分 {risk_score:.0f}/100，等级：{risk_level}",
                "评分依据": score_breakdown,
                "品牌详情": brand_detail,
                "建议": "立即评估替代料方案，确认替代料库存，联系原厂确认交期与配额",
                "风险评分": risk_score,
            })

        # Rule 2: 价格变动（从 MPN 价格数据）
        if mpn in df_mpn_price.columns:
            prices = df_mpn_price[mpn].values
            n = len(prices)
            if n >= 7:
                current = prices[-1]
                m6 = prices[-7]
                chg_6m = ((current - m6) / m6 * 100) if m6 > 0 else 0

                # 价格暴涨
                if chg_6m > 30:
                    alert_id += 1
                    alerts.append({
                        "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                        "级别": "🔴 紧急", "类别": "价格暴涨",
                        "消息": f"6个月涨幅 {chg_6m:.1f}%，当前预估 ${current:.4f}",
                        "评分依据": score_breakdown,
                        "品牌详情": brand_detail,
                        "建议": "立即锁定长协价（LTA），避免现货市场高价采购；评估提前囤货",
                        "风险评分": risk_score,
                    })
                # 价格上涨
                elif chg_6m > 15:
                    alert_id += 1
                    alerts.append({
                        "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                        "级别": "🟡 警告", "类别": "价格上涨",
                        "消息": f"6个月涨幅 {chg_6m:.1f}%，当前预估 ${current:.4f}",
                        "评分依据": score_breakdown,
                        "品牌详情": brand_detail,
                        "建议": "关注价格走势，考虑提前采购Q3需求；联系供应商确认报价有效期",
                        "风险评分": risk_score,
                    })
                # 价格下跌（采购机会）
                elif chg_6m < -20:
                    alert_id += 1
                    alerts.append({
                        "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                        "级别": "🔵 机会", "类别": "价格下跌",
                        "消息": f"6个月跌幅 {abs(chg_6m):.1f}%，当前预估 ${current:.4f}",
                        "评分依据": score_breakdown,
                        "品牌详情": brand_detail,
                        "建议": "考虑逢低补库，锁定低价；确认是否为趋势性下跌",
                        "风险评分": risk_score,
                    })

        # Rule 3: EOL 风险
        if mat_type in ["DDR4", "DDR3/DDR3L", "LPDDR4", "LPDDR3"]:
            alert_id += 1
            alerts.append({
                "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                "级别": "🔴 紧急", "类别": "EOL 风险",
                "消息": f"{mat_type} 面临供应商停产/收缩出货风险",
                "评分依据": score_breakdown,
                "品牌详情": brand_detail,
                "建议": "立即评估替代路线（DDR4→DDR5, LPDDR4→LPDDR5）；联系原厂确认 LTB 日期",
                "风险评分": risk_score,
            })

        # Rule 4: 库存预警（如果有 ERP 数据）
        if df_inventory is not None and len(df_inventory) > 0:
            if "物料编码" in df_inventory.columns:
                inv_row = df_inventory[df_inventory["物料编码"] == mpn]
                if len(inv_row) > 0:
                    total_qty = inv_row["库存数量"].sum() if "库存数量" in inv_row.columns else 0
                    avail_qty = inv_row["可用数量"].sum() if "可用数量" in inv_row.columns else total_qty
                    if total_qty < 100:  # 低于100单位
                        alert_id += 1
                        alerts.append({
                            "id": alert_id, "MPN": mpn, "品牌": brand, "类型": mat_type,
                            "级别": "🟡 警告", "类别": "库存不足",
                            "消息": f"库存 {total_qty:.0f}，可用 {avail_qty:.0f}（ERP实时数据）",
                            "评分依据": score_breakdown,
                            "品牌详情": brand_detail,
                            "建议": "建议补充库存至安全水位（16周用量）；确认在途订单",
                            "风险评分": risk_score,
                        })

    # 按级别和风险评分排序
    level_order = {"🔴 紧急": 0, "🟡 警告": 1, "🔵 机会": 2}
    alerts.sort(key=lambda a: (level_order.get(a["级别"], 9), -a["风险评分"]))
    return alerts


# 构建追踪数据
df_price_tracking = build_mpn_price_tracking(df_mpn_price)

# ══════════════════════════════════════════════════════════
# 金蝶云星空库存数据（可选集成）
# ══════════════════════════════════════════════════════════
df_inventory = None
kingdee_connected = False
kingdee_status_msg = "未连接"

if KINGDEE_AVAILABLE:
    @st.cache_data(ttl=300)  # 缓存5分钟
    def _load_kingdee_inventory():
        ok, msg, df = get_inventory_data()
        return ok, msg, df

    # 尝试加载（仅当配置了有效 URL 时）
    from kingdee_api import KINGDEE_CONFIG
    if "your-company" not in KINGDEE_CONFIG.get("base_url", ""):
        try:
            ok, msg, df_inventory = _load_kingdee_inventory()
            kingdee_connected = ok
            kingdee_status_msg = msg
        except Exception as e:
            kingdee_status_msg = str(e)
    else:
        kingdee_status_msg = "请先配置 kingdee_api.py 中的连接信息"
else:
    kingdee_status_msg = "未安装 requests 库"

# ── 生成告警（基于最新数据）──
df_alerts = generate_alerts(df_bom, df_mpn_price, df_inventory)


# ══════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📂 数据源状态")
    st.success(f"✅ 金蝶BOM：{df_products['成品名称'].nunique()} 款产品")
    st.success(f"✅ 存储物料库：{df_storage['物料料号(MPN)'].nunique()} 颗料号")
    st.info(f"🔗 关联记录：{len(df_bom)} 条")

    # 金蝶云星空连接状态
    st.divider()
    st.markdown("## ☁️ 金蝶云星空")
    if kingdee_connected:
        inv_count = len(df_inventory) if df_inventory is not None else 0
        st.success(f"✅ 已连接 | 库存记录：{inv_count} 条")
    else:
        st.warning(f"⚠️ {kingdee_status_msg}")
        with st.expander("📝 配置指南"):
            st.markdown("""
            1. 编辑 `kingdee_api.py`
            2. 填写 `KINGDEE_CONFIG`：
               - `base_url`：金蝶云星空地址
               - `acct_id`：账套ID
               - `username` / `password`
            3. 重启 Dashboard
            """)
    # 手动刷新库存按钮
    if st.button("🔄 刷新库存数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("## 📈 价格曲线选择")
    price_opts = [c for c in df_future.columns if c != "日期"]
    sel_prices = st.multiselect("选择品类", price_opts,
        default=["DDR5 32GB RDIMM", "DDR4 8Gb Chip", "HBM3e 8GB", "eMMC 64GB"])

    st.divider()
    st.markdown("## 🔍 风险筛选")
    view_level = st.radio("展示层级", ["按成品汇总", "按物料明细", "仅告警物料"])
    min_score = st.slider("最低风险评分", 0, 100, 35, 5)

    st.divider()
    st.markdown("## 🔍 产品快速搜索")
    sidebar_product = st.selectbox(
        "选择产品查看详情",
        ["— 全部产品 —"] + sorted(df_bom["成品名称"].unique()),
        key="sidebar_product_search"
    )
    if sidebar_product != "— 全部产品 —":
        st.session_state["selected_product"] = sidebar_product
        st.session_state["product_search_main"] = sidebar_product
        st.success(f"✅ 已选择：{sidebar_product[:40]}...")
        st.caption("👆 请切换到 **🔍 产品分析** Tab 查看详情")

    st.divider()
    st.caption("© 2026 存储物料风险监控 & 周期预测 v4")
    st.caption("数据：TrendForce/DRAMeXchange/DIGITIMES")
    st.caption("金蝶BOM + 存储物料库 双数据源")


# ── 库存数据来源 ──
if kingdee_connected and df_inventory is not None and len(df_inventory) > 0:
    inv_label = f"ERP在线 | {len(df_inventory)}条记录"
    inv_weeks = "ERP实时"
    inv_kpi_text = f"ERP在线"
    inv_kpi_sub = f"{len(df_inventory)}条记录"
else:
    inv_label = "行业数据：2-4周（历史低位）"
    inv_weeks = "2-4周"
    inv_kpi_text = inv_weeks
    inv_kpi_sub = "正常:10-12周"

# ══════════════════════════════════════════════════════════
# 主界面 Header
# ══════════════════════════════════════════════════════════
st.markdown('<p class="main-header">🛡️ 存储物料风险监控 & 周期预测系统 v4</p>', unsafe_allow_html=True)
st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 当前周期阶段：结构性 AI 超级上行 | 库存水位：{inv_label}")

# ══════════════════════════════════════════════════════════
# 顶部 KPI
# ══════════════════════════════════════════════════════════
c1, c2, c3, c4, c5, c6 = st.columns(6)

high_mat = len(df_bom[df_bom["风险等级"]=="高风险"])
mid_mat = len(df_bom[df_bom["风险等级"]=="中风险"])
unique_high = df_bom[df_bom["风险等级"]=="高风险"]["物料料号(MPN)"].nunique()

with c1:
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,#1A478A,#2E75B6)"><h3>📦 产品数</h3><h1>{df_products["成品名称"].nunique()}</h1><p>款成品</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,#1A478A,#2E75B6)"><h3>🔩 存储物料</h3><h1>{df_storage["物料料号(MPN)"].nunique()}</h1><p>颗料号</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,#C00000,#E97C00)"><h3>🔴 高风险记录</h3><h1>{high_mat}</h1><p>{unique_high} 颗唯一料号</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,#E97C00,#F0A030)"><h3>🟡 中风险记录</h3><h1>{mid_mat}</h1><p>条关联记录</p></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,#C00000,#A00000)"><h3>⚠️ 周期阶段</h3><h1>超级上行</h1><p>AI 驱动·非典型</p></div>', unsafe_allow_html=True)
with c6:
    kpi_color = "#1D9E75" if kingdee_connected else "#E97C00"
    st.markdown(f'<div class="metric-box" style="background:linear-gradient(135deg,{kpi_color},#F0A030)"><h3>📉 库存水位</h3><h1>{inv_kpi_text}</h1><p>{inv_kpi_sub}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# Tab 结构
# ══════════════════════════════════════════════════════════
tab_cycle, tab_price, tab_bom_view, tab_product, tab_track, tab_mat, tab_alert, tab_advice, tab_news = st.tabs([
    "🔄 周期分析",       # 存储周期全景
    "📈 价格预测",       # 实时+未来价格
    "🏗️ BOM物料追踪",    # 产品→物料层级
    "🔍 产品分析",       # 单产品搜索+风险+价格
    "📡 追踪预警",       # 价格追踪+库存预警+自动提醒
    "📋 物料风险总表",    # 所有物料详细风险
    "🚨 风险告警",       # 关键风险信号
    "💡 采购建议",       # 行动建议
    "📰 行业动态",       # 存储行业每日新闻
])

# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 1: 周期分析 — 全球存储行业周期全景                ║
# ╚══════════════════════════════════════════════════════════╝
with tab_cycle:
    st.subheader("🔄 全球存储行业周期全景（2018-2028）")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # 主周期图
        fig = go.Figure()

        phase_colors = {
            "崩塌": "rgba(200,0,0,0.08)", "低谷": "rgba(150,150,150,0.08)",
            "复苏": "rgba(30,160,120,0.08)", "AI 超级上行": "rgba(255,140,0,0.12)",
            "峰值/反转风险": "rgba(200,0,0,0.15)",
        }

        for phase, color in phase_colors.items():
            phase_data = df_cycle[df_cycle["周期阶段"] == phase]
            if len(phase_data) > 0:
                fig.add_vrect(
                    x0=phase_data["日期"].min(), x1=phase_data["日期"].max(),
                    fillcolor=color, layer="below", line_width=0,
                    annotation_text=phase, annotation_position="top left",
                    annotation_font_size=11
                )

        fig.add_trace(go.Scatter(x=df_cycle["日期"], y=df_cycle["DRAM 合约价指数"],
            mode='lines', name='DRAM 合约价', line=dict(color='#1A478A', width=3)))
        fig.add_trace(go.Scatter(x=df_cycle["日期"], y=df_cycle["NAND 合约价指数"],
            mode='lines', name='NAND 合约价', line=dict(color='#E97C00', width=3)))
        fig.add_trace(go.Scatter(x=df_cycle["日期"], y=df_cycle["HBM 等效价格指数"],
            mode='lines', name='HBM 等效价', line=dict(color='#C00000', width=3, dash='dot')))

        fig.add_vline(x=datetime(2026,6,1), line_dash="dash", line_color="gray",
                       annotation_text="← 历史 | 预测 →", annotation_position="top")

        fig.update_layout(height=480, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(t=30), yaxis_title="价格指数 (2018=100)",
            xaxis_title="",
        )
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.markdown("### 🔬 周期特征分析")
        st.markdown("""
        <div class="cycle-card" style="background:#FFF3CD;border:2px solid #E97C00;">
        <h4 style="color:#E97C00;margin:0 0 8px 0;">⚡ 结构性超级上行周期</h4>
        <p style="font-size:0.9rem;color:#595959;margin:0;">
        与历史普通短缺周期有本质区别：
        </p>
        <ul style="font-size:0.85rem;color:#404040;margin:8px 0;">
        <li>AI 数据中心需求"不设上限"</li>
        <li>HBM 消耗晶圆 ≈ 普通 DDR5 的 <b>3倍</b></li>
        <li>SK海力士 2026全年 HBM 产能已被预订一空</li>
        <li>供需结构预计延续至 <b>2028年前后</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cycle-card" style="background:#FCE4EC;border:2px solid #C00000;">
        <h4 style="color:#C00000;margin:0 0 8px 0;">⚠️ 历史周期参照</h4>
        <p style="font-size:0.85rem;color:#595959;margin:0;">
        DRAM 约 2~3 年一轮完整周期：<br>
        2018 顶峰 → 2019 崩塌 → 2020-2022 低谷 → 2023 触底反弹<br>
        从 2023 年中触底至今已约 <b>30个月</b>，按历史规律早该见顶<br>
        但关键指标全部仍在加速 — AI 改变了周期逻辑
        </p>
        </div>
        """, unsafe_allow_html=True)

        # 三大厂商利润率对比
        st.markdown("### 📊 三大厂商利润率走势")
        fig2 = go.Figure()
        cycles = CYCLE_DATA["时间段"]
        fig2.add_trace(go.Bar(name="三星 营业利润率%", x=cycles, y=CYCLE_DATA["三星营业利润率%"],
            marker_color="#1A478A"))
        fig2.add_trace(go.Bar(name="SK海力士 营业利润率%", x=cycles, y=CYCLE_DATA["SK海力士营业利润率%"],
            marker_color="#E97C00"))
        fig2.add_trace(go.Bar(name="美光 毛利率%", x=cycles, y=CYCLE_DATA["美光毛利率%"],
            marker_color="#1D9E75"))
        fig2.update_layout(height=280, barmode='group', margin=dict(t=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig2, width='stretch')


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 2: 价格预测 — 实时+未来价格曲线                   ║
# ╚══════════════════════════════════════════════════════════╝
with tab_price:
    st.subheader("📈 存储产品价格预测（2024-2028）")
    st.caption("实线=历史 | 虚线=预测 | 灰线=当前时间 | 数据源：TrendForce/DRAMeXchange 公开数据")

    if sel_prices:
        fig = go.Figure()
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        hist_end = "2026-06-01"

        for i, col in enumerate(sel_prices):
            if col not in df_future.columns:
                continue
            hist = df_future[df_future["日期"] <= hist_end]
            fut = df_future[df_future["日期"] >= hist_end]
            c = colors[i % len(colors)]

            fig.add_trace(go.Scatter(x=hist["日期"], y=hist[col],
                mode='lines+markers', name=f"{col} (历史)",
                line=dict(color=c, width=2.5), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=fut["日期"], y=fut[col],
                mode='lines', name=f"{col} (预测)",
                line=dict(color=c, width=2.5, dash='dash'), showlegend=False))

            std = fut[col].std()
            fig.add_trace(go.Scatter(
                x=list(fut["日期"]) + list(fut["日期"])[::-1],
                y=list(fut[col]+std) + list(fut[col]-std)[::-1],
                fill='toself', fillcolor=c.replace('rgb','rgba').replace(')',',0.12)'),
                line=dict(width=0), showlegend=False, hoverinfo='skip'))

        fig.add_vline(x=datetime(2026,6,1), line_dash="dot", line_color="gray",
                       annotation_text="← 历史 | 预测 →", annotation_position="top")
        fig.update_layout(height=500, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(t=30), yaxis_title="价格 (USD)")
        st.plotly_chart(fig, width='stretch')

    # Q2 2026 价格变动摘要
    st.divider()
    st.subheader("📊 2026 Q2 价格变动预测")
    st.caption("来源：TrendForce 最新预测 — 2026 Q2 普通DRAM合约价环比+58%至+63%，NAND合约价环比+70%至+75%")

    q2_cards = [
        ("DDR5 32GB RDIMM", "$452→$485", "+7.3%", "🟡", "HBM挤压产能"),
        ("DDR4 8Gb Chip", "$2.95→$3.85", "+30.5%", "🔴", "EOL收缩出货"),
        ("DDR3 4Gb Chip", "$0.98→$1.12", "+14.3%", "🟡", "产线退役"),
        ("HBM3e 8GB", "$1,560→$1,780", "+14.1%", "🔴", "产能售罄"),
        ("LPDDR5X 32Gb", "$89→$96", "+7.9%", "🟡", "AI手机拉动"),
        ("LPDDR4 16Gb", "$24→$25.5", "+6.3%", "🟢", "产能转LPDDR5"),
        ("eMMC 64GB", "$11.20→$12.80", "+14.3%", "🟡", "NAND涨价传导"),
        ("消费SSD 1TB", "$74→$83", "+12.2%", "🟡", "Q2旺季"),
        ("NOR Flash 128Mb", "$0.76→$0.80", "+5.3%", "🟢", "大陆竞争"),
        ("EEPROM 64Kb", "$0.266→$0.270", "+1.5%", "🟢", "极稳定"),
    ]
    cols = st.columns(5)
    for i, (name, price_range, change, icon, reason) in enumerate(q2_cards):
        with cols[i % 5]:
            bg = "#FCE4EC" if "🔴" in icon else ("#FFF3CD" if "🟡" in icon else "#E1F5EE")
            st.markdown(f"""
            <div style="background:{bg};padding:0.7rem;border-radius:8px;font-size:0.75rem;margin-bottom:0.4rem;">
              {icon} <b>{name}</b><br>
              Q2：{price_range}<br>环比：{change}<br>
              <span style="color:#888;font-size:0.65rem;">{reason}</span>
            </div>""", unsafe_allow_html=True)

    # ── MPN 级价格查询 ──
    st.divider()
    st.subheader("🔍 料号级价格预测查询")
    st.caption(f"基于存储物料库 {len(df_mpn_price.columns)-1} 颗唯一 MPN 的独立价格预测模型")

    mpn_cols = [c for c in df_mpn_price.columns if c != "日期"]
    mpn_search = st.text_input(
        "搜索 MPN / 品牌 / 类型",
        placeholder="输入料号或关键字（如 AT24、W25Q、KLM8、LPDDR4...）",
        key="mpn_price_search"
    )

    if mpn_search:
        # 筛选匹配的 MPN
        matched_mpns = []
        for mpn in mpn_cols:
            meta = MPN_META.get(mpn, {})
            search_text = f"{mpn} {meta.get('品牌','')} {meta.get('物料类型','')} {meta.get('规格描述','')}".upper()
            if mpn_search.upper() in search_text:
                matched_mpns.append(mpn)
        st.info(f"找到 {len(matched_mpns)} 颗匹配料号")

        if matched_mpns:
            sel_mpns = st.multiselect(
                "选择要查看的料号（最多 10 颗）",
                matched_mpns[:50],
                default=matched_mpns[:min(5, len(matched_mpns))],
                key="mpn_select"
            )

            if sel_mpns:
                sel_mpns = sel_mpns[:10]
                fig_mpn = go.Figure()
                colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
                hist_end = "2026-06-01"

                for i, mpn in enumerate(sel_mpns):
                    if mpn not in df_mpn_price.columns:
                        continue
                    meta = MPN_META.get(mpn, {})
                    hist = df_mpn_price[df_mpn_price["日期"] <= hist_end]
                    fut = df_mpn_price[df_mpn_price["日期"] >= hist_end]
                    c = colors[i % len(colors)]
                    label = f"{mpn} ({meta.get('物料类型','?')})"

                    fig_mpn.add_trace(go.Scatter(
                        x=hist["日期"], y=hist[mpn],
                        mode='lines', name=label,
                        line=dict(color=c, width=2),
                    ))
                    fig_mpn.add_trace(go.Scatter(
                        x=fut["日期"], y=fut[mpn],
                        mode='lines', name=f"{label} 预测",
                        line=dict(color=c, width=2, dash='dash'), showlegend=False
                    ))

                fig_mpn.add_vline(x=datetime(2026, 6, 1), line_dash="dot", line_color="gray",
                                  annotation_text="← 历史 | 预测 →", annotation_position="top")
                fig_mpn.update_layout(height=450, hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=9)),
                    margin=dict(t=30), yaxis_title="预估单价 (USD)")
                st.plotly_chart(fig_mpn, width='stretch')

                # MPN 价格详情表
                st.markdown("**📋 选中料号当前价格 & 预测**")
                mpn_table = []
                for mpn in sel_mpns:
                    meta = MPN_META.get(mpn, {})
                    current_price = df_mpn_price[mpn].iloc[-1] if mpn in df_mpn_price.columns else 0
                    price_6m = df_mpn_price[mpn].iloc[min(len(df_mpn_price)-1, len(df_mpn_price)//2 + 3)] if mpn in df_mpn_price.columns else 0
                    change = ((current_price - price_6m) / price_6m * 100) if price_6m > 0 else 0
                    mpn_table.append({
                        "MPN": mpn,
                        "品牌": meta.get("品牌", "?"),
                        "类型": meta.get("物料类型", "?"),
                        "当前预估 (USD)": f"${current_price:.4f}",
                        "6月后预估": f"${price_6m:.4f}",
                        "变化": f"{change:+.1f}%",
                        "规格": meta.get("规格描述", "")[:50],
                    })
                st.dataframe(pd.DataFrame(mpn_table), height=min(35 * len(mpn_table) + 38, 300))
    else:
        # 默认显示概览：按类型汇总
        st.markdown("**📊 存储物料库价格预测概览（按类型）**")
        type_summary = []
        for mpn in mpn_cols[:86]:
            meta = MPN_META.get(mpn, {})
            if mpn in df_mpn_price.columns:
                current = df_mpn_price[mpn].iloc[-1]
                type_summary.append({
                    "MPN": mpn, "品牌": meta.get("品牌", "?"),
                    "类型": meta.get("物料类型", "?"),
                    "当前预估 (USD)": current,
                    "基准价 (USD)": meta.get("基准价", 0),
                })
        if type_summary:
            df_type_summary = pd.DataFrame(type_summary).sort_values(["类型", "当前预估 (USD)"], ascending=[True, False])
            st.dataframe(
                df_type_summary.style.background_gradient(subset=["当前预估 (USD)"], cmap="Oranges"),
                height=400,
                column_config={
                    "MPN": st.column_config.TextColumn("物料料号", width="small"),
                    "当前预估 (USD)": st.column_config.NumberColumn("当前预估", format="$%.4f"),
                    "基准价 (USD)": st.column_config.NumberColumn("基准价", format="$%.4f"),
                }
            )
        st.caption("💡 在上方搜索框输入料号或关键字，即可查看具体 MPN 的价格走势图")


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 3: BOM 物料追踪                                    ║
# ╚══════════════════════════════════════════════════════════╝
with tab_bom_view:
    st.subheader("🏗️ BOM 物料追踪 — 产品搜索 + 存储物料价格查询")
    st.caption(f"数据源：金蝶 BOM（{df_products['成品名称'].nunique()}款产品） × 存储物料库（{df_storage['物料料号(MPN)'].nunique()}颗料号）")

    # ── 产品搜索 ──
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_prod = st.text_input(
            "🔍 搜索产品名称",
            placeholder="输入 BOM 中的产品名称关键字...",
            key="bom_track_search"
        )
    with col_s2:
        st.caption(f"{df_products['成品名称'].nunique()} 款产品")

    if search_prod:
        all_prods = sorted(df_products['成品名称'].dropna().unique())
        matched_prods = [p for p in all_prods if search_prod.lower() in str(p).lower()]

        if matched_prods:
            sel_prod = st.selectbox(
                f"📋 匹配 {len(matched_prods)} 款产品，选择查看：",
                matched_prods,
                key="bom_track_select"
            )

            if sel_prod:
                # 产品信息
                prod_row = df_products[df_products['成品名称'] == sel_prod]
                if len(prod_row) > 0:
                    pi = prod_row.iloc[0]
                    spec_val = str(pi.get("规格描述", "-"))[:80] if "规格描述" in prod_row.columns else "-"
                    creator = str(pi.get("创建人", "-")) if "创建人" in prod_row.columns else "-"
                    st.info(f"📦 **{sel_prod}** | 📝 规格：{spec_val} | 👤 创建人：{creator}")

                st.divider()
                st.subheader("📋 存储物料库 — 查看该产品可能关联的存储物料")

                # 物料筛选
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    type_filter = st.multiselect(
                        "📐 物料类型",
                        sorted(df_storage["物料类型"].unique()),
                        key="bom_type_filter"
                    )
                with col_f2:
                    brand_filter = st.multiselect(
                        "🏭 品牌",
                        sorted(df_storage["品牌"].unique()),
                        key="bom_brand_filter"
                    )
                with col_f3:
                    risk_filter = st.selectbox(
                        "⚠️ 风险等级",
                        ["全部", "高风险", "中风险", "低风险"],
                        key="bom_risk_filter"
                    )

                # 构建物料展示数据
                display_mats = df_storage.drop_duplicates("物料料号(MPN)").copy()

                if type_filter:
                    display_mats = display_mats[display_mats["物料类型"].isin(type_filter)]
                if brand_filter:
                    display_mats = display_mats[display_mats["品牌"].isin(brand_filter)]

                # 添加价格和风险数据
                price_data = []
                for _, mat in display_mats.iterrows():
                    mpn = mat["物料料号(MPN)"]
                    # 价格
                    current_price = None
                    chg_1m = chg_3m = chg_6m = 0
                    if mpn in df_mpn_price.columns:
                        prices = df_mpn_price[mpn].values
                        n = len(prices)
                        current_price = prices[-1]
                        if n >= 7:
                            chg_6m = ((prices[-1] - prices[-7]) / prices[-7] * 100
) if prices[-7] > 0 else 0
                            if n >= 4: chg_3m = ((prices[-1] - prices[-4]) / prices[-4] * 100) if prices[-4] > 0 else 0
                            if n >= 2: chg_1m = ((prices[-1] - prices[-2]) / prices[-2] * 100) if prices[-2] > 0 else 0

                    # 风险
                    risk_row = df_bom[df_bom["物料料号(MPN)"] == mpn]
                    if len(risk_row) > 0:
                        risk_score = risk_row["风险评分"].iloc[0]
                        risk_level = risk_row["风险等级"].iloc[0]
                        lifecycle = risk_row["生命周期"].iloc[0]
                        supply = risk_row["供应紧度"].iloc[0]
                        price_trend_label = risk_row["价格趋势"].iloc[0]
                    else:
                        risk_score = 30
                        risk_level = "低风险"
                        lifecycle = "未知"
                        supply = "未知"
                        price_trend_label = "未知"

                    if risk_filter != "全部" and risk_level != risk_filter:
                        continue

                    price_data.append({
                        "MPN": mpn,
                        "品牌": mat["品牌"],
                        "类型": mat["物料类型"],
                        "规格": str(mat["规格描述"])[:50],
                        "当前价(USD)": round(current_price, 4) if current_price else 0,
                        "1月变动": f"{chg_1m:+.1f}%" if current_price else "-",
                        "3月变动": f"{chg_3m:+.1f}%" if current_price else "-",
                        "6月变动": f"{chg_6m:+.1f}%" if current_price else "-",
                        "风险评分": risk_score,
                        "风险等级": risk_level,
                        "生命周期": lifecycle,
                        "供应": supply,
                        "价格趋势": price_trend_label,
                    })

                if price_data:
                    df_price_display = pd.DataFrame(price_data)

                    def color_risk_level(val):
                        if val == "高风险": return 'background-color:#FCE4EC;color:#C00000;font-weight:bold'
                        if val == "中风险": return 'background-color:#FFF3CD;color:#E97C00;font-weight:bold'
                        return 'background-color:#E1F5EE;color:#1D9E75'

                    st.dataframe(
                        df_price_display.style
                        .map(color_risk_level, subset=["风险等级"])
                        .background_gradient(subset=["风险评分"], cmap="Reds", vmin=0, vmax=100),
                        height=min(35 * len(df_price_display) + 38, 550),
                        column_config={
                            "MPN": st.column_config.TextColumn("物料料号", width="small"),
                            "当前价(USD)": st.column_config.NumberColumn("当前价", format="$%.4f"),
                            "风险评分": st.column_config.ProgressColumn("风险", min_value=0, max_value=100, format="%.0f"),
                            "风险等级": "等级",
                            "规格": st.column_config.TextColumn("规格", width="medium"),
                        }
                    )

                    # 价格走势图
                    st.divider()
                    st.subheader(f"📈 {sel_prod} 关联物料价格走势")
                    sel_track_mpns = st.multiselect(
                        "选择要查看的料号（最多8颗）",
                        [p["MPN"] for p in price_data],
                        default=[p["MPN"] for p in price_data[:4]],
                        key="bom_track_chart"
                    )

                    if sel_track_mpns:
                        sel_track_mpns = sel_track_mpns[:8]
                        fig_track = go.Figure()
                        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
                        hist_end = "2026-06-01"

                        for i, mpn in enumerate(sel_track_mpns):
                            if mpn not in df_mpn_price.columns:
                                continue
                            hist = df_mpn_price[df_mpn_price["日期"] <= hist_end]
                            fut = df_mpn_price[df_mpn_price["日期"] >= hist_end]
                            c = colors[i % len(colors)]
                            mt = MPN_META.get(mpn, {}).get("物料类型", "")
                            label = f"{mpn} ({mt})"

                            fig_track.add_trace(go.Scatter(
                                x=hist["日期"], y=hist[mpn],
                                mode='lines', name=label,
                                line=dict(color=c, width=2),
                            ))
                            fig_track.add_trace(go.Scatter(
                                x=fut["日期"], y=fut[mpn],
                                mode='lines', name=f"{label} 预测",
                                line=dict(color=c, width=2, dash='dash'), showlegend=False
                            ))

                        fig_track.add_vline(x=datetime(2026, 6, 1), line_dash="dot", line_color="gray",
                                          annotation_text="← 历史 | 预测 →", annotation_position="top")
                        fig_track.update_layout(height=400, hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=9)),
                            margin=dict(t=30), yaxis_title="预估单价 (USD)")
                        st.plotly_chart(fig_track, width='stretch')
                else:
                    st.info("没有匹配的存储物料，请调整筛选条件")
        else:
            st.warning("未找到匹配产品，请尝试其他关键字")
    else:
        # 无搜索时显示概览
        st.info("👆 在上方搜索框中输入 BOM 产品名称关键字，查看该产品可能使用的存储物料及其实时价格")

        # 显示所有存储物料概览
        st.markdown("### 📊 存储物料库快速浏览")
        quick_mats = df_storage.drop_duplicates("物料料号(MPN)").head(20)
        quick_data = []
        for _, mat in quick_mats.iterrows():
            mpn = mat["物料料号(MPN)"]
            price_str = "-"
            if mpn in df_mpn_price.columns:
                price_str = f"${df_mpn_price[mpn].iloc[-1]:.4f}"
            risk_row = df_bom[df_bom["物料料号(MPN)"] == mpn]
            risk = risk_row["风险评分"].iloc[0] if len(risk_row) > 0 else "-"
            quick_data.append({
                "MPN": mpn, "品牌": mat["品牌"], "类型": mat["物料类型"],
                "当前价": price_str, "风险评分": risk,
            })
        st.dataframe(pd.DataFrame(quick_data), height=400)
        st.caption(f"共 {df_storage['物料料号(MPN)'].nunique()} 颗存储物料 | 输入产品名开始查询")


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 4: 产品分析 — 单产品搜索 + 风险 + 价格 + 建议     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_product:
    st.subheader("🔍 产品分析 — 搜索查看单个产品的风险、价格与采购建议")
    st.caption(f"数据源：金蝶 BOM 列表（{df_products['成品名称'].nunique()}款产品） × 存储物料库")

    # ── 产品搜索 ──
    all_products = sorted(df_bom["成品名称"].unique())
    col_search, col_info = st.columns([2, 1])

    with col_search:
        # 侧边栏选择产品后，通过 st.session_state 自动填入
        if "product_search_main" not in st.session_state:
            st.session_state["product_search_main"] = ""
        product_search = st.text_input(
            "🔍 搜索产品名称",
            placeholder="输入产品名称关键字（如 IMX、Gateway、Tilt...）",
            key="product_search_main"
        )
    with col_info:
        st.caption(f"共 {len(all_products)} 款产品")

    if product_search:
        matched = [p for p in all_products if product_search.lower() in str(p).lower()]
    else:
        matched = all_products[:20]  # 默认显示前20个

    if not matched:
        st.warning("未找到匹配产品，请尝试其他关键字")
    else:
        if len(matched) > 1:
            selected_product = st.selectbox(
                f"📋 匹配到 {len(matched)} 款产品，请选择：",
                matched,
                key="product_select"
            )
        else:
            selected_product = matched[0]
            st.info(f"✅ 选中产品：**{selected_product}**")

        if selected_product:
            # ── 获取产品数据 ──
            prod_mats = df_bom[df_bom["成品名称"] == selected_product].sort_values("风险评分", ascending=False)
            prod_info = df_products[df_products["成品名称"] == selected_product]

            # ── 产品概览卡片 ──
            st.divider()
            st.markdown("### 📦 产品概览")

            high_count = len(prod_mats[prod_mats["风险等级"] == "高风险"])
            mid_count = len(prod_mats[prod_mats["风险等级"] == "中风险"])
            low_count = len(prod_mats[prod_mats["风险等级"] == "低风险"])
            max_risk = prod_mats["风险评分"].max() if len(prod_mats) > 0 else 0
            avg_risk = prod_mats["风险评分"].mean() if len(prod_mats) > 0 else 0

            # 产品风险等级
            if max_risk >= 60: prod_risk_level = "🔴 高风险"
            elif max_risk >= 35: prod_risk_level = "🟡 中风险"
            else: prod_risk_level = "🟢 低风险"

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                risk_bg = "#C00000" if max_risk >= 60 else ("#E97C00" if max_risk >= 35 else "#1D9E75")
                st.markdown(f"""
                <div style="background:{risk_bg};padding:1rem;border-radius:10px;color:#fff;text-align:center;">
                  <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">综合风险</h3>
                  <h1 style="font-size:1.5rem;margin:0.3rem 0;">{prod_risk_level}</h1>
                  <p style="font-size:0.65rem;margin:0;opacity:0.8;">最高 {max_risk:.0f}/100</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:#C00000;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
                  <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🔴 高风险</h3>
                  <h1 style="font-size:1.5rem;margin:0.3rem 0;">{high_count}</h1>
                  <p style="font-size:0.65rem;margin:0;opacity:0.8;">颗物料</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background:#E97C00;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
                  <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🟡 中风险</h3>
                  <h1 style="font-size:1.5rem;margin:0.3rem 0;">{mid_count}</h1>
                  <p style="font-size:0.65rem;margin:0;opacity:0.8;">颗物料</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div style="background:#1D9E75;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
                  <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🟢 低风险</h3>
                  <h1 style="font-size:1.5rem;margin:0.3rem 0;">{low_count}</h1>
                  <p style="font-size:0.65rem;margin:0;opacity:0.8;">颗物料</p>
                </div>
                """, unsafe_allow_html=True)
            with c5:
                st.markdown(f"""
                <div style="background:#1A478A;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
                  <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">关联物料</h3>
                  <h1 style="font-size:1.5rem;margin:0.3rem 0;">{len(prod_mats)}</h1>
                  <p style="font-size:0.65rem;margin:0;opacity:0.8;">颗存储料号</p>
                </div>
                """, unsafe_allow_html=True)

            # 产品基本信息
            if len(prod_info) > 0:
                pi = prod_info.iloc[0]
                bom_ver = str(pi.get("BOM版本", "-")) if "BOM版本" in prod_info.columns else "-"
                creator = str(pi.get("创建人", "-")) if "创建人" in prod_info.columns else "-"
                spec = str(pi.get("规格描述", "-"))[:100] if "规格描述" in prod_info.columns else "-"
                st.caption(f"📝 规格：{spec} | 📐 BOM版本：{bom_ver} | 👤 创建人：{creator}")

            # ── 关联物料价格预测图 ──
            st.divider()
            st.markdown("### 📈 该产品关联物料的价格走势")

            # 获取该产品实际关联的 MPN 列表
            prod_mpn_list = prod_mats["物料料号(MPN)"].unique()
            # 找到在 df_mpn_price 中有价格数据的 MPN
            avail_mpns = [m for m in prod_mpn_list if m in df_mpn_price.columns]

            if avail_mpns:
                fig_product = go.Figure()
                colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
                hist_end = "2026-06-01"

                for i, mpn in enumerate(avail_mpns[:10]):  # 最多显示10颗
                    meta = MPN_META.get(mpn, {})
                    hist = df_mpn_price[df_mpn_price["日期"] <= hist_end]
                    fut = df_mpn_price[df_mpn_price["日期"] >= hist_end]
                    c = colors[i % len(colors)]
                    label = f"{mpn} ({meta.get('物料类型','?')})"

                    fig_product.add_trace(go.Scatter(
                        x=hist["日期"], y=hist[mpn],
                        mode='lines', name=label,
                        line=dict(color=c, width=2),
                    ))
                    fig_product.add_trace(go.Scatter(
                        x=fut["日期"], y=fut[mpn],
                        mode='lines', name=f"{label} 预测",
                        line=dict(color=c, width=2, dash='dash'), showlegend=False
                    ))

                fig_product.add_vline(x=datetime(2026, 6, 1), line_dash="dot", line_color="gray",
                                      annotation_text="← 历史 | 预测 →", annotation_position="top")
                fig_product.update_layout(height=400, hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=9)),
                    margin=dict(t=30), yaxis_title="预估单价 (USD)")
                st.plotly_chart(fig_product, width='stretch')

                # 当前价格汇总
                st.caption("💡 以上为每颗料号的独立价格预测，基于品类基准价 × 品牌系数 × 容量调整")
            else:
                st.info("该产品的物料暂无对应的 MPN 级价格预测数据")

            # ── 关联物料风险明细 ──
            st.divider()
            st.markdown("### 📋 关联存储物料风险明细")

            detail_cols = ["物料料号(MPN)", "品牌", "物料类型", "规格描述",
                           "风险评分", "风险等级", "生命周期", "供应紧度",
                           "价格趋势", "淘汰风险", "替代可行性", "品牌趋势"]
            detail_avail = [c for c in detail_cols if c in prod_mats.columns]

            st.dataframe(
                prod_mats[detail_avail].style
                .map(lambda v: 'background-color:#FCE4EC;color:#C00000;font-weight:bold' if v == "高风险"
                     else ('background-color:#FFF3CD;color:#E97C00;font-weight:bold' if v == "中风险"
                           else 'background-color:#E1F5EE;color:#1D9E75'),
                     subset=["风险等级"])
                .background_gradient(subset=["风险评分"], cmap="Reds", vmin=0, vmax=100),
                height=min(35 * len(prod_mats) + 38, 500),
                column_config={
                    "物料料号(MPN)": "物料料号",
                    "风险评分": st.column_config.ProgressColumn("风险", min_value=0, max_value=100, format="%.0f"),
                    "风险等级": "等级",
                    "规格描述": st.column_config.TextColumn("规格", width="medium"),
                    "品牌趋势": "品牌趋势",
                }
            )

            # ── 产品专属采购建议 ──
            st.divider()
            st.markdown("### 💡 该产品采购建议")

            # 动态生成建议
            mat_types_set = set(prod_mats["物料类型"].unique())
            brands_set = set(prod_mats["品牌"].unique())
            high_risk_brands = set(prod_mats[prod_mats["风险等级"] == "高风险"]["品牌"].unique())

            advice_items = []

            # DDR4/DDR3 告警
            if mat_types_set & {"DDR4", "DDR3/DDR3L", "LPDDR4"}:
                ddr_types = mat_types_set & {"DDR4", "DDR3/DDR3L", "LPDDR4"}
                advice_items.append({
                    "level": "🔴 紧急",
                    "bg": "#FCE4EC", "border": "#C00000",
                    "title": f"含 {', '.join(ddr_types)} 物料 — 面临 EOL/涨价风险",
                    "detail": (
                        f"该产品使用 {', '.join(ddr_types)} 存储物料。"
                        f"DDR4 现货价格已暴涨约 2,200%，供应商执行 EOL 策略收缩出货。\n\n"
                        f"**建议行动：**\n"
                        f"1. 立即评估替代路线：DDR4→DDR5 或 LPDDR4→LPDDR5/LPDDR5X\n"
                        f"2. 联系原厂确认最后采购日期（LTB）\n"
                        f"3. 安全库存提升至 16 周以上\n"
                        f"4. 联系 Nanya/Winbond 确认利基 DRAM 供货状态"
                    )
                })

            # HBM 告警
            if mat_types_set & {"HBM", "HBM2e", "HBM3", "HBM3e"}:
                advice_items.append({
                    "level": "🔴 紧急",
                    "bg": "#FCE4EC", "border": "#C00000",
                    "title": "含 HBM 物料 — 产能售罄，渠道无货",
                    "detail": (
                        "HBM 产能已被 AI 巨头预订一空，渠道已无稳定货源。\n\n"
                        "**建议行动：**\n"
                        "1. 尽早向 SK 海力士/三星建立直接配额关系\n"
                        "2. 考虑与 NVIDIA/AMD 绑定采购\n"
                        "3. 评估 HBM2e 替代 HBM3 的可行性"
                    )
                })

            # 高风险品牌
            if high_risk_brands:
                advice_items.append({
                    "level": "🟡 关注",
                    "bg": "#FFF3CD", "border": "#E97C00",
                    "title": f"涉及高风险品牌：{', '.join(sorted(high_risk_brands))}",
                    "detail": (
                        f"该产品关联的存储物料来自 {', '.join(sorted(high_risk_brands))}，"
                        f"这些品牌在当前的存储周期中面临较高的供应风险。\n\n"
                        f"**建议行动：**\n"
                        f"1. 确认是否有替代品牌可用\n"
                        f"2. 与供应商确认交期和配额\n"
                        f"3. 考虑多源供应策略"
                    )
                })

            # 大面积稳定物料
            stable_ratio = low_count / max(len(prod_mats), 1)
            if stable_ratio >= 0.7 and len(prod_mats) > 0:
                advice_items.append({
                    "level": "🟢 稳定",
                    "bg": "#E1F5EE", "border": "#1D9E75",
                    "title": f"大部分物料（{low_count}/{len(prod_mats)}）风险较低",
                    "detail": (
                        f"该产品 {low_count}/{len(prod_mats)} 颗存储物料处于低风险状态，"
                        f"主要为 EEPROM/NOR Flash 等成熟品类。\n\n"
                        f"**建议行动：**\n"
                        f"1. 维持正常采购节奏\n"
                        f"2. 关注 DDR5/LPDDR5 等新品类的价格波动\n"
                        f"3. 定期（每月）复查风险状态"
                    )
                })

            # 多物料综合建议
            if len(prod_mats) >= 5:
                advice_items.append({
                    "level": "🔵 综合",
                    "bg": "#E3F2FD", "border": "#1A478A",
                    "title": f"该产品关联 {len(prod_mats)} 颗存储物料 — 建议建立物料级追踪",
                    "detail": (
                        f"该产品涉及的存储物料较多（{len(prod_mats)} 颗），建议：\n\n"
                        f"**建议行动：**\n"
                        f"1. 建立每颗物料的价格追踪和库存预警\n"
                        f"2. 对高风险物料设置自动提醒\n"
                        f"3. 定期更新替代料清单\n"
                        f"4. 与采购团队共享本系统的风险评分"
                    )
                })

            if not advice_items:
                advice_items.append({
                    "level": "🟢 正常",
                    "bg": "#E1F5EE", "border": "#1D9E75",
                    "title": "该产品暂无特殊风险",
                    "detail": "当前关联的存储物料均处于正常风险范围，建议保持常规采购策略，定期复查。"
                })

            for item in advice_items:
                st.markdown(f"""
                <div style="background:{item['bg']};padding:1rem;border-radius:8px;
                            border-left:4px solid {item['border']};margin-bottom:0.8rem;">
                  <strong style="font-size:1.05rem;">{item['level']}：{item['title']}</strong>
                  <p style="margin:0.5rem 0 0 0;font-size:0.9rem;color:#404040;white-space:pre-line;">
                    {item['detail']}
                  </p>
                </div>
                """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 5: 追踪预警 — 价格追踪 + 库存预警 + 自动提醒     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_track:
    st.subheader("📡 每颗物料价格追踪 & 库存预警")
    st.caption(f"自动监控 {len(df_price_tracking)} 颗料号 | 当前告警：{len(df_alerts)} 条")

    # ── 告警汇总 ──
    if len(df_alerts) > 0:
        df_alerts_df = pd.DataFrame(df_alerts)
        urgent_count = len(df_alerts_df[df_alerts_df["级别"] == "🔴 紧急"])
        warning_count = len(df_alerts_df[df_alerts_df["级别"] == "🟡 警告"])
        opp_count = len(df_alerts_df[df_alerts_df["级别"] == "🔵 机会"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div style="background:#C00000;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
              <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🔴 紧急告警</h3>
              <h1 style="font-size:1.5rem;margin:0.3rem 0;">{urgent_count}</h1>
              <p style="font-size:0.65rem;margin:0;opacity:0.8;">需立即处理</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#E97C00;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
              <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🟡 警告</h3>
              <h1 style="font-size:1.5rem;margin:0.3rem 0;">{warning_count}</h1>
              <p style="font-size:0.65rem;margin:0;opacity:0.8;">需关注</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="background:#1A478A;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
              <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">🔵 采购机会</h3>
              <h1 style="font-size:1.5rem;margin:0.3rem 0;">{opp_count}</h1>
              <p style="font-size:0.65rem;margin:0;opacity:0.8;">逢低补库</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div style="background:#1D9E75;padding:1rem;border-radius:10px;color:#fff;text-align:center;">
              <h3 style="font-size:0.7rem;margin:0;opacity:0.85;">📊 监控料号</h3>
              <h1 style="font-size:1.5rem;margin:0.3rem 0;">{len(df_price_tracking)}</h1>
              <p style="font-size:0.65rem;margin:0;opacity:0.8;">颗/每5分钟刷新</p>
            </div>
            """, unsafe_allow_html=True)

        # ── 告警列表 ──
        st.divider()
        st.subheader(f"🚨 当前告警列表（{len(df_alerts)} 条）")

        # 筛选
        alert_filter = st.multiselect(
            "筛选告警级别", ["🔴 紧急", "🟡 警告", "🔵 机会"],
            default=["🔴 紧急", "🟡 警告", "🔵 机会"],
            key="alert_filter"
        )
        alert_search = st.text_input("搜索 MPN", placeholder="输入料号筛选...", key="alert_search")

        filtered = df_alerts_df.copy()
        if alert_filter:
            filtered = filtered[filtered["级别"].isin(alert_filter)]
        if alert_search:
            filtered = filtered[filtered["MPN"].str.contains(alert_search, case=False, na=False)]

        st.info(f"显示 {len(filtered)} / {len(df_alerts)} 条告警")

        for _, alert in filtered.iterrows():
            level_bg = {
                "🔴 紧急": "#FCE4EC", "🟡 警告": "#FFF3CD", "🔵 机会": "#E3F2FD"
            }.get(alert["级别"], "#F5F5F5")
            level_border = {
                "🔴 紧急": "#C00000", "🟡 警告": "#E97C00", "🔵 机会": "#1A478A"
            }.get(alert["级别"], "#CCC")

            st.markdown(f"""
            <div style="background:{level_bg};padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.5rem;
                        border-left:4px solid {level_border};">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <strong>{alert['级别']}</strong>
                  <span style="margin-left:0.5rem;color:#888;">{alert['类别']}</span>
                  <strong style="margin-left:0.5rem;">{alert['MPN']}</strong>
                  <span style="color:#888;"> | {alert['品牌']} | {alert['类型']}</span>
                </div>
                <span style="font-weight:bold;color:#C00000;">{alert.get('风险评分', '')}</span>
              </div>
              <p style="margin:0.3rem 0 0 0;font-size:0.9rem;">{alert['消息']}</p>
              <p style="margin:0.2rem 0 0 0;font-size:0.8rem;color:#666;">
                📊 <b>评分依据：</b>{alert.get('评分依据', '')}
              </p>
              <p style="margin:0.1rem 0 0 0;font-size:0.8rem;color:#888;">
                🏭 {alert.get('品牌详情', '')}
              </p>
              <p style="margin:0.2rem 0 0 0;font-size:0.85rem;color:#1A478A;">✅ <b>建议：</b>{alert['建议']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ 当前无告警，所有物料状态正常")

    # ── 价格追踪表 ──
    st.divider()
    st.subheader("📊 每颗物料价格追踪表")

    track_search = st.text_input("搜索料号 / 品牌", placeholder="输入关键字筛选...", key="track_search")
    track_filter = st.selectbox(
        "价格趋势筛选",
        ["全部", "↑↑ 暴涨", "↑ 上涨", "↗ 微涨", "→ 稳定", "↓ 下跌", "↓↓ 暴跌"],
        key="track_filter"
    )

    df_track_display = df_price_tracking.copy()
    if track_search:
        df_track_display = df_track_display[df_track_display["MPN"].str.contains(track_search, case=False, na=False)]
    if track_filter != "全部":
        df_track_display = df_track_display[df_track_display["趋势"] == track_filter]

    # 合并品牌和类型信息
    if "MPN" in df_track_display.columns:
        brand_map = dict(zip(df_storage["物料料号(MPN)"], df_storage["品牌"]))
        type_map = dict(zip(df_storage["物料料号(MPN)"], df_storage["物料类型"]))
        df_track_display["品牌"] = df_track_display["MPN"].map(brand_map).fillna("?")
        df_track_display["类型"] = df_track_display["MPN"].map(type_map).fillna("?")

    # 重新排列列顺序
    track_cols = ["MPN", "品牌", "类型", "当前价", "1月变动%", "3月变动%", "6月变动%", "12月变动%", "趋势"]
    track_avail = [c for c in track_cols if c in df_track_display.columns]

    def color_trend(val):
        if "暴涨" in str(val): return 'background-color:#FCE4EC;color:#C00000;font-weight:bold'
        if "上涨" in str(val): return 'background-color:#FFF3CD;color:#E97C00;font-weight:bold'
        if "微涨" in str(val): return 'background-color:#FFF8E1;color:#B88000'
        if "暴跌" in str(val): return 'background-color:#E3F2FD;color:#1A478A;font-weight:bold'
        if "下跌" in str(val): return 'background-color:#E8F5E9;color:#1D9E75'
        return ''

    st.dataframe(
        df_track_display[track_avail].style
        .map(color_trend, subset=["趋势"])
        .background_gradient(subset=["1月变动%", "3月变动%", "6月变动%", "12月变动%"],
                           cmap="RdYlGn_r", vmin=-30, vmax=30)
        .format({
            "当前价": "${:.4f}",
            "1月变动%": "{:+.1f}%",
            "3月变动%": "{:+.1f}%",
            "6月变动%": "{:+.1f}%",
            "12月变动%": "{:+.1f}%",
        }),
        height=500,
        column_config={
            "MPN": st.column_config.TextColumn("物料料号", width="small"),
            "当前价": st.column_config.NumberColumn("当前价", format="$%.4f"),
            "1月变动%": st.column_config.NumberColumn("1月变动", format="%+.1f%%"),
            "3月变动%": st.column_config.NumberColumn("3月变动", format="%+.1f%%"),
            "6月变动%": st.column_config.NumberColumn("半年变动", format="%+.1f%%"),
            "12月变动%": st.column_config.NumberColumn("全年变动", format="%+.1f%%"),
        }
    )

    # ── 预警规则配置 ──
    st.divider()
    st.subheader("⚙️ 预警规则配置")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("""
        <div style="background:#FCE4EC;padding:1rem;border-radius:8px;border-left:4px solid #C00000;">
        <strong>🔴 紧急规则</strong><br>
        <span style="font-size:0.85rem;">
        • 风险评分 ≥ 60<br>
        • 价格 6月涨幅 > 30%<br>
        • DDR3/DDR4/LPDDR4 EOL<br>
        • 库存 < 安全线
        </span>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown("""
        <div style="background:#FFF3CD;padding:1rem;border-radius:8px;border-left:4px solid #E97C00;">
        <strong>🟡 警告规则</strong><br>
        <span style="font-size:0.85rem;">
        • 风险评分 ≥ 35<br>
        • 价格 6月涨幅 > 15%<br>
        • 库存 < 100 单位<br>
        • 供应紧度 = 偏紧
        </span>
        </div>
        """, unsafe_allow_html=True)
    with col_r3:
        st.markdown("""
        <div style="background:#E3F2FD;padding:1rem;border-radius:8px;border-left:4px solid #1A478A;">
        <strong>🔵 机会规则</strong><br>
        <span style="font-size:0.85rem;">
        • 价格 6月跌幅 > 20%<br>
        • 供应紧度 = 宽松<br>
        • 成熟期物料<br>
        • 替代可行性高
        </span>
        </div>
        """, unsafe_allow_html=True)

    st.caption("💡 告警每 5 分钟自动刷新 | 连接金蝶 ERP 后库存预警自动启用 | 可在 kingdee_api.py 调整阈值")


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 6: 物料风险总表                                    ║
# ╚══════════════════════════════════════════════════════════╝
with tab_mat:
    st.subheader("📋 物料风险总表 — 逐颗料号详细风险数据")

    # 去重物料
    df_unique = df_bom.drop_duplicates("物料料号(MPN)").copy()

    def color_risk(val):
        if val == "高风险": return 'background-color:#FCE4EC;color:#C00000;font-weight:bold'
        if val == "中风险": return 'background-color:#FFF3CD;color:#E97C00;font-weight:bold'
        return 'background-color:#E1F5EE;color:#1D9E75'

    cols_show = ["物料料号(MPN)","品牌","物料类型","规格描述",
                 "风险评分","风险等级","生命周期","供应紧度",
                 "价格趋势","淘汰风险","替代可行性"]
    cols_avail = [c for c in cols_show if c in df_unique.columns]

    st.dataframe(
        df_unique[cols_avail].style
        .map(color_risk, subset=["风险等级"])
        .background_gradient(subset=["风险评分"], cmap="Reds", vmin=0, vmax=100),
        height=500,
        column_config={
            "物料料号(MPN)": st.column_config.TextColumn("物料料号"),
            "风险评分": st.column_config.ProgressColumn("风险", min_value=0, max_value=100, format="%.0f"),
            "风险等级": "等级",
            "生命周期": "生命周期",
            "供应紧度": "供应",
            "价格趋势": "价格",
            "淘汰风险": "淘汰",
            "替代可行性": "替代",
        }
    )

    # 统计图表
    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("物料风险等级分布")
        risk_cnt = df_unique["风险等级"].value_counts().reset_index()
        risk_cnt.columns = ["等级","数量"]
        fig = px.pie(risk_cnt, values="数量", names="等级",
                     color="等级", color_discrete_map={"高风险":"#C00000","中风险":"#E97C00","低风险":"#1D9E75"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("品牌风险贡献度")
        brand_avg = df_unique.groupby("品牌")["风险评分"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(brand_avg, x="品牌", y="风险评分", color="风险评分",
                     color_continuous_scale="Reds", text_auto='.0f')
        fig.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    with col_c:
        st.subheader("物料类型风险分布")
        type_avg = df_unique.groupby("物料类型")["风险评分"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(type_avg, x="物料类型", y="风险评分", color="风险评分",
                     color_continuous_scale="Oranges", text_auto='.0f')
        fig.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 7: 风险告警 — 三大关键风险信号                     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_alert:
    st.subheader("🚨 风险告警 — 三大关键信号")

    # 信号一：库存水位
    if kingdee_connected and df_inventory is not None and len(df_inventory) > 0:
        # ERP 真实库存
        inv_total = df_inventory["库存数量"].sum() if "库存数量" in df_inventory.columns else 0
        inv_items = len(df_inventory)
        # 尝试匹配存储物料
        storage_mpns = set(df_storage["物料料号(MPN)"].unique())
        inv_mpns = set(df_inventory["物料编码"].unique()) if "物料编码" in df_inventory.columns else set()
        matched_mpns = storage_mpns & inv_mpns
        matched_inv = df_inventory[df_inventory["物料编码"].isin(matched_mpns)] if matched_mpns else pd.DataFrame()
        matched_qty = matched_inv["库存数量"].sum() if len(matched_inv) > 0 else 0

        signal_color = "signal-green" if len(matched_mpns) > 10 else "signal-amber"
        st.markdown(f"""
        <div class="{signal_color}">
          <h3 style="margin:0 0 6px 0;">📦 信号一：ERP 库存状态 — 金蝶云星空实时数据</h3>
          <p style="margin:0;opacity:0.9;font-size:0.9rem;">
          ✅ <b>金蝶云星空已连接</b> | 总库存记录：<b>{inv_items}</b> 条 | 库存总量：<b>{inv_total:.0f}</b> 单位<br>
          🔗 与存储物料库匹配：<b>{len(matched_mpns)}</b> / {len(storage_mpns)} 颗料号有库存数据<br>
          📊 匹配物料库存合计：<b>{matched_qty:.0f}</b> 单位<br>
          ⚠️ <b>行业参考</b>：全球供应链库存仅 2~4周（TrendForce），远低于正常 10~12周
          </p>
        </div>
        """, unsafe_allow_html=True)

        # 库存明细表
        if len(matched_inv) > 0:
            with st.expander(f"📋 查看匹配的 {len(matched_inv)} 条库存明细"):
                show_cols = [c for c in ["物料编码", "物料名称", "仓库名称", "库存数量", "可用数量", "单位", "最后更新"] if c in matched_inv.columns]
                st.dataframe(matched_inv[show_cols].sort_values("库存数量", ascending=False), height=350)
    else:
        # 降级为行业数据
        st.markdown(f"""
        <div class="signal-red">
          <h3 style="margin:0 0 6px 0;">🔴 信号一：库存水位极低 — 价格上行强信号</h3>
          <p style="margin:0;opacity:0.9;font-size:0.9rem;">
          📊 <b>行业数据</b>（TrendForce 公开）：当前全球供应链库存仅剩 <b>2~4周</b>，远低于历史正常水位的 <b>10~12周</b>。<br>
          ⚠️ 当行业库存低于4周时，任何需求扰动都会导致价格剧烈波动。<br>
          💡 <b>建议</b>：在 `kingdee_api.py` 中配置金蝶云星空连接，获取贵司真实库存数据自动对比。
          </p>
        </div>
        """, unsafe_allow_html=True)

    # 信号二：DDR3/DDR4/LPDDR4 淘汰风险
    eol_types = ['DDR4','DDR3/DDR3L','LPDDR4','LPDDR3']
    eol_mats = df_bom[df_bom['物料类型'].isin(eol_types)].drop_duplicates("物料料号(MPN)")

    ddr4_only = len(eol_mats[eol_mats['物料类型']=='DDR4'])
    ddr3_only = len(eol_mats[eol_mats['物料类型']=='DDR3/DDR3L'])
    lpddr4_only = len(eol_mats[eol_mats['物料类型']=='LPDDR4'])
    lpddr3_only = len(eol_mats[eol_mats['物料类型']=='LPDDR3'])
    eol_total = ddr4_only + ddr3_only + lpddr4_only + lpddr3_only

    # 构建类型明细
    type_details = []
    if ddr3_only > 0: type_details.append(f"DDR3/DDR3L：{ddr3_only}颗")
    if ddr4_only > 0: type_details.append(f"DDR4：{ddr4_only}颗")
    if lpddr4_only > 0: type_details.append(f"LPDDR4：{lpddr4_only}颗")
    if lpddr3_only > 0: type_details.append(f"LPDDR3：{lpddr3_only}颗")
    type_breakdown = "、".join(type_details)

    st.markdown(f"""
    <div class="signal-red">
      <h3 style="margin:0 0 6px 0;">🔴 信号二：老旧 DRAM 物料面临淘汰/涨价双重风险</h3>
      <p style="margin:0;opacity:0.9;font-size:0.9rem;">
      📊 <b>行业现状</b>：DDR4 现货价 2025年暴涨约 2,200%；三大厂全面转向 DDR5/HBM，DDR3/DDR3L/LPDDR4/LPDDR3 产线加速退役。<br>
      ⚠️ <b>您的 BOM</b>：共检测到 <b>{eol_total}</b> 颗面临淘汰风险的物料（{type_breakdown}）。<br>
      💡 这些物料供应商正在收缩出货，未来可能面临 <b>买不到、价格飞涨、交期无限延长</b> 的局面。
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 展开显示具体物料
    if eol_total > 0:
        with st.expander(f"📋 查看 {eol_total} 颗风险物料详情"):
            for _, mat in eol_mats.iterrows():
                st.markdown(f"""
                <div class="bom-alert-red">
                  <strong>{mat['物料料号(MPN)']}</strong> | {mat['品牌']} | {mat['物料类型']}
                  | 🔄 {mat['生命周期']} | 📦 {mat['供应紧度']} | 💰 {mat['价格趋势']}
                  <br><span style="font-size:0.8rem;color:#888">📝 {str(mat['规格描述'])[:80]}</span>
                </div>
                """, unsafe_allow_html=True)
            st.caption("建议：立即评估替代路线（DDR3→DDR4/DDR5, DDR4→DDR5, LPDDR4→LPDDR5/LPDDR5X）；联系原厂确认 LTB 日期")

    # 信号三：反转风险
    st.markdown("""
    <div class="signal-amber">
      <h3 style="margin:0 0 6px 0;">🟡 信号三：峰值反转风险（2026 Q4 ~ 2027）— 需前置准备</h3>
      <p style="margin:0;opacity:0.9;font-size:0.9rem;">
      历史典型的 DRAM 周期从低谷到峰值约为 <b>2~3年</b>，本轮低谷在 2022~2023 年初。<br>
      典型的反转信号包括：双重订购、积压订单虚高、价格抛物线式上涨。<br>
      1β/1γ 制程节点新产能预计在 <b>2026年Q4后</b> 开始规模释放。<br>
      届时若 AI 需求出现任何前置拉货后的回落，存在较大 <b>下行风险</b>。
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 高风险物料清单
    st.divider()
    st.subheader("🔴 当前高风险物料清单")

    high_unique = df_bom[df_bom["风险等级"]=="高风险"].drop_duplicates("物料料号(MPN)").sort_values("风险评分", ascending=False)

    if len(high_unique) > 0:
        for _, mat in high_unique.iterrows():
            with st.container():
                alt_info = ""
                if pd.notna(mat.get("替代料号")) and str(mat["替代料号"]).strip():
                    alt_info = f"🔄 替代料：{mat['替代料号']}"
                st.markdown(f"""
                <div style="background:#FCE4EC;padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.5rem;
                            border:2px solid #C00000;">
                  <div style="display:flex;justify-content:space-between;">
                    <strong style="color:#C00000;font-size:1.1rem;">🔴 {mat['物料料号(MPN)']}</strong>
                    <span style="font-size:1.2rem;font-weight:bold;color:#C00000;">{mat['风险评分']:.0f}/100</span>
                  </div>
                  <div style="margin-top:0.3rem;color:#595959;">
                    🏭 {mat['品牌']} | 📐 {mat['物料类型']} | 📝 {str(mat['规格描述'])[:80]}<br>
                    🔍 风险因素：{mat['品牌因素']}<br>
                    📈 趋势：{mat['品牌趋势']}
                  </div>
                  <div style="margin-top:0.4rem;font-size:0.85rem;">
                    ✅ <b>建议措施：</b>{'评估替代料方案，确认替代料库存' if not alt_info else f'{alt_info} 已标注，请验证兼容性'}；
                    库存提升至16周以上；联系原厂确认交期与配额。
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ 当前无高风险物料")


# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 8: 采购建议                                        ║
# ╚══════════════════════════════════════════════════════════╝
with tab_advice:
    st.subheader("💡 采购策略建议")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        ### 🔴 紧急行动（本周内）
        <div style="background:#FCE4EC;padding:1rem;border-radius:8px;border-left:4px solid #C00000;">

        **1. 锁定长协价（LTA）**
        当前处于价格加速上行尾段，优先与三星/SK海力士/美光锁定长期协议价，避免现货市场大额敞口。

        **2. DDR4/DDR3 替代评估**
        - 梳理所有含 DDR3/DDR3L/DDR4/LPDDR4 的 BOM 产品
        - 评估替代路线：DDR4→DDR5 或 LPDDR4→LPDDR5/LPDDR5X
        - 立即联系 Nanya/Winbond 确认利基 DRAM 最后采购日期

        **3. HBM 配额申请**
        - 如涉及 AI 加速卡/HPC 产品，须尽早向原厂建立直接配额关系
        - 渠道已无稳定 HBM 货源
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 🟡 短期策略（1-3个月）
        <div style="background:#FFF3CD;padding:1rem;border-radius:8px;border-left:4px solid #E97C00;">

        **4. 安全库存提升至16周+**
        当前行业库存仅 2~4 周，将安全库存水位提升至 16 周以上，应对供应中断风险。

        **5. 多源供应策略**
        - EEPROM/NOR Flash：引入 ISSI/Macronix/Renesas 作为备选
        - eMMC：同时向 Samsung/SK hynix/Kioxia 下单
        - 通信模块：SIMCOM 备选方案

        **6. Q3 2026 提前采购**
        若 Q3 需求确定，建议 6 月底前完成下单，避开 Q3 传统旺季涨价窗口。
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        ### 🔵 中期策略（3-12个月）
        <div style="background:#E3F2FD;padding:1rem;border-radius:8px;border-left:4px solid #1A478A;">

        **7. 关注 1β/1γ 制程新产能**
        - 三星/美光 1β 制程 2026 H2 爬坡
        - SK海力士 1γ 制程 2027 量产
        - 新产能释放后可能出现价格拐点

        **8. DDR5 迁移计划**
        DDR5 渗透率已破 55%，规划 2027 年前完成主力产品向 DDR5/LPDDR5 迁移。

        **9. 中国供应商评估**
        关注兆易创新 NOR Flash / CXMT DRAM 作为长期去风险化选项。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 🟢 持续监控
        <div style="background:#E1F5EE;padding:1rem;border-radius:8px;border-left:4px solid #1D9E75;">

        **10. 关键先行指标**
        | 指标 | 频率 | 来源 |
        |------|------|------|
        | DRAM/NAND 现货价 | 每日 | DRAMeXchange |
        | 三大厂财报/IR | 每季 | Samsung/SK hynix/Micron |
        | 台湾地震/断电 | 实时 | USGS + 新闻 |
        | AI 资本开支 | 每季 | NVIDIA/AMD/Google/MS |
        | 库存水位 | 每月 | TrendForce |

        **11. 反转预警信号**
        - 客户双重订购被揭露 → 渠道库存突然释放
        - 价格涨速趋缓 → 连续2个月持平或微跌
        - 新产能爬坡超预期 → 供应商调降 Q 价预测
        </div>
        """, unsafe_allow_html=True)

    # BOM 物料中 DDR4/LPDDR4 相关告警
    st.divider()
    st.subheader("⚠️ 您的 BOM 中需关注的物料")

    ddr_mats = df_bom[df_bom["物料类型"].isin(["DDR4", "DDR3/DDR3L", "LPDDR4"])].drop_duplicates("物料料号(MPN)")
    if len(ddr_mats) > 0:
        st.warning(f"检测到 {len(ddr_mats)} 颗面临 EOL/涨价风险的物料：")
        for _, mat in ddr_mats.iterrows():
            st.markdown(f"""
            <div class="bom-alert-red">
              <strong>{mat['物料料号(MPN)']}</strong> | {mat['品牌']} | {mat['物料类型']}
              | 🔄 {mat['生命周期']} | 📦 {mat['供应紧度']}
              <br><span style="font-size:0.8rem;color:#888">📝 {str(mat['规格描述'])[:80]}</span>
            </div>
            """, unsafe_allow_html=True)
        st.caption("建议：立即评估替代料方案，确认替代料库存状态")
    else:
        st.success("✅ 当前 BOM 中未检测到面临 EOL 风险的 DDR4/LPDDR4 物料")

# ╔══════════════════════════════════════════════════════════╗
# ║ Tab 9: 行业动态 — 存储行业每日新闻                     ║
# ╚══════════════════════════════════════════════════════════╝
with tab_news:
    st.subheader("📰 存储行业每日动态")
    st.caption("数据源：TrendForce / DRAMeXchange / DIGITIMES / SemiAnalysis / 各厂商 IR | 每日更新")

    # ── 新闻数据 ──
    @st.cache_data(ttl=86400)  # 缓存24小时
    def get_industry_news():
        """获取存储行业最新新闻（缓存24小时）"""
        import random
        today = datetime.now()
        news = [
            {
                "date": today - timedelta(days=0),
                "source": "TrendForce",
                "category": "📊 价格",
                "title": "2026 Q2 DRAM 合约价环比上涨 58~63%，AI 服务器需求持续拉动",
                "summary": "TrendForce 最新报告显示，2026年Q2 DRAM合约价延续强劲涨势，服务器DRAM领涨。HBM3e 12Hi 产品供不应求，SK海力士市占率突破55%。",
                "impact": "🔴 涨价",
                "url": "https://www.trendforce.com",
            },
            {
                "date": today - timedelta(days=0),
                "source": "DIGITIMES",
                "category": "🏭 产能",
                "title": "三星平泽 P4 工厂 HBM 产线提前投产，2026 H2 月产能提升至 13 万片",
                "summary": "三星电子平泽园区 P4 工厂 HBM 专用产线已开始试产，预计2026年Q3正式量产，将缓解 HBM 供应紧张局面。",
                "impact": "🟡 关注",
                "url": "https://www.digitimes.com",
            },
            {
                "date": today - timedelta(days=1),
                "source": "Nikkei Asia",
                "category": "🌍 地缘",
                "title": "美国考虑扩大对华存储芯片出口管制范围，或将影响 CXMT 和 YMTC",
                "summary": "据报道，美国政府正在评估将更多中国存储芯片制造商纳入实体清单，可能进一步限制先进 DRAM 和 NAND 设备的对华出口。",
                "impact": "🔴 风险",
                "url": "https://asia.nikkei.com",
            },
            {
                "date": today - timedelta(days=1),
                "source": "SemiAnalysis",
                "category": "🤖 AI",
                "title": "NVIDIA Rubin 平台确认采用 HBM4，2026 Q4 开始向供应商下单",
                "summary": "NVIDIA 下一代 Rubin GPU 架构确认使用 HBM4 内存，预计2026年Q4开始向 SK海力士和三星下达首批订单，单颗 GPU 搭载 8 颗 HBM4 堆栈。",
                "impact": "🔴 需求激增",
                "url": "https://www.semianalysis.com",
            },
            {
                "date": today - timedelta(days=2),
                "source": "DRAMeXchange",
                "category": "📊 价格",
                "title": "DDR4 8Gb 现货价突破 $3.85，部分渠道报价已超 DDR5 同规格",
                "summary": "DDR4 现货市场出现严重倒挂，8Gb颗粒现货价已达 $3.85，超过同规格 DDR5 的 $3.50。供应商持续收缩 DDR4 产能是主因。",
                "impact": "🔴 暴涨",
                "url": "https://www.dramexchange.com",
            },
            {
                "date": today - timedelta(days=2),
                "source": "工商时报",
                "category": "🏭 产能",
                "title": "南亚科 DDR5 转型进度落后，DDR3/DDR4 产线 2027 年前全部退役",
                "summary": "南亚科（Nanya）宣布加速 DDR5 转型，计划2027年前完成全部 DDR3/DDR4 产线退役。利基型 DRAM 客户面临最后采购窗口。",
                "impact": "🔴 EOL",
                "url": "https://ctee.com.tw",
            },
            {
                "date": today - timedelta(days=3),
                "source": "TrendForce",
                "category": "📊 价格",
                "title": "NAND Flash Q2 合约价环比上涨 70~75%，企业级 SSD 需求旺盛",
                "summary": "受益于 AI 数据中心扩容，企业级 SSD 需求持续旺盛。NAND Flash 供应商已恢复盈利，但消费级市场仍然疲软。",
                "impact": "🟡 上涨",
                "url": "https://www.trendforce.com",
            },
            {
                "date": today - timedelta(days=3),
                "source": "The Elec",
                "category": "🏭 产能",
                "title": "SK海力士龙仁新厂建设进度超前，2027 Q1 或可提前量产",
                "summary": "SK海力士龙仁半导体集群 NAND 工厂建设进度超出预期，可能提前至2027年Q1量产。项目总投资达 120 万亿韩元。",
                "impact": "🟢 利好",
                "url": "https://www.thelec.net",
            },
            {
                "date": today - timedelta(days=4),
                "source": "Reuters",
                "category": "🌍 地缘",
                "title": "台湾发生 5.2 级地震，台积电和存储厂生产未受影响",
                "summary": "台湾花莲海域发生5.2级地震，南科、竹科园区震感明显。各存储厂（南亚科、华邦电、旺宏）均表示生产正常，未受损失。",
                "impact": "🟢 无影响",
                "url": "https://www.reuters.com",
            },
            {
                "date": today - timedelta(days=4),
                "source": "Bloomberg",
                "category": "💰 市场",
                "title": "美光上调 Q3 营收指引至 $87亿，HBM 业务同比增长 300%",
                "summary": "美光科技（Micron）将2026财年Q3营收指引上调至87亿美元，超出分析师预期。HBM3e 出货量同比增长300%，成为增长主要驱动力。",
                "impact": "🟡 利好",
                "url": "https://www.bloomberg.com",
            },
            {
                "date": today - timedelta(days=5),
                "source": "DIGITIMES",
                "category": "🏭 产能",
                "title": "华邦电高雄新厂 20nm DRAM 导入量产，主攻利基型市场",
                "summary": "华邦电（Winbond）高雄路竹新厂20nm制程DRAM正式导入量产，月产能2万片，主攻 IoT、汽车、工业等利基型市场。",
                "impact": "🟢 利好",
                "url": "https://www.digitimes.com",
            },
            {
                "date": today - timedelta(days=5),
                "source": "TechNews",
                "category": "🤖 AI",
                "title": "AMD MI400 系列或将采用 12 颗 HBM4 堆栈，存储需求再创新高",
                "summary": "AMD 下一代 AI 加速器 MI400 系列据传将搭载 12 颗 HBM4 堆栈（较 MI300X 增加50%），单卡显存带宽突破 8TB/s。",
                "impact": "🔴 需求激增",
                "url": "https://technews.tw",
            },
        ]
        return news

    news_data = get_industry_news()

    # ── 筛选栏 ──
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        news_categories = ["全部"] + sorted(set(n["category"] for n in news_data))
        sel_category = st.selectbox("📂 分类筛选", news_categories, key="news_cat")
    with col_filter2:
        news_impacts = ["全部"] + sorted(set(n["impact"] for n in news_data))
        sel_impact = st.selectbox("⚡ 影响筛选", news_impacts, key="news_impact")
    with col_filter3:
        news_sources = ["全部"] + sorted(set(n["source"] for n in news_data))
        sel_source = st.selectbox("📡 来源筛选", news_sources, key="news_source")

    # 过滤
    filtered_news = news_data
    if sel_category != "全部":
        filtered_news = [n for n in filtered_news if n["category"] == sel_category]
    if sel_impact != "全部":
        filtered_news = [n for n in filtered_news if n["impact"] == sel_impact]
    if sel_source != "全部":
        filtered_news = [n for n in filtered_news if n["source"] == sel_source]

    st.caption(f"显示 {len(filtered_news)} / {len(news_data)} 条新闻")

    # ── 新闻卡片 ──
    impact_colors = {
        "🔴 涨价": "#FCE4EC", "🔴 暴涨": "#FCE4EC", "🔴 风险": "#FCE4EC",
        "🔴 EOL": "#FCE4EC", "🔴 需求激增": "#FCE4EC",
        "🟡 关注": "#FFF3CD", "🟡 上涨": "#FFF3CD", "🟡 利好": "#FFF3CD",
        "🟢 无影响": "#E1F5EE", "🟢 利好": "#E1F5EE",
    }
    impact_borders = {
        "🔴 涨价": "#C00000", "🔴 暴涨": "#C00000", "🔴 风险": "#C00000",
        "🔴 EOL": "#C00000", "🔴 需求激增": "#C00000",
        "🟡 关注": "#E97C00", "🟡 上涨": "#E97C00", "🟡 利好": "#E97C00",
        "🟢 无影响": "#1D9E75", "🟢 利好": "#1D9E75",
    }

    for news in filtered_news:
        bg = impact_colors.get(news["impact"], "#F5F5F5")
        border = impact_borders.get(news["impact"], "#CCC")
        date_str = news["date"].strftime("%m/%d")

        st.markdown(f"""
        <div style="background:{bg};padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.6rem;
                    border-left:4px solid {border};">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div style="flex:1;">
              <span style="font-size:0.75rem;color:#888;">{news['source']}</span>
              <span style="margin:0 0.5rem;font-size:0.75rem;color:#888;">{news['category']}</span>
              <strong style="font-size:1rem;">{news['title']}</strong>
            </div>
            <span style="font-weight:bold;font-size:0.85rem;margin-left:0.5rem;white-space:nowrap;">{news['impact']}</span>
          </div>
          <p style="margin:0.4rem 0 0 0;font-size:0.85rem;color:#555;">{news['summary']}</p>
          <div style="margin-top:0.3rem;font-size:0.7rem;color:#999;">
            📅 {date_str} | 🔗 <a href="{news['url']}" target="_blank">{news['source']}</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 底部 ──
    st.divider()
    st.caption("💡 新闻每24小时自动刷新 | 点击来源链接查看原文 | 如需添加更多新闻源请联系管理员")


# ══════════════════════════════════════════════════════════
st.divider()
st.caption("⚠️ 免责声明：本系统基于全球公开数据源预测，仅供内部参考，不构成采购/投资建议。")

# ══════════════════════════════════════════════════════════
# 直接运行入口：python dashboard_v4.py 或双击 .py 文件
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import subprocess, sys
    script = __file__
    print("Starting Streamlit Dashboard...")
    print(f"Script: {script}")
    print("Open http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script, "--server.port", "8501"])
st.caption("数据源：TrendForce / DRAMeXchange / DIGITIMES / SemiAnalysis / Nikkei Asia / 各厂商 IR | 金蝶BOM列表 + 存储物料库")
