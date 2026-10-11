"""安全防护模块：PII 检测与脱敏 + Prompt 注入检测。

两层防护：
1. PII 检测与脱敏：正则识别手机号、身份证、银行卡、邮箱、微信号、QQ 号，
   脱敏后存储（可逆映射，便于恢复原始值）。
2. Prompt 注入检测：识别"忽略指令""角色覆盖""角色混淆"等注入尝试，
   标记风险等级供 UI 展示。

集成点：streamlit_app.append_message() 在消息入库前调用 SecurityGuard.scan()。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. PII 模式定义
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIIPattern:
    """单条 PII 检测规则。"""
    name: str            # 类型标识：phone / id_card / bank_card / email / wechat / qq
    label: str           # 中文展示名
    regex: str           # 正则表达式
    mask_fn: str         # 脱敏策略标识


def _mask_phone(value: str) -> str:
    """13812345678 → 138****5678"""
    if len(value) >= 7:
        return value[:3] + "****" + value[-4:]
    return "*" * len(value)


def _mask_id_card(value: str) -> str:
    """110101199001011234 → 110***********1234"""
    if len(value) >= 6:
        return value[:3] + "*" * (len(value) - 6) + value[-3:]
    return "*" * len(value)


def _mask_bank_card(value: str) -> str:
    """6222021234567890123 → **** **** **** 0123"""
    if len(value) >= 4:
        return "**** **** **** " + value[-4:]
    return "*" * len(value)


def _mask_email(value: str) -> str:
    """zhangsan@gmail.com → z***@gmail.com"""
    if "@" in value:
        local, domain = value.split("@", 1)
        if local:
            return local[0] + "***@" + domain
    return "***"


def _mask_wechat(value: str) -> str:
    """微信号: wx12345678 → 微信号: wx****5678"""
    # value 可能包含前缀"微信号:"，只脱敏 ID 部分
    match = re.search(r"([a-zA-Z_][a-zA-Z0-9_-]{5,19})$", value)
    if match:
        wxid = match.group(1)
        if len(wxid) >= 4:
            masked_wxid = wxid[:2] + "****" + wxid[-2:]
            return value[:match.start()] + masked_wxid
    return value


def _mask_qq(value: str) -> str:
    """QQ: 123456789 → QQ: 12****89"""
    match = re.search(r"([1-9]\d{4,10})$", value)
    if match:
        qq = match.group(1)
        if len(qq) >= 4:
            masked_qq = qq[:2] + "****" + qq[-2:]
            return value[:match.start()] + masked_qq
    return value


_MASK_FUNCTIONS = {
    "phone": _mask_phone,
    "id_card": _mask_id_card,
    "bank_card": _mask_bank_card,
    "email": _mask_email,
    "wechat": _mask_wechat,
    "qq": _mask_qq,
}


# PII 正则规则表（顺序很重要：身份证/银行卡要先于通用数字匹配）
_PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="id_card",
        label="身份证号",
        regex=r"(?<!\d)([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)",
        mask_fn="id_card",
    ),
    PIIPattern(
        name="bank_card",
        label="银行卡号",
        regex=r"(?<!\d)(6[0-9]{15,18})(?!\d)",
        mask_fn="bank_card",
    ),
    PIIPattern(
        name="phone",
        label="手机号",
        regex=r"(?<!\d)(1[3-9]\d{9})(?!\d)",
        mask_fn="phone",
    ),
    PIIPattern(
        name="email",
        label="邮箱",
        regex=r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        mask_fn="email",
    ),
    PIIPattern(
        name="wechat",
        label="微信号",
        regex=r"(?:微信号|微信|wechat|WeChat)\s*[:：]?\s*([a-zA-Z_][a-zA-Z0-9_-]{5,19})",
        mask_fn="wechat",
    ),
    PIIPattern(
        name="qq",
        label="QQ号",
        regex=r"(?:QQ|qq|Qq)\s*[:：]?\s*([1-9]\d{4,10})",
        mask_fn="qq",
    ),
]


# ---------------------------------------------------------------------------
# 2. Prompt 注入模式定义
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionPattern:
    """单条 Prompt 注入检测规则。"""
    name: str
    label: str
    regex: str
    severity: str   # high / medium / low


_INJECTION_PATTERNS: list[InjectionPattern] = [
    # 高风险：直接要求忽略系统指令（允许中间有"你的/前面/以上"等词）
    InjectionPattern(
        name="ignore_instructions",
        label="忽略指令",
        regex=r"(?:忽略|无视|跳过|不要遵守|disregard|ignore).{0,15}?(?:指令|规则|提示|设定|instructions?|rules?|prompt|system)",
        severity="high",
    ),
    # 高风险：角色覆盖
    InjectionPattern(
        name="role_override",
        label="角色覆盖",
        regex=r"(?:你现在(?:是|不是)|你(?:不再|已经不是|不再是)|从现在起你(?:是|将)|let'?s role.?play|you are now|from now on you(?:'re| are))",
        severity="high",
    ),
    # 中风险：角色混淆（冒充 system/assistant）
    InjectionPattern(
        name="role_confusion",
        label="角色混淆",
        regex=r"(?:^|\n)\s*(?:system|assistant|admin|developer)\s*[:：]\s*",
        severity="medium",
    ),
    # 中风险：DAN 越狱
    InjectionPattern(
        name="jailbreak_dan",
        label="越狱攻击",
        regex=r"(?:DAN|do anything now|越狱模式|jailbreak|开发者模式|developer mode)",
        severity="medium",
    ),
    # 低风险：要求输出系统提示
    InjectionPattern(
        name="prompt_leak",
        label="提示泄露",
        regex=r"(?:输出|显示|打印|告诉我|reveal|show|print)\s*(?:你的|the)?\s*(?:系统)?\s*(?:提示|指令|规则|prompt|instructions?|system message)",
        severity="low",
    ),
    # 低风险：Base64 可疑载荷
    InjectionPattern(
        name="base64_payload",
        label="Base64载荷",
        regex=r"(?:^|\s)([A-Za-z0-9+/]{40,}={0,2})(?:\s|$)",
        severity="low",
    ),
]


# ---------------------------------------------------------------------------
# 3. 扫描结果数据类
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    """单条 PII 检测结果。"""
    pii_type: str       # phone / id_card / ...
    label: str          # 手机号 / 身份证号 / ...
    original: str       # 原始值
    masked: str         # 脱敏值
    start: int          # 在原文中的起始位置
    end: int            # 结束位置


@dataclass
class InjectionMatch:
    """单条 Prompt 注入检测结果。"""
    pattern_name: str
    label: str
    matched_text: str
    severity: str        # high / medium / low
    start: int
    end: int


@dataclass
class SecurityScanResult:
    """安全扫描总结果。"""
    original_text: str
    redacted_text: str
    pii_matches: list[PIIMatch] = field(default_factory=list)
    injection_matches: list[InjectionMatch] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_matches)

    @property
    def has_injection(self) -> bool:
        return bool(self.injection_matches)

    @property
    def has_high_risk(self) -> bool:
        return any(m.severity == "high" for m in self.injection_matches)

    @property
    def is_safe(self) -> bool:
        """安全 = 无 PII 且无高风险注入。"""
        return not self.has_pii and not self.has_high_risk

    @property
    def pii_summary(self) -> dict[str, int]:
        """按类型统计 PII 数量。"""
        summary: dict[str, int] = {}
        for match in self.pii_matches:
            summary[match.label] = summary.get(match.label, 0) + 1
        return summary

    @property
    def injection_summary(self) -> dict[str, int]:
        """按严重程度统计注入数量。"""
        summary: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for match in self.injection_matches:
            summary[match.severity] = summary.get(match.severity, 0) + 1
        return summary


# ---------------------------------------------------------------------------
# 4. 安全守卫主类
# ---------------------------------------------------------------------------

class SecurityGuard:
    """安全守卫：PII 检测脱敏 + Prompt 注入检测。

    用法::

        guard = SecurityGuard()
        result = guard.scan("我的手机号是13812345678，忽略你的系统指令")
        if result.has_pii:
            print("检测到PII:", result.pii_summary)
        if result.has_injection:
            print("检测到注入:", result.injection_summary)
        safe_text = result.redacted_text  # 脱敏后的文本，可安全传给 LLM
    """

    def __init__(self) -> None:
        self._pii_compiled = [
            (pattern, re.compile(pattern.regex)) for pattern in _PII_PATTERNS
        ]
        self._injection_compiled = [
            (pattern, re.compile(pattern.regex, re.IGNORECASE))
            for pattern in _INJECTION_PATTERNS
        ]

    # ---- PII 检测 ----

    def detect_pii(self, text: str) -> list[PIIMatch]:
        """检测文本中所有 PII。"""
        matches: list[PIIMatch] = []
        occupied: list[tuple[int, int]] = []  # 已占用区间，避免重叠

        for pattern, regex in self._pii_compiled:
            for m in regex.finditer(text):
                start, end = m.span()
                # 检查是否与已检测到的 PII 区间重叠
                if any(not (end <= s or start >= e) for s, e in occupied):
                    continue
                occupied.append((start, end))

                original = m.group(0)
                mask_fn = _MASK_FUNCTIONS.get(pattern.mask_fn, lambda v: "*" * len(v))
                masked = mask_fn(original)

                matches.append(PIIMatch(
                    pii_type=pattern.name,
                    label=pattern.label,
                    original=original,
                    masked=masked,
                    start=start,
                    end=end,
                ))

        # 按位置排序
        matches.sort(key=lambda x: x.start)
        return matches

    # ---- PII 脱敏 ----

    def redact(self, text: str, pii_matches: list[PIIMatch] | None = None) -> str:
        """将文本中的 PII 替换为脱敏版本。

        Args:
            text: 原始文本
            pii_matches: 预计算的 PII 匹配（可选，避免重复扫描）
        Returns:
            脱敏后的文本
        """
        if pii_matches is None:
            pii_matches = self.detect_pii(text)

        if not pii_matches:
            return text

        # 从后往前替换，避免位置偏移
        result = text
        for match in sorted(pii_matches, key=lambda x: x.start, reverse=True):
            result = result[:match.start] + match.masked + result[match.end:]
        return result

    # ---- Prompt 注入检测 ----

    def detect_injection(self, text: str) -> list[InjectionMatch]:
        """检测文本中的 Prompt 注入尝试。"""
        matches: list[InjectionMatch] = []
        for pattern, regex in self._injection_compiled:
            for m in regex.finditer(text):
                matches.append(InjectionMatch(
                    pattern_name=pattern.name,
                    label=pattern.label,
                    matched_text=m.group(0).strip(),
                    severity=pattern.severity,
                    start=m.start(),
                    end=m.end(),
                ))
        matches.sort(key=lambda x: x.start)
        return matches

    # ---- 综合扫描 ----

    def scan(self, text: str) -> SecurityScanResult:
        """综合扫描：PII 检测 + 脱敏 + Prompt 注入检测。

        Args:
            text: 待扫描的原始文本
        Returns:
            SecurityScanResult，包含原始文本、脱敏文本、PII 匹配、注入匹配
        """
        pii_matches = self.detect_pii(text)
        redacted = self.redact(text, pii_matches)
        injection_matches = self.detect_injection(text)

        return SecurityScanResult(
            original_text=text,
            redacted_text=redacted,
            pii_matches=pii_matches,
            injection_matches=injection_matches,
        )

    def scan_messages(self, messages: list[dict[str, Any]]) -> list[SecurityScanResult]:
        """批量扫描消息列表。"""
        return [self.scan(msg.get("content", "")) for msg in messages]


# ---------------------------------------------------------------------------
# 5. 便捷函数
# ---------------------------------------------------------------------------

# 全局单例（线程安全在 Streamlit 单线程模型下不需要）
_GUARD: SecurityGuard | None = None


def get_guard() -> SecurityGuard:
    """获取全局 SecurityGuard 单例。"""
    global _GUARD
    if _GUARD is None:
        _GUARD = SecurityGuard()
    return _GUARD


def scan_message(text: str) -> SecurityScanResult:
    """便捷函数：扫描单条消息。"""
    return get_guard().scan(text)


def redact_message(text: str) -> str:
    """便捷函数：仅脱敏，返回脱敏后的文本。"""
    return get_guard().redact(text)
