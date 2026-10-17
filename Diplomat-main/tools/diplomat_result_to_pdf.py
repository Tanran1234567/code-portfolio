"""
将前端导出的 diplomat_result_*.json 转换为 PDF。

用法（仓库根目录）：
  python tools/diplomat_result_to_pdf.py ^
    --input diplomat_result_demo-001_2026-01-18T02-34-03-015Z.json ^
    --output diplomat_result_demo-001.pdf

依赖：
  pip install -r backend/requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _try_register_cn_font() -> Tuple[str, Optional[str]]:
    """
    尝试注册中文字体。
    返回：(font_name, font_path_used)
    """
    # 优先：微软雅黑（Windows 常见）
    candidates: List[Tuple[str, Path, int]] = []
    win_font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates.extend(
        [
            ("CNFont", win_font_dir / "msyh.ttc", 0),  # Microsoft YaHei
            ("CNFont", win_font_dir / "msyh.ttf", 0),
            ("CNFont", win_font_dir / "simsun.ttc", 0),  # SimSun
            ("CNFont", win_font_dir / "simhei.ttf", 0),  # SimHei
            ("CNFont", win_font_dir / "simkai.ttf", 0),  # KaiTi
        ]
    )

    # 兼容：用户提供自定义字体文件（同目录下放一个 font.ttf / font.otf）
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            ("CNFont", script_dir / "font.ttf", 0),
            ("CNFont", script_dir / "font.otf", 0),
        ]
    )

    for font_name, font_path, sub_idx in candidates:
        if not font_path.exists():
            continue
        try:
            # reportlab 对 ttc 支持依赖 subfontIndex 参数；对 ttf/otf 则忽略即可
            if font_path.suffix.lower() == ".ttc":
                pdfmetrics.registerFont(
                    TTFont(font_name, str(font_path), subfontIndex=sub_idx)
                )
            else:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name, str(font_path)
        except Exception:
            continue

    # fallback：Helvetica（不保证中文能正常显示）
    return "Helvetica", None


def _h(text: str) -> str:
    # Paragraph 支持少量 HTML；把换行变成 <br/>
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _esc(text: str) -> str:
    # 给我们自己拼的 <b> / <font> 用：只转义内容，不转义标记
    return escape(text or "", {"'": "&apos;", '"': "&quot;"})


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _group_speeches_by_round(
    speeches: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for s in speeches or []:
        try:
            r = int(s.get("round", 0) or 0)
        except Exception:
            r = 0
        out.setdefault(r, []).append(s)
    for r in out:
        # 尽量按时间排序（若无 timestamp 则保持原顺序）
        out[r].sort(key=lambda x: _safe_str(x.get("timestamp", "")))
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def build_pdf(data: Dict[str, Any], output_path: Path) -> None:
    font_name, font_path_used = _try_register_cn_font()

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "CNBase",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "CNTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "CNH2",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#1f2937"),
    )
    small = ParagraphStyle(
        "CNSmall",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "CNMono",
        parent=styles["Code"],
        fontName=font_name,
        fontSize=9.5,
        leading=12,
        spaceAfter=6,
    )
    table_cell = ParagraphStyle(
        "CNTableCell",
        parent=base,
        fontName=font_name,
        fontSize=9.5,
        leading=12,
        spaceAfter=0,
        wordWrap="CJK",  # 关键：中文自动换行
    )

    scenario_id = _safe_str(data.get("scenario_id", ""))
    exported_at = _safe_str(data.get("exported_at", ""))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Diplomat Result - {scenario_id}",
        author="Diplomat",
    )

    story = []

    story.append(
        Paragraph(_h(f"Diplomat 结果报告（{scenario_id or '未命名场景'}）"), title)
    )
    meta_lines = []
    if exported_at:
        meta_lines.append(f"exported_at：{exported_at}")
    meta_lines.append(
        f"generated_at：{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    if font_path_used:
        meta_lines.append(f"font：{font_name} ({font_path_used})")
    else:
        meta_lines.append(f"font：{font_name}（未检测到中文字体时可能出现乱码）")
    story.append(Paragraph(_h(" / ".join(meta_lines)), small))

    # 输入信息
    inp = data.get("input") or {}
    background = (inp.get("background") or {}) if isinstance(inp, dict) else {}
    issues = (inp.get("issues") or {}) if isinstance(inp, dict) else {}
    rubrics = (inp.get("rubrics") or []) if isinstance(inp, dict) else []

    story.append(Paragraph("输入配置", h2))
    story.append(
        Paragraph(
            _h(
                f"scenario_id：{_safe_str(background.get('scenario_id') or scenario_id)}"
            ),
            base,
        )
    )

    world_background = _safe_str(background.get("world_background", ""))
    if world_background:
        story.append(Paragraph("世界观（world_background）", h2))
        story.append(Paragraph(_h(world_background), base))

    countries = background.get("countries") if isinstance(background, dict) else None
    if isinstance(countries, dict) and countries:
        story.append(Paragraph("国家 Profiles", h2))
        rows = [["国家", "profile"]]
        for k in sorted(countries.keys(), key=lambda x: str(x)):
            profile = countries.get(k, {}) or {}
            profile_text = _safe_str(profile.get("profile", ""))
            rows.append([_safe_str(k), Paragraph(_h(profile_text), table_cell)])
        tbl = Table(rows, colWidths=[32 * mm, 148 * mm], repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), font_name),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 6))

    issue_list = issues.get("issues") if isinstance(issues, dict) else None
    if isinstance(issue_list, list) and issue_list:
        story.append(Paragraph("议题（Issues）", h2))
        for it in issue_list:
            if not isinstance(it, dict):
                continue
            title_txt = _safe_str(it.get("title", ""))
            iid = _safe_str(it.get("issue_id", ""))
            desc = _safe_str(it.get("description", ""))
            line = f"{iid}：{title_txt}"
            if desc:
                line += f"（{desc}）"
            story.append(Paragraph(_h(line), base))

    if isinstance(rubrics, list) and rubrics:
        story.append(Paragraph("评分维度（Rubrics）", h2))
        rows = [["id", "title", "weight"]]
        for r in rubrics:
            if not isinstance(r, dict):
                continue
            rows.append(
                [
                    _safe_str(r.get("id", "")),
                    _safe_str(r.get("title", "")),
                    _safe_str(r.get("weight", "")),
                ]
            )
        tbl = Table(rows, colWidths=[20 * mm, 120 * mm, 40 * mm])
        tbl.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), font_name),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(tbl)
        story.append(PageBreak())

    # speeches
    speeches = data.get("speeches") or []
    story.append(Paragraph(f"Speeches（共 {len(speeches)} 条）", h2))
    grouped = _group_speeches_by_round(speeches if isinstance(speeches, list) else [])
    for r, items in grouped.items():
        if r <= 0:
            continue
        story.append(Paragraph(f"第 {r} 轮", h2))
        for s in items:
            speaker = _safe_str(s.get("speaker", ""))
            ts = _safe_str(s.get("timestamp", ""))
            content = _safe_str(s.get("content", ""))
            story.append(
                Paragraph(
                    f'<b>{_esc(speaker)}</b>  <font color="#6b7280">{_esc(ts)}</font>',
                    base,
                )
            )
            story.append(Paragraph(_h(content), base))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # round summaries
    summaries = data.get("round_summaries") or []
    story.append(Paragraph(f"Round Summaries（共 {len(summaries)} 条）", h2))
    if isinstance(summaries, list) and summaries:
        for sm in summaries:
            if not isinstance(sm, dict):
                continue
            r = _safe_str(sm.get("round", ""))
            summary_txt = _safe_str(sm.get("summary", ""))
            story.append(Paragraph(_h(f"第 {r} 轮摘要"), h2))
            # summary 里常含 markdown（** / *），这里用等宽块直接输出以保真
            story.append(Preformatted(summary_txt, mono))
            sig = sm.get("signals")
            if isinstance(sig, dict) and sig:
                story.append(Paragraph("signals", h2))
                story.append(
                    Preformatted(json.dumps(sig, ensure_ascii=False, indent=2), mono)
                )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("无", base))

    story.append(PageBreak())

    # evaluation
    eval_obj = data.get("evaluation")
    story.append(Paragraph("Evaluation", h2))
    if isinstance(eval_obj, dict):
        scores = eval_obj.get("scores") or {}
        feedback = eval_obj.get("feedback") or {}
        if isinstance(scores, dict) and scores:
            story.append(Paragraph("scores", h2))
            rows = [["agent", "score"]]
            for k in sorted(scores.keys(), key=lambda x: str(x)):
                rows.append([_safe_str(k), _safe_str(scores.get(k))])
            tbl = Table(rows, colWidths=[60 * mm, 60 * mm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, -1), font_name),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("LEADING", (0, 0), (-1, -1), 12),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 10))
        if isinstance(feedback, dict) and feedback:
            story.append(Paragraph("feedback", h2))
            for k in sorted(feedback.keys(), key=lambda x: str(x)):
                story.append(Paragraph(_h(f"<b>{_safe_str(k)}</b>"), base))
                story.append(Paragraph(_h(_safe_str(feedback.get(k, ""))), base))
                story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("无", base))

    def _page_footer(canvas, doc_):  # type: ignore[no-redef]
        canvas.saveState()
        canvas.setFont(font_name, 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert diplomat_result_*.json to PDF")
    ap.add_argument("--input", "-i", required=True, help="输入 JSON 文件路径")
    ap.add_argument("--output", "-o", required=True, help="输出 PDF 文件路径")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit("输入 JSON 顶层必须是 object")

    build_pdf(data, out_path)
    print(f"OK: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
