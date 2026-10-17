from __future__ import annotations

import json
from typing import Optional, Tuple
from pathlib import Path
from typing import Dict, Any, Optional
from ..llm_client import LLMClient
from ..models import EvaluationInput, EvaluationOutput, PromptDebug
from ..prompt_loader import format_user, load_prompt, to_pretty_json


# def evaluate(payload: EvaluationInput) -> Tuple[EvaluationOutput, Optional[PromptDebug]]:
#     """
#     评分者模块核心逻辑入口。
#     输入：rubrics + speeches
#     输出：scores + feedback
#     """
#     llm = LLMClient()
#     if llm.api_key and llm.api_base:
#         print(1)
#         cfg = load_prompt("evaluator")
#         system = cfg.system
#         user = format_user(
#             cfg.user_template,
#             {
#                 "rubrics_json": to_pretty_json(payload.rubrics),
#                 "speeches_json": to_pretty_json([s.model_dump() for s in payload.speeches]),
#             },
#         )
#         temp = float(cfg.temperature)
#         dbg = PromptDebug(system=system, user=user, model=(cfg.model or llm.model), temperature=temp)
#         text = llm.chat(system=system, user=user, temperature=temp, timeout=90) or ""

#         print(text)
#         print(f"字符串首字符编码: {ord(text[0]) if text else '空'}")
#         print(f"字符串首字符实际值: {repr(text[0]) if text else '空'}")
        
#         try:
#             obj = json.loads(text) if text else None
#             print("JSON解析结果：", obj, "｜ 解析结果类型：", type(obj))
#             if isinstance(obj, dict) and "scores" in obj and "feedback" in obj:

#                 return EvaluationOutput(**obj), dbg
#         except Exception as e:
#             print("解析失败异常：", type(e).__name__, "｜ 异常详情：", e)
#             pass

#     speakers = sorted({s.speaker for s in payload.speeches} or {"A", "B", "C", "D"})
#     return EvaluationOutput(
#         scores={sp: 70 for sp in speakers},
#         feedback={sp: "[stub] feedback" for sp in speakers},
#     ), None

# ========== 跨平台同目录Prompt加载函数（核心修改） ==========
_EVALUATOR_PROMPTS_CACHE: Optional[Dict[str, Any]] = None  # 缓存，避免重复读文件

def load_evaluator_prompts(prompt_name: str) -> Dict[str, Any]:
    """
    从【当前文件同目录】的prompt.json（单文件数组格式）加载指定name的Prompt配置
    跨平台兼容（Windows/Linux/Mac），无需硬编码绝对路径
    :param prompt_name: 要加载的Prompt名称（r1/r2/.../r10/summary）
    :return: 匹配的Prompt配置字典
    :raise: FileNotFoundError 若prompt.json不存在，ValueError 若名称未找到
    """
    global _EVALUATOR_PROMPTS_CACHE
    # 关键：获取当前函数所在文件的目录 → 拼接同目录下的prompt.json
    # __file__ 是Python内置变量，指向当前文件的绝对路径，跨平台自动适配
    current_dir = Path(__file__).parent  # 当前文件目录 = evaluator文件夹
    prompt_file = current_dir / "prompt.json"  # 同目录下的prompt.json，自动适配路径分隔符（\ /）
    
    # 缓存机制：首次读取后存入内存，11次调用仅读1次文件，提升效率
    if _EVALUATOR_PROMPTS_CACHE is None:
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt文件未找到！请确认prompt.json在{current_dir}目录下\n"
                f"预期路径：{prompt_file}"
            )
        # 读取并解析JSON数组（指定utf-8编码，避免中文乱码）
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts_list = json.load(f)
        # 转换为{name: 配置}的字典，方便快速匹配r1-r10/summary
        _EVALUATOR_PROMPTS_CACHE = {p["name"]: p for p in prompts_list}
    
    # 匹配指定名称的Prompt配置
    if prompt_name not in _EVALUATOR_PROMPTS_CACHE:
        raise ValueError(
            f"在prompt.json中未找到名称为'{prompt_name}'的配置\n"
            f"已存在的配置名称：{list(_EVALUATOR_PROMPTS_CACHE.keys())}"
        )
    return _EVALUATOR_PROMPTS_CACHE[prompt_name]
# =============================================================



def evaluate(payload: EvaluationInput) -> Tuple[EvaluationOutput, Optional[PromptDebug]]:
    """
    评分者模块核心逻辑入口（分10次单项评分+1次反馈，同目录加载Prompt，跨平台兼容）
    输入：rubrics + speeches（r1-r10共10条评分标准）
    输出：scores（10项单项分累加总分） + feedback（summary Prompt生成的评语）
    主函数与prompt.json同属evaluator文件夹，自动适配Windows/Linux/Mac
    """
    llm = LLMClient()
    # 基础校验：LLM配置无效直接走兜底逻辑
    filtered_speeches = [
        s for s in payload.speeches
        if s.stage and s.stage.startswith("negotiation_round_")
    ]
    if not (llm.api_key and llm.api_base):
        speakers = sorted({s.speaker for s in filtered_speeches} or {"A", "B", "C", "D"})
        # s: class Speech 找 s.stage 是 negotiate 的即可
        return EvaluationOutput(
            scores={sp: 70 for sp in speakers},
            feedback={sp: "[stub] feedback" for sp in speakers},
        ), None

    # 初始化：总分字典（初始0分）、调试信息列表、评分项列表（r1-r10）
    speakers = sorted({s.speaker for s in filtered_speeches} or {"A", "B", "C", "D"})
    total_scores = {sp: 0 for sp in speakers}
    debug_list = []
    score_items = [f"r{i}" for i in range(1, 11)]  # r1到r10

    try:
        # 第一步：循环调用r1-r10，获取单项分并累加
        for item in score_items:
            # 加载当前分项的Prompt配置（同目录prompt.json，跨平台）
            cfg = load_evaluator_prompts(item)
            # 渲染User Prompt（复用模板）
            user_prompt = format_user(
                cfg["user_template"],
                {
                    "rubrics_json": to_pretty_json(payload.rubrics),
                    "speeches_json": to_pretty_json([s.model_dump() for s in filtered_speeches]),
                },
            )
            # 提取Prompt配置参数
            system_prompt = cfg["system"]
            temp = float(cfg["temperature"])
            model = cfg["model"] or llm.model

            # 构造调试信息并记录
            debug = PromptDebug(
                system=system_prompt,
                user=user_prompt,
                model=model,
                temperature=temp
            )
            debug_list.append({"item": item, "debug": debug})

            # 调用LLM获取当前分项评分
            llm_response = llm.chat(
                system=system_prompt,
                user=user_prompt,
                temperature=temp,
                timeout=90
            ) or ""
            print(f"【{item}】LLM返回结果：{llm_response}")

            # 解析单项分并安全累加
            score_obj = json.loads(llm_response) if llm_response else None
            if not isinstance(score_obj, dict) or "scores" not in score_obj:
                print(f"【{item}】解析失败：非预期JSON格式，跳过当前分项")
                continue
            # 遍历每个发言者，累加有效分数（0-10分限制，非数字记0）
            for sp in speakers:
                raw_score = score_obj["scores"].get(sp, 0)
                # 安全处理：非数字转0，超出0-10范围自动截断
                valid_score = int(raw_score) if isinstance(raw_score, (int, float)) else 0
                valid_score = max(0, min(10, valid_score))
                total_scores[sp] += valid_score

        # 第二步：单独调用summary，获取最终反馈
        summary_cfg = load_evaluator_prompts("summary")
        summary_user = format_user(
            summary_cfg["user_template"],
            {
                "rubrics_json": to_pretty_json(payload.rubrics),
                "speeches_json": to_pretty_json([s.model_dump() for s in filtered_speeches]),
            },
        )
        # 构造summary调试信息
        summary_debug = PromptDebug(
            system=summary_cfg["system"],
            user=summary_user,
            model=summary_cfg["model"] or llm.model,
            temperature=float(summary_cfg["temperature"])
        )
        debug_list.append({"item": "summary", "debug": summary_debug})
        # 调用LLM获取反馈
        summary_response = llm.chat(
            system=summary_cfg["system"],
            user=summary_user,
            temperature=float(summary_cfg["temperature"]),
            timeout=90
        ) or ""
        print(f"【summary】LLM返回反馈：{summary_response}")
        # 解析反馈（失败则用默认值）
        feedback_obj = json.loads(summary_response) if summary_response else None
        raw_feedback = (
            feedback_obj["feedback"]
            if (isinstance(feedback_obj, dict) and "feedback" in feedback_obj)
            else {sp: "[默认反馈] 综合各维度表现，需结合分项评分优化谈判策略" for sp in speakers}
        )

        final_feedback = {
            sp: raw_feedback[sp]["overall"] if isinstance(raw_feedback[sp], dict) and "overall" in raw_feedback[sp] else raw_feedback[sp]
            for sp in speakers
        }

        # 第三步：构造最终调试信息和返回结果
        final_debug = PromptDebug(
            system=f"分10项评分（r1-r10）+ summary反馈 | 同目录加载Prompt（跨平台）",
            user=f"汇总{len(debug_list)}次LLM调用调试信息：{debug_list}",
            model=llm.model,
            temperature=0.2
        )
        return EvaluationOutput(
            scores=total_scores,
            feedback=final_feedback
        ), final_debug

    except Exception as e:
        # 异常处理：保留已累加分数，返回异常反馈，不中断流程
        print(f"【评分流程异常】{type(e).__name__}：{str(e)}")
        error_feedback = {sp: f"[评分异常] 流程中断：{str(e)[:60]}..." for sp in speakers}
        return EvaluationOutput(scores=total_scores, feedback=error_feedback), None