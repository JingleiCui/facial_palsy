# -*- coding: utf-8 -*-
"""
临床评分报告生成器
==================

生成可解释性HTML报告，包含:
1. 诊断结果概览
2. Sunnybrook评分详情
3. House-Brackmann评分详情
4. 各动作分析详情
5. 可视化图表
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np


class ClinicalReportGenerator:
    """
    临床评分报告生成器

    生成综合性HTML报告，展示:
    - 诊断结果
    - 评分依据
    - 可视化结果
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
            self,
            examination_id: str,
            sunnybrook_result: Any,
            hb_result: Any,
            action_summaries: List[Dict[str, Any]],
            meta: Dict[str, Any] = None,
            ground_truth: Dict[str, Any] = None
    ) -> str:
        """
        生成完整HTML报告

        Args:
            examination_id: 检查ID
            sunnybrook_result: Sunnybrook评分结果
            hb_result: House-Brackmann评分结果
            action_summaries: 各动作分析摘要
            meta: 元信息
            ground_truth: 真实标签(如果有)

        Returns:
            报告文件路径
        """
        html = self._build_html(
            examination_id, sunnybrook_result, hb_result,
            action_summaries, meta, ground_truth
        )

        report_path = self.output_dir / f"{examination_id}_clinical_report.html"
        report_path.write_text(html, encoding='utf-8')

        return str(report_path)

    def _build_html(
            self,
            examination_id: str,
            sunnybrook_result: Any,
            hb_result: Any,
            action_summaries: List[Dict[str, Any]],
            meta: Dict[str, Any],
            ground_truth: Dict[str, Any]
    ) -> str:
        """构建HTML内容"""

        # CSS样式
        css = self._get_css()

        # 患者信息
        patient_info = self._build_patient_info(examination_id, meta, ground_truth)

        # 诊断结果概览
        diagnosis_summary = self._build_diagnosis_summary(sunnybrook_result, hb_result)

        # Sunnybrook详情
        sunnybrook_details = self._build_sunnybrook_details(sunnybrook_result)

        # House-Brackmann详情
        hb_details = self._build_hb_details(hb_result)

        # 动作分析详情
        actions_details = self._build_actions_details(action_summaries)

        # 与真实标签对比
        comparison = ""
        if ground_truth:
            comparison = self._build_comparison(sunnybrook_result, hb_result, ground_truth)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>面瘫临床评分报告 - {examination_id}</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>面瘫临床评分报告</h1>
            <p class="subtitle">基于House-Brackmann和Sunnybrook评分系统</p>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        {patient_info}

        <section class="diagnosis-summary">
            <h2>📋 诊断结果概览</h2>
            {diagnosis_summary}
        </section>

        {comparison}

        <section class="sunnybrook-section">
            <h2>📊 Sunnybrook评分详情</h2>
            {sunnybrook_details}
        </section>

        <section class="hb-section">
            <h2>📈 House-Brackmann评分详情</h2>
            {hb_details}
        </section>

        <section class="actions-section">
            <h2>🎬 动作分析详情</h2>
            {actions_details}
        </section>

        <footer>
            <p>本报告由H-GFA Net系统自动生成，仅供参考，最终诊断请以医生判断为准。</p>
        </footer>
    </div>

    <script>
        // 可折叠区域
        document.querySelectorAll('.collapsible').forEach(item => {{
            item.addEventListener('click', function() {{
                this.classList.toggle('active');
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                }}
            }});
        }});
    </script>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        header .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }

        header .timestamp {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
        }

        section {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        section h2 {
            color: #2c3e50;
            font-size: 1.4em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }

        .patient-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .info-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }

        .info-item label {
            display: block;
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }

        .info-item value {
            display: block;
            font-size: 1.1em;
            font-weight: 500;
            color: #2c3e50;
        }

        .diagnosis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .diagnosis-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 2px solid transparent;
        }

        .diagnosis-card.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .diagnosis-card h3 {
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            opacity: 0.8;
        }

        .diagnosis-card .value {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .diagnosis-card .description {
            font-size: 0.9em;
        }

        .score-bar {
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .score-bar-fill.good { background: #28a745; }
        .score-bar-fill.warning { background: #ffc107; }
        .score-bar-fill.danger { background: #dc3545; }

        .details-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .details-table th,
        .details-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        .details-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        .details-table tr:hover {
            background: #f8f9fa;
        }

        .evidence-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            font-size: 0.9em;
        }

        .evidence-box code {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }

        .action-card {
            border: 1px solid #e9ecef;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
        }

        .action-header {
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .action-header:hover {
            background: #e9ecef;
        }

        .action-header h4 {
            margin: 0;
            color: #2c3e50;
        }

        .action-content {
            padding: 20px;
            border-top: 1px solid #e9ecef;
        }

        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .indicator-item {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }

        .indicator-item .name {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }

        .indicator-item .value {
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }

        .status-badge.normal { background: #d4edda; color: #155724; }
        .status-badge.mild { background: #fff3cd; color: #856404; }
        .status-badge.moderate { background: #ffeeba; color: #856404; }
        .status-badge.severe { background: #f8d7da; color: #721c24; }

        .comparison-section {
            background: #fff3cd;
            border: 1px solid #ffc107;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            text-align: center;
        }

        .comparison-item {
            padding: 15px;
        }

        .comparison-item.match { background: #d4edda; border-radius: 8px; }
        .comparison-item.mismatch { background: #f8d7da; border-radius: 8px; }

        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }

        .collapsible {
            cursor: pointer;
        }

        .collapsible:after {
            content: '\\25BC';
            float: right;
            margin-left: 5px;
        }

        .collapsible.active:after {
            content: '\\25B2';
        }

        .collapse-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }

        .image-container {
            text-align: center;
            margin: 15px 0;
        }

        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        @media (max-width: 768px) {
            .diagnosis-grid {
                grid-template-columns: 1fr;
            }

            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _build_patient_info(
            self,
            examination_id: str,
            meta: Dict[str, Any],
            ground_truth: Dict[str, Any]
    ) -> str:
        """构建患者信息区"""
        meta = meta or {}

        items = [
            ("检查ID", examination_id),
            ("患者ID", meta.get('patient_id', 'N/A')),
            ("检查时间", meta.get('capture_datetime', 'N/A')),
            ("数据来源", meta.get('source', 'N/A')),
        ]

        if ground_truth:
            items.extend([
                ("真实面瘫状态", "有面瘫" if ground_truth.get('has_palsy') else "无面瘫"),
                ("真实患侧", ground_truth.get('palsy_side', 'N/A')),
                ("真实HB分级", ground_truth.get('hb_grade', 'N/A')),
                ("真实Sunnybrook", ground_truth.get('sunnybrook_score', 'N/A')),
            ])

        info_html = '<div class="patient-info">'
        for label, value in items:
            info_html += f'''
            <div class="info-item">
                <label>{label}</label>
                <value>{value}</value>
            </div>
            '''
        info_html += '</div>'

        return f'''
        <section class="patient-section">
            <h2>👤 患者信息</h2>
            {info_html}
        </section>
        '''

    def _build_diagnosis_summary(
            self,
            sunnybrook_result: Any,
            hb_result: Any
    ) -> str:
        """构建诊断结果概览"""

        # 提取结果
        if hasattr(hb_result, 'grade_roman'):
            hb_grade = hb_result.grade_roman
            hb_desc = hb_result.description
            hb_pct = hb_result.composite_function_pct
        else:
            hb_grade = "N/A"
            hb_desc = "无数据"
            hb_pct = 0

        if hasattr(sunnybrook_result, 'composite_score'):
            sb_score = sunnybrook_result.composite_score
            affected_side = sunnybrook_result.affected_side
        else:
            sb_score = 0
            affected_side = "none"

        # 判断是否有面瘫
        has_palsy = sb_score < 90 or (hasattr(hb_result, 'grade') and hb_result.grade.value > 1)
        palsy_status = "检测到面瘫" if has_palsy else "未检测到面瘫"

        # 患侧显示
        side_display = {
            "left": "左侧",
            "right": "右侧",
            "none": "无/对称"
        }.get(affected_side, affected_side)

        # 评分条颜色
        sb_color = "good" if sb_score >= 80 else ("warning" if sb_score >= 50 else "danger")

        return f'''
        <div class="diagnosis-grid">
            <div class="diagnosis-card primary">
                <h3>面瘫状态</h3>
                <div class="value">{palsy_status}</div>
                <div class="description">患侧: {side_display}</div>
            </div>

            <div class="diagnosis-card">
                <h3>House-Brackmann分级</h3>
                <div class="value">{hb_grade}</div>
                <div class="description">{hb_desc.split(' - ')[0] if ' - ' in str(hb_desc) else hb_desc}</div>
                <div class="score-bar">
                    <div class="score-bar-fill {sb_color}" style="width: {hb_pct}%"></div>
                </div>
            </div>

            <div class="diagnosis-card">
                <h3>Sunnybrook综合评分</h3>
                <div class="value">{sb_score:.0f}</div>
                <div class="description">满分100分</div>
                <div class="score-bar">
                    <div class="score-bar-fill {sb_color}" style="width: {sb_score}%"></div>
                </div>
            </div>

            <div class="diagnosis-card">
                <h3>置信度</h3>
                <div class="value">{getattr(hb_result, 'confidence', 0.8) * 100:.0f}%</div>
                <div class="description">评估可靠性</div>
            </div>
        </div>
        '''

    def _build_sunnybrook_details(self, sunnybrook_result: Any) -> str:
        """构建Sunnybrook评分详情"""

        if not sunnybrook_result:
            return '<p>无Sunnybrook评分数据</p>'

        # 静态对称性
        resting = getattr(sunnybrook_result, 'resting_symmetry', None)
        resting_html = '<p>无数据</p>'
        if resting:
            resting_html = f'''
            <table class="details-table">
                <tr>
                    <th>部位</th>
                    <th>状态</th>
                    <th>评分</th>
                    <th>依据</th>
                </tr>
                <tr>
                    <td>👁 眼 (睑裂)</td>
                    <td><span class="status-badge {'normal' if resting.eye_score == 0 else 'mild'}">{resting.eye_status.name}</span></td>
                    <td>{resting.eye_score}</td>
                    <td>{resting.eye_evidence.get('interpretation', 'N/A')}</td>
                </tr>
                <tr>
                    <td>👃 颊 (鼻唇沟)</td>
                    <td><span class="status-badge {'normal' if resting.cheek_score == 0 else ('severe' if resting.cheek_score == 2 else 'mild')}">{resting.cheek_status.name}</span></td>
                    <td>{resting.cheek_score}</td>
                    <td>{resting.cheek_evidence.get('interpretation', 'N/A')}</td>
                </tr>
                <tr>
                    <td>👄 嘴</td>
                    <td><span class="status-badge {'normal' if resting.mouth_score == 0 else 'mild'}">{resting.mouth_status.name}</span></td>
                    <td>{resting.mouth_score}</td>
                    <td>{resting.mouth_evidence.get('interpretation', 'N/A')}</td>
                </tr>
                <tr style="background: #f8f9fa; font-weight: bold;">
                    <td colspan="2">静态对称性总分</td>
                    <td colspan="2">{resting.total_weighted}/20 (原始分 {resting.total_raw} × 5)</td>
                </tr>
            </table>
            '''

        # 自主运动
        voluntary = getattr(sunnybrook_result, 'voluntary_movement', None)
        voluntary_html = '<p>无数据</p>'
        if voluntary:
            voluntary_html = f'''
            <table class="details-table">
                <tr>
                    <th>动作</th>
                    <th>运动等级</th>
                    <th>评分</th>
                    <th>功能百分比</th>
                </tr>
                <tr>
                    <td>🔼 Brow (抬眉)</td>
                    <td><span class="status-badge {'normal' if voluntary.brow_score == 5 else ('severe' if voluntary.brow_score <= 2 else 'mild')}">{voluntary.brow_level.name}</span></td>
                    <td>{voluntary.brow_score}/5</td>
                    <td>{voluntary.brow_evidence.get('function_pct', 'N/A'):.1f}%</td>
                </tr>
                <tr>
                    <td>👁 Eye Closure (闭眼)</td>
                    <td><span class="status-badge {'normal' if voluntary.eye_closure_score == 5 else ('severe' if voluntary.eye_closure_score <= 2 else 'mild')}">{voluntary.eye_closure_level.name}</span></td>
                    <td>{voluntary.eye_closure_score}/5</td>
                    <td>{voluntary.eye_closure_evidence.get('function_pct', 'N/A'):.1f}%</td>
                </tr>
                <tr>
                    <td>😊 Smile (微笑)</td>
                    <td><span class="status-badge {'normal' if voluntary.smile_score == 5 else ('severe' if voluntary.smile_score <= 2 else 'mild')}">{voluntary.smile_level.name}</span></td>
                    <td>{voluntary.smile_score}/5</td>
                    <td>{voluntary.smile_evidence.get('function_pct', 'N/A'):.1f}%</td>
                </tr>
                <tr>
                    <td>😤 Snarl (皱鼻)</td>
                    <td><span class="status-badge {'normal' if voluntary.snarl_score == 5 else ('severe' if voluntary.snarl_score <= 2 else 'mild')}">{voluntary.snarl_level.name}</span></td>
                    <td>{voluntary.snarl_score}/5</td>
                    <td>{voluntary.snarl_evidence.get('function_pct', 'N/A'):.1f}%</td>
                </tr>
                <tr>
                    <td>😗 Lip Pucker (撅嘴)</td>
                    <td><span class="status-badge {'normal' if voluntary.lip_pucker_score == 5 else ('severe' if voluntary.lip_pucker_score <= 2 else 'mild')}">{voluntary.lip_pucker_level.name}</span></td>
                    <td>{voluntary.lip_pucker_score}/5</td>
                    <td>{voluntary.lip_pucker_evidence.get('function_pct', 'N/A'):.1f}%</td>
                </tr>
                <tr style="background: #f8f9fa; font-weight: bold;">
                    <td colspan="2">自主运动总分</td>
                    <td colspan="2">{voluntary.total_weighted}/100 (原始分 {voluntary.total_raw} × 4)</td>
                </tr>
            </table>
            '''

        # 联带运动
        synkinesis = getattr(sunnybrook_result, 'synkinesis', None)
        synkinesis_html = '<p>无数据</p>'
        if synkinesis:
            synkinesis_html = f'''
            <div class="evidence-box">
                <strong>联带运动总分:</strong> {synkinesis.total_score}/15<br>
                <small>联带运动是指做一个动作时，其他面部区域出现不自主的运动。分数越低越好。</small>
            </div>
            '''

        # 综合评分公式
        composite = getattr(sunnybrook_result, 'composite_score', 0)
        resting_total = getattr(resting, 'total_weighted', 0) if resting else 0
        voluntary_total = getattr(voluntary, 'total_weighted', 0) if voluntary else 0
        synkinesis_total = getattr(synkinesis, 'total_score', 0) if synkinesis else 0

        formula_html = f'''
        <div class="evidence-box">
            <strong>综合评分计算公式:</strong><br>
            <code>综合评分 = 自主运动评分 - 静态对称性评分 - 联带运动评分</code><br>
            <code>{composite:.0f} = {voluntary_total} - {resting_total} - {synkinesis_total}</code>
        </div>
        '''

        return f'''
        <h3>1. 静态对称性评分 (Resting Symmetry)</h3>
        <p>评估静息状态下面部各部位的对称性。评分越低越好 (0-20分)。</p>
        {resting_html}

        <h3>2. 自主运动评分 (Voluntary Movement)</h3>
        <p>评估各标准动作的运动能力。评分越高越好 (20-100分)。</p>
        {voluntary_html}

        <h3>3. 联带运动评分 (Synkinesis)</h3>
        <p>评估运动时是否有不自主的联带运动。评分越低越好 (0-15分)。</p>
        {synkinesis_html}

        <h3>4. 综合评分</h3>
        {formula_html}
        '''

    def _build_hb_details(self, hb_result: Any) -> str:
        """构建House-Brackmann评分详情"""

        if not hb_result:
            return '<p>无House-Brackmann评分数据</p>'

        # 分支评估
        branches = []
        if hasattr(hb_result, 'temporal_branch') and hb_result.temporal_branch:
            branches.append(('颞支 (Temporal)', '额部运动', hb_result.temporal_branch))
        if hasattr(hb_result, 'zygomatic_branch') and hb_result.zygomatic_branch:
            branches.append(('颧支 (Zygomatic)', '眼部闭合', hb_result.zygomatic_branch))
        if hasattr(hb_result, 'buccal_branch') and hb_result.buccal_branch:
            branches.append(('颊支 (Buccal)', '中面部运动', hb_result.buccal_branch))
        if hasattr(hb_result, 'marginal_mandibular_branch') and hb_result.marginal_mandibular_branch:
            branches.append(('下颌缘支 (Marginal)', '口部运动', hb_result.marginal_mandibular_branch))

        branch_rows = ""
        for name, func, branch in branches:
            grade_str = f"Grade {branch.grade.value}"
            pct = branch.function_pct
            badge_class = 'normal' if pct >= 75 else ('mild' if pct >= 50 else ('moderate' if pct >= 25 else 'severe'))
            branch_rows += f'''
            <tr>
                <td>{name}</td>
                <td>{func}</td>
                <td><span class="status-badge {badge_class}">{grade_str}</span></td>
                <td>{pct:.1f}%</td>
                <td>{branch.description}</td>
            </tr>
            '''

        branches_html = f'''
        <table class="details-table">
            <tr>
                <th>神经分支</th>
                <th>主要功能</th>
                <th>分级</th>
                <th>功能百分比</th>
                <th>状态</th>
            </tr>
            {branch_rows}
        </table>
        '''

        # 分级标准
        grade_criteria = f'''
        <div class="evidence-box">
            <strong>House-Brackmann分级标准:</strong><br>
            <table class="details-table" style="margin-top: 10px;">
                <tr><th>分级</th><th>功能</th><th>描述</th></tr>
                <tr><td>I</td><td>100%</td><td>正常</td></tr>
                <tr><td>II</td><td>75-99%</td><td>轻度功能异常</td></tr>
                <tr><td>III</td><td>50-74%</td><td>中度功能异常</td></tr>
                <tr><td>IV</td><td>25-49%</td><td>中重度功能异常</td></tr>
                <tr><td>V</td><td>1-24%</td><td>重度功能异常</td></tr>
                <tr><td>VI</td><td>0%</td><td>完全麻痹</td></tr>
            </table>
        </div>
        '''

        # 临床特征
        features = getattr(hb_result, 'clinical_features', {})
        features_html = ""
        if features:
            features_html = '<div class="evidence-box"><strong>当前分级临床特征:</strong><ul>'
            for key, value in features.items():
                features_html += f'<li><strong>{key}:</strong> {value}</li>'
            features_html += '</ul></div>'

        return f'''
        <h3>各神经分支评估</h3>
        {branches_html}

        <h3>分级标准参考</h3>
        {grade_criteria}

        {features_html}
        '''

    def _build_actions_details(self, action_summaries: List[Dict[str, Any]]) -> str:
        """构建动作分析详情"""

        if not action_summaries:
            return '<p>无动作分析数据</p>'

        cards = ""
        for action in action_summaries:
            action_name = action.get('action_name', 'Unknown')
            highlights = action.get('highlights', {})

            # 指标显示
            indicators_html = '<div class="indicator-grid">'
            for key, value in highlights.items():
                if isinstance(value, (int, float)):
                    indicators_html += f'''
                    <div class="indicator-item">
                        <div class="name">{key}</div>
                        <div class="value">{value:.4f}</div>
                    </div>
                    '''
            indicators_html += '</div>'

            # 图片链接
            files = action.get('files', {})
            image_html = ""
            if files.get('peak_vis'):
                image_path = f"actions/{action_name}/{files['peak_vis']}"
                image_html = f'''
                <div class="image-container">
                    <a href="{image_path}" target="_blank">
                        <img src="{image_path}" alt="{action_name} peak frame">
                    </a>
                    <p><small>点击查看大图</small></p>
                </div>
                '''

            cards += f'''
            <div class="action-card">
                <div class="action-header collapsible">
                    <h4>{action_name}</h4>
                    <span class="status-badge normal">已分析</span>
                </div>
                <div class="collapse-content">
                    <div class="action-content">
                        {image_html}
                        <h5>关键指标</h5>
                        {indicators_html}
                    </div>
                </div>
            </div>
            '''

        return cards

    def _build_comparison(
            self,
            sunnybrook_result: Any,
            hb_result: Any,
            ground_truth: Dict[str, Any]
    ) -> str:
        """构建与真实标签的对比"""

        gt_hb = ground_truth.get('hb_grade')
        gt_sb = ground_truth.get('sunnybrook_score')
        gt_palsy = ground_truth.get('has_palsy')
        gt_side = ground_truth.get('palsy_side')

        pred_hb = hb_result.grade_roman if hasattr(hb_result, 'grade_roman') else 'N/A'
        pred_sb = sunnybrook_result.composite_score if hasattr(sunnybrook_result, 'composite_score') else 0
        pred_side = sunnybrook_result.affected_side if hasattr(sunnybrook_result, 'affected_side') else 'none'

        # 判断匹配
        hb_match = str(gt_hb) == str(pred_hb) if gt_hb else True
        sb_match = abs(float(gt_sb or 0) - pred_sb) < 15 if gt_sb else True
        side_match = gt_side == pred_side if gt_side else True

        return f'''
        <section class="comparison-section">
            <h2>🔍 与真实标签对比</h2>
            <div class="comparison-grid">
                <div class="comparison-item {'match' if hb_match else 'mismatch'}">
                    <h4>House-Brackmann</h4>
                    <p>预测: <strong>{pred_hb}</strong></p>
                    <p>真实: <strong>{gt_hb or 'N/A'}</strong></p>
                    <p>{'✓ 匹配' if hb_match else '✗ 不匹配'}</p>
                </div>
                <div class="comparison-item {'match' if sb_match else 'mismatch'}">
                    <h4>Sunnybrook</h4>
                    <p>预测: <strong>{pred_sb:.0f}</strong></p>
                    <p>真实: <strong>{gt_sb or 'N/A'}</strong></p>
                    <p>{'✓ 接近' if sb_match else '✗ 差异较大'}</p>
                </div>
                <div class="comparison-item {'match' if side_match else 'mismatch'}">
                    <h4>患侧</h4>
                    <p>预测: <strong>{pred_side}</strong></p>
                    <p>真实: <strong>{gt_side or 'N/A'}</strong></p>
                    <p>{'✓ 匹配' if side_match else '✗ 不匹配'}</p>
                </div>
            </div>
        </section>
        '''


# ============ 便捷函数 ============

def generate_clinical_report(
        output_dir: str,
        examination_id: str,
        sunnybrook_result: Any,
        hb_result: Any,
        action_summaries: List[Dict[str, Any]],
        meta: Dict[str, Any] = None,
        ground_truth: Dict[str, Any] = None
) -> str:
    """
    便捷函数：生成临床评分报告
    """
    generator = ClinicalReportGenerator(output_dir)
    return generator.generate_report(
        examination_id, sunnybrook_result, hb_result,
        action_summaries, meta, ground_truth
    )


if __name__ == "__main__":
    print("Report Generator - 测试")

    # 测试生成空报告
    from sunnybrook_scorer import SunnybrookScorer, SunnybrookResult
    from house_brackmann_scorer import HouseBrackmannScorer

    # 模拟数据
    neutral = {
        'eye_area_ratio': 0.85,
        'nlf_length_ratio': 0.88,
        'oral_height_diff': -0.02,
    }

    actions = {
        'RaiseEyebrow': {'lift_ratio': 0.7, 'function_pct': 70},
        'CloseEyeSoftly': {'closure_ratio': 0.8, 'both_complete_closure': True},
        'Smile': {'oral_excursion_ratio': 0.75},
        'ShrugNose': {'nostril_flare_ratio': 0.8},
        'LipPucker': {'pucker_symmetry': 0.85},
    }

    sb_scorer = SunnybrookScorer()
    sb_result = sb_scorer.compute_score(neutral, actions)

    hb_scorer = HouseBrackmannScorer()
    hb_result = hb_scorer.compute_score(neutral, actions)

    action_summaries = [
        {'action_name': 'NeutralFace', 'highlights': neutral, 'files': {}},
        {'action_name': 'Smile', 'highlights': actions['Smile'], 'files': {}},
    ]

    meta = {
        'patient_id': 'TEST001',
        'capture_datetime': '2025-01-01 10:00:00',
        'source': 'test'
    }

    ground_truth = {
        'has_palsy': True,
        'palsy_side': 'left',
        'hb_grade': 'III',
        'sunnybrook_score': 65
    }

    report_path = generate_clinical_report(
        output_dir='/tmp/test_report',
        examination_id='TEST001',
        sunnybrook_result=sb_result,
        hb_result=hb_result,
        action_summaries=action_summaries,
        meta=meta,
        ground_truth=ground_truth
    )

    print(f"报告已生成: {report_path}")