---
name: research_analysis
description: >-
  Provides structured methodology for research, evidence gathering, source evaluation,
  and structured synthesis for ResearcherAgent and SupervisorAgent in the AB SMART AI system.
version: 1.0.0
---

# Research Analysis Skill

This skill guides the structured exploration, information extraction, factual verification,
and evidence synthesis for research-oriented agent tasks.

---

## When to Use This Skill

Activate this skill whenever:
- An agent is assigned a research, analysis, survey, or competitive comparison task.
- Evidence, citations, or verifiable data points are required before planning or execution.
- Complex technical topics require structured decomposition and evidence synthesis.

---

## Execution Workflow

1. **Scope Clarification**:
   - Determine core research questions, target entities, and constraints.
   - Identify whether real external web tools or mock information sources are permitted.

2. **Information Gathering**:
   - Use authorized read-only tools (`read_file`, `web_search`, external app lookup).
   - Collect concrete data points, metrics, and architecture patterns.

3. **Evidence Evaluation**:
   - Verify accuracy and check for bias or hallucinated data.
   - Tag each piece of evidence with its source origin and trust level.

4. **Structured Synthesis**:
   - Format results into clear sections:
     - Executive Overview
     - Key Findings & Comparison Table
     - Evidence & Verification
     - Technical Limitations

---

## Input & Output Expectations

- **Input**: Research topic, target domains, specific questions.
- **Output**: Markdown or JSON structured research report with citations and identified risks.
