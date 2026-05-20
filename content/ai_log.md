# AI Usage Log

Every prompt used in this project, what we used the output for, and what was changed. This log is part of the deliverable and demonstrates responsible AI usage with human oversight.

## Format

Each entry uses the same five fields. Add a new entry under the relevant phase as you go — don't try to reconstruct it at the end.

```
### [Date] — [Brief title]
- **Phase:** Research / EDA / Modelling / App / Presentation
- **Tool & Prompt:** [Which AI, what we asked]
- **Why we used AI:** [What the AI helped with that we couldn't do faster ourselves]
- **What we kept:** [Specific output we used, mostly unchanged]
- **What we modified:** [What we rewrote, restructured, or fact-checked]
- **What we rejected:** [Suggestions we did not use, and why]
```

---

## Research Phase

### 2026-05-20 — Research section synthesis
- **Phase:** Research
- **Tool & Prompt:** Claude. "Synthesise the existing Research_Findings.docx with the eight academic papers into one research section. Ensure inline citations and address all requirements in student_task.pdf."
- **Why we used AI:** To merge our existing notes with several peer-reviewed papers into a coherent, cited section faster than reading every paper end-to-end.
- **What we kept:** Section structure (GLMs → Interpretability → WoE/IV → Metrics → Regulatory). The metrics section retained the core explanations from our prior notes. Citations and the WoE/IV theoretical framing (Sudjianto & Burakov, 2025; Sharma et al., 2023) were AI-sourced.
- **What we modified:** Tightened the regulatory section, removed inline tool attributions from the existing draft (those moved here, where they belong). Mapped regulatory concerns to actual loan-book columns ourselves.
- **What we rejected:** A suggestion to lengthen the GLM section with deeper mathematical derivations — kept it accessible for the non-technical audience the task targets.

### [Add next research entry here]

---

## EDA Phase

*Add entries as we use AI for EDA work.*

---

## Modelling Phase

*Add entries as we use AI for modelling work.*

---

## App Development Phase

### 2026-05-20 — Streamlit app shell scaffolding
- **Phase:** App
- **Tool & Prompt:** Claude. "Build the Streamlit app shell with the agreed file structure: views, content, core, components, utils."
- **Why we used AI:** Scaffolding 20+ Python files with consistent conventions, caching, and stubs is mechanical work where AI is faster than typing manually.
- **What we kept:** Directory structure, navigation via `st.navigation`, cached data and content loaders, reusable chart and KPI components.
- **What we modified:** [To be filled in as we iterate on the shell.]
- **What we rejected:** [To be filled in as we iterate.]

---

## Presentation Phase

*Add entries when working on slides and the video script.*
