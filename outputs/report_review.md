# Report Review: Report_Draft_0318_v2.1.docx

Reviewed against MSIN0097 coursework brief. Items sorted by severity.

---

## CRITICAL ISSUES (fix before submission)

### 1. Word count on cover page is wrong
The cover page says **1988** but the body text (Sections 1–4 including tables) is approximately **2,300+ words** by naive count. You need to:
- Check the actual word count in Word (select Sections 1–4 only, use Review > Word Count or the status bar)
- Update the cover page number to match
- If over 2,000, trim. The easiest places to cut are Section 2.2 (Block descriptions are verbose) and the overlapping opening of Section 3 (see issue #6 below)

### 2. Empty pages between Section 3 and Section 4
There are **17+ blank paragraph/page breaks** between the end of Section 3 (after Table 6) and the start of Section 4. These appear as empty pages in the document. Delete all the blank paragraphs/page breaks between Section 3's last line and the "4 Reflection and conclusion" heading. A single page break is sufficient.

### 3. Empty heading between Section 2.3 and Section 3
There is a stray empty `#` heading (line 278 in the markdown) between Section 2.3 and Section 3. This likely renders as a blank heading or orphan page break in Word. Remove it.

### 4. Anonymity check
The brief states: *"Anonymity: required — your names should not appear anywhere on your submission."* Your cover page includes names and student numbers. **Check with your module team** whether the cover page is an exception (it often is for group coursework), but if not, you may need to replace names with student numbers only, or remove them and use only "Team 16."

---

## IMPORTANT ISSUES (strongly recommended fixes)

### 5. Spelling consistency — American vs British English
The report mixes American and British spelling. Pick one and stick with it. Since the majority of the report uses British English, change these American spellings:
- Line 196: "standardized" → "standardised"
- Line 276: "organized" → "organised"
- Line 298: "behavior" → "behaviour"
- Line 300: "judgment" → "judgement" (or the other way around, but be consistent with line 516 which uses "judgement")

### 6. Section 2.3 and Section 3 overlap
Section 2.3 (Results and findings) and Section 3 (Comparative analysis) make several of the same points:
- Both say all agents converged on tuned logistic regression
- Both say intermediate decisions mattered more than final model choice
- Both discuss missingness divergence

This overlap uses word budget. Consider either:
- **Option A:** Shorten Section 2.3 to a brief summary paragraph (~3 sentences: all completed, strengths differed, details in Section 3) and move the substantive analysis entirely to Section 3
- **Option B:** Keep Section 2.3 as the high-level summary but remove the repeated sentences from Section 3's opening paragraphs

This could save 50–80 words, which helps if you're over 2,000.

### 7. Forward reference to Table 6
Section 2.3 (line 260) says "as shown in Table 6" but Table 6 appears at the very end of Section 3, several pages later. Either:
- Change to "as summarised in Table 6 (Section 3)" to make the forward reference explicit
- Or move the reference to a table that's closer, or restructure

### 8. Block 1 formatting inconsistency
Block 1 uses blockquote formatting (`>`) while Blocks 2–4 use bold text. Make all four blocks consistent — likely just remove the `>` from Block 1 and make it bold like the others.

---

## MINOR SUGGESTIONS (nice to have)

### 9. Table caption style
Some table captions use italics (*Table 1. ...*) and some are plain. Make all consistent. Convention is italicised captions above tables.

### 10. Consider adding the GitHub repo link more visibly
The repo link is currently in footnote 1, which is good and word-count-free. You could also add it to the appendix introduction or the cover page if you want it more visible to the marker.

### 11. "Conclusion" sub-heading in Section 1
The lit review ends with a bold "Conclusion" sub-heading. This is fine, but since no other section has an explicit "Conclusion" sub-heading, you could consider removing the label and just having the final paragraph flow naturally. Minor style point.

### 12. Table 6 scale
The +/++/+++ scale explanation appears after Table 6 as a note. Consider placing it as an italicised caption line immediately below the table title (before the table body) so the reader understands the scale before reading the data.

---

## CHECKLIST AGAINST COURSEWORK BRIEF

| Requirement | Status | Notes |
|-------------|--------|-------|
| 2,000 words max | ⚠️ CHECK | Cover says 1988 but actual count appears higher. Verify in Word. |
| Literature review (30%) | ✅ | 16 papers cited, covers reasoning/tool-use, multi-agent, benchmarks, challenges. Strong. |
| At least 10 academic papers | ✅ | 16 references. |
| Key themes discussed | ✅ | Planning/tool-use, verification, human-in-loop, reproducibility, failure modes all covered. |
| Taxonomies/approaches identified | ✅ | ReAct, Toolformer, RAG, multi-agent workflows, self-reflection. |
| Consistent referencing style | ✅ | Harvard style throughout. |
| Practical exploration (40%) | ✅ | 3 agents, 8 tasks across 4 blocks, clear spec and evidence. |
| At least 3 agent tools | ✅ | Claude Code, Codex, Antigravity. |
| At least 4 task types | ✅ | All 8 task types covered (ingestion, EDA, baseline, tuning, comparison, packaging, documentation, plus missingness). |
| Task specs and success criteria | ✅ | Defined in Section 2.2 blocks + Appendix A prompts. |
| Evidence captured | ✅ | CSVs, JSONs, plots, run logs referenced; appendices included. |
| Failures recorded | ⚠️ PARTIAL | Default threshold failure mode described well. Could mention the Codex indentation fix and XGB deprecation warning more explicitly as recorded failures. |
| Reproducibility addressed | ✅ | Fixed seeds, pinned dependencies, run instructions. |
| Comparative analysis (20%) | ✅ | 6 tables, consistent framework, failure modes identified. |
| Correctness | ✅ | Discussed (all pipelines ran). |
| Statistical validity | ✅ | Splits, metrics, leakage discipline discussed. |
| Reproducibility | ✅ | Seeds, requirements, README assessed. |
| Code quality | ✅ | Table 5 covers evidence trails and tuning configs. |
| Efficiency | ✅ | Tuning config counts compared. |
| Safety/compliance | ✅ | "No major safety or compliance issues observed." |
| At least 1 table or figure | ✅ | 6 tables. |
| Reflection and conclusion (10%) | ✅ | Findings synthesis, lessons, 4-point playbook. |
| Playbook with workflow/checklist/failures/when-not-to-use | ✅ | All four covered as numbered items. |
| Appendices | ✅ | A: Prompts, B: Run logs. |
| Bibliography with 10+ refs | ✅ | 16 references. |
| Repo link (recommended) | ✅ | Footnote 1 links to GitHub. |
| Anonymous submission | ⚠️ CHECK | Names on cover page — verify if this is acceptable. |

---

## SUMMARY OF ACTIONS

**Must fix (before submission):**
1. Remove blank pages/paragraphs between Sections 3 and 4
2. Remove stray empty heading between Sections 2.3 and 3
3. Verify and update word count on cover page
4. Check anonymity requirement with module team

**Strongly recommended:**
5. Fix American/British spelling inconsistencies (4 words)
6. Reduce Section 2.3/3 overlap to save words if over limit
7. Fix Block 1 formatting to match Blocks 2–4
8. Clarify the forward reference to Table 6

---
---

# Second Review: Report_Draft_0318_v2.2.docx

Compared against v2.1 review and the coursework brief.

## What was fixed since v2.1

| v2.1 Issue | Status in v2.2 |
|------------|----------------|
| #2 Empty pages between Sections 3 and 4 | ✅ FIXED — blank paragraphs removed |
| #3 Stray empty heading between 2.3 and 3 | ✅ FIXED — removed |
| #6 Section 2.3 / 3 overlap | ✅ FIXED — Section 2.3 is now a concise 3-sentence bridge |
| #7 Forward reference to Table 6 | ✅ FIXED — removed with the shortened Section 2.3 |

## Remaining issues from v2.1

### 1. Word count on cover page — STILL WRONG
Cover page still says **1988**. Estimated body word count is now ~2,100–2,200 (the trim from Section 2.3 saved ~100 words). You must verify in Word and update this number.

### 2. Spelling inconsistencies — STILL PRESENT (3 words)
- Line 196: **"standardized"** → should be "standardised"
- Line 263: **"organization"** → should be "organisation"
- Line 286: **"behavior"** → should be "behaviour"

The rest of the report uses British English ("organised", "specialisation", "judgement", "optimised"). Fix these three words to match.

### 3. Block 1 formatting — STILL INCONSISTENT
Block 1 still uses blockquote (`>`) formatting while Blocks 2–4 use bold. Remove the `>` from Block 1 so it matches.

### 4. Anonymity — STILL NEEDS CHECKING
Names and student numbers remain on the cover page. Check with module team.

## New issues in v2.2

### 5. CRITICAL — Redundant phrase in Section 3 opening sentence
The first sentence of Section 3 reads:

> "Completing the full workflow **with coded missing-value structures and routed responses** already represents a meaningful result given the survey's **coded missing-value structure and the need to preserve routed responses**."

The same concept ("coded missing-value structure" and "routed responses") appears **twice in one sentence**. This looks like a merge error when the suggested edit was applied. Fix by replacing with:

> "Completing the full workflow already represents a meaningful result given the survey's coded missing-value structure and the need to preserve routed responses."

### 6. Stray asterisk artifact before Section 4
Between the Table 6 note and Section 4, there is a stray `*\*` (an escaped asterisk). This likely renders as a lone asterisk or formatting artifact in Word. Delete it.

---

## Updated summary of actions for v2.2

**Must fix before submission:**
1. Fix the redundant sentence in Section 3 opening (issue #5 above — copy-paste the corrected version)
2. Delete the stray `*\*` between Table 6 and Section 4
3. Verify actual word count in Word and update cover page
4. Fix 3 American spellings: "standardized", "organization", "behavior"

**Recommended:**
5. Fix Block 1 formatting (`>` → bold, matching Blocks 2–4)
6. Check anonymity with module team

**Overall assessment:** The report is in strong shape. The Section 2.3/3 restructuring reads cleanly. All brief requirements are met (16 references, 3 agents, 8 tasks, 6 tables, playbook, appendices, repo link). The remaining issues are minor — fixing items 1–4 above should take under 5 minutes.
