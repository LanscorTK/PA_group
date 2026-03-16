const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, LevelFormat, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, Header, Footer, PageBreak
} = require('docx');
const fs = require('fs');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };
const headerFill = { fill: "E8EEF4", type: ShadingType.CLEAR };

function cell(text, bold = false, shade = null, colSpan = 1) {
  return new TableCell({
    borders: cellBorders,
    shading: shade ? shade : { fill: "FFFFFF", type: ShadingType.CLEAR },
    columnSpan: colSpan,
    children: [new Paragraph({
      children: [new TextRun({ text, bold, size: 20 })]
    })]
  });
}

function hcell(text) {
  return cell(text, true, headerFill);
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    ...opts,
    children: [new TextRun({ text, size: 24, ...(opts.run || {}) })],
  });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 24 })]
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F3864", font: "Arial" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5090", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "MSIN0097 Predictive Analytics — Group Coursework 2025-26", size: 18, color: "666666" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 20, color: "666666" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 20, color: "666666" }),
            new TextRun({ text: " of ", size: 20, color: "666666" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20, color: "666666" })
          ]
        })]
      })
    },
    children: [

      // ── Title Block ──────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 480, after: 120 },
        children: [new TextRun({ text: "Practical Exploration and Benchmarking", bold: true, size: 40, font: "Arial", color: "1F3864" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "AI Coding Agents for End-to-End Data Science Workflows", size: 28, font: "Arial", color: "444444" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        children: [new TextRun({ text: "MSIN0097 Predictive Analytics 2025-26  |  UCL", size: 22, font: "Arial", color: "888888", italics: true })]
      }),

      // ── Section 1: Benchmark Overview ────────────────────────────────────────
      h1("1. Benchmark Overview"),

      p("Three AI coding agents — Claude Code, OpenAI Codex, and Antigravity — were benchmarked against an identical end-to-end data science pipeline. All agents operated on the same dataset (UK Participation Survey 2024-25; 34,378 rows, 11 features), the same binary classification target (CARTS_NET: arts engagement vs. non-engagement), and the same frozen eight-step prompt protocol (Steps 0-7). No manual intervention was permitted except to recover from hard crashes; any deviation was logged."),

      p("The benchmark was designed to be fair and comparable across agents in three ways. First, the dataset and target were fixed before any agent was run, so no agent received a task tailored to its strengths. Second, prompts were delivered in a fixed sequence with no agent-specific wording. Third, evaluation metrics were defined in the protocol, not chosen post-hoc. This setup allowed head-to-head comparison of agent outputs on identical inputs."),

      p("The primary goal was not to identify which agent produced the highest accuracy, but to assess end-to-end data science workflow capability: could each agent ingest, clean, explore, model, and document a realistic ML task with methodological discipline? Key dimensions monitored included execution stability, missingness handling coherence, class-imbalance awareness, evaluation rigour, code reproducibility, and documentation quality."),

      // ── Section 2: Task Coverage ─────────────────────────────────────────────
      h1("2. Task Coverage and Task Specification"),

      p("Tasks were grouped into four blocks, each targeting a distinct stage of the data science workflow."),

      h2("Block 1: Data Preparation"),

      p("Task spec: agents were required to load the tab-delimited dataset, verify shape (34,378 rows × 11 columns), confirm column types, and handle coded missing values per variable (e.g. −3 = Not applicable; 3 = No & Missing). The cleaned dataset was to be saved as a named DataFrame (participation_clean)."),

      p("Success criteria: all 11 variables present, coded invalids removed or recoded, target variable restricted to values 1 and 2 only, no data leakage into test set."),

      p("Evidence collected: final row counts, COHAB recoding strategy, missingness summary CSVs (Codex). Failure modes monitored: silent row dropping, over-aggressive purging, inconsistent handling of high-missingness variables. Reproducibility note: row counts diverged across agents (Claude Code: 29,848; Codex: ~34,338; Antigravity: ~33,000+), reflecting genuinely different design decisions rather than protocol deviations."),

      h2("Block 2: Analytical Exploration"),

      p("Task spec: agents were to generate exploratory data analysis (EDA) with visualisations and written insights, including class distribution, feature distributions, and feature-target relationships."),

      p("Success criteria: minimum 3 plots with interpretive commentary; class imbalance acknowledged explicitly; at least one feature-target relationship visualised."),

      p("Evidence collected: plot files (Claude Code: 5 plots; Codex: 3 plots; Antigravity: 3 plots + 4 inline). Failure modes monitored: omitting class imbalance from commentary, generating plots without interpretation. Reproducibility check: all plots were saved to evidence folders; inline outputs are present in notebooks."),

      h2("Block 3: Modelling and Improvement"),

      p("Task spec: agents were to build a Logistic Regression baseline with an imbalance-aware evaluation harness (stratified split, macro-averaged metrics), then attempt improvement via LR tuning and XGBoost, comparing results and selecting a final model with a documented rationale."),

      p("Success criteria: baseline evaluated on validation set only; test set used only once for the final comparison; model selection justified against defined criteria; XGBoost attempted."),

      p("Evidence collected: model comparison CSVs, tuning results, scoring frameworks. Failure modes monitored: test-set leakage, XGBoost failure to detect minority class, metric inconsistency. All three agents attempted the full modelling block; Codex's XGBoost produced minority-class recall of 0.077, essentially failing to detect non-engagers."),

      h2("Block 4: Reproducibility and Communication"),

      p("Task spec: agents were to produce a requirements.txt, a README with run instructions (seed=42, relative paths), and a non-technical stakeholder report (~400 words) for a government arts department."),

      p("Success criteria: another user can reproduce the notebook end-to-end using the README; the stakeholder report avoids jargon and frames findings in policy terms."),

      p("Evidence collected: requirements.txt, README.md, and report files for all three agents. Failure modes monitored: absolute paths, missing seeds, reports that re-state model metrics rather than policy implications. Antigravity used an unusual script-based notebook assembly workflow (add_cell.py + separate .py/.md files), which adds a reproducibility complexity not present in the other two agents."),

      // ── Section 3: Evaluation Approach ───────────────────────────────────────
      h1("3. Evaluation Approach"),

      p("Evaluation was structured around two complementary criteria sets."),

      p("Objective criteria assessed whether the task ran and whether outputs met the specification: did the notebook execute top-to-bottom without errors? Were all required DataFrames, files, and plots produced? Were the correct train/validation/test splits applied? These were binary pass/fail checks across all four task blocks."),

      p("Quality criteria assessed the methodological soundness of agent decisions. For data preparation: was the missingness strategy coherent and justified, or arbitrary? For EDA: did plots address class imbalance and feature-target relationships, or were they superficial? For modelling: were evaluation metrics appropriate for a severely imbalanced dataset (~91-93% engaged class)? Was the evaluation harness leakage-free? For documentation: did the stakeholder report communicate actionable findings, or simply restate metrics?"),

      p("Particular emphasis was placed on three cross-cutting properties. Class-imbalance awareness: all four evaluation blocks were assessed for whether agents explicitly recognised the dominant engaged class and adjusted methodology accordingly (e.g. stratified splits, recall-weighted metrics). Evaluation discipline: test-set use was audited to ensure it occurred only at the final model comparison stage. Metric consistency: agents used different averaging conventions (macro vs. minority-class), making standardisation a prerequisite for any cross-agent comparison."),

      // ── Section 4: Main Findings ──────────────────────────────────────────────
      h1("4. Main Findings"),

      h2("Overall Stability"),

      p("Claude Code was the most methodologically stable agent overall. It completed all eight steps without protocol deviations, produced the most evidence artefacts (5 plots, 2 summary CSVs), and applied the most structured model selection framework (weighted scoring across recall, ROC-AUC, and interpretability). Codex was equally stable in execution but produced a weaker modelling outcome due to XGBoost failure. Antigravity completed all steps but introduced workflow idiosyncrasy through its script-based notebook construction, making direct replication harder to verify."),

      h2("Strengths by Area"),

      new Table({
        columnWidths: [2200, 2380, 2380, 2400],
        margins: { top: 80, bottom: 80, left: 160, right: 160 },
        rows: [
          new TableRow({
            tableHeader: true,
            children: [hcell("Area"), hcell("Claude Code"), hcell("Codex"), hcell("Antigravity")]
          }),
          new TableRow({ children: [hcell("Cleaning"), cell("Tiered strategy; documented"), cell("Conservative; max row retention"), cell("Balanced approach")] }),
          new TableRow({ children: [hcell("EDA"), cell("Strongest (5 plots, feature-target analysis)"), cell("Adequate (3 plots)"), cell("Adequate (3 plots + inline)")] }),
          new TableRow({ children: [hcell("Modelling"), cell("Coherent; LR selected with rationale"), cell("LR selected; XGBoost failed"), cell("XGBoost selected; strong recall claimed")] }),
          new TableRow({ children: [hcell("Documentation"), cell("Academic/policy balanced"), cell("Risk-modelling framing"), cell("Policy-advocacy framing")] }),
        ]
      }),

      new Paragraph({ spacing: { after: 200 }, children: [] }),

      h2("Task Types That Exposed Differences Most Clearly"),

      p("Missingness handling was the single most differentiating task. Claude Code dropped 13.1% of rows, Codex dropped fewer than 40, and Antigravity sat between them. The decision directly affected training data volume and potential selection bias — yet no agent provided a formal justification for its chosen threshold. This is a significant gap in methodological transparency."),

      p("XGBoost behaviour revealed the most dramatic quality divergence: Codex's model had a minority-class recall of 0.077 (near random), while Antigravity claimed ~0.71 recall from its XGBoost. Given that both agents used the same algorithm on datasets derived from the same source, this gap almost certainly reflects differences in class-weight configuration or hyperparameter defaults, but neither agent logged the specific configuration that caused the divergence."),

      h2("Common Failure Modes"),

      bullet("Metric inconsistency: Claude Code used macro-averaged PR-AUC, Codex used minority-class PR-AUC. These are not directly comparable; cross-agent comparison of reported values is misleading without standardisation."),
      bullet("Shallow missingness justification: all agents recoded or dropped values without formally testing the impact of their chosen strategy on class balance or downstream model performance."),
      bullet("XGBoost configuration opacity: no agent logged hyperparameter grids or class-weight settings in a way that would allow independent replication of the tuned model."),
      bullet("Evaluation discipline partially maintained: all agents used stratified splits and held out a test set, but metric averaging conventions were inconsistent even within agents across the validation and test phases."),

      p("Failures were primarily detected through cross-agent comparison of reported metrics and inspection of run logs and evidence CSVs. No agent corrected its own metric convention during execution."),

      // ── Section 5: Short Takeaway ─────────────────────────────────────────────
      h1("5. Short Takeaway"),

      p("This benchmark demonstrates that current AI coding agents can complete a realistic end-to-end data science pipeline reliably, but with significant variation in methodological rigour. All three agents cleared the execution bar; the meaningful differences emerged in the quality of decisions made along the way — particularly around missingness, metric selection, and model configuration transparency."),

      p("The most important limitation of this benchmark is that it was run once per agent, without replication. Given that agent outputs can vary across runs, the observed differences may partially reflect stochastic variation rather than stable capability gaps. A multi-run protocol with fixed random seeds for both the data split and the agent session would strengthen reproducibility claims."),

      p("These findings motivate the comparative analysis that follows, which applies a standardised evaluation framework across agents to enable fairer metric-level comparison."),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("outputs/practical_exploration_benchmarking.docx", buffer);
  console.log("Done: outputs/practical_exploration_benchmarking.docx");
});
