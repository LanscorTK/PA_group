const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat
} = require('docx');
const fs = require('fs');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

function hCell(text, width) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "D9E1F2", type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, size: 20 })]
    })]
  });
}

function dCell(text, width, bold = false, align = AlignmentType.CENTER) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold, size: 20 })]
    })]
  });
}

function body(text, spaceAfter = 160) {
  return new Paragraph({
    spacing: { after: spaceAfter },
    children: [new TextRun({ text, size: 24 })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: 28 })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, italics: true, size: 24 })]
  });
}

// Table 1: Final model performance comparison
const perfTable = new Table({
  columnWidths: [2000, 1840, 1840, 1840, 1840],
  margins: { top: 80, bottom: 80, left: 150, right: 150 },
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        hCell("Agent", 2000),
        hCell("F2", 1840),
        hCell("Recall", 1840),
        hCell("Precision", 1840),
        hCell("ROC-AUC", 1840),
      ]
    }),
    new TableRow({ children: [dCell("Claude Code", 2000, false, AlignmentType.LEFT), dCell("0.452", 1840), dCell("0.769", 1840), dCell("0.170", 1840), dCell("0.829", 1840)] }),
    new TableRow({ children: [dCell("Codex", 2000, false, AlignmentType.LEFT), dCell("0.464", 1840), dCell("0.713", 1840), dCell("0.194", 1840), dCell("0.831", 1840)] }),
    new TableRow({ children: [dCell("Antigravity", 2000, false, AlignmentType.LEFT), dCell("0.488", 1840), dCell("0.735", 1840), dCell("0.208", 1840), dCell("0.853", 1840)] }),
  ]
});

// Table 2: Missingness comparison
const missTable = new Table({
  columnWidths: [2200, 2580, 2380, 2200],
  margins: { top: 80, bottom: 80, left: 150, right: 150 },
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        hCell("Agent", 2200),
        hCell("Rows after EDA filter", 2580),
        hCell("Rows after cleaning", 2380),
        hCell("Drop rate", 2200),
      ]
    }),
    new TableRow({ children: [dCell("Claude Code", 2200, false, AlignmentType.LEFT), dCell("34,338", 2580), dCell("29,073", 2380), dCell("15.3%", 2200)] }),
    new TableRow({ children: [dCell("Codex", 2200, false, AlignmentType.LEFT), dCell("34,338", 2580), dCell("28,995", 2380), dCell("15.5%", 2200)] }),
    new TableRow({ children: [dCell("Antigravity", 2200, false, AlignmentType.LEFT), dCell("34,338", 2580), dCell("24,867", 2380), dCell("27.6%", 2200)] }),
  ]
});

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    children: [

      // Section title
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 0, after: 200 },
        children: [new TextRun({ text: "Practical Exploration and Benchmarking", bold: true, size: 32, font: "Arial" })]
      }),

      // ── 1. Benchmark Overview ──────────────────────────────────────────────
      h2("Benchmark Overview"),

      body("This section reports the practical benchmarking of three AI coding agents — Claude Code (Anthropic), Codex (OpenAI), and Antigravity — on a shared data science pipeline. Each agent worked independently on the same task: predicting arts under-engagement using the 2024-25 UK Participation Survey. The dataset contained 34,378 respondents and 15 predictor variables drawn from DCMS data, with a binary target (CARTS_NET) indicating physical arts engagement in the preceding year. Approximately 91% of respondents were classified as engaged, making the minority class — under-engaged individuals — the target of policy interest."),

      body("Each agent received an identical sequence of eight prompts, from dataset ingestion to a non-technical report for a government arts department. The protocol was frozen before any agent ran: the same random seed (42), the same train/validation/test split proportions (70/15/15), and the same evaluation priorities applied throughout. No manual code changes were permitted except to correct clear execution errors. The objective was not simply to compare final model accuracy, but to assess how each agent navigated the full workflow — including decisions around missing data, class imbalance, and documentation — that a practitioner would face on real survey data."),

      // ── 2. Task Coverage ──────────────────────────────────────────────────
      h2("Task Coverage and Specification"),

      body("The pipeline was organised into four task blocks, each with its own specification, success criteria, and monitored failure modes."),

      h3("Data Preparation"),

      body("Agents were required to load the dataset, verify schema and column types, and apply a tiered missingness strategy guided by the data dictionary. The survey uses coded values rather than standard NaN markers: -3 indicates 'not applicable' (a routing skip), while -4, -5, 997, and 999 indicate non-informative responses such as refusals or 'don't know'. Success required producing a clean dataframe without leaking target labels into preprocessing. The primary failure modes monitored were overly aggressive row removal (which can introduce selection bias by discarding respondents with systematically incomplete profiles) and treating all coded values as equivalent (which conflates informative skips with uninformative refusals)."),

      body("Reproducibility was enforced through seed-fixed splits and documented per-variable decisions. All three agents recorded their missingness strategies, though the level of detail varied across run logs and evidence files."),

      h3("Analytical Exploration"),

      body("Each agent was asked to produce visualisations that revealed the class imbalance and the relationships between predictor variables and the binary target. The minimum bar was set at the target distribution and at least two feature-level plots. A higher-quality output would connect those distributions to the policy framing — identifying which demographic or socioeconomic groups were most associated with under-engagement. The failure mode monitored here was superficial output: charts that print frequency counts without interpreting what they mean for intervention design."),

      h3("Modelling and Improvement"),

      body("The modelling block ran in two phases. First, a baseline logistic regression was trained and evaluated on the validation set at the default decision threshold. Second, agents tuned both logistic regression and XGBoost, optimising threshold and hyperparameters on the validation set only, before evaluating all models once on the held-out test set. F2 score (beta = 2) was specified as the primary metric, which penalises missed under-engaged individuals more heavily than false positives. The critical failure modes were: using the default 0.5 threshold on a 9%-minority dataset (which yields recall near zero), running the test set during tuning (data leakage), and selecting a model on accuracy rather than recall-weighted metrics."),

      h3("Reproducibility and Communication"),

      body("Each agent was asked to package the experiment with a requirements.txt, a README with run instructions, and a policy-facing report of approximately 400 words. Success required the notebook to run top-to-bottom without intervention and the written report to accurately reflect the analysis — not describe a different study. Evidence artefacts were also expected: metric tables, tuning logs, and EDA plots stored in a dedicated evidence folder."),

      // ── 3. Evaluation Approach ────────────────────────────────────────────
      h2("Evaluation Approach"),

      body("Evaluation operated at two levels. At the objective level, the question was straightforward: did the step run, and did the output match the specification? This covered whether the correct dataframe shape was produced, whether the evaluation harness applied the right metrics, and whether model artefacts were saved correctly."),

      body("At the quality level, the focus shifted to the reasoning behind decisions. On missingness, the question was whether the agent read and applied the data dictionary or simply dropped all non-positive values. On EDA, the question was whether plots were chosen to illuminate the under-engagement problem or generated by default. On modelling, the key criteria were class imbalance awareness (use of recall-weighted metrics and threshold tuning), leakage discipline (test set reserved for final evaluation only), metric consistency (the same metrics applied to all models so comparisons are meaningful), and documentation alignment (the written report accurately reflects what the model produced)."),

      body("Accuracy was explicitly deprioritised as an evaluation criterion. On a dataset where 91% of respondents are engaged, a model that predicts 'engaged' for everyone achieves 91% accuracy while identifying zero under-engaged individuals. All meaningful comparisons therefore used F2, recall, balanced accuracy, and ROC-AUC."),

      // ── 4. Main Findings ──────────────────────────────────────────────────
      h2("Main Findings"),

      body("All three agents completed the full pipeline. Given that the dataset uses a non-standard coded-value scheme for missing data, this is not trivial: a default pandas dropna() call would silently remove tens of thousands of valid rows, and several variables require interpretation of the data dictionary before a sensible cleaning strategy can be designed. That all three agents navigated this without protocol errors is a meaningful result in itself."),

      body("The sharpest differences emerged in data preparation. Claude Code and Codex adopted similar strategies, recoding high-prevalence -3 codes — in particular CULTSATIS, where 72% of respondents were routed past the question — as an 'Unknown' or 'RoutingSkip' category rather than treating them as missing. Both dropped around 15% of rows after the initial EDA filter (see Table 1). Antigravity dropped more broadly, including -3 codes for most variables, resulting in a training set of 24,867 rows compared to approximately 29,000 for the other two — a reduction of roughly 4,000 additional observations that may skew the model toward a particular respondent profile."),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 160, after: 80 },
        children: [new TextRun({ text: "Table 1. Dataset sizes after each cleaning stage", italics: true, size: 20 })]
      }),
      missTable,
      new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),

      body("On modelling, the gap between agents was smaller. All three selected tuned logistic regression as their final model, with test-set F2 scores of 0.452, 0.464, and 0.488 respectively. Recall — the proportion of under-engaged respondents correctly identified — ranged from 0.713 to 0.769 across agents, meaning each model caught roughly three in four individuals in the minority class. Antigravity's model recorded the highest ROC-AUC (0.853), while Claude Code's achieved the highest recall (0.769) at the cost of lower precision (0.170). Table 2 summarises the final test-set performance."),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 160, after: 80 },
        children: [new TextRun({ text: "Table 2. Final model test-set performance (tuned logistic regression, all agents)", italics: true, size: 20 })]
      }),
      perfTable,
      new Paragraph({ spacing: { after: 160 }, children: [new TextRun("")] }),

      body("XGBoost performance was considerably more variable. Claude Code's tuned XGBoost produced an F2 of 0.212 on the test set — well below its own logistic regression baseline at the same threshold — while Codex and Antigravity achieved 0.452 and 0.427 respectively. The divergence appears linked to scale_pos_weight configuration and grid search scope: Claude Code's XGBoost search found that no configuration meaningfully exceeded the logistic regression, whereas the other two found workable configurations through broader search strategies. This variability suggests that tree-based models are more sensitive to implementation choices when the agent is operating autonomously."),

      body("The single most consistent failure mode across all three agents was the default decision threshold. At threshold 0.5, all three baseline logistic regressions produced recall between 0.027 and 0.059 — almost all under-engaged respondents were missed. Each agent corrected this through threshold tuning on the validation set, settling on operating thresholds between 0.52 and 0.59. The correction worked, but it required explicit guidance in the protocol; the problem would not have been identified without the F2 and recall metrics specified in the task brief."),

      body("EDA depth differed noticeably. Claude Code produced four visualisations including a correlation heatmap across predictors, while Codex and Antigravity each produced three plots focused on target distribution and selected demographic comparisons. None of the agents produced plots that were directly wrong, but the depth of analysis — how far beyond summary counts the agent went in connecting variables to the under-engagement problem — varied."),

      body("Documentation quality also split along similar lines. Claude Code and Codex each generated structured CSV evidence files covering tuning results, test comparisons, and model selection scores, creating an auditable analysis trail. Antigravity's evidence was embedded primarily in the notebook itself, with fewer standalone artefacts. Antigravity also used a distinctive notebook construction method — assembling the final notebook from separate Python scripts via an add_cell.py helper — rather than writing cells directly. The end result was functionally equivalent, but the intermediate structure is harder to inspect."),

      // ── 5. Takeaway ───────────────────────────────────────────────────────
      h2("Summary and Limitations"),

      body("Taken together, the experiments show that current AI coding agents can complete a realistic end-to-end data science pipeline on survey data with significant class imbalance. The differences between agents were more visible in intermediate decisions — how to handle missing codes, how thoroughly to explore features, how to structure evidence — than in the final model choice, where all three converged on the same algorithm. The main limitation of this benchmark is that each agent ran the pipeline once. Without replicated runs under varied prompt phrasings, it is not possible to separate genuine capability differences from run-to-run variation or prompt sensitivity. The comparative analysis section takes the results from these experiments and evaluates them against a consistent cross-agent framework."),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("practical_exploration_benchmarking.docx", buffer);
  console.log("Created practical_exploration_benchmarking.docx");
});
