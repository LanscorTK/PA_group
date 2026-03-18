#!/bin/bash
# prep_repo.sh — Review and optionally clean the repo for submission
# Run: bash outputs/prep_repo.sh
# This script only PRINTS recommendations. Pass --clean to act.

set -e
cd "$(dirname "$0")/.."

echo "=== FILES TO KEEP (benchmarking harness + evidence) ==="
echo ""
echo "  agents/prompts/            14 frozen prompt files (task specs)"
echo "  agents/claude_code/        notebook, build script, evidence, run log, report, README"
echo "  agents/codex/              notebook, build script, evidence, run log, report, README"
echo "  agents/antigravity/        notebook, build script, evidence, run log, report, README"
echo "  data/                      dataset + data dictionary"
echo "  docs/Pipeline.md           frozen benchmark protocol"
echo "  docs/PROJECT_OVERVIEW.md   project overview"
echo "  docs/cross_agent_comparison.md  detailed cross-agent comparison"
echo "  README.md                  repo-level documentation"
echo "  requirements.txt           shared Python dependencies"
echo "  benchmark_notes.md         protocol constraints"
echo "  .gitignore                 git config"
echo ""

echo "=== FILES TO CONSIDER REMOVING (not benchmarking harness) ==="
echo ""

REMOVE_CANDIDATES=(
  "docs/MSIN0097_ Predictive Analytics 25-26 Group Coursework.pdf"
  "docs/Report Draft 0318.docx"
  "docs/Pipeline_260316.docx"
  "docs/practical_exploration_benchmarking_structure.md"
  "docs/initial_prompt"
  "agents/codex/evidence_codex/participation_clean.csv"
  "agents/antigravity/report_data.md"
  "agents/antigravity/report_data_files"
  "agents/claude_code/best_model_name.txt"
  "outputs/generate_report.js"
  "outputs/create_practical_exploration.js"
  "outputs/report_2-3.docx"
  "outputs/appendices.docx"
  "logs/.gitkeep"
)

for f in "${REMOVE_CANDIDATES[@]}"; do
  if [ -e "$f" ]; then
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "  [$SIZE]  $f"
  fi
done

echo ""
echo "=== REPO SIZE ==="
du -sh .git
echo "Total tracked files:"
git ls-files | wc -l | tr -d ' '
echo ""

if [ "$1" = "--clean" ]; then
  echo "=== REMOVING FILES FROM GIT TRACKING ==="
  for f in "${REMOVE_CANDIDATES[@]}"; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      echo "  Removing: $f"
      git rm --cached -r "$f" 2>/dev/null || true
    fi
  done
  echo ""
  echo "Done. Review with 'git status', then commit if satisfied."
  echo "The files are still on disk — only removed from git tracking."
else
  echo "To remove these from git (keeps local copies), run:"
  echo "  bash outputs/prep_repo.sh --clean"
fi
