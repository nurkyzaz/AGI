#!/bin/bash
LOG=~/agi/ibrg_llm/logs/orch.log; cd ~/agi
q(){ squeue -h -u nurkyz 2>/dev/null | wc -l; }
room(){ while [ "$(q)" -gt 3 ]; do sleep 60; done; }
sub(){ room; sbatch "$1" >> "$LOG" 2>&1; echo "submitted $1 $(date)" >> "$LOG"; sleep 30; }
sub ibrg_llm/scale_q05b.sbatch
sub ibrg_llm/scale_q15b.sbatch
for i in $(seq 1 160); do
  grep -q BOTH_OK ~/agi/ibrg_llm/logs/bc_smoke.out 2>/dev/null && break
  grep -qE "Traceback|Error:|AssertionError" ~/agi/ibrg_llm/logs/bc_smoke.out 2>/dev/null && { echo "B/C smoke FAILED; skipping $(date)" >> "$LOG"; exit 1; }
  sleep 30
done
sub ibrg_llm/phase.sbatch
sub ibrg_llm/ablsweep.sbatch
echo "orchestration complete $(date)" >> "$LOG"
