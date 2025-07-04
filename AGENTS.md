# AGENTS.md – Photo Restoration Pipeline

## Scope
All files under this repository.

## Coding Conventions
- Black formatting, 120-char lines.
- Use type hints everywhere (`PEP 484`).
- No hard-coded paths outside `input/` and `output/`.

## Pull-Request Rules
1. Title starts with `feat:`, `fix:`, or `docs:`.
2. PR body must list affected modules and a test plan.

## Data-Flow Overview
`raw_image` ➡ **MIRNet** ➡ `stage1_out` ➡ **Real-ESRGAN** ➡ `final_out` (shown inline).

## Deferred Features
- SwinIR refinement
- Batch mode
