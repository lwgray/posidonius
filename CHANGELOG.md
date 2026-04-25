# Changelog

All notable changes to Posidonius are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Batch parallel experiment platform**: launch N experiments across N independent Marcus instances simultaneously
- **Epictetus auto-audit**: automatically runs code quality audit before teardown on experiment completion
- Dismiss button to remove failed or stopped experiments from the dashboard
- `--marcus-python` flag with conda/venv/pyenv path examples in docs

### Fixed
- `spawn.log` now captured; fixed pretrust race condition on parallel batch launch
- MLflow fluent API replaced with `MlflowClient` to fix metric logging in parallel batch runs
- Configured `marcus_python` interpreter now used correctly in subprocess launch
- `--marcus-python` path validated up front with a clear error message

---

## [0.1.1] - 2026-03-31

### Added
- **Auto-advance mode**: hands-free batch experiments run sequentially without manual intervention
- **Pause/resume toggle** for auto-advance pipeline (#6)
- **Pipeline event logging** with full UI redesign (#5)
- Auto-confirm for Claude trust/permission prompts in tmux panes
- Board metrics panel scoped to active project only

### Changed
- UX overhaul: app layout, component structure, and test cleanup

### Fixed
- Tmux attach and export buttons restored to Running dashboard
- Trust prompt polling exits early once Claude has started normally
- Repeated polling for trust prompts after session launch

---

## [0.1.0] - 2026-03-26

### Added
- Initial release of Posidonius — experiment dashboard for multi-agent coordination at scale
- Sequential experiment pipeline: run scaling tests (e.g. 3 → 5 → 10 agents) automatically
- Live agent monitoring via real-time tmux pane capture
- Agent status detection: working / waiting / complete / error / idle
- Interactive terminal access to individual agent panes via WebSocket
- MLflow experiment tracking with parent/child run structure
- Optimal agent count estimation via Marcus MCP CPM analysis
- Task progress strip: completed tasks, percentage, active agents, blockers
- History tab with past experiments and metrics comparison
- Export: download all agent output as a ZIP archive
- FastAPI backend with WebSocket support
- Pre-commit hooks: mypy (strict), black, isort, flake8, bandit, detect-secrets, pydocstyle
- CI/CD workflows: tests, pre-commit, version gate

[Unreleased]: https://github.com/lwgray/posidonius/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/lwgray/posidonius/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/lwgray/posidonius/releases/tag/v0.1.0
