# Contributing to Posidonius

Welcome to Posidonius! We're excited you're interested in contributing. This guide covers everything you need to get started, whether you're improving the experiment pipeline, the dashboard, or the MLflow integration.

## Branching Strategy

- **`main`**: Production-ready code. Protected — no direct pushes allowed.
- **`develop`**: Primary development branch. All PRs should target this branch.
- **Feature branches**: Work in your fork's feature branches, created from `develop`.

**Quick workflow:**
1. Fork the Posidonius repository
2. Clone your fork and add the upstream remote
3. Always branch from `develop`
4. Submit PRs targeting `develop`

## Ways to Contribute

Posidonius needs more than code:

- **Bug reports**: Found something broken? Open an issue with steps to reproduce.
- **Documentation**: Improve setup guides, add experiment examples, clarify behavior.
- **Testing**: Write tests, improve coverage, report edge cases.
- **Experiment ideas**: Propose new metrics, visualizations, or pipeline features.
- **Code**: Bug fixes, features, performance improvements.

## Development Setup

### Prerequisites

- Python 3.11+
- tmux (`brew install tmux` on macOS)
- [Marcus](https://github.com/lwgray/marcus) MCP server (for running experiments)
- MLflow
- Git

### Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/posidonius.git
cd posidonius
git remote add upstream https://github.com/lwgray/posidonius.git
git checkout develop

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify
pytest tests/
```

### Development Workflow

```bash
# 1. Keep develop in sync
git checkout develop
git pull upstream develop
git push origin develop

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and verify
pytest tests/                    # Run tests
mypy posidonius/                 # Type checking
pre-commit run --all-files       # All quality checks

# 4. Commit with conventional commits
git add .
git commit -m "feat(pipeline): auto-advance to next run on completion"

# 5. Stay up to date
git fetch upstream
git rebase upstream/develop

# 6. Push and open PR targeting develop
git push origin feature/your-feature-name
```

## Code Quality

We use pre-commit hooks that run automatically before every commit:

- **MyPy**: Static type checking (strict mode)
- **Black**: Code formatting
- **isort**: Import ordering
- **Flake8**: Linting
- **Bandit**: Security checks
- **detect-secrets**: Prevents committing credentials
- **pydocstyle**: Docstring validation

### Running Checks Manually

```bash
pre-commit run --all-files        # All hooks
mypy posidonius/                  # Type checking only
pytest --cov=posidonius --cov-report=html  # Tests with coverage
```

### Quality Standards

All code must pass:

1. **Type safety**: MyPy strict mode with no errors
2. **Formatting**: Black applied
3. **Import order**: isort organized
4. **Linting**: Flake8 clean
5. **Security**: No secrets in code
6. **Tests**: 80% minimum coverage for new code

## Coding Standards

```python
# Good: typed, documented
def launch_experiment_run(
    config: ExperimentConfig,
    agent_count: int,
    mlflow_run_id: str,
) -> ExperimentRun:
    """
    Launch a single experiment run with the given agent count.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration
    agent_count : int
        Number of agents to spawn for this run
    mlflow_run_id : str
        Parent MLflow run ID for metric logging

    Returns
    -------
    ExperimentRun
        Run handle with tmux session reference and status

    Raises
    ------
    ExperimentLaunchError
        If tmux session creation or agent spawn fails
    """
    ...

# Bad: untyped, undocumented
def launch(cfg, n, run):
    ...
```

- Always use type hints (strict mypy enforced)
- NumPy-style docstrings on all public functions and classes
- Use structured logging, not `print`
- Never use naive `datetime.now()` — use timezone-aware datetimes
- Update `CHANGELOG.md` for any user-facing change

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

```bash
# Good
git commit -m "feat(mlflow): log per-agent task completion metrics"
git commit -m "fix(tmux): handle session cleanup on experiment abort"
git commit -m "docs(quickstart): add conda environment example"

# Bad
git commit -m "fixed stuff"
git commit -m "WIP"
```

## Testing

```
tests/
├── unit/        # Fast, isolated — mock MLflow and tmux
└── integration/ # Requires Marcus MCP and MLflow running
```

```bash
pytest                            # All tests
pytest tests/unit/                # Fast tests only
pytest --cov=posidonius           # With coverage
pytest -k "test_pipeline"         # Filter by name
```

Write tests for every new feature. Aim for 80% coverage on changed code.

## Pull Request Process

### Before Submitting

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest` passes
- [ ] `mypy posidonius/` passes
- [ ] `CHANGELOG.md` updated (if user-facing change)
- [ ] PR targets the `develop` branch

### PR Description Template

```markdown
## What
Brief description of the change.

## Why
The problem this solves or feature it adds.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor

## Testing
How you verified this works.
```

### After Merge

```bash
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
git checkout develop
git pull upstream develop
git push origin develop
```

## Getting Help

- **[GitHub Issues](https://github.com/lwgray/posidonius/issues)**: Bug reports and feature requests
- **[GitHub Discussions](https://github.com/lwgray/posidonius/discussions)**: Questions and ideas
- **[Marcus Repo](https://github.com/lwgray/marcus)**: For questions about the underlying coordination system

## Recognition

Contributors are listed in `CONTRIBUTORS.md` and credited in release notes for significant contributions.

---

Thank you for helping make Posidonius better!
