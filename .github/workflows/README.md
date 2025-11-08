# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

The CI pipeline runs on every push and pull request to `main` or `develop` branches.

**Jobs:**
1. **Lint** - Code quality checks
   - Black (code formatting)
   - isort (import sorting)
   - Ruff (linting)
   - MyPy (type checking, non-blocking)

2. **Test** - Automated testing
   - Runs on Python 3.10 and 3.11
   - Smoke tests (`test_smoke.py`)
   - Feature tests (`test_features.py`)
   - Code coverage reporting

3. **Integration** - Integration tests
   - Runs only on pushes to `main`
   - Requires POC data to be present

### CD Pipeline (`.github/workflows/cd.yml`)

The CD pipeline handles automated deployment.

**Triggers:**
- Pushes to `main` branch
- Git tags matching `v*.*.*`
- Manual workflow dispatch

**Jobs:**
1. **Build** - Build and package artifacts
   - Runs tests
   - Creates build artifacts
   - Uploads artifacts for deployment

2. **Deploy Staging** - Deploy to staging environment
   - Runs on pushes to `main`
   - Validates pipeline
   - Deploys to staging
   - Runs smoke tests

3. **Deploy Production** - Deploy to production
   - Runs on version tags or manual trigger
   - Full test suite
   - Production deployment
   - Health checks
   - Release notes generation

4. **Notify** - Send deployment notifications
   - Sends notifications after deployment
   - Configure with Slack, email, etc.

## Setup

### Required Secrets

Configure these secrets in GitHub repository settings:

- `SLACK_WEBHOOK_URL` (optional) - For deployment notifications

### Environment Variables

Update environment URLs in `cd.yml`:
- `staging` environment URL
- `production` environment URL

### Customization

1. **Deployment Commands**: Update deployment steps in `cd.yml` with your specific deployment logic
2. **Test Coverage**: Configure Codecov or similar service for coverage tracking
3. **Notifications**: Add notification logic for Slack, email, or other services

## Local Testing

Test workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or download from https://github.com/nektos/act/releases

# Run CI workflow
act push

# Run specific job
act -j lint
```

## Pre-commit Hooks

Install pre-commit hooks to run checks before committing:

```bash
pip install pre-commit
pre-commit install
```

This will run code quality checks automatically on `git commit`.

