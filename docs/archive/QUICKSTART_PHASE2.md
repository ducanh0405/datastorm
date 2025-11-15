# 🚀 Quick Start: Phase 2 Features (Data Quality Monitoring)

**5 minutes to production-grade data quality monitoring!**

---

## 📝 TL;DR (Too Long; Didn't Read)

```bash
# 1. Pull latest
git pull origin main

# 2. Setup GX (one-time)
python scripts/setup_great_expectations.py

# 3. Run pipeline with GX
python run_modern_pipeline_v2.py --full-data --use-v2

# DONE! 🎉
```

---

## 📚 Full Guide (3 Steps)

### Step 1: Update Repository

```bash
cd /path/to/datastorm
git pull origin main
pip install --upgrade pandas==2.3.3
```

**Verify**:
```bash
git log --oneline -6
```

Should show:
```
4e72b9d docs: Phase 2 completion report
ea1c5e2 feat: Enhanced pipeline runner with GX and CLI sampling
bfefe50 feat: Enhanced orchestrator with full GX integration
f76266a feat: Add Great Expectations integration module
40be65e feat: Add standalone data quality validation runner
fbf4ca6 feat: Add Great Expectations setup script
```

---

### Step 2: Setup Great Expectations (One-Time)

```bash
python scripts/setup_great_expectations.py
```

**Expected**: ✓ Creates `great_expectations/` directory with:
- Expectation suites (150 validation rules)
- Checkpoints (automated validation)
- Data docs (HTML reports)

**Time**: ~30 seconds

---

### Step 3: Run Pipeline

#### Option A: Quick Test (Recommended First)
```bash
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2
```
- Uses 10% of data
- Completes in ~5 minutes
- Full GX validation
- Perfect for testing

#### Option B: Full Production Run
```bash
python run_modern_pipeline_v2.py --full-data --use-v2
```
- Uses 100% of data
- Completes in ~30-60 minutes
- Full GX validation
- Production-ready

---

## 🎯 Common Use Cases

### Use Case 1: Daily Development
```bash
# Morning: Quick test with sample
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2

# If successful: Full run
python run_modern_pipeline_v2.py --full-data --use-v2
```

---

### Use Case 2: Check Data Quality Only
```bash
# Validate existing feature table
python scripts/run_data_quality_check.py --verbose

# View detailed report
# Windows: start great_expectations/uncommitted/data_docs/local_site/index.html
# Mac: open great_expectations/uncommitted/data_docs/local_site/index.html
```

---

### Use Case 3: Debug Pipeline Issues
```bash
# 1. Check data quality first
python scripts/run_data_quality_check.py --verbose

# 2. If quality OK, run pipeline with logging
python run_modern_pipeline_v2.py --full-data --use-v2 2>&1 | tee pipeline_debug.log

# 3. Review logs
tail -100 logs/pipeline.log
cat pipeline_debug.log
```

---

### Use Case 4: CI/CD Integration
```bash
# In CI/CD script (e.g., GitHub Actions)
python scripts/run_data_quality_check.py

# Exit code 0 = pass, 1 = fail
if [ $? -eq 0 ]; then
    echo "Quality check passed, deploying..."
else
    echo "Quality check failed, blocking deployment"
    exit 1
fi
```

---

## 🛠️ Available Commands

### Pipeline Runners
```bash
# V1: Original (no GX)
python run_modern_pipeline.py --full-data

# V2: Enhanced (with GX)
python run_modern_pipeline_v2.py --full-data --use-v2

# V2 with sampling
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2
```

### Data Quality
```bash
# Setup GX (one-time)
python scripts/setup_great_expectations.py

# Run validation
python scripts/run_data_quality_check.py

# Verbose validation
python scripts/run_data_quality_check.py --verbose
```

### Apply Phase 1 Fixes
```bash
# LightGBM stability fix
python scripts/apply_phase1_lightgbm_fix.py
```

---

## ⚡ Performance Tips

### Tip 1: Use Sampling for Development
```bash
# Start with 10% for quick iteration
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2

# Increase gradually: 20%, 50%, 100%
python run_modern_pipeline_v2.py --full-data --sample 0.2 --use-v2
```

### Tip 2: Skip GX for Speed (if needed)
```bash
# Use v1 orchestrator (no GX overhead)
python run_modern_pipeline.py --full-data

# ~10% faster than v2
```

### Tip 3: Enable Caching
```bash
# Default: caching enabled
python run_modern_pipeline_v2.py --full-data --use-v2

# Only use --no-cache when you need fresh results
python run_modern_pipeline_v2.py --full-data --use-v2 --no-cache
```

---

## ❌ Common Issues

### Issue: "GX not setup"
```bash
# Solution
python scripts/setup_great_expectations.py
```

### Issue: "Validation failed"
```bash
# Check what failed
python scripts/run_data_quality_check.py --verbose

# View detailed report
open great_expectations/uncommitted/data_docs/local_site/index.html
```

### Issue: "Import error: great_expectations"
```bash
# Install GX
pip install great-expectations==0.18.19
```

### Issue: "Pipeline too slow"
```bash
# Use sampling
python run_modern_pipeline_v2.py --full-data --sample 0.2 --use-v2
```

---

## 📊 Monitoring & Reports

### Real-Time Monitoring
```bash
# Watch pipeline logs
tail -f logs/pipeline.log
```

### Quality Reports
```bash
# Text summary
cat reports/quality_summary.txt

# HTML report (GX data docs)
open great_expectations/uncommitted/data_docs/local_site/index.html
```

### Model Metrics
```bash
# View training metrics
cat reports/metrics/model_metrics.json | python -m json.tool
```

---

## 📞 Support

**Documentation**:
- Full details: `PHASE2_COMPLETION_REPORT.md`
- Phase 1 fixes: `PHASE1_FIXES.md`
- Original guide: `README.md`

**Troubleshooting**:
1. Check logs: `logs/pipeline.log`
2. Review GX reports: `great_expectations/uncommitted/data_docs/`
3. Run diagnostics: `python scripts/run_data_quality_check.py --verbose`

**Contact**: ducanh0405@gmail.com

---

**✅ You're ready to go! Start with the TL;DR commands above.**
