# CLOUD DEPLOYMENT GUIDE
**Production Deployment on GCP, AWS, Azure**

## OVERVIEW

Deploy SmartGrocy pipeline to cloud for:
- ✅ Automatic daily forecasts
- ✅ Scalable processing
- ✅ Web dashboard access
- ✅ Automated retraining
- ✅ 24/7 availability

---

## OPTION 1: GOOGLE CLOUD PLATFORM (GCP)

### Architecture

```
Cloud Storage        Cloud Run         Cloud Scheduler
(Data & Models)  →  (Forecast API)  ← (Daily Trigger)
                         |
                         ↓
                  BigQuery (Results)
                         |
                         ↓
                  Looker Studio (Dashboard)
```

### Step 1: Setup GCP Project (10 min)

```bash
# 1.1. Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 1.2. Initialize project
gcloud init
gcloud config set project smartgrocy-prod

# 1.3. Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### Step 2: Dockerize Application (15 min)

**Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run forecast service
CMD ["python", "scripts/api/forecast_service.py"]
```

**Build and push:**
```bash
# Build
docker build -t gcr.io/smartgrocy-prod/forecast-api .

# Push to GCR
docker push gcr.io/smartgrocy-prod/forecast-api
```

### Step 3: Deploy to Cloud Run (5 min)

```bash
# Deploy
gcloud run deploy forecast-api \
  --image gcr.io/smartgrocy-prod/forecast-api \
  --platform managed \
  --region asia-southeast1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --allow-unauthenticated

# Output: Service URL: https://forecast-api-xxx.run.app
```

### Step 4: Schedule Daily Forecasts (10 min)

```bash
# Create Cloud Scheduler job
gcloud scheduler jobs create http daily-forecast \
  --schedule "0 2 * * *" \
  --uri "https://forecast-api-xxx.run.app/forecast" \
  --http-method POST \
  --time-zone "Asia/Ho_Chi_Minh"

echo "✅ Daily forecast scheduled for 2 AM"
```

### Step 5: Setup Storage & BigQuery (15 min)

```bash
# 5.1. Create storage bucket
gsutil mb -l asia-southeast1 gs://smartgrocy-data

# 5.2. Upload initial data
gsutil cp -r data/* gs://smartgrocy-data/

# 5.3. Create BigQuery dataset
bq mk --dataset smartgrocy-prod:forecasts

# 5.4. Create table
bq mk --table \
  smartgrocy-prod:forecasts.predictions \
  product_id:STRING,forecast_date:DATE,forecast_q50:FLOAT64
```

---

## OPTION 2: AWS (Amazon Web Services)

### Architecture

```
S3 Bucket         ECS Fargate      EventBridge
(Data)        →  (Container)   ← (Scheduler)
                      |
                      ↓
                  RDS/Redshift
                      |
                      ↓
                QuickSight (Dashboard)
```

### Quick Start

```bash
# 1. Install AWS CLI
pip install awscli
aws configure

# 2. Create ECR repository
aws ecr create-repository --repository-name smartgrocy

# 3. Build and push Docker image
aws ecr get-login-password | docker login --username AWS --password-stdin xxx.dkr.ecr.region.amazonaws.com
docker build -t smartgrocy .
docker tag smartgrocy:latest xxx.dkr.ecr.region.amazonaws.com/smartgrocy:latest
docker push xxx.dkr.ecr.region.amazonaws.com/smartgrocy:latest

# 4. Create ECS cluster
aws ecs create-cluster --cluster-name smartgrocy-cluster

# 5. Create task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 6. Schedule task
aws events put-rule --schedule-expression "cron(0 2 * * ? *)" --name daily-forecast
```

---

## OPTION 3: AZURE

### Architecture

```
Blob Storage      Container Instances    Logic Apps
(Data)        →   (Compute)          ← (Scheduler)
                       |
                       ↓
                 Azure SQL
                       |
                       ↓
                 Power BI (Dashboard)
```

### Quick Start

```bash
# 1. Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login

# 2. Create resource group
az group create --name smartgrocy-rg --location southeastasia

# 3. Create container registry
az acr create --resource-group smartgrocy-rg --name smartgrocyacr --sku Basic

# 4. Build and push
az acr build --registry smartgrocyacr --image forecast-api:latest .

# 5. Deploy container
az container create \
  --resource-group smartgrocy-rg \
  --name forecast-api \
  --image smartgrocyacr.azurecr.io/forecast-api:latest \
  --cpu 2 --memory 4

# 6. Schedule with Logic App (via portal)
```

---

## CI/CD AUTOMATION

### GitHub Actions Deployment

**`.github/workflows/deploy.yml`:**
```yaml
name: Deploy to Cloud

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Cloud CLI
        uses: google-github-actions/setup-gcloud@v0
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: smartgrocy-prod
      
      - name: Build Docker
        run: docker build -t gcr.io/smartgrocy-prod/forecast-api .
      
      - name: Push to GCR
        run: docker push gcr.io/smartgrocy-prod/forecast-api
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy forecast-api \
            --image gcr.io/smartgrocy-prod/forecast-api \
            --platform managed \
            --region asia-southeast1
```

---

## MONITORING & ALERTING

### Setup Monitoring

```bash
# GCP: Enable Cloud Monitoring
gcloud services enable monitoring.googleapis.com

# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-threshold-value=0.1

# AWS: CloudWatch Alarms
aws cloudwatch put-metric-alarm \
  --alarm-name high-error-rate \
  --metric-name ErrorCount \
  --threshold 10
```

### Log Analysis

```bash
# GCP: View logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# AWS: CloudWatch Logs
aws logs tail /aws/ecs/smartgrocy --follow

# Azure: Container Logs
az container logs --resource-group smartgrocy-rg --name forecast-api
```

---

## COST OPTIMIZATION

### Estimated Monthly Costs

**GCP (Small deployment):**
- Cloud Run: $10-30 (2 vCPU, 2GB RAM, 1hr/day)
- Cloud Storage: $5 (50GB data)
- BigQuery: $10 (1GB queries/day)
- **Total: ~$25-50/month**

**AWS (Small deployment):**
- ECS Fargate: $30-50
- S3: $5
- RDS: $15-30
- **Total: ~$50-85/month**

**Tips to reduce costs:**
- Use spot instances
- Schedule compute (only during business hours)
- Optimize Docker image size
- Use auto-scaling

---

## SECURITY BEST PRACTICES

1. **Authentication:**
   - Use API keys for service-to-service
   - OAuth for user access
   - Service accounts with minimal permissions

2. **Data Encryption:**
   - Enable encryption at rest (automatic on GCP/AWS/Azure)
   - Use HTTPS for all API calls
   - Store secrets in Secret Manager

3. **Network Security:**
   - Use VPC/VNet
   - Restrict IP access
   - Enable Cloud Armor / WAF

4. **Compliance:**
   - Enable audit logging
   - Regular security scans
   - Data residency compliance

---

## SUPPORT & RESOURCES

**Documentation:**
- GCP: https://cloud.google.com/run/docs
- AWS: https://aws.amazon.com/ecs/
- Azure: https://azure.microsoft.com/en-us/services/container-instances/

**Community:**
- Stack Overflow: [gcp], [aws], [azure] tags
- Discord: SmartGrocy Community
- GitHub: Issues & Discussions
