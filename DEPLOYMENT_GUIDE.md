# 🚀 Car Analysis Intelligence Platform - Deployment Guide

## 📋 Environment Variables Setup

### 🔧 **Required Environment Variables**

```bash
# Essential Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database_name

# Optional Application Settings
DEBUG_MODE=false
PYTHONUNBUFFERED=1
```

## 🌐 **Platform-Specific Deployment**

### 🐳 **Docker Compose (Recommended for Local/VPS)**

Create `.env` file:
```bash
DATABASE_URL=postgresql://autointel:secure_password@postgres:5432/car_analysis
DEBUG_MODE=false
```

Deploy:
```bash
docker-compose up -d
```

### ☁️ **Heroku**

Set environment variables:
```bash
heroku config:set DATABASE_URL=your_postgres_url
heroku config:set PORT=8501
heroku config:set PYTHONUNBUFFERED=1
```

Deploy:
```bash
git push heroku main
```

### 🚄 **Railway**

In Railway dashboard, set:
```bash
DATABASE_URL=${{Postgres.DATABASE_URL}}
PYTHONUNBUFFERED=1
```

Deploy:
- Connect GitHub repository
- Railway auto-deploys on push

### 🎨 **Render**

Environment variables:
```bash
DATABASE_URL=${{DATABASE_URL}}  # From Render PostgreSQL service
PYTHON_VERSION=3.11
```

Deploy:
- Connect GitHub repository
- Select Docker deployment

### ☁️ **Google Cloud Run**

Set environment variables:
```bash
DATABASE_URL=postgresql://user:pass@/dbname?host=/cloudsql/project:region:instance
GOOGLE_CLOUD_PROJECT=your-project-id
```

Deploy:
```bash
gcloud run deploy --source .
```

### 🔶 **AWS ECS/Fargate**

Environment variables in task definition:
```json
{
  "environment": [
    {"name": "DATABASE_URL", "value": "postgresql://..."},
    {"name": "AWS_REGION", "value": "us-east-1"},
    {"name": "PYTHONUNBUFFERED", "value": "1"}
  ]
}
```

### 🌊 **DigitalOcean App Platform**

In app spec:
```yaml
envs:
- key: DATABASE_URL
  value: ${{db.DATABASE_URL}}
- key: PYTHONUNBUFFERED
  value: "1"
```

## 🗄️ **Database Setup**

### **Option 1: Use Platform Database**
Most hosting platforms offer managed PostgreSQL:
- Heroku Postgres
- Railway PostgreSQL
- Render PostgreSQL
- Google Cloud SQL
- AWS RDS

### **Option 2: External Database**
Use services like:
- [Supabase](https://supabase.com) (Free tier available)
- [Neon](https://neon.tech) (Free tier available)
- [ElephantSQL](https://www.elephantsql.com) (Free tier available)

### **Database Connection Format**
```bash
DATABASE_URL=postgresql://username:password@host:port/database_name

# Example with Supabase:
DATABASE_URL=postgresql://postgres:your_password@db.xxx.supabase.co:5432/postgres

# Example with Railway:
DATABASE_URL=postgresql://postgres:password@containers-us-west-xxx.railway.app:7xxx/railway
```

## ⚙️ **Optional Configuration**

### **Debug Mode**
```bash
DEBUG_MODE=true  # Shows additional N-gram analysis info
```

### **Performance Tuning**
```bash
MAX_RECORDS_LIMIT=10000  # Limit processing for large datasets
```

### **Feature Toggles**
```bash
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_NER_ANALYSIS=true
ENABLE_TOPIC_MODELING=true
ENABLE_KEYWORD_ANALYSIS=true
ENABLE_NGRAM_ANALYSIS=true
```

## 🔒 **Security Best Practices**

1. **Never commit `.env` files** to version control
2. **Use strong database passwords**
3. **Restrict database access** to your application only
4. **Use HTTPS** in production (most platforms handle this automatically)
5. **Set unique SECRET_KEY** for each deployment

## 🚀 **Quick Start Commands**

### **Local Development with Docker**
```bash
cp env.template .env
# Edit .env with your settings
docker-compose up -d
```

### **Deploy to Heroku**
```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:mini
heroku config:set PYTHONUNBUFFERED=1
git push heroku main
```

### **Deploy to Railway**
```bash
# Just connect your GitHub repo in Railway dashboard
# Add PostgreSQL plugin
# Set environment variables in Railway UI
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **Database Connection Error**
   - Check DATABASE_URL format
   - Verify database is accessible
   - Ensure database exists

2. **Module Import Errors**
   - Verify PYTHONPATH=/app is set
   - Check all dependencies in requirements.txt

3. **Port Issues**
   - Most platforms auto-assign PORT
   - Streamlit uses 8501 by default

4. **Memory Issues**
   - Increase container memory limits
   - Consider using MAX_RECORDS_LIMIT

## 📞 **Support**

- 📧 GitHub Issues: [autointel/issues](https://github.com/olastephen/autointel/issues)
- 📖 Documentation: README.md
- 🐳 Docker: DOCKER_README.md
