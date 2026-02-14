# Railway Deployment Guide

## Why Railway?

Railway offers the most generous free tier among cloud platforms:
- **$5/month credit** (enough for small projects)
- **512MB RAM, 1GB disk** free forever
- **Docker support** with custom Dockerfiles
- **PostgreSQL database** included
- **Auto-deploy from GitHub**
- **Custom domains** with SSL
- **Environment variables** management
- **Logs and monitoring**

## Step-by-Step Deployment

### 1. Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (recommended for auto-deploy)
3. Verify your email

### 2. Deploy from GitHub

1. Click **"New Project"** → **"Deploy from GitHub repo"**
2. Connect your GitHub account
3. Select `Abood991B/CortexaAI` repository
4. Railway will auto-detect your `railway.json` and `Dockerfile`

### 3. Configure Environment Variables

In your Railway project dashboard:

1. Go to **"Variables"** tab
2. Add these variables:

```
GOOGLE_API_KEY=your_google_api_key_here
PORT=8000
HOST=0.0.0.0
DEFAULT_MODEL_PROVIDER=google
DEFAULT_MODEL_NAME=gemini-2.0-flash
LOG_LEVEL=INFO
ENABLE_CACHING=true
```

### 4. Add PostgreSQL (Optional but Recommended)

Your app uses SQLite by default, but for production:

1. In Railway dashboard: **"Add Plugin"** → **"PostgreSQL"**
2. Railway will automatically set `DATABASE_URL`
3. Your app will auto-detect and use PostgreSQL

### 5. Deploy

Click **"Deploy"** - Railway will:
- Build your Docker image
- Start the container
- Run health checks
- Provide a public URL: `https://cortexaai-production.up.railway.app`

## Alternative: Manual Docker Deploy

If you prefer manual control:

```bash
# Railway CLI installation
npm install -g @railway/cli
railway login

# Link to existing project or create new
railway link

# Deploy
railway up
```

## Cost Comparison

| Feature | Railway Free | Render Free | Railway Paid |
|---------|-------------|-------------|--------------|
| RAM | 512MB | 512MB | 1GB ($5) |
| Disk | 1GB | 1GB | 32GB ($5) |
| Bandwidth | 512GB/month | 750GB/month | Unlimited |
| Database | PostgreSQL ($5) | None | PostgreSQL included |
| Custom Domain | Yes | Yes | Yes |
| Docker | Yes | Yes | Yes |

## Troubleshooting

### Build Fails
- Check Railway build logs
- Ensure `Dockerfile` is in project root
- Verify all dependencies in `requirements.txt`

### App Won't Start
- Check environment variables are set correctly
- Verify `GOOGLE_API_KEY` is valid
- Check `/health` endpoint in Railway logs

### Database Issues
- If using PostgreSQL, ensure `DATABASE_URL` is set
- Your app auto-detects between SQLite and PostgreSQL