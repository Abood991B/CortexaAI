# Koyeb Deployment Guide for CortexaAI

This guide will help you deploy CortexaAI to Koyeb.com.

## Prerequisites

1. A Koyeb account (sign up at https://www.koyeb.com)
2. A GitHub/GitLab repository with your code
3. API keys for at least one LLM provider (Google AI or OpenAI)

## Step 1: Prepare Your Repository

The repository is already configured with:
- ✅ Production Dockerfile
- ✅ Static file serving setup
- ✅ CORS configuration
- ✅ Environment variable support
- ✅ Docker Compose for local testing

**Important:** Make sure your `.env` file is NOT committed to Git (it should be in `.gitignore`). 
For Koyeb deployment, you'll set environment variables in the Koyeb dashboard instead.

## Step 2: Set Up Your Repository

1. Push your code to GitHub/GitLab
2. Make sure all your changes are committed and pushed

## Step 3: Deploy on Koyeb

### Option A: Deploy via Koyeb Dashboard

1. Log in to your Koyeb account
2. Click "Create App" or "New App"
3. Connect your Git repository
4. Select your repository and branch
5. Configure the following:

   **Build Settings:**
   - Build Type: Dockerfile
   - Dockerfile Path: `Dockerfile` (root of repository)

   **Environment Variables:**
   Add these required variables:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   # OR
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Optional variables:
   ```
   LANGSMITH_API_KEY=your_langsmith_key (optional, for tracing)
   CORS_ORIGINS=https://your-app-name.koyeb.app (optional, defaults to *)
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

6. Click "Deploy"

### Option B: Deploy via Koyeb CLI

1. Install Koyeb CLI:
   ```bash
   curl https://www.koyeb.com/cli.sh | sh
   ```

2. Login to Koyeb:
   ```bash
   koyeb login
   ```

3. Create and deploy your app:
   ```bash
   koyeb app create cortexaai
   koyeb service create cortexaai \
     --app cortexaai \
     --git github/yourusername/cortexaai \
     --git-branch main \
     --dockerfile Dockerfile \
     --env GOOGLE_API_KEY=your_key_here
   ```

## Step 4: Configure Environment Variables

In the Koyeb dashboard, go to your app → Settings → Environment Variables and add:

### Required:
- `GOOGLE_API_KEY` or `OPENAI_API_KEY` (at least one)

### Optional:
- `LANGSMITH_API_KEY` - For LangSmith tracing
- `CORS_ORIGINS` - Comma-separated list of allowed origins (defaults to `*`)
- `ENVIRONMENT` - Set to `production`
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `DEFAULT_MODEL_PROVIDER` - `google` or `openai` (default: `google`)
- `DEFAULT_MODEL_NAME` - Model name to use

## Step 5: Configure Resources

In Koyeb dashboard → App Settings → Resources:

- **Memory**: Minimum 512MB (1GB recommended)
- **CPU**: 0.5 vCPU minimum (1 vCPU recommended)
- **Scaling**: 
  - Min instances: 1
  - Max instances: Configure based on your needs

## Step 6: Access Your Application

Once deployed, Koyeb will provide you with a URL like:
```
https://your-app-name.koyeb.app
```

Your application will be available at this URL.

## Step 7: Verify Deployment

1. Visit your Koyeb URL
2. Check the health endpoint: `https://your-app-name.koyeb.app/health`
3. Check API docs: `https://your-app-name.koyeb.app/docs`
4. Test the frontend interface

## Troubleshooting

### Build Fails

- Check that all dependencies are in `requirements.txt`
- Verify the Dockerfile is in the root directory
- Check build logs in Koyeb dashboard

### Application Won't Start

- Verify environment variables are set correctly
- Check application logs in Koyeb dashboard
- Ensure at least one LLM API key is configured

### Frontend Not Loading

- Verify the frontend build completed successfully
- Check that static files are being served correctly
- Review browser console for errors

### CORS Errors

- Set `CORS_ORIGINS` environment variable to your Koyeb domain
- Or leave it unset to allow all origins (default)

## Monitoring

Koyeb provides built-in monitoring:
- View logs in the Koyeb dashboard
- Monitor metrics and performance
- Set up alerts for errors

## Custom Domain

To use a custom domain:
1. Go to App Settings → Domains
2. Add your custom domain
3. Configure DNS as instructed by Koyeb

## Updating Your Deployment

Simply push changes to your connected Git branch, and Koyeb will automatically rebuild and redeploy your application.

## Support

- Koyeb Documentation: https://www.koyeb.com/docs
- Koyeb Community: https://www.koyeb.com/community
- Project Issues: Open an issue in your repository

