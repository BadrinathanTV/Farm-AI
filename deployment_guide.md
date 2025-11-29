# 🚀 Deploying Farm-AI to Google Cloud Platform (GCP)

This guide will help you deploy your Farm-AI assistant to **Google Cloud Run**, a serverless platform that scales automatically.

## Prerequisites
1.  **Google Cloud Account**: [Sign up here](https://cloud.google.com/) (Free tier available).
2.  **Google Cloud SDK**: Install the `gcloud` CLI tool.
3.  **Docker**: Ensure Docker is running on your machine.

## Step 1: Setup GCP Project
1.  Create a new project in the [GCP Console](https://console.cloud.google.com/).
2.  Enable the **Cloud Run API** and **Container Registry API**.
    ```bash
    gcloud services enable run.googleapis.com containerregistry.googleapis.com
    ```
3.  Initialize the gcloud CLI:
    ```bash
    gcloud init
    ```

## Step 2: Build and Push Docker Image
We need to upload your app's "blueprint" (Docker image) to Google's storage (Container Registry).

1.  **Tag your image**:
    Replace `[PROJECT-ID]` with your actual GCP project ID.
    ```bash
    gcloud builds submit --tag gcr.io/[PROJECT-ID]/farm-ai
    ```
    *(This command zips your code, sends it to Google, builds the Docker container there, and stores it.)*

## Step 3: Deploy to Cloud Run
Now, we tell Google to run that image.

1.  **Deploy**:
    ```bash
    gcloud run deploy farm-ai \
      --image gcr.io/[PROJECT-ID]/farm-ai \
      --platform managed \
      --region us-central1 \
      --allow-unauthenticated \
      --set-env-vars OPENAI_API_KEY=your_actual_api_key_here
    ```

2.  **Success!**
    The terminal will output a URL (e.g., `https://farm-ai-xyz-uc.a.run.app`). Click it to see your live app!

## 💡 Tips for Students
-   **Cost**: Cloud Run has a generous free tier (2 million requests/month). You likely won't pay anything.
-   **Database**: This guide runs the app, but for a production database, you should use **MongoDB Atlas** (Cloud) instead of a local container. You would just update the `MONGO_URI` environment variable in the deploy command.
