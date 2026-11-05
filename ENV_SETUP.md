# Environment Configuration Guide

## Google Cloud Vision API Setup

The PiZeroCam package uses Google Cloud Vision API for OCR text detection in pH analysis.

### Step 1: Get Google Cloud Credentials

1. **Go to Google Cloud Console**: https://console.cloud.google.com/

2. **Create a new project** (or select an existing one, go to step 5)

3. **Enable the Vision API**: (for new applications only)
   - Go to "APIs & Services" → "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"

4. **Create a Service Account**: (for new applications only)
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Give it a name (e.g., "pizerocam-vision")
   - Grant it the "Cloud Vision API User" role
   - Click "Done"

5. **Generate a Key/Download Existing Key (go to last point)**:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key" (for new applications only)
   - Choose "JSON" format (for new applications only)
   - Download the JSON file (e.g., `pizerocam-vision-key.json`)

### Step 2: Configure in PiZeroCam

You have **two options**:

#### Option A: Using .env file (Recommended)

1. Create a file named `.env` in the project root directory:

```bash
cd /path/to/your/pizerocam_folder
touch .env
```

2. Add the following content to `.env`:

```bash
# Google Cloud Vision API Credentials
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-credentials.json"

# Example:
# GOOGLE_APPLICATION_CREDENTIALS="/Users/macbook_m2/credentials/pizerocam-vision-key.json"
```

3. Replace `/path/to/your/google-cloud-credentials.json` with the actual path to your downloaded JSON file.

**Note**: The `.env` file is automatically ignored by git, so your credentials won't be committed.

#### Option B: Using Environment Variable

Set the environment variable before running your scripts:

**macOS/Linux:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-credentials.json"
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\google-cloud-credentials.json"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\google-cloud-credentials.json
```

### Step 3: Verify Setup

Test that the credentials are working:

```python
from google.cloud import vision
import os

# Check if credentials are set
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if creds_path:
    print(f"✓ Credentials path: {creds_path}")
    print(f"✓ File exists: {os.path.exists(creds_path)}")
    
    # Try to create a client
    try:
        client = vision.ImageAnnotatorClient()
        print("✓ Google Cloud Vision API client created successfully!")
    except Exception as e:
        print(f"✗ Error creating client: {e}")
else:
    print("✗ GOOGLE_APPLICATION_CREDENTIALS not set")
```

### Troubleshooting

**Error: "DefaultCredentialsError: Could not automatically determine credentials"**
- Make sure `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
- Verify the JSON file exists at the specified path
- Check that the file is readable

**Error: "google.auth.exceptions.RefreshError: The credentials do not contain the `project_id` field"**
- Make sure you downloaded a service account JSON key (not an API key)
- The JSON file should contain a `project_id` field

**Error: "PermissionDenied: Cloud Vision API has not been used in project"**
- Go back to Google Cloud Console and enable the Vision API for your project

### Where Credentials Are Used

The credentials are used in these modules:
- `src/image_req_client/ph_grid_color_reader.py` - Main pH analysis with OCR
- `src/image_req_client/photo_analyzer.py` - Legacy pH analyzers

Both modules automatically load credentials from the `.env` file using `python-dotenv`.

## Other Environment Variables

### Server Configuration (Optional)

```bash
# Port the client connects to (default: 2222)
SERVER_PORT=2222
```

## Security Best Practices

1. **Never commit** your `.env` file or Google Cloud credentials to version control
2. **Restrict permissions** on the JSON key file:
   ```bash
   chmod 600 /path/to/your/google-cloud-credentials.json
   ```
3. **Use separate service accounts** for development and production
4. **Rotate keys** periodically from the Google Cloud Console
5. **Monitor API usage** in the Google Cloud Console to detect unauthorized use

## Quick Start Template

Here's a complete `.env` file template:

```bash
# ============================================
# PiZeroCam Environment Configuration
# ============================================

# Google Cloud Vision API Credentials (REQUIRED for pH analysis)
GOOGLE_APPLICATION_CREDENTIALS="/Users/macbook_m2/credentials/pizerocam-vision-key.json"

# Server Configuration (optional)
SERVER_PORT=2222
```

Save this as `.env` in your project root and update the paths accordingly.


