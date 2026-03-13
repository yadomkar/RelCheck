#!/bin/bash
# ============================================================
# RelCheck — One-time GitHub setup script
# Run this once from your CS298 folder to push everything to GitHub
#
# Usage:
#   1. Create a new repo on github.com called "RelCheck" (public, no README)
#   2. Replace YOUR_GITHUB_USERNAME below
#   3. Run: bash setup.sh
# ============================================================

GITHUB_USERNAME="siddhipatil503"   # ← CHANGE THIS
REPO_NAME="RelCheck"
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "🚀 Setting up RelCheck GitHub repo..."
echo "Remote: ${REMOTE_URL}"
echo ""

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git initialized"
fi

# Create eval and figures directories (with .gitkeep so git tracks them)
mkdir -p eval figures
touch eval/.gitkeep figures/.gitkeep

# Create relcheck/__init__.py so it's a proper Python package
touch relcheck/__init__.py

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv/
venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# Large model files (don't commit)
*.bin
*.safetensors
*.pt
*.pth

# Results (generated — commit selectively)
# eval/*.csv   ← uncomment to also ignore CSVs

# macOS
.DS_Store

# API keys (NEVER commit)
secrets.py
api_keys.py
EOF
echo "✅ .gitignore created"

# Stage everything
git add README.md requirements.txt setup.sh relcheck/ images/ \
        RelCheck_Master.ipynb relcheck_blip2_probe.ipynb \
        CONTEXT.md eval/.gitkeep figures/.gitkeep .gitignore \
        relcheck/__init__.py 2>/dev/null

git status

echo ""
echo "Ready to commit and push. Run:"
echo "  git commit -m 'Initial RelCheck commit — all pipeline modules'"
echo "  git branch -M main"
echo "  git remote add origin ${REMOTE_URL}"
echo "  git push -u origin main"
echo ""
echo "Or press Enter to do it automatically now."
read -p "Auto-push? (y/n): " confirm

if [ "$confirm" = "y" ]; then
    git commit -m "Initial RelCheck commit — all pipeline modules"
    git branch -M main
    git remote remove origin 2>/dev/null
    git remote add origin "${REMOTE_URL}"
    git push -u origin main
    echo ""
    echo "✅ Pushed to ${REMOTE_URL}"
    echo "Open: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
else
    echo "Run the git commands above manually when ready."
fi
