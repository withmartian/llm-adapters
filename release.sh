#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting release process..."

# Step 1: Generate documentation
echo "📝 Generating documentation..."
python docs/generate.py

# Step 2: Check if there are changes to commit
if [[ -n $(git status docs/index.md --porcelain) ]]; then
    echo "📋 Committing documentation changes..."
    git add docs/index.md
    git commit -m "docs: update supported models table"
    
    echo "⬆️  Pushing documentation changes..."
    git push
else
    echo "✅ No documentation changes to commit"
fi

# Step 3: Build and publish
echo "📦 Building and publishing package..."
poetry publish --build

echo "✨ Release complete!" 