#!/bin/bash

# Push the repository to GitHub
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME

# Check if a username was provided
if [ -z "$1" ]; then
  echo "Error: Please provide your GitHub username"
  echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
  exit 1
fi

# Assign the GitHub username to a variable
GITHUB_USERNAME=$1

# Repository name
REPO_NAME="multimodal-rag-system"

# Set the remote repository URL
echo "Setting up remote repository..."
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Verify the remote was added
echo "Verifying remote..."
git remote -v

# Push the code to GitHub
echo "Pushing code to GitHub..."
git push -u origin main

# Check if the push was successful
if [ $? -eq 0 ]; then
  echo "Success! Your code has been pushed to GitHub."
  echo "You can access your repository at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
else
  echo "Error: Failed to push to GitHub."
  echo "Please make sure you have:"
  echo "1. Created the repository on GitHub: $REPO_NAME"
  echo "2. Have the correct access permissions"
  echo "3. The repository is empty (no README, LICENSE, etc.)"
  
  # If there's an authentication issue, provide additional instructions
  echo ""
  echo "If you're having authentication issues:"
  echo "1. You may need to create a Personal Access Token (PAT) on GitHub"
  echo "2. Go to GitHub > Settings > Developer settings > Personal access tokens"
  echo "3. Generate a new token with repo permissions"
  echo "4. Use the token as your password when prompted"
  echo ""
  echo "Alternatively, you can set up SSH access:"
  echo "1. Run: git remote set-url origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
  echo "2. Make sure your SSH key is set up and added to your GitHub account"
  echo "3. Run: git push -u origin main"
fi 