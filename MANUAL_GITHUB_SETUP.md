# Manual GitHub Repository Setup Guide

Since we're having issues with the automatic GitHub repository creation, here's a detailed manual guide to create and push to a GitHub repository.

## Step 1: Create a Repository on GitHub

1. Open your web browser and go to [GitHub](https://github.com/)
2. Log in to your GitHub account
3. Click the "+" icon in the top-right corner, then select "New repository"
4. Enter the following repository details:
   - Repository name: `multimodal-rag-system`
   - Description: `A Python-based Retrieval-Augmented Generation (RAG) system for complex websites using visual processing`
   - Visibility: Public (or Private if you prefer)
   - DO NOT initialize the repository with a README, .gitignore, or license
5. Click "Create repository"

## Step 2: Push Your Code to GitHub

After creating the repository, you'll see instructions for pushing an existing repository. Follow these steps:

1. Copy your GitHub username from your profile (you'll need it for the commands)

2. Open your terminal in the project directory and run the following commands, replacing `YOUR_USERNAME` with your actual GitHub username:

```bash
# Set your GitHub username
GITHUB_USERNAME="YOUR_USERNAME"

# Add the GitHub repository as a remote
git remote add origin https://github.com/$GITHUB_USERNAME/multimodal-rag-system.git

# Verify the remote was added successfully
git remote -v

# Push your code to GitHub
git push -u origin main
```

3. If prompted for your GitHub username and password:
   - Enter your GitHub username
   - For the password, use a Personal Access Token (PAT) instead of your regular password

## Step 3: Creating a Personal Access Token (If Needed)

If you're prompted for authentication and your regular password doesn't work, GitHub requires a Personal Access Token:

1. Go to [GitHub Settings > Developer Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token" and authenticate if prompted
3. Give your token a descriptive name (e.g., "Multimodal RAG System")
4. Set an expiration date as needed
5. Select at least the "repo" scope for full repository access
6. Click "Generate token"
7. Copy the generated token immediately (you won't be able to see it again)
8. Use this token as your password when prompted during git push

## Step 4: Using SSH Instead of HTTPS (Alternative Method)

If you prefer using SSH or are having issues with HTTPS:

1. Make sure you have an SSH key set up on your computer and added to your GitHub account
2. If not, follow [GitHub's SSH key generation guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
3. Change the remote URL to use SSH:

```bash
# Remove the existing HTTPS remote
git remote remove origin

# Add the SSH remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin git@github.com:YOUR_USERNAME/multimodal-rag-system.git

# Push your code
git push -u origin main
```

## Step 5: Verify the Repository

After successfully pushing your code:

1. Refresh your GitHub repository page in your browser
2. Verify that all files and directories have been pushed correctly
3. Check that the repository structure matches your local project

## Troubleshooting

If you encounter issues:

- Make sure the repository name on GitHub matches exactly: `multimodal-rag-system`
- Verify you have permission to push to the repository (you should if you created it)
- Check your internet connection
- If using a corporate network, check if there are any firewall or proxy issues
- Try pushing a single test file first to verify connectivity

For more help, refer to [GitHub's documentation on pushing to a repository](https://docs.github.com/en/get-started/importing-your-projects-to-github/importing-source-code-to-github/adding-locally-hosted-code-to-github). 