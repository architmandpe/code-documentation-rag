import os
import tempfile
from typing import List, Dict, Any
from github import Github
import git
from pathlib import Path

class GitHubLoader:
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        if github_token:
            self.github_client = Github(github_token)
        else:
            self.github_client = Github()
    
    def clone_repository(self, repo_url: str) -> str:
        """Clone a GitHub repository to a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            git.Repo.clone_from(repo_url, temp_dir)
            return temp_dir
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")
    
    def load_repository_files(self, repo_path: str, extensions: List[str]) -> List[Dict[str, Any]]:
        """Load all relevant files from the repository"""
        documents = []
        repo_path = Path(repo_path)
        
        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                # Skip hidden directories and common non-source directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(skip in str(file_path) for skip in ['node_modules', '__pycache__', 'venv', 'env']):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append({
                        'content': content,
                        'metadata': {
                            'file_path': str(file_path.relative_to(repo_path)),
                            'file_type': file_path.suffix,
                            'language': self._detect_language(file_path.suffix),
                            'file_name': file_path.name
                        }
                    })
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
        
        return documents
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        from config.settings import settings
        return settings.CODE_EXTENSIONS.get(extension, 'unknown')
    
    def get_repository_info(self, repo_url: str) -> Dict[str, Any]:
        """Get repository metadata"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        owner = repo_url.split('/')[-2]
        
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            return {
                'name': repo.name,
                'description': repo.description,
                'language': repo.language,
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'topics': repo.get_topics()
            }
        except:
            return {'name': repo_name, 'owner': owner}