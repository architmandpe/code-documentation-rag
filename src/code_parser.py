import ast
import re
from typing import List, Dict, Any, Optional

class CodeParser:
    def __init__(self):
        self.parsers = {}
        # Note: tree-sitter-languages removed due to deployment compatibility
    
    def parse_code(self, code: str, language: str) -> Dict[str, Any]:
        """Parse code and extract structural information"""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': [],
            'docstrings': []
        }
        
        if language == 'python':
            result = self._parse_python(code)
        else:
            # Use regex-based parsing for all other languages
            result = self._parse_with_regex(code, language)
        
        return result
    
    def _parse_python(self, code: str) -> Dict[str, Any]:
        """Parse Python code using AST"""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': [],
            'docstrings': []
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'decorators': [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list],
                        'line_start': node.lineno,
                        'line_end': node.end_lineno
                    }
                    result['functions'].append(func_info)
                    if func_info['docstring']:
                        result['docstrings'].append(func_info['docstring'])
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'line_start': node.lineno,
                        'line_end': node.end_lineno
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                    
                    result['classes'].append(class_info)
                    if class_info['docstring']:
                        result['docstrings'].append(class_info['docstring'])
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    else:
                        module = node.module or ''
                        imports = [f"{module}.{alias.name}" for alias in node.names]
                    result['imports'].extend(imports)
        
        except SyntaxError as e:
            print(f"Python parsing error: {str(e)}")
        
        # Extract comments
        comment_pattern = r'#.*$'
        for line in code.split('\n'):
            match = re.search(comment_pattern, line)
            if match:
                result['comments'].append(match.group())
        
        return result
    
    def _parse_with_regex(self, code: str, language: str) -> Dict[str, Any]:
        """Enhanced regex-based parsing for different languages"""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': [],
            'docstrings': []
        }
        
        # Language-specific patterns
        patterns = self._get_language_patterns(language)
        
        # Extract functions
        if patterns.get('functions'):
            for pattern in patterns['functions']:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    func_name = match.group(1) if match.groups() else match.group()
                    result['functions'].append({
                        'name': func_name,
                        'line_start': code[:match.start()].count('\n') + 1
                    })
        
        # Extract classes
        if patterns.get('classes'):
            for pattern in patterns['classes']:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    class_name = match.group(1) if match.groups() else match.group()
                    result['classes'].append({
                        'name': class_name,
                        'line_start': code[:match.start()].count('\n') + 1
                    })
        
        # Extract imports
        if patterns.get('imports'):
            for pattern in patterns['imports']:
                matches = re.findall(pattern, code, re.MULTILINE)
                result['imports'].extend(matches)
        
        # Extract comments
        if patterns.get('comments'):
            for pattern in patterns['comments']:
                matches = re.findall(pattern, code, re.MULTILINE)
                result['comments'].extend(matches)
        
        # Extract docstrings/documentation
        if patterns.get('docstrings'):
            for pattern in patterns['docstrings']:
                matches = re.findall(pattern, code, re.DOTALL)
                result['docstrings'].extend(matches)
        
        return result
    
    def _get_language_patterns(self, language: str) -> Dict[str, List[str]]:
        """Get regex patterns for different languages"""
        patterns = {
            'javascript': {
                'functions': [
                    r'function\s+(\w+)\s*\(',
                    r'const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>',
                    r'(?:export\s+)?(?:async\s+)?function\s+(\w+)',
                    r'(\w+)\s*:\s*(?:async\s+)?function\s*\('
                ],
                'classes': [r'class\s+(\w+)(?:\s+extends\s+\w+)?'],
                'imports': [
                    r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
                    r'require\s*\(["\']([^"\']+)["\']\)',
                    r'import\s*\(["\']([^"\']+)["\']\)'
                ],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': [r'/\*\*[\s\S]*?\*/']
            },
            'typescript': {
                'functions': [
                    r'function\s+(\w+)\s*\(',
                    r'const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>',
                    r'(?:export\s+)?(?:async\s+)?function\s+(\w+)',
                    r'(\w+)\s*:\s*(?:async\s+)?function\s*\('
                ],
                'classes': [r'class\s+(\w+)(?:\s+(?:extends|implements)\s+\w+)?'],
                'imports': [
                    r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
                    r'import\s*\(["\']([^"\']+)["\']\)'
                ],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': [r'/\*\*[\s\S]*?\*/']
            },
            'java': {
                'functions': [
                    r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?'
                ],
                'classes': [
                    r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)',
                    r'interface\s+(\w+)'
                ],
                'imports': [r'import\s+([\w\.]+);'],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': [r'/\*\*[\s\S]*?\*/']
            },
            'cpp': {
                'functions': [
                    r'(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:const)?\s*{',
                    r'(?:template\s*<[^>]+>\s*)?(?:\w+\s+)*(\w+)\s*\([^)]*\)'
                ],
                'classes': [
                    r'class\s+(\w+)',
                    r'struct\s+(\w+)'
                ],
                'imports': [r'#include\s*[<"]([^>"]+)[>"]'],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': [r'/\*\*[\s\S]*?\*/']
            },
            'go': {
                'functions': [r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\('],
                'classes': [r'type\s+(\w+)\s+struct'],
                'imports': [r'import\s+"([^"]+)"', r'import\s+\(([^)]+)\)'],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': []
            },
            'rust': {
                'functions': [r'fn\s+(\w+)\s*(?:<[^>]+>)?\s*\('],
                'classes': [r'struct\s+(\w+)', r'enum\s+(\w+)', r'trait\s+(\w+)'],
                'imports': [r'use\s+([\w:]+);'],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/'],
                'docstrings': [r'///.*$', r'/\*\*[\s\S]*?\*/']
            },
            'ruby': {
                'functions': [r'def\s+(\w+)'],
                'classes': [r'class\s+(\w+)'],
                'imports': [r'require\s+["\']([^"\']+)["\']'],
                'comments': [r'#.*$'],
                'docstrings': []
            },
            'php': {
                'functions': [r'function\s+(\w+)\s*\('],
                'classes': [r'class\s+(\w+)'],
                'imports': [r'(?:require|include)(?:_once)?\s*["\']([^"\']+)["\']'],
                'comments': [r'//.*$', r'/\*[\s\S]*?\*/', r'#.*$'],
                'docstrings': [r'/\*\*[\s\S]*?\*/']
            }
        }
        
        # Default patterns for unknown languages
        default_patterns = {
            'functions': [r'(?:def|function|func)\s+(\w+)\s*\('],
            'classes': [r'(?:class|struct)\s+(\w+)'],
            'imports': [r'(?:import|include|require|use)\s+["\']?([^"\';\s]+)'],
            'comments': [r'//.*$', r'#.*$', r'/\*[\s\S]*?\*/'],
            'docstrings': []
        }
        
        return patterns.get(language, default_patterns)
    
    def extract_api_references(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract API function calls and method invocations"""
        api_refs = []
        
        # Common API patterns
        api_patterns = [
            r'(\w+)\.(\w+)\(',  # object.method()
            r'(\w+)\(',  # function()
            r'new\s+(\w+)\(',  # new Class()
        ]
        
        for pattern in api_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                api_refs.append({
                    'text': match.group(),
                    'type': 'api_call',
                    'position': match.span()
                })
        
        return api_refs
