import tree_sitter_languages as tsl
from typing import List, Dict, Any, Optional
import ast
import re

class CodeParser:
    def __init__(self):
        self.parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        languages = ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust']
        for lang in languages:
            try:
                self.parsers[lang] = tsl.get_parser(lang)
            except:
                print(f"Parser for {lang} not available")
    
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
        elif language in self.parsers:
            result = self._parse_with_tree_sitter(code, language)
        else:
            result = self._basic_parse(code)
        
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
    
    def _parse_with_tree_sitter(self, code: str, language: str) -> Dict[str, Any]:
        """Parse code using tree-sitter"""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': [],
            'docstrings': []
        }
        
        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(code, 'utf8'))
            
            # Query patterns for different languages
            queries = self._get_language_queries(language)
            
            for query_type, query_pattern in queries.items():
                if query_pattern:
                    query = parser.query(query_pattern)
                    captures = query.captures(tree.root_node)
                    
                    for node, name in captures:
                        text = code[node.start_byte:node.end_byte]
                        result[query_type].append(text)
        
        except Exception as e:
            print(f"Tree-sitter parsing error for {language}: {str(e)}")
        
        return result
    
    def _get_language_queries(self, language: str) -> Dict[str, str]:
        """Get tree-sitter query patterns for different languages"""
        queries = {
            'javascript': {
                'functions': '(function_declaration name: (identifier) @func)',
                'classes': '(class_declaration name: (identifier) @class)',
                'imports': '(import_statement) @import',
                'comments': '(comment) @comment'
            },
            'typescript': {
                'functions': '(function_declaration name: (identifier) @func)',
                'classes': '(class_declaration name: (type_identifier) @class)',
                'imports': '(import_statement) @import',
                'comments': '(comment) @comment'
            },
            'java': {
                'functions': '(method_declaration name: (identifier) @method)',
                'classes': '(class_declaration name: (identifier) @class)',
                'imports': '(import_declaration) @import',
                'comments': '(comment) @comment'
            }
        }
        
        return queries.get(language, {})
    
    def _basic_parse(self, code: str) -> Dict[str, Any]:
        """Basic parsing using regex patterns"""
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': [],
            'docstrings': []
        }
        
        # Basic regex patterns
        patterns = {
            'functions': r'(?:def|function|func)\s+(\w+)\s*\(',
            'classes': r'(?:class|struct)\s+(\w+)',
            'imports': r'(?:import|include|require|use)\s+["\']?([^"\';\s]+)',
            'comments': r'(?://|#|/\*|\*).*'
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            result[key] = matches
        
        return result
    
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