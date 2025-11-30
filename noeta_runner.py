"""
Noeta Runner - Main execution script for Noeta DSL
"""
import sys
import os
from pathlib import Path

from noeta_lexer import Lexer
from noeta_parser import Parser
from noeta_codegen import CodeGenerator

def compile_noeta(source_code: str) -> str:
    """Compile Noeta source code to Python code."""
    try:
        # Lexical analysis
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parsing
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Code generation
        generator = CodeGenerator()
        python_code = generator.generate(ast)
        
        return python_code
    except Exception as e:
        raise RuntimeError(f"Compilation error: {str(e)}")

def execute_noeta(source_code: str, verbose: bool = True):
    """Compile and execute Noeta source code."""
    try:
        # Compile to Python
        python_code = compile_noeta(source_code)
        
        if verbose:
            print("=" * 60)
            print("Generated Python Code:")
            print("=" * 60)
            print(python_code)
            print("=" * 60)
            print("Execution Output:")
            print("=" * 60)
        
        # Execute the generated Python code
        exec(python_code, globals())
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

def main():
    """Main entry point for command-line execution."""
    if len(sys.argv) < 2:
        print("Usage: noeta <noeta_file>")
        print("   or: noeta -c '<noeta_code>'")
        sys.exit(1)
    
    if sys.argv[1] == '-c':
        # Execute code from command line
        if len(sys.argv) < 3:
            print("Error: No code provided after -c")
            sys.exit(1)
        source_code = sys.argv[2]
    else:
        # Execute code from file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: File '{file_path}' not found")
            sys.exit(1)
        
        with open(file_path, 'r') as f:
            source_code = f.read()
    
    # Execute the Noeta code
    exit_code = execute_noeta(source_code)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
