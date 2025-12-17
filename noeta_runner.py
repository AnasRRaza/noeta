"""
Noeta Runner - Main execution script for Noeta DSL
"""
import sys
import os
from pathlib import Path

from noeta_lexer import Lexer
from noeta_parser import Parser
from noeta_codegen import CodeGenerator
from noeta_semantic import SemanticAnalyzer
from noeta_errors import NoetaError

def compile_noeta(source_code: str) -> str:
    """Compile Noeta source code to Python code."""
    try:
        # Lexical analysis
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()

        # Parsing (pass source code for error context)
        parser = Parser(tokens, source_code)
        ast = parser.parse()

        # Semantic validation (NEW: catch errors at compile-time)
        analyzer = SemanticAnalyzer(source_code)
        errors = analyzer.analyze(ast)

        if errors:
            # Raise the first error (could show all errors in future)
            raise errors[0]

        # Code generation
        generator = CodeGenerator()
        python_code = generator.generate(ast)

        return python_code
    except NoetaError:
        # Re-raise NoetaError as-is (already formatted)
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected compilation error: {str(e)}\nPlease report this as a bug")

def execute_noeta(source_code: str, verbose: bool = False):
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

    except NoetaError as e:
        # NoetaError is already beautifully formatted
        print(str(e), file=sys.stderr)
        return 1
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
