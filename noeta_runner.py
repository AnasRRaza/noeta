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
from noeta_errors import NoetaError, create_multi_error

def compile_noeta(source_code: str, enable_type_check: bool = False) -> str:
    """
    Compile Noeta source code to Python code.

    Args:
        source_code: Noeta source code to compile
        enable_type_check: If True, enables compile-time type checking (reads file schemas)

    Returns:
        Generated Python code
    """
    try:
        # Lexical analysis
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()

        # Parsing (pass source code for error context)
        parser = Parser(tokens, source_code)
        ast = parser.parse()

        # Semantic validation (with optional type checking)
        analyzer = SemanticAnalyzer(source_code, enable_type_check=enable_type_check)
        errors = analyzer.analyze(ast)

        if errors:
            # Show all errors if multiple, or single error if just one
            if len(errors) == 1:
                raise errors[0]
            else:
                raise create_multi_error(errors)

        # Code generation
        generator = CodeGenerator()
        python_code = generator.generate(ast)

        return python_code
    except NoetaError:
        # Re-raise NoetaError as-is (already formatted)
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected compilation error: {str(e)}\nPlease report this as a bug")

def execute_noeta(source_code: str, verbose: bool = False, enable_type_check: bool = False):
    """
    Compile and execute Noeta source code.

    Args:
        source_code: Noeta source code to execute
        verbose: If True, shows generated Python code
        enable_type_check: If True, enables compile-time type checking

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Compile to Python (with optional type checking)
        python_code = compile_noeta(source_code, enable_type_check=enable_type_check)

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
    import argparse

    parser = argparse.ArgumentParser(
        description='Noeta DSL Compiler and Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python noeta_runner.py script.noeta
  python noeta_runner.py script.noeta -v
  python noeta_runner.py script.noeta --type-check
  python noeta_runner.py -c 'load "data.csv" as d\\ndescribe d'
  python noeta_runner.py -c 'load "data.csv" as d\\nselect d {price} as result' --type-check
        """
    )
    parser.add_argument('file', nargs='?', help='Noeta file to execute')
    parser.add_argument('-c', '--code', help='Execute inline Noeta code')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show generated Python code')
    parser.add_argument('--type-check', action='store_true',
                       help='Enable compile-time type checking (reads file schemas)')

    args = parser.parse_args()

    # Get source code from file or inline
    if args.code:
        source_code = args.code
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' not found", file=sys.stderr)
            sys.exit(1)

        with open(file_path, 'r') as f:
            source_code = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Execute the Noeta code with specified options
    exit_code = execute_noeta(source_code, verbose=args.verbose,
                             enable_type_check=args.type_check)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
