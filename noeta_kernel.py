"""
Noeta Jupyter Kernel
"""
import sys
import json
from ipykernel.kernelbase import Kernel
from noeta_runner import compile_noeta
import io
import contextlib

class NoetaKernel(Kernel):
    implementation = 'Noeta'
    implementation_version = '1.0'
    language = 'noeta'
    language_version = '1.0'
    language_info = {
        'name': 'noeta',
        'mimetype': 'text/x-noeta',
        'file_extension': '.noeta',
        'pygments_lexer': 'python',  # Use Python highlighting for now
        'codemirror_mode': 'python'
    }
    banner = "Noeta - Data Analysis DSL Kernel"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.namespace = {}
        # Add get_ipython to namespace so generated code can detect Jupyter environment
        self.namespace['get_ipython'] = lambda: self
    
    def do_execute(self, code, silent, store_history=True,
                   user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {}
            }
        
        # Capture stdout
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            # Compile Noeta to Python
            python_code = compile_noeta(code)
            
            # Capture output while executing
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                # Execute in the kernel's namespace
                exec(python_code, self.namespace)
            
            # Send output
            output = stdout.getvalue()
            if output and not silent:
                stream_content = {
                    'name': 'stdout',
                    'text': output
                }
                self.send_response(self.iopub_socket, 'stream', stream_content)
            
            error_output = stderr.getvalue()
            if error_output:
                stream_content = {
                    'name': 'stderr',
                    'text': error_output
                }
                self.send_response(self.iopub_socket, 'stream', stream_content)
            
            # Check if matplotlib figure exists
            # Import matplotlib from the execution namespace to check for figures
            if 'plt' in self.namespace:
                import matplotlib.pyplot as plt
                import base64
                from io import BytesIO

                # Check if there are any figures
                if plt.get_fignums():
                    # Save and display each figure
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        buf = BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)

                        # Send image data
                        image_data = base64.b64encode(buf.read()).decode()
                        content = {
                            'data': {
                                'image/png': image_data
                            },
                            'metadata': {}
                        }
                        self.send_response(self.iopub_socket, 'display_data', content)
                        buf.close()

                    # Clear all figures for next execution
                    plt.close('all')
            
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {}
            }
            
        except Exception as e:
            # Send error message
            if not silent:
                err_content = {
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': [str(e)]
                }
                self.send_response(self.iopub_socket, 'error', err_content)
            
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [str(e)]
            }
    
    def do_complete(self, code, cursor_pos):
        """Basic completion support."""
        # Simple keyword completion
        keywords = [
            'load', 'select', 'filter', 'sort', 'join', 'groupby',
            'sample', 'dropna', 'fillna', 'mutate', 'apply',
            'describe', 'summary', 'info', 'outliers', 'quantile',
            'normalize', 'binning', 'rolling', 'hypothesis',
            'boxplot', 'heatmap', 'pairplot', 'timeseries', 'pie',
            'save', 'export_plot', 'as', 'by', 'with', 'on'
        ]
        
        # Find the word being typed
        text_before = code[:cursor_pos]
        word_start = len(text_before)
        for i in range(len(text_before) - 1, -1, -1):
            if text_before[i] in ' \t\n':
                word_start = i + 1
                break
        else:
            word_start = 0
        
        word = text_before[word_start:]
        
        # Find matching completions
        matches = [kw for kw in keywords if kw.startswith(word)]
        
        return {
            'matches': matches,
            'cursor_start': word_start,
            'cursor_end': cursor_pos,
            'metadata': {},
            'status': 'ok'
        }

def main():
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=NoetaKernel)

if __name__ == '__main__':
    main()
