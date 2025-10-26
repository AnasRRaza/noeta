"""
Install Noeta Jupyter Kernel
"""
import json
import os
import sys
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

def install_kernel():
    """Install the Noeta kernel specification."""
    
    # Get the directory where this script is located
    kernel_dir = Path(__file__).parent
    
    # Create kernel.json specification
    kernel_spec = {
        "argv": [
            sys.executable,
            str(kernel_dir / "noeta_kernel.py"),
            "-f",
            "{connection_file}"
        ],
        "display_name": "Noeta",
        "language": "noeta",
        "name": "noeta"
    }
    
    # Create a temporary directory for the kernel spec
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_dir = Path(tmpdir) / "noeta"
        spec_dir.mkdir()
        
        # Write kernel.json
        with open(spec_dir / "kernel.json", "w") as f:
            json.dump(kernel_spec, f, indent=2)
        
        # Install the kernel spec
        km = KernelSpecManager()
        km.install_kernel_spec(str(spec_dir), "noeta", user=True, replace=True)
        
    print("Noeta kernel installed successfully!")
    print("You can now use it in Jupyter by selecting 'Noeta' when creating a new notebook.")

if __name__ == "__main__":
    install_kernel()
