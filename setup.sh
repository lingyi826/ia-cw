#!/usr/bin/env bash
set -e

echo "-> Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    echo "Error: Python 3.11+ is required. Found version $PYTHON_VERSION"
    exit 1
fi
echo "-> Python version $PYTHON_VERSION check passed."

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "-> Creating virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
else
    echo "-> Virtual environment ($VENV_DIR) already exists."
fi

PIP_EXE="$VENV_DIR/bin/pip"
PYTHON_EXE="$VENV_DIR/bin/python"

echo "-> Upgrading pip and setuptools..."
"$PYTHON_EXE" -m pip install --upgrade pip setuptools==84.2.1

# ==========================================
# CONFIGURE YOUR TWO SUBFOLDERS HERE
# ==========================================
PROJECT_DIRS=(
    "src"
)

echo "-> Configuring Python path for project subfolders..."
"$PYTHON_EXE" - "${PROJECT_DIRS[@]}" << 'EOF'
import sys, sysconfig, pathlib

site_packages = pathlib.Path(sysconfig.get_path('purelib'))
pth_file = site_packages / 'project_subfolders.pth'
root = pathlib.Path.cwd()

lines = []
for folder in sys.argv[1:]:
    full_path = root / folder
    if full_path.exists():
        lines.append(str(full_path))
        print(f'   Adding to Python path: {full_path}')
    else:
        print(f'   Warning: Directory {full_path} does not exist yet.')

pth_file.write_text('\n'.join(lines) + '\n')
print(f'-> Created configuration at {pth_file}')
EOF

# Common dependencies
COMMON_DEPS=(
    "filelock==4.2.10"
    "fsspec==2026.9.0"
    "jinja2==3.1.6"
    "markupsafe==3.0.4"
    "mpmath==1.3.0"
    "pyyaml==6.0.3"
    "sympy==1.14.0"
    "torch==2.15.0"
    "typing-extensions==4.16.0"
)

# Version-specific dependencies based on Python lockfile markers
if [ "$PYTHON_MINOR" -eq 11 ]; then
    echo "-> Configuring for Python 3.11 (Lockfile markers < 3.12)..."
    VERSION_SPECIFIC_DEPS=(
        "networkx==3.6.1"
        "numpy==2.4.6"
    )
else
    echo "-> Configuring for Python 3.12+ (Lockfile markers >= 3.12)..."
    VERSION_SPECIFIC_DEPS=(
        "networkx==3.7"
        "numpy==2.5.4"
    )
fi

CUDA_DEPS=(
    "cuda-bindings==13.5.0"
    "cuda-pathfinder==1.8.6"
    "cuda-toolkit==13.2.2"
    "nvidia-cublas==13.4.1.3"
    "nvidia-cuda-cccl==13.3.5.0.6"
    "nvidia-cuda-cupti==13.2.86"
    "nvidia-cuda-nvrtc==13.2.86"
    "nvidia-cuda-runtime==13.2.86"
    "nvidia-cudnn-cu13==9.26.0.51"
    "nvidia-cufft==12.2.0.57"
    "nvidia-cufile==1.17.1.22"
    "nvidia-cusolver==12.2.0.11"
    "nvidia-cusparse==12.7.10.12"
    "nvidia-cusparselt-cu13==0.8.1"
    "nvidia-nccl-cu13==2.30.7"
    "nvidia-nvjitlink==13.5.37"
    "nvidia-nvshmem-cu13==3.7.2"
    "nvidia-nvtx==13.2.86"
    "triton==3.9.0"
)

OS_TYPE="$(uname -s)"
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "-> macOS detected: Skipping CUDA binaries..."
    PACKAGES_TO_INSTALL=("${COMMON_DEPS[@]}" "${VERSION_SPECIFIC_DEPS[@]}")
else
    echo "-> Linux/Unix detected: Installing full stack including CUDA libraries..."
    PACKAGES_TO_INSTALL=("${COMMON_DEPS[@]}" "${VERSION_SPECIFIC_DEPS[@]}" "${CUDA_DEPS[@]}")
fi

echo "-> Installing packages in a single command..."
"$PIP_EXE" install "${PACKAGES_TO_INSTALL[@]}"

echo ""
echo "✅ Setup complete successfully!"
echo "To activate your environment, run:"
echo "  source .venv/bin/activate"
