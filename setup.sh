#!/bin/bash
# filepath: setup_environment.sh

# Document Analyzer - Environment Setup Script
# This script creates a virtual environment and installs all required dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Detect the Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    fi
    
    case $OS in
        *"Ubuntu"*|*"Debian"*)
            print_status "Detected Ubuntu/Debian system"
            sudo apt-get update
            sudo apt-get install -y \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                curl \
                wget \
                git \
                pkg-config \
                libffi-dev \
                libssl-dev
            ;;
        *"CentOS"*|*"Red Hat"*|*"Fedora"*)
            print_status "Detected RedHat/CentOS/Fedora system"
            sudo yum install -y \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++ \
                make \
                curl \
                wget \
                git \
                openssl-devel \
                libffi-devel
            ;;
        *"Arch"*)
            print_status "Detected Arch Linux system"
            sudo pacman -S --noconfirm \
                python-pip \
                python-virtualenv \
                base-devel \
                curl \
                wget \
                git \
                openssl \
                libffi
            ;;
        *)
            print_warning "Unknown Linux distribution. Attempting to continue..."
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        print_warning "Existing virtual environment found. Removing..."
        rm -rf venv
    fi
    
    # Create new virtual environment
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment and install Python packages
install_python_deps() {
    print_status "Activating virtual environment and installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip, setuptools, and wheel
    print_status "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    # Install main requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing main requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found. Installing core dependencies manually..."
        pip install \
            pandas \
            numpy \
            requests \
            python-dotenv \
            PyPDF2 \
            langchain \
            chromadb \
            sentence-transformers \
            ollama
    fi
    
    # Install evaluation requirements if they exist
    if [ -f "evaluation_requirements.txt" ]; then
        print_status "Installing evaluation requirements..."
        pip install -r evaluation_requirements.txt
    else
        print_status "Installing evaluation dependencies manually..."
        pip install \
            matplotlib \
            seaborn \
            scikit-learn \
            psutil
    fi
    
    print_success "Python dependencies installed"
}

# Install Ollama (optional)
install_ollama() {
    print_status "Checking for Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama is already installed"
    else
        read -p "Do you want to install Ollama? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Installing Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh
            print_success "Ollama installed"
            
            print_status "Starting Ollama service..."
            ollama serve &
            sleep 5
            
            print_status "Pulling recommended model (gemma2:2b)..."
            ollama pull gemma2:2b
        else
            print_warning "Skipping Ollama installation"
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p documents
    mkdir -p output
    mkdir -p chroma_db
    mkdir -p analysis_results
    
    print_success "Directories created"
}

# Set up configuration files
setup_config() {
    print_status "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env configuration file..."
        cat > .env << EOL
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b
OLLAMA_TIMEOUT=60

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database Configuration
VECTOR_DB_PATH=./chroma_db
COLLECTION_NAME=documents

# Processing Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100
MAX_SECTIONS=8
MAX_SUBSECTIONS=5
TOP_K_RETRIEVAL=10

# Performance Configuration
MAX_PROCESSING_TIME=180
MAX_MEMORY_USAGE=4096
CPU_THREADS=4
BATCH_SIZE=5
MAX_FILE_SIZE=50
MAX_DOCUMENTS=10

# Directories
DOCUMENTS_DIR=./documents
OUTPUT_DIR=./output
EOL
        print_success ".env file created"
    else
        print_success ".env file already exists"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "
import sys
import pkg_resources

required_packages = [
    'pandas', 'numpy', 'requests', 'matplotlib', 
    'seaborn', 'scikit-learn', 'psutil'
]

missing_packages = []
for package in required_packages:
    try:
        pkg_resources.get_distribution(package)
        print(f'✅ {package} - OK')
    except pkg_resources.DistributionNotFound:
        missing_packages.append(package)
        print(f'❌ {package} - MISSING')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All packages installed successfully!')
"
    
    print_success "Installation verification completed"
}

# Main installation process
main() {
    echo "================================================"
    echo "Document Analyzer - Environment Setup"
    echo "================================================"
    echo
    
    check_python
    install_system_deps
    create_venv
    install_python_deps
    create_directories
    setup_config
    install_ollama
    verify_installation
    
    echo
    echo "================================================"
    print_success "Setup completed successfully!"
    echo "================================================"
    echo
    echo "To activate the virtual environment, run:"
    echo "  source venv/bin/activate"
    echo
    echo "To run the document analyzer:"
    echo "  python document_analyzer.py input.json output.json"
    echo
    echo "To run evaluation demo:"
    echo "  python demo_evaluation.py"
    echo
    echo "To run detailed analysis:"
    echo "  python run_detailed_analysis.py"
    echo
}

# Run main function
main "$@"