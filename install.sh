#!/bin/bash
set -e

echo "🚀 Installing Your Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${YELLOW}Python version: $python_version${NC}"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}❌ Python 3.8+ required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}📦 Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install system dependencies
echo -e "${YELLOW}🔧 Checking system dependencies...${NC}"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ FFmpeg not found. Please install it:${NC}"
    echo -e "${YELLOW}Ubuntu/Debian: sudo apt install ffmpeg${NC}"
    echo -e "${YELLOW}macOS: brew install ffmpeg${NC}"
    exit 1
else
    echo -e "${GREEN}✅ FFmpeg found${NC}"
fi

# Check Java (for benepar)
if ! command -v java &> /dev/null; then
    echo -e "${YELLOW}⚠️  Java not found. Installing default-jdk...${NC}"
    sudo apt install -y default-jdk || echo -e "${RED}Please install Java manually${NC}"
else
    echo -e "${GREEN}✅ Java found${NC}"
fi

# Install Python packages
echo -e "${YELLOW}📦 Installing Python packages...${NC}"
pip install -r requirements.txt

# Download spaCy model
echo -e "${YELLOW}📦 Downloading spaCy model...${NC}"
python -m spacy download en_core_web_sm

# Download NLTK data
echo -e "${YELLOW}📦 Downloading NLTK data...${NC}"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data models logs

echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${YELLOW}To activate the environment, run: source venv/bin/activate${NC}"
