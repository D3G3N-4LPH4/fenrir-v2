#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
#               FENRIR TRADING BOT - QUICK SETUP SCRIPT
# ═══════════════════════════════════════════════════════════════════════════
#
# This script helps you get FENRIR up and running quickly.
# Run with: bash setup.sh
#
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "🐺 FENRIR Trading Bot - Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    echo "   Please install Python 3.9 or higher: https://python.org"
    exit 1
fi

echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Setup environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo ""
    echo "🔐 IMPORTANT: Edit .env to add your configuration:"
    echo "   - SOLANA_RPC_URL (get from QuickNode, Helius, etc.)"
    echo "   - WALLET_PRIVATE_KEY (for live trading only)"
    echo ""
    echo "   Run: nano .env"
else
    echo "ℹ️  .env file already exists (not overwriting)"
fi
echo ""

# Setup config file
if [ ! -f "config.json" ]; then
    cp config.example.json config.json
    echo "✅ Created config.json from template"
else
    echo "ℹ️  config.json already exists (not overwriting)"
fi
echo ""

# Final instructions
echo "═══════════════════════════════════════════════════════════════"
echo "🎉 Setup complete!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Configure your settings:"
echo "   nano .env"
echo ""
echo "2. Test in simulation mode (SAFE - no real trades):"
echo "   python -m fenrir --mode simulation"
echo ""
echo "3. When ready for live trading:"
echo "   python -m fenrir --mode conservative"
echo ""
echo "📚 For full documentation, read: README.md"
echo ""
echo "⚠️  REMEMBER:"
echo "   - Start with simulation mode"
echo "   - Use only funds you can afford to lose"
echo "   - Memecoins are extremely high risk"
echo "   - This is educational software, not financial advice"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🐺 Happy hunting!"
