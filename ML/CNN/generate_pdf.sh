#!/bin/bash

# CNN Report PDF Generator
# This script converts the CNN_REPORT.md to PDF

echo "🔄 Converting CNN_REPORT.md to PDF..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is not installed"
    echo "📦 Install with: brew install pandoc (macOS)"
    exit 1
fi

# Convert markdown to PDF
pandoc CNN_REPORT.md \
    -o CNN_Report.pdf \
    --pdf-engine=xelatex \
    --toc \
    --toc-depth=2 \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --highlight-style=tango

if [ $? -eq 0 ]; then
    echo "✅ PDF generated successfully: CNN_Report.pdf"
else
    echo "❌ Error generating PDF"
    echo "Alternative: Use an online markdown to PDF converter"
    exit 1
fi
