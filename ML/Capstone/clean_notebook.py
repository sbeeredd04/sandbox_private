#!/usr/bin/env python3
"""
Script to remove duplicate sections from milestone2_cnn.ipynb
"""
import json
import sys

def clean_notebook(input_file, output_file):
    # Read the notebook
    with open(input_file, 'r') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    cleaned_cells = []
    
    skip_mode = False
    for i, cell in enumerate(cells):
        # Check if this is the start of duplicate section
        if cell['cell_type'] == 'markdown' and 'source' in cell:
            content = ''.join(cell['source'])
            
            # Start skipping at duplicate "## 5. Model Training"
            if '## 5. Model Training' in content and i > 100:  # After section 6
                print(f"Skipping duplicate section starting at cell {i}: ## 5. Model Training")
                skip_mode = True
                continue
            
            # Stop skipping when we reach Cross-Validation
            if '## 9. Cross-Validation' in content:
                print(f"Resuming at cell {i}: ## 9. Cross-Validation")
                skip_mode = False
        
        # Add cell if not in skip mode
        if not skip_mode:
            cleaned_cells.append(cell)
        else:
            # Log what we're skipping
            if cell['cell_type'] == 'markdown' and 'source' in cell:
                content = ''.join(cell['source'])[:100]
                print(f"  Skipping cell {i}: {content}")
    
    # Update notebook
    nb['cells'] = cleaned_cells
    
    # Save cleaned notebook
    with open(output_file, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\nCleaned notebook saved to: {output_file}")
    print(f"Original cells: {len(cells)}")
    print(f"Cleaned cells: {len(cleaned_cells)}")
    print(f"Removed cells: {len(cells) - len(cleaned_cells)}")

if __name__ == '__main__':
    input_file = '/home/sbeeredd/sandbox_private/ML/Capstone/milestone2_cnn.ipynb'
    output_file = '/home/sbeeredd/sandbox_private/ML/Capstone/milestone2_cnn_clean.ipynb'
    
    clean_notebook(input_file, output_file)
    print("\n✅ Done! Review milestone2_cnn_clean.ipynb and if it looks good, rename it to milestone2_cnn.ipynb")

