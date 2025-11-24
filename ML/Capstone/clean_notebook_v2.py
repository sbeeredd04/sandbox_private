#!/usr/bin/env python3
"""
Script to remove duplicate sections from milestone2_cnn.ipynb
Removes cells 41-50 which are duplicates of Section 6 content
"""
import json
import shutil

def clean_notebook(input_file, output_file, backup=True):
    # Backup original
    if backup:
        shutil.copy(input_file, input_file + '.backup')
        print(f"✓ Backup created: {input_file}.backup")
    
    # Read the notebook
    with open(input_file, 'r') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    print(f"\nOriginal notebook has {len(cells)} cells")
    
    # Print cells 41-50 to confirm what we're removing
    print("\nCells to remove (41-50):")
    for i in range(41, min(51, len(cells))):
        cell = cells[i]
        if cell['cell_type'] == 'markdown' and cell['source']:
            content = ''.join(cell['source'])[:80].replace('\n', ' ')
            print(f"  Cell {i}: [markdown] {content}")
        else:
            content = (''.join(cell['source'])[:60] if cell['source'] else 'empty').replace('\n', ' ')
            print(f"  Cell {i}: [code] {content}...")
    
    # Remove cells 41-50 (duplicate sections 5, 7, 8)
    cleaned_cells = cells[:41] + cells[51:]
    
    # Update notebook
    nb['cells'] = cleaned_cells
    
    # Save cleaned notebook
    with open(output_file, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Cleaned notebook saved to: {output_file}")
    print(f"   Original cells: {len(cells)}")
    print(f"   Cleaned cells: {len(cleaned_cells)}")
    print(f"   Removed cells: {len(cells) - len(cleaned_cells)}")
    
    # Show new structure
    print("\nNew section structure:")
    for i, cell in enumerate(cleaned_cells):
        if cell['cell_type'] == 'markdown' and cell['source']:
            content = ''.join(cell['source'])
            if content.startswith('##'):
                title = content.split('\n')[0]
                print(f"  Cell {i}: {title}")

if __name__ == '__main__':
    input_file = '/home/sbeeredd/sandbox_private/ML/Capstone/milestone2_cnn.ipynb'
    output_file = '/home/sbeeredd/sandbox_private/ML/Capstone/milestone2_cnn.ipynb'
    
    clean_notebook(input_file, output_file, backup=True)
    print("\n🎉 Done! Duplicate sections removed.")

