import sys
import nbformat

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_cell.py <notebook_path> <cell_type> < <content_file>")
        sys.exit(1)
        
    nb_path = sys.argv[1]
    cell_type = sys.argv[2]
    content = sys.stdin.read().strip()
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    if cell_type == 'markdown':
        nb.cells.append(nbformat.v4.new_markdown_cell(content))
    elif cell_type == 'code':
        nb.cells.append(nbformat.v4.new_code_cell(content))
        
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully added {cell_type} cell to {nb_path}.")
