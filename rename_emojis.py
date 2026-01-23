
import os
import re
import sys

# Force utf-8 for stdout if possible, or just ignore errors
sys.stdout.reconfigure(encoding='utf-8')

def rename_files():
    base_dir = r"C:\Users\Mohamed Abdalkader\Desktop\Face Recognition - New Streamlit\pages"
    if not os.path.exists(base_dir):
        print("Pages directory not found")
        return

    count = 0
    for filename in os.listdir(base_dir):
        # Check if file has non-ascii characters
        if not filename.isascii() or any(ord(c) > 127 for c in filename):
            old_path = os.path.join(base_dir, filename)
            
            # Remove non-ascii
            clean_name = re.sub(r'[^\x00-\x7F]+', '', filename)
            
            # Cleanup underscores (e.g. 1__Face.py -> 1_Face.py)
            clean_name = re.sub(r'_{2,}', '_', clean_name)
            
            # Ensure it looks like "N_Name.py"
            # If it starts with "1_Face...", good.
            
            new_path = os.path.join(base_dir, clean_name)
            
            try:
                print(f"Renaming: {filename.encode('utf-8', 'replace').decode()} -> {clean_name}")
                os.rename(old_path, new_path)
                count += 1
            except OSError as e:
                print(f"Error renaming: {e}")
    
    print(f"Renamed {count} files.")

if __name__ == "__main__":
    rename_files()
