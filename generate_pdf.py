from PIL import Image
import glob
import sys

def main():
    png_files = sorted(glob.glob("Figures/*.png"))
    if not png_files:
        print("No PNG files found in Figures/ directory.")
        sys.exit(1)
        
    images = [Image.open(f).convert('RGB') for f in png_files]
    out_path = "Executive_Plots_Summary.pdf"
    
    images[0].save(out_path, save_all=True, append_images=images[1:])
    print(f"Successfully generated {out_path} from {len(png_files)} plots.")

if __name__ == "__main__":
    main()
