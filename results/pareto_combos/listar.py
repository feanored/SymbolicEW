import os
import subprocess
import numpy as np
from natsort import natsorted
from PIL import Image # pip install pillow

pasta = "."
imagens = natsorted([f for f in os.listdir(pasta) if f.endswith('.png')])

# Renomear
def rename():
    for i, img in enumerate(imagens, 1):
        antigo = os.path.join(pasta, img)
        novo = os.path.join(pasta, f"{i:06d}.png")
        os.rename(antigo, novo)
        if i % 500 == 0:
            print(f"Renomeadas {i}...")

    print(f"✅ {len(imagens)} imagens renomeadas!")

# Gerar o vídeo
def video():
    result = subprocess.run(
        r'ffmpeg -framerate 2 -i "%06d.png" -c:v libx264 -pix_fmt yuv420p output.mp4',
        shell=True,
        cwd=pasta,
    )
    if result.returncode == 0:
        print(f"✅ Vídeo gerado com sucesso!")

# Gerar um gif
def gif():
    image_files = natsorted([os.path.join(pasta, f) for f in os.listdir(pasta) if (f.endswith(('.png')) and np.random.random() < 0.2)])
    images = [Image.open(img).convert('RGBA') for img in image_files]
    images[0].save(os.path.join(pasta, "output.gif"), save_all=True, append_images=images[1:], duration=500, loop=0)
    print(f"✅ GIF gerado com sucesso!")

if __name__ == "__main__":
    rename()
    # video()
    gif()