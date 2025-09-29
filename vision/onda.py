import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def create_wave_cmap():
    # Cores oceânicas para as ondas 3D
    colors = ['#000080', '#0080ff', '#00c0ff', '#40e0ff', '#80ffff', '#ffffff']
    return LinearSegmentedColormap.from_list('ocean_waves', colors, N=256)

def process_frame_3d_waves(frame, time_factor=0):
    # Converter e redimensionar para performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar para 60x60 para melhor performance no 3D
    small = cv2.resize(gray, (60, 60))
    blurred = cv2.GaussianBlur(small, (5, 5), 0)

    # Normalizar
    normalized = blurred.astype(np.float32) / 255.0

    # Criar coordenadas X, Y
    height, width = normalized.shape
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)

    # Altura das ondas baseada nos pixels + animação temporal
    Z = normalized * 3 + 0.5 * np.sin(X + time_factor) * np.cos(Y + time_factor * 0.8)

    return X, Y, Z

def main():
    video_path = "/Users/nicolasalan/Documents/stacks/vision/video2.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        return

    cmap = create_wave_cmap()

    # Configurar plot 3D
    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processar frame
        X, Y, Z = process_frame_3d_waves(frame, frame_count * 0.2)

        # Limpar e plotar
        ax.clear()

        # Surface plot das ondas
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9,
                              linewidth=0, antialiased=True, shade=True)

        # Configurações visuais
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(-1, 4)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altura da Onda')
        ax.set_title(f'Ondas 3D do Vídeo - Frame {frame_count}', fontsize=14)

        # Ângulo de visão
        # ax.view_init(elev=45, azim=frame_count * 2)  # Rotação automática

        # Remover grid para visual mais limpo
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.pause(0.05)
        frame_count += 1

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
