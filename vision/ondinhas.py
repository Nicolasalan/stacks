import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def lerp(t, a, b):
    return t * (b - a) + a

def map_range(t, a, b, c, d):
    return (t - a) / (b - a) * (d - c) + c

def process_frame_waves(frame, t=0):
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar para melhor performance (opcional)
    h, w = gray.shape
    if w > 400:
        scale = 400 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h))
        h, w = new_h, new_w

    # Configurações das ondas
    h_divisions = 60
    h_division_size = h / h_divisions
    max_amp = h_division_size / 2

    # Frequência fixa
    freq = 100
    phase = t / 2

    # Criar figura para desenhar
    fig_data = np.zeros((h, w, 3), dtype=np.uint8)  # RGB

    for h_div in range(h_divisions):
        y_center = int((h_division_size / 2) + h_div * h_division_size)

        # Pontos da onda para esta linha horizontal
        wave_points = []

        for x in range(w):
            # Calcular ângulo e valor do seno
            angle = map_range(x, 0, w, 0, np.pi * 2)
            sin_value = np.sin(phase + angle * freq)

            # Pegar valor de cinza do pixel
            if y_center < h:
                gray_value = gray[y_center, x]
            else:
                gray_value = 0

            # Amplitude baseada no valor de cinza (invertido: escuro = mais amplitude)
            amplitude = map_range(gray_value, 0, 255, max_amp, 0)

            # Posição final do ponto da onda
            y_wave = int(y_center + sin_value * amplitude)
            wave_points.append((x, y_wave))

        # Desenhar a linha da onda
        for i in range(len(wave_points) - 1):
            x1, y1 = wave_points[i]
            x2, y2 = wave_points[i + 1]

            # Desenhar linha entre pontos (interpolação simples)
            if 0 <= y1 < h and 0 <= y2 < h:
                # Cor branca para as linhas das ondas
                steps = max(abs(x2 - x1), abs(y2 - y1), 1)
                for step in range(steps):
                    t_interp = step / steps if steps > 0 else 0
                    x_interp = int(lerp(t_interp, x1, x2))
                    y_interp = int(lerp(t_interp, y1, y2))

                    if 0 <= x_interp < w and 0 <= y_interp < h:
                        fig_data[y_interp, x_interp] = [255, 255, 255]  # Branco

    return fig_data

def main():
    video_path = "/Users/nicolasalan/Documents/stacks/vision/video2.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        return

    # Obter FPS do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback para 30fps

    frame_delay = 1.0 / fps

    # Configurar matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    t = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processar frame
        wave_image = process_frame_waves(frame, t)

        # Mostrar
        ax.clear()
        ax.imshow(wave_image, aspect='auto')
        ax.axis('off')
        ax.set_title('Ondas Horizontais do Vídeo', color='white', fontsize=14)

        plt.pause(frame_delay)  # FPS correto do vídeo
        t += 1

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
