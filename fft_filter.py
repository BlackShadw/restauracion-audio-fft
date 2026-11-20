#!/usr/bin/env python3
"""
Restauración de Audio mediante Filtrado en el Dominio de la Frecuencia con FFT
===============================================================================

Script principal para filtrado espectral de interferencias en señales de audio.
Elimina hum de red (60 Hz), tonos de alta frecuencia y ruido blanco gaussiano
mediante filtros muesca (notch filters) en el dominio de la frecuencia.

Autor: Proyecto Fourier DSP
Fecha: Noviembre 2026
Licencia: MIT
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io import wavfile
import argparse
import sys
import os


def generar_senal_base(fs=44100, duracion=3.0):
    """
    Genera una señal sintética de audio limpio compuesta por armónicos de voz.
    
    Parámetros:
        fs (int): Frecuencia de muestreo en Hz (default: 44100)
        duracion (float): Duración en segundos (default: 3.0)
    
    Retorna:
        tuple: (t, s) - vector de tiempo y señal limpia
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    N = len(t)
    
    # Señal limpia s(t): combinación armónica (440 Hz y 880 Hz)
    s = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    return t, s


def contaminar_senal(s, t, fs=44100):
    """
    Agrega interferencias controladas a la señal limpia.
    
    Interferencias:
    - Hum de red: 60 Hz (amplitud 0.5)
    - Tono agudo: 2500 Hz (amplitud 0.4)
    - Ruido blanco gaussiano: σ=0.05
    
    Parámetros:
        s (array): Señal limpia
        t (array): Vector de tiempo
        fs (int): Frecuencia de muestreo
    
    Retorna:
        array: Señal contaminada x[n]
    """
    N = len(s)
    
    ruido_hum = 0.5 * np.sin(2 * np.pi * 60 * t)
    ruido_tono = 0.4 * np.sin(2 * np.pi * 2500 * t)
    ruido_blanco = 0.05 * np.random.normal(0, 1, N)
    
    x = s + ruido_hum + ruido_tono + ruido_blanco
    
    return x


def calcular_snr(senal_limpia, senal_ruidosa):
    """
    Calcula la relación señal-ruido (SNR) en dB.
    
    SNR = 10 * log10(P_señal / P_ruido)
    
    Parámetros:
        senal_limpia (array): Señal original sin ruido
        senal_ruidosa (array): Señal con ruido
    
    Retorna:
        float: SNR en dB
    """
    potencia_senal = np.sum(senal_limpia**2)
    potencia_ruido = np.sum((senal_ruidosa - senal_limpia)**2)
    
    if potencia_ruido == 0:
        return float('inf')
    
    return 10 * np.log10(potencia_senal / potencia_ruido)


def crear_filtro_notch(frecuencias, fs, N, ancho_muesca=12):
    """
    Crea un filtro muesca (notch) espectral para anular frecuencias específicas.
    
    El filtro anula componentes en [f - ancho/2, f + ancho/2] Hz
    para cada frecuencia objetivo, considerando frecuencias positivas y negativas.
    
    Parámetros:
        frecuencias (list): Lista de frecuencias a eliminar en Hz
        fs (int): Frecuencia de muestreo
        N (int): Número de muestras (longitud FFT)
        ancho_muesca (float): Ancho de la muesca en Hz (default: 12)
    
    Retorna:
        array: Función de transferencia H[k] (1 = pasa, 0 = atenúa)
    """
    H = np.ones(N)
    freqs = fftfreq(N, 1/fs)
    
    for f_target in frecuencias:
        # Máscara para frecuencias positivas y negativas (simetría hermitiana)
        mask = (np.abs(freqs - f_target) <= ancho_muesca / 2) | \
               (np.abs(freqs + f_target) <= ancho_muesca / 2)
        H[mask] = 0.0
    
    return H


def filtrar_espectral(x, fs, frecuencias_notch=[60, 2500], ancho_muesca=12):
    """
    Aplica filtrado en el dominio de la frecuencia usando FFT.
    
    Proceso:
    1. FFT de la señal contaminada
    2. Multiplicación punto a punto con filtro notch
    3. IFFT para reconstrucción temporal
    
    Parámetros:
        x (array): Señal contaminada
        fs (int): Frecuencia de muestreo
        frecuencias_notch (list): Frecuencias a eliminar
        ancho_muesca (float): Ancho de banda del filtro notch
    
    Retorna:
        tuple: (y, Y, X, H) - señal filtrada, espectro filtrado, espectro original, filtro
    """
    N = len(x)
    
    # 3. Transformada Rápida de Fourier (FFT)
    X = fft(x)
    frecuencias = fftfreq(N, 1/fs)
    
    # 4. Diseño del Filtro Muesca (Notch Filter) Espectral
    H = crear_filtro_notch(frecuencias_notch, fs, N, ancho_muesca)
    
    # 5. Filtrado por Multiplicación Espectral e IFFT
    Y = X * H
    y = np.real(ifft(Y))
    
    return y, Y, X, H


def procesar_archivo_wav(archivo_entrada, archivo_salida, frecuencias_notch, ancho_muesca=12):
    """
    Procesa un archivo WAV real aplicando filtrado notch.
    
    Parámetros:
        archivo_entrada (str): Ruta al archivo WAV de entrada
        archivo_salida (str): Ruta al archivo WAV de salida
        frecuencias_notch (list): Frecuencias a eliminar en Hz
        ancho_muesca (float): Ancho de muesca en Hz
    """
    print(f"Cargando: {archivo_entrada}")
    
    # Leer archivo WAV
    fs, data = wavfile.read(archivo_entrada)
    
    # Convertir a mono si es estéreo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Normalizar a float [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    
    print(f"  Frecuencia de muestreo: {fs} Hz")
    print(f"  Duración: {len(data)/fs:.2f} s")
    print(f"  Frecuencias a eliminar: {frecuencias_notch} Hz")
    
    # Aplicar filtrado
    y, _, _, _ = filtrar_espectral(data, fs, frecuencias_notch, ancho_muesca)
    
    # Normalizar y convertir a int16 para guardar
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767).astype(np.int16)
    
    # Guardar
    wavfile.write(archivo_salida, fs, y_int16)
    print(f"Guardado: {archivo_salida}")


def demo_sintetica():
    """
    Ejecuta la demostración completa con señal sintética.
    Reproduce el experimento del reporte técnico.
    """
    print("=" * 70)
    print("DEMOSTRACIÓN: Restauración de Audio con FFT - Señal Sintética")
    print("=" * 70)
    
    # Parámetros del reporte
    fs = 44100
    duracion = 3.0
    
    print("\n1. Generando señal base (voz sintética: 440 Hz + 880 Hz)...")
    t, s = generar_senal_base(fs, duracion)
    
    print("2. Inyectando interferencias controladas...")
    print("   - Hum de red: 60 Hz (amplitud 0.5)")
    print("   - Tono agudo: 2500 Hz (amplitud 0.4)")
    print("   - Ruido blanco gaussiano: sigma=0.05")
    x = contaminar_senal(s, t, fs)
    
    print("3. Calculando FFT...")
    print("4. Aplicando filtro notch espectral (60 Hz y 2500 Hz, ancho 12 Hz)...")
    y, Y, X, H = filtrar_espectral(x, fs, [60, 2500], 12)
    
    print("5. Evaluando rendimiento (SNR)...")
    snr_ini = calcular_snr(s, x)
    snr_fin = calcular_snr(s, y)
    ganancia = snr_fin - snr_ini
    
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"SNR Inicial:  {snr_ini:>8.2f} dB")
    print(f"SNR Final:    {snr_fin:>8.2f} dB")
    print(f"Ganancia SNR: {ganancia:>8.2f} dB")
    print("=" * 70)
    
    # Verificar atenuación de interferencias
    print("\nAtenuación de componentes espurias:")
    N = len(s)
    freqs = fftfreq(N, 1/fs)
    
    for f_target in [60, 2500]:
        idx = np.argmin(np.abs(freqs - f_target))
        pot_orig = np.abs(X[idx])**2
        pot_filt = np.abs(Y[idx])**2
        atenuacion = 10 * np.log10(pot_filt / pot_orig) if pot_orig > 0 else -np.inf
        print(f"  {f_target} Hz: {atenuacion:.2f} dB")
    
    return snr_ini, snr_fin, ganancia


def main():
    parser = argparse.ArgumentParser(
        description="Restauración de Audio mediante Filtrado FFT (Notch Filter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python fft_filter.py --demo                    # Ejecuta demo sintética
  python fft_filter.py -i audio.wav -o limpio.wav -f 60,2500
  python fft_filter.py -i ruido.wav -o salida.wav -f 50,100,150 -w 10
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Ejecuta demostración con señal sintética (reporte técnico)')
    parser.add_argument('-i', '--input', type=str,
                        help='Archivo WAV de entrada')
    parser.add_argument('-o', '--output', type=str,
                        help='Archivo WAV de salida')
    parser.add_argument('-f', '--frecuencias', type=str,
                        help='Frecuencias a eliminar separadas por comas (ej: 60,2500)')
    parser.add_argument('-w', '--ancho', type=float, default=12,
                        help='Ancho de muesca en Hz (default: 12)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_sintetica()
        return
    
    if not args.input or not args.output:
        parser.print_help()
        print("\nError: Se requieren -i/--input y -o/--output para procesar archivos.")
        sys.exit(1)
    
    if not args.frecuencias:
        print("Error: Se requieren frecuencias con -f/--frecuencias")
        sys.exit(1)
    
    try:
        frecuencias = [float(f.strip()) for f in args.frecuencias.split(',')]
    except ValueError:
        print("Error: Formato de frecuencias inválido. Use: 60,2500")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Archivo no encontrado: {args.input}")
        sys.exit(1)
    
    procesar_archivo_wav(args.input, args.output, frecuencias, args.ancho)
    print("¡Procesamiento completado!")


if __name__ == "__main__":
    main()