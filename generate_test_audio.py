#!/usr/bin/env python3
"""
Generador de Archivos de Prueba WAV
====================================

Crea archivos de audio sintéticos para validar el filtrado FFT:
- original.wav: Señal contaminada (hum 60Hz + tono 2.5kHz + ruido blanco)
- filtrado.wav: Señal restaurada después de aplicar filtro notch

Autor: Proyecto Fourier DSP
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.io import wavfile
import os


def generar_senal_base(fs=44100, duracion=3.0):
    """Genera señal limpia: armónicos 440 Hz (A4) + 880 Hz (A5)"""
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    # Voz sintética: fundamental 440 Hz + armónico 880 Hz
    s = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return t, s


def contaminar_senal(s, t, fs=44100, seed=42):
    """Inyecta interferencias controladas"""
    np.random.seed(seed)  # Reproducibilidad
    N = len(s)
    
    ruido_hum = 0.5 * np.sin(2 * np.pi * 60 * t)       # Hum red 60 Hz
    ruido_tono = 0.4 * np.sin(2 * np.pi * 2500 * t)    # Tono 2.5 kHz
    ruido_blanco = 0.05 * np.random.normal(0, 1, N)    # Ruido gaussiano
    
    x = s + ruido_hum + ruido_tono + ruido_blanco
    return x


def filtrar_espectral(x, fs, frecuencias_notch=[60, 2500], ancho_muesca=12):
    """Aplica filtro notch en dominio frecuencial"""
    N = len(x)
    X = fft(x)
    H = np.ones(N)
    freqs = fftfreq(N, 1/fs)
    
    for f_target in frecuencias_notch:
        mask = (np.abs(freqs - f_target) <= ancho_muesca / 2) | \
               (np.abs(freqs + f_target) <= ancho_muesca / 2)
        H[mask] = 0.0
    
    Y = X * H
    y = np.real(ifft(Y))
    return y


def normalizar_guardar(senal, fs, ruta, nombre):
    """Normaliza a int16 y guarda como WAV"""
    # Normalizar a [-1, 1] evitando clipping
    max_val = np.max(np.abs(senal))
    if max_val > 0:
        senal_norm = senal / max_val * 0.95  # 95% para headroom
    else:
        senal_norm = senal
    
    senal_int16 = (senal_norm * 32767).astype(np.int16)
    wavfile.write(ruta, fs, senal_int16)
    print(f"  [OK] {nombre} guardado: {ruta} ({len(senal)/fs:.2f}s, {fs}Hz)")


def main():
    print("=" * 60)
    print("GENERADOR DE ARCHIVOS DE PRUEBA - Restauración Audio FFT")
    print("=" * 60)
    
    # Configuración
    fs = 44100
    duracion = 3.0
    output_dir = "data"
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[CONFIG] Configuracion:")
    print(f"   Frecuencia de muestreo: {fs} Hz")
    print(f"   Duracion: {duracion} s")
    print(f"   Directorio salida: {output_dir}/")
    
    # 1. Generar señal base limpia
    print("\n[1] Generando senal base (440 Hz + 880 Hz)...")
    t, s = generar_senal_base(fs, duracion)
    
    # 2. Contaminar
    print("[2] Inyectando interferencias...")
    print("   - Hum 60 Hz (amplitud 0.5)")
    print("   - Tono 2500 Hz (amplitud 0.4)")
    print("   - Ruido blanco sigma=0.05")
    x = contaminar_senal(s, t, fs)
    
    # 3. Filtrar
    print("[3] Aplicando filtro notch FFT (60 Hz, 2500 Hz, ancho 12 Hz)...")
    y = filtrar_espectral(x, fs, [60, 2500], 12)
    
    # 4. Guardar archivos
    print("\n[4] Guardando archivos WAV...")
    normalizar_guardar(x, fs, f"{output_dir}/original.wav", "original.wav (contaminado)")
    normalizar_guardar(y, fs, f"{output_dir}/filtrado.wav", "filtrado.wav (restaurado)")
    normalizar_guardar(s, fs, f"{output_dir}/limpio_referencia.wav", "limpio_referencia.wav (original)")
    
    # 5. Métricas de validación
    print("\n[5] Metricas de validacion:")
    def snr(clean, noisy):
        p_signal = np.sum(clean**2)
        p_noise = np.sum((noisy - clean)**2)
        return 10 * np.log10(p_signal / p_noise) if p_noise > 0 else float('inf')
    
    snr_ini = snr(s, x)
    snr_fin = snr(s, y)
    print(f"   SNR Entrada:  {snr_ini:>7.2f} dB")
    print(f"   SNR Salida:   {snr_fin:>7.2f} dB")
    print(f"   Ganancia SNR: {snr_fin - snr_ini:>7.2f} dB")
    
    # Verificar atenuación en frecuencias notch
    N = len(x)
    freqs = fftfreq(N, 1/fs)
    X = fft(x)
    Y = fft(y)
    
    print("\n   Atenuación en frecuencias objetivo:")
    for f_target in [60, 2500]:
        idx = np.argmin(np.abs(freqs - f_target))
        pot_orig = np.abs(X[idx])**2
        pot_filt = np.abs(Y[idx])**2
        att = 10 * np.log10(pot_filt / pot_orig) if pot_orig > 0 else -np.inf
        print(f"     {f_target} Hz: {att:>7.2f} dB")
    
    print("\n" + "=" * 60)
    print("[OK] Archivos generados exitosamente en ./data/")
    print("=" * 60)
    print("\n[INFO] Para probar el filtrado:")
    print(f"   python fft_filter.py -i {output_dir}/original.wav -o {output_dir}/test_salida.wav -f 60,2500")


if __name__ == "__main__":
    main()