#!/usr/bin/env python3
"""
Interfaz Gráfica Simple para Restauración de Audio con FFT
===========================================================
Una GUI amigable para usar el filtrado notch sin línea de comandos.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys

# Añadir el directorio actual al path para importar fft_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fft_filter import (
    demo_sintetica,
    procesar_archivo_wav,
    generar_senal_base,
    contaminar_senal,
    filtrar_espectral,
    calcular_snr
)


class AudioRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Restauración de Audio - Filtro Notch FFT")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.frequencies = tk.StringVar(value="60, 2500")
        self.notch_width = tk.DoubleVar(value=12.0)
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook (pestañas)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === Pestaña 1: Procesar Archivo ===
        tab_file = ttk.Frame(notebook, padding=15)
        notebook.add(tab_file, text="📁 Procesar Archivo WAV")
        self.setup_file_tab(tab_file)
        
        # === Pestaña 2: Demo Sintética ===
        tab_demo = ttk.Frame(notebook, padding=15)
        notebook.add(tab_demo, text="🧪 Demo Sintética")
        self.setup_demo_tab(tab_demo)
        
        # === Pestaña 3: Generar Pruebas ===
        tab_gen = ttk.Frame(notebook, padding=15)
        notebook.add(tab_gen, text="🎼 Generar Archivos de Prueba")
        self.setup_generate_tab(tab_gen)
        
        # === Pestaña 4: Ayuda/Info ===
        tab_help = ttk.Frame(notebook, padding=15)
        notebook.add(tab_help, text="ℹ️ Ayuda")
        self.setup_help_tab(tab_help)
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Listo")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        
    def setup_file_tab(self, parent):
        # Título
        ttk.Label(parent, text="Procesar Archivo de Audio Real", font=('Segoe UI', 14, 'bold')).pack(pady=(0, 20))
        
        # Frame archivo entrada
        frame_in = ttk.LabelFrame(parent, text="Archivo de Entrada (contaminado)", padding=10)
        frame_in.pack(fill=tk.X, pady=5)
        
        ttk.Entry(frame_in, textvariable=self.input_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(frame_in, text="📂 Examinar", command=self.browse_input).pack(side=tk.RIGHT)
        
        # Frame archivo salida
        frame_out = ttk.LabelFrame(parent, text="Archivo de Salida (restaurado)", padding=10)
        frame_out.pack(fill=tk.X, pady=5)
        
        ttk.Entry(frame_out, textvariable=self.output_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(frame_out, text="💾 Guardar como...", command=self.browse_output).pack(side=tk.RIGHT)
        
        # Frame frecuencias
        frame_freq = ttk.LabelFrame(parent, text="Frecuencias a Eliminar (Hz)", padding=10)
        frame_freq.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_freq, text="Separadas por comas:").pack(anchor=tk.W)
        ttk.Entry(frame_freq, textvariable=self.frequencies, width=50).pack(fill=tk.X, pady=5)
        
        # Presets rápidos
        preset_frame = ttk.Frame(frame_freq)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="🔌 Hum 50Hz (EU)", command=lambda: self.frequencies.set("50, 100, 150")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="🔌 Hum 60Hz (US)", command=lambda: self.frequencies.set("60, 120, 180")).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="🎵 Demo (60, 2500)", command=lambda: self.frequencies.set("60, 2500")).pack(side=tk.LEFT, padx=5)
        
        # Frame ancho muesca
        frame_width = ttk.LabelFrame(parent, text="Ancho de Muesca (Hz)", padding=10)
        frame_width.pack(fill=tk.X, pady=5)
        
        self.width_label_var = tk.StringVar(value="12 Hz")
        width_scale = ttk.Scale(frame_width, from_=1, to=50, variable=self.notch_width, orient=tk.HORIZONTAL)
        width_scale.pack(fill=tk.X, padx=5)
        ttk.Label(frame_width, textvariable=self.width_label_var).pack()
        
        def update_width_label(*args):
            self.width_label_var.set(f"{self.notch_width.get():.0f} Hz")
        self.notch_width.trace_add('write', update_width_label)
        
        # Botón procesar
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        self.process_btn = ttk.Button(btn_frame, text="🚀 PROCESAR AUDIO", command=self.process_file, style='Accent.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=5, ipadx=20, ipady=5)
        
        ttk.Button(btn_frame, text="🧹 Limpiar", command=self.clear_file_tab).pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        # Área de log
        log_frame = ttk.LabelFrame(parent, text="Registro de Proceso", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
    def setup_demo_tab(self, parent):
        ttk.Label(parent, text="Demostración con Señal Sintética", font=('Segoe UI', 14, 'bold')).pack(pady=(0, 10))
        
        info_text = """Esta demo reproduce el experimento del reporte técnico:
• Genera una señal limpia: 440 Hz (A4) + 880 Hz (A5)
• Inyecta interferencias: Hum 60Hz + Tono 2500Hz + Ruido blanco
• Aplica filtro notch FFT y muestra métricas SNR"""
        
        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=600).pack(pady=10, anchor=tk.W)
        
        ttk.Button(parent, text="▶ EJECUTAR DEMO", command=self.run_demo, style='Accent.TButton').pack(pady=10, ipadx=30, ipady=10)
        
        # Resultados
        result_frame = ttk.LabelFrame(parent, text="Resultados", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.demo_log = scrolledtext.ScrolledText(result_frame, height=15, wrap=tk.WORD, font=('Consolas', 10))
        self.demo_log.pack(fill=tk.BOTH, expand=True)
        self.demo_log.config(state=tk.DISABLED)
        
    def setup_generate_tab(self, parent):
        ttk.Label(parent, text="Generar Archivos WAV de Prueba", font=('Segoe UI', 14, 'bold')).pack(pady=(0, 10))
        
        info_text = """Crea 3 archivos en la carpeta data/:
• original.wav - Señal contaminada (para testear)
• filtrado.wav - Resultado esperado tras filtrado
• limpio_referencia.wav - Señal original sin ruido"""
        
        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=600).pack(pady=10, anchor=tk.W)
        
        ttk.Button(parent, text="🎼 GENERAR ARCHIVOS", command=self.generate_files, style='Accent.TButton').pack(pady=10, ipadx=20, ipady=10)
        
        self.gen_log = scrolledtext.ScrolledText(parent, height=15, wrap=tk.WORD, font=('Consolas', 9))
        self.gen_log.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gen_log.config(state=tk.DISABLED)
        
    def setup_help_tab(self, parent):
        help_text = """🎵 RESTAURACIÓN DE AUDIO - FILTRO NOTCH FFT
=================================================

¿QUÉ HACE ESTA HERRAMIENTA?
Elimina interferencias tonales fijas de grabaciones de audio:
• Hum de red eléctrica (50 Hz en Europa, 60 Hz en América)
• Armónicos de la red (100, 150, 200... Hz o 120, 180, 240... Hz)
• Tonos de prueba, acoplamientos, pitidos de alta frecuencia

CÓMO FUNCIONA:
1. Convierte el audio al dominio de la frecuencia (FFT)
2. Anula selectivamente las frecuencias indicadas (filtro muesca/notch)
3. Reconstruye el audio limpio (IFFT)

PARÁMETROS:
• Frecuencias: Lista separada por comas (ej: 60, 120, 2500)
• Ancho de muesca: Qué tan ancho es el filtro (default 12 Hz)
  - Más ancho = elimina más alrededor, pero puede afectar señal válida
  - Menos ancho = más preciso, pero puede dejar residuos

FORMATOS SOPORTADOS:
• WAV (PCM 16-bit, 24-bit, 32-bit float)
• Mono o Estéreo (se convierte a mono automáticamente)

EJEMPLOS DE USO:
• Eliminar hum 60Hz:          60
• Hum 60Hz + armónicos:       60, 120, 180, 240
• Hum 50Hz (Europa):          50, 100, 150, 200
• Demo técnico (60Hz + 2.5k): 60, 2500

RESULTADOS ESPERADOS (demo):
• SNR inicial:  ~-14 dB
• SNR final:    ~+20 dB  
• Mejora:       ~+34 dB
• Atenuación 60Hz:  >60 dB
• Atenuación 2.5kHz: >60 dB

REQUISITOS:
• Python 3.8+
• numpy, scipy
• pip install -r requirements.txt

ARCHIVOS DEL PROYECTO:
• fft_filter.py        - Motor principal (CLI + funciones)
• generate_test_audio.py - Generador de archivos de prueba
• gui_interface.py     - Esta interfaz gráfica
• data/original.wav    - Ejemplo contaminado
• data/filtrado.wav    - Ejemplo restaurado
• docs/reporte_tecnico.md - Documentación completa"""
        
        help_display = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=('Consolas', 9))
        help_display.pack(fill=tk.BOTH, expand=True, pady=10)
        help_display.insert(tk.END, help_text)
        help_display.config(state=tk.DISABLED)
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de audio contaminado",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Sugerir nombre de salida
            base, ext = os.path.splitext(filename)
            self.output_file.set(f"{base}_limpio{ext}")
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Guardar audio restaurado como",
            defaultextension=".wav",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            
    def log_message(self, text_widget, message):
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        text_widget.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
    def clear_file_tab(self):
        self.input_file.set("")
        self.output_file.set("")
        self.frequencies.set("60, 2500")
        self.notch_width.set(12.0)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set("Listo")
        
    def process_file(self):
        if self.is_processing:
            return
            
        # Validaciones
        if not self.input_file.get():
            messagebox.showerror("Error", "Seleccione un archivo de entrada")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "Especifique un archivo de salida")
            return
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "El archivo de entrada no existe")
            return
            
        try:
            freqs = [float(f.strip()) for f in self.frequencies.get().split(',') if f.strip()]
            if not freqs:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Formato de frecuencias inválido. Use: 60, 2500")
            return
            
        # Ejecutar en hilo separado
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED, text="⏳ Procesando...")
        self.status_var.set("Procesando...")
        
        thread = threading.Thread(target=self._process_file_thread, args=(freqs,))
        thread.daemon = True
        thread.start()
        
    def _process_file_thread(self, freqs):
        try:
            self.log_message(self.log_text, f"📂 Cargando: {os.path.basename(self.input_file.get())}")
            self.log_message(self.log_text, f"🎯 Frecuencias a eliminar: {freqs} Hz")
            self.log_message(self.log_text, f"📏 Ancho de muesca: {self.notch_width.get()} Hz")
            self.log_message(self.log_text, "-" * 50)
            
            # Procesar
            procesar_archivo_wav(
                self.input_file.get(),
                self.output_file.get(),
                freqs,
                self.notch_width.get()
            )
            
            self.log_message(self.log_text, "-" * 50)
            self.log_message(self.log_text, f"✅ ¡Completado! Guardado en: {os.path.basename(self.output_file.get())}")
            self.status_var.set("Completado")
            
            messagebox.showinfo("Éxito", f"Audio procesado correctamente:\n{self.output_file.get()}")
            
        except Exception as e:
            self.log_message(self.log_text, f"❌ ERROR: {str(e)}")
            self.status_var.set("Error")
            messagebox.showerror("Error", f"Error al procesar:\n{str(e)}")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL, text="🚀 PROCESAR AUDIO"))
            
    def run_demo(self):
        if self.is_processing:
            return
            
        self.is_processing = True
        self.demo_log.config(state=tk.NORMAL)
        self.demo_log.delete(1.0, tk.END)
        self.demo_log.config(state=tk.DISABLED)
        self.status_var.set("Ejecutando demo...")
        
        thread = threading.Thread(target=self._run_demo_thread)
        thread.daemon = True
        thread.start()
        
    def _run_demo_thread(self):
        try:
            # Redirigir prints a nuestra log
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capturar salida
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer):
                demo_sintetica()
                
            output = output_buffer.getvalue()
            for line in output.strip().split('\n'):
                self.log_message(self.demo_log, line)
                
            self.status_var.set("Demo completada")
            
        except Exception as e:
            self.log_message(self.demo_log, f"❌ ERROR: {str(e)}")
            self.status_var.set("Error en demo")
        finally:
            self.is_processing = False
            
    def generate_files(self):
        if self.is_processing:
            return
            
        self.is_processing = True
        self.gen_log.config(state=tk.NORMAL)
        self.gen_log.delete(1.0, tk.END)
        self.gen_log.config(state=tk.DISABLED)
        self.status_var.set("Generando archivos...")
        
        thread = threading.Thread(target=self._generate_files_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_files_thread(self):
        try:
            from generate_test_audio import main as gen_main
            import io
            import sys
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                gen_main()
                
            output = output_buffer.getvalue()
            for line in output.strip().split('\n'):
                self.log_message(self.gen_log, line)
                
            self.status_var.set("Archivos generados en ./data/")
            messagebox.showinfo("Éxito", "Archivos de prueba generados en carpeta data/")
            
        except Exception as e:
            self.log_message(self.gen_log, f"❌ ERROR: {str(e)}")
            self.status_var.set("Error generando")
            messagebox.showerror("Error", f"Error al generar:\n{str(e)}")
        finally:
            self.is_processing = False


def main():
    root = tk.Tk()
    
    # Configurar estilo para botón accent
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
    
    # Icono (opcional)
    try:
        root.iconbitmap(default='')  # Placeholder
    except:
        pass
    
    app = AudioRestorationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()