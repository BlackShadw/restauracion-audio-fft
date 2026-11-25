#!/usr/bin/env python3
"""
Interfaz Moderna y Elegante para Restauración de Audio FFT
==========================================================
Diseño limpio, intuitivo y visualmente atractivo.
Con reproducción de audio integrada y comparación 3D completa.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
import numpy as np
import tempfile
import wave

# Matplotlib para gráficas bonitas
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fft_filter import (
    demo_sintetica,
    procesar_archivo_wav,
    generar_senal_base,
    contaminar_senal,
    filtrar_espectral,
    calcular_snr
)
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.io import wavfile
from scipy.signal import spectrogram

# Audio playback - try multiple backends
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

AUDIO_BACKEND = None
if HAS_SOUNDDEVICE:
    AUDIO_BACKEND = 'sounddevice'
elif HAS_SIMPLEAUDIO:
    AUDIO_BACKEND = 'simpleaudio'
elif HAS_PYGAME:
    AUDIO_BACKEND = 'pygame'


# ============================================================
# PALETA DE COLORES MODERNA
# ============================================================
COLORS = {
    'bg': '#0d1117',
    'bg_secondary': '#161b22',
    'bg_tertiary': '#21262d',
    'fg': '#e6edf3',
    'fg_muted': '#8b949e',
    'accent': '#58a6ff',
    'accent_hover': '#79c0ff',
    'accent_dim': '#1f6feb',
    'success': '#3fb950',
    'success_hover': '#56d364',
    'warning': '#d29922',
    'danger': '#f85149',
    'danger_hover': '#ff7b72',
    'border': '#30363d',
    'border_focus': '#58a6ff',
    'surface': '#161b22',
    'overlay': '#21262d',
}

CUSTOM_CMAP = LinearSegmentedColormap.from_list('audio_cmap', [
    '#0d1117', '#1f1f3f', '#3a0ca3', '#7209b7', '#f72585', '#ff6b6b', '#ffd60a'
])


class ModernStyle:
    @staticmethod
    def setup(root):
        style = ttk.Style()
        style.theme_create('modern', parent='alt', settings={
            'TFrame': {'configure': {'background': COLORS['bg']}},
            'TLabel': {'configure': {'background': COLORS['bg'], 'foreground': COLORS['fg'], 'font': ('Segoe UI', 10)}},
            'TLabelframe': {'configure': {'background': COLORS['bg_secondary'], 'foreground': COLORS['fg'], 'bordercolor': COLORS['border'], 'lightcolor': COLORS['border'], 'darkcolor': COLORS['border'], 'font': ('Segoe UI', 10, 'bold')}},
            'TLabelframe.Label': {'configure': {'background': COLORS['bg_secondary'], 'foreground': COLORS['accent'], 'font': ('Segoe UI', 10, 'bold')}},
            'TEntry': {'configure': {'fieldbackground': COLORS['bg_tertiary'], 'foreground': COLORS['fg'], 'bordercolor': COLORS['border'], 'lightcolor': COLORS['border'], 'darkcolor': COLORS['border'], 'insertcolor': COLORS['fg'], 'font': ('Segoe UI', 10)}},
            'TCombobox': {'configure': {'fieldbackground': COLORS['bg_tertiary'], 'foreground': COLORS['fg'], 'background': COLORS['bg_tertiary'], 'bordercolor': COLORS['border'], 'arrowcolor': COLORS['fg'], 'font': ('Segoe UI', 10)}},
            'TButton': {'configure': {'background': COLORS['bg_tertiary'], 'foreground': COLORS['fg'], 'bordercolor': COLORS['border'], 'font': ('Segoe UI', 10), 'padding': (12, 8)}, 'map': {'background': [('active', COLORS['accent_dim']), ('pressed', COLORS['accent'])], 'foreground': [('active', '#ffffff')]}},
            'TCheckbutton': {'configure': {'background': COLORS['bg_secondary'], 'foreground': COLORS['fg'], 'font': ('Segoe UI', 9)}, 'map': {'background': [('active', COLORS['bg_secondary'])]}},
            'TRadiobutton': {'configure': {'background': COLORS['bg_secondary'], 'foreground': COLORS['fg'], 'font': ('Segoe UI', 9)}},
            'TScale': {'configure': {'background': COLORS['bg_secondary'], 'troughcolor': COLORS['bg_tertiary'], 'sliderrelief': 'flat', 'borderwidth': 0}},
            'TNotebook': {'configure': {'background': COLORS['bg'], 'borderwidth': 0, 'tabmargins': [2, 5, 2, 0]}},
            'TNotebook.Tab': {'configure': {'background': COLORS['bg_tertiary'], 'foreground': COLORS['fg_muted'], 'padding': (16, 8), 'font': ('Segoe UI', 9), 'borderwidth': 0}, 'map': {'background': [('selected', COLORS['bg_secondary']), ('active', COLORS['overlay'])], 'foreground': [('selected', COLORS['accent']), ('active', COLORS['fg'])]}},
            'Vertical.TScrollbar': {'configure': {'background': COLORS['bg_tertiary'], 'troughcolor': COLORS['bg'], 'bordercolor': COLORS['bg'], 'arrowcolor': COLORS['fg_muted'], 'width': 8}},
            'Horizontal.TScrollbar': {'configure': {'background': COLORS['bg_tertiary'], 'troughcolor': COLORS['bg'], 'bordercolor': COLORS['bg'], 'arrowcolor': COLORS['fg_muted'], 'width': 8}},
        })
        style.theme_use('modern')
        root.configure(bg=COLORS['bg'])
        root.option_add('*Font', ('Segoe UI', 10))
        return style


class ModernButton(ttk.Button):
    def __init__(self, parent, variant='primary', **kwargs):
        self.variant = variant
        style_name = f'{variant}.TButton'
        style = ttk.Style()
        if variant == 'primary':
            style.configure(style_name, background=COLORS['accent'], foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=(20, 12), borderwidth=0)
            style.map(style_name, background=[('active', COLORS['accent_hover']), ('pressed', COLORS['accent_dim'])])
        elif variant == 'success':
            style.configure(style_name, background=COLORS['success'], foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=(20, 12), borderwidth=0)
            style.map(style_name, background=[('active', COLORS['success_hover'])])
        elif variant == 'danger':
            style.configure(style_name, background=COLORS['danger'], foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=(20, 12), borderwidth=0)
            style.map(style_name, background=[('active', COLORS['danger_hover'])])
        elif variant == 'warning':
            style.configure(style_name, background=COLORS['warning'], foreground='#ffffff', font=('Segoe UI', 10, 'bold'), padding=(20, 12), borderwidth=0)
            style.map(style_name, background=[('active', COLORS['warning'])])
        elif variant == 'ghost':
            style.configure(style_name, background=COLORS['bg_tertiary'], foreground=COLORS['fg'], font=('Segoe UI', 10), padding=(16, 10), borderwidth=1)
            style.map(style_name, background=[('active', COLORS['overlay'])], foreground=[('active', COLORS['accent'])])
        elif variant == 'outline':
            style.configure(style_name, background=COLORS['bg_secondary'], foreground=COLORS['accent'], font=('Segoe UI', 10, 'bold'), padding=(16, 10), borderwidth=2)
            style.map(style_name, background=[('active', COLORS['accent'])], foreground=[('active', '#ffffff')])
        super().__init__(parent, style=style_name, **kwargs)


class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Audio Restoration Studio")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.frequencies = tk.StringVar(value="60, 120, 180, 2500")
        self.notch_width = tk.DoubleVar(value=12.0)
        self.is_processing = False
        
        # Datos audio
        self.audio_data = None
        self.filtered_data = None
        self.fs = None
        self.freqs_notch = [60, 120, 180, 2500]
        
        # Playback state
        self.playback_state = 'stopped'
        self.current_playback = None
        self.playback_thread = None
        self.stop_playback_flag = False
        self.removed_relative_volume = tk.BooleanVar(value=True)
        
        # Comparison mode
        self.compare_mode = tk.StringVar(value="side_by_side")
        
        # Camera
        self.cam_vars = {
            'elev': tk.DoubleVar(value=30),
            'azim': tk.DoubleVar(value=45),
            'dist': tk.DoubleVar(value=10),
        }
        
        # Animation
        self.anim_var = tk.BooleanVar(value=False)
        self.anim_speed = tk.DoubleVar(value=1.0)
        
        # Setup estilo
        ModernStyle.setup(root)
        
        # Build UI
        self.build_ui()
        self.animate_entrance()
        
    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main = ttk.Frame(self.root)
        main.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        
        # HEADER
        self.build_header(main)
        
        # CONTENIDO PRINCIPAL
        content = ttk.Frame(main)
        content.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(16, 0))
        content.columnconfigure(0, weight=0, minsize=340)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        
        # Panel lateral
        self.build_sidebar(content)
        
        # Panel central (gráficas)
        self.build_main_panel(content)
        
        # FOOTER / STATUS BAR
        self.build_status_bar(main)
        
    def build_header(self, parent):
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        header.columnconfigure(1, weight=1)
        
        title_frame = ttk.Frame(header)
        title_frame.grid(row=0, column=0, sticky='w')
        
        ttk.Label(title_frame, text="🎵", font=('Segoe UI', 28)).pack(side='left', padx=(0, 10))
        
        text_frame = ttk.Frame(title_frame)
        text_frame.pack(side='left')
        
        ttk.Label(text_frame, text="Audio Restoration Studio", font=('Segoe UI', 20, 'bold'), foreground=COLORS['fg']).pack(anchor='w')
        ttk.Label(text_frame, text="FFT Notch Filter • Remove hum, tones & interference", font=('Segoe UI', 10), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        actions = ttk.Frame(header)
        actions.grid(row=0, column=1, sticky='e')
        
        ModernButton(actions, variant='ghost', text="📁 Open", command=self.browse_input).pack(side='left', padx=4)
        ModernButton(actions, variant='ghost', text="🧪 Demo", command=self.run_demo).pack(side='left', padx=4)
        ModernButton(actions, variant='outline', text="✨ Generate Test Files", command=self.generate_test_files).pack(side='left', padx=4)
        
    def build_sidebar(self, parent):
        sidebar = ttk.Frame(parent, style='Card.TFrame', padding=0)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 16))
        sidebar.rowconfigure(1, weight=1)
        
        self.sidebar_notebook = ttk.Notebook(sidebar)
        self.sidebar_notebook.pack(fill='both', expand=True)
        
        # Tab 1: Archivo
        tab_file = ttk.Frame(self.sidebar_notebook, padding=16)
        self.sidebar_notebook.add(tab_file, text="📁  File")
        self.build_file_tab(tab_file)
        
        # Tab 2: Filtro
        tab_filter = ttk.Frame(self.sidebar_notebook, padding=16)
        self.sidebar_notebook.add(tab_filter, text="🎛️  Filter")
        self.build_filter_tab(tab_filter)
        
        # Tab 3: Visualización
        tab_viz = ttk.Frame(self.sidebar_notebook, padding=16)
        self.sidebar_notebook.add(tab_viz, text="📈  View")
        self.build_viz_tab(tab_viz)
        
    def build_file_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        
        # Input file
        self.build_file_field(parent, "Input Audio (Noisy)", self.input_file, "Select noisy WAV file", self.browse_input, "📂")
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=16)
        
        # Output file
        self.build_file_field(parent, "Output Audio (Clean)", self.output_file, "Save cleaned WAV as", self.browse_output, "💾")
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=16)
        
        # Info del archivo cargado
        self.file_info_frame = ttk.Frame(parent)
        self.file_info_frame.pack(fill='x')
        self.file_info_labels = {}
        for label, key in [("Duration", "dur"), ("Sample Rate", "sr"), ("Channels", "ch")]:
            row = ttk.Frame(self.file_info_frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=label, foreground=COLORS['fg_muted'], font=('Segoe UI', 9)).pack(side='left')
            self.file_info_labels[key] = ttk.Label(row, text="—", font=('Segoe UI', 9, 'bold'))
            self.file_info_labels[key].pack(side='right')
            
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=16)
        
        # Botón principal procesar
        self.process_btn = ModernButton(parent, variant='success', text="✨  Process & Visualize", command=self.process_file, state='normal')
        self.process_btn.pack(fill='x', pady=(0, 8))
        
        ModernButton(parent, variant='ghost', text="👁️  Preview Only (No Filter)", command=self.load_preview).pack(fill='x', pady=(0, 16))
        
        # === PLAYBACK CONTROLS ===
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8)
        
        playback_frame = ttk.Frame(parent)
        playback_frame.pack(fill='x')
        
        ttk.Label(playback_frame, text="🔊  Audio Playback", font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 8))
        
        backend_text = f"Backend: {AUDIO_BACKEND or 'None (install sounddevice/simpleaudio/pygame)'}"
        backend_color = COLORS['success'] if AUDIO_BACKEND else COLORS['danger']
        self.backend_label = ttk.Label(playback_frame, text=backend_text, font=('Segoe UI', 7), foreground=backend_color)
        self.backend_label.pack(anchor='w', pady=(0, 8))
        
        # Playback buttons row 1
        btn_row1 = ttk.Frame(playback_frame)
        btn_row1.pack(fill='x', pady=2)
        btn_row1.columnconfigure(0, weight=1)
        btn_row1.columnconfigure(1, weight=1)
        btn_row1.columnconfigure(2, weight=1)
        
        self.play_orig_btn = ModernButton(btn_row1, variant='outline', text="🎵 Original", command=lambda: self.play_audio('original'), width=12)
        self.play_orig_btn.grid(row=0, column=0, padx=2, sticky='ew')
        
        self.play_filt_btn = ModernButton(btn_row1, variant='success', text="✨ Filtered", command=lambda: self.play_audio('filtered'), width=12, state='disabled')
        self.play_filt_btn.grid(row=0, column=1, padx=2, sticky='ew')
        
        self.play_removed_btn = ModernButton(btn_row1, variant='ghost', text="🗑️ Removed", command=lambda: self.play_audio('removed'), width=12, state='disabled')
        self.play_removed_btn.grid(row=0, column=2, padx=2, sticky='ew')
        
        # Playback buttons row 2 - Stop + Volume
        btn_row2 = ttk.Frame(playback_frame)
        btn_row2.pack(fill='x', pady=(8, 0))
        btn_row2.columnconfigure(0, weight=1)
        btn_row2.columnconfigure(1, weight=1)
        
        self.stop_btn = ModernButton(btn_row2, variant='danger', text="⏹ Stop", command=self.stop_audio, width=12, state='disabled')
        self.stop_btn.grid(row=0, column=0, padx=2, sticky='ew')
        
        # Volume control
        vol_frame = ttk.Frame(btn_row2)
        vol_frame.grid(row=0, column=1, sticky='e', padx=2)
        ttk.Label(vol_frame, text="🔊", font=('Segoe UI', 9)).pack(side='left')
        self.volume_var = tk.DoubleVar(value=0.7)
        vol_scale = ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=self.volume_var, orient='horizontal', length=100)
        vol_scale.pack(side='left', padx=4)
        self.vol_label = ttk.Label(vol_frame, text="70%", font=('Segoe UI', 8), foreground=COLORS['accent'], width=4)
        self.vol_label.pack(side='left')
        self.volume_var.trace_add('write', lambda *a: self.vol_label.config(text=f"{int(self.volume_var.get()*100)}%"))
        
        # Relative volume checkbox for removed audio
        rel_frame = ttk.Frame(playback_frame)
        rel_frame.pack(fill='x', pady=(4, 0))
        ttk.Checkbutton(rel_frame, text="🔗 Removed at relative level (not boosted)", variable=self.removed_relative_volume).pack(anchor='w')
        ttk.Label(rel_frame, text="   When ON: removed noise plays at actual level vs original", font=('Segoe UI', 7), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        # Progress bar for playback
        self.playback_progress = ttk.Progressbar(playback_frame, mode='determinate')
        self.playback_progress.pack(fill='x', pady=(8, 0))
        self.playback_progress['value'] = 0
        
    def build_file_field(self, parent, label_text, var, tooltip, command, icon):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=(0, 4))
        
        ttk.Label(frame, text=label_text, font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        input_row = ttk.Frame(frame)
        input_row.pack(fill='x', pady=(4, 0))
        input_row.columnconfigure(0, weight=1)
        
        entry = ttk.Entry(input_row, textvariable=var, font=('Segoe UI', 9))
        entry.grid(row=0, column=0, sticky='ew', padx=(0, 8))
        
        ModernButton(input_row, variant='outline', text=icon, width=3, command=command).grid(row=0, column=1)
        
    def build_filter_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        
        ttk.Label(parent, text="Frequencies to Remove (Hz)", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        freq_entry = ttk.Entry(parent, textvariable=self.frequencies, font=('Segoe UI', 10))
        freq_entry.pack(fill='x', pady=(4, 12))
        
        ttk.Label(parent, text="Quick Presets", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w', pady=(0, 8))
        
        presets = [
            ("🔌  Mains 50 Hz (EU)", "50, 100, 150, 200, 250"),
            ("🔌  Mains 60 Hz (US)", "60, 120, 180, 240, 300"),
            ("🎵  Demo (60 + 2500)", "60, 2500"),
            ("🎹  Piano A4 (440 Hz)", "440, 880, 1320"),
            ("🔧  Custom...", None),
        ]
        
        for label, value in presets:
            btn = ModernButton(parent, variant='ghost' if value else 'outline', text=label)
            if value:
                btn.configure(command=lambda v=value: self.frequencies.set(v))
            btn.pack(fill='x', pady=2)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=16)
        
        ttk.Label(parent, text="Notch Width", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        width_frame = ttk.Frame(parent)
        width_frame.pack(fill='x', pady=(4, 0))
        width_frame.columnconfigure(0, weight=1)
        
        self.width_value_label = ttk.Label(width_frame, text="12 Hz", font=('Segoe UI', 10, 'bold'), foreground=COLORS['accent'])
        self.width_value_label.pack(anchor='e')
        
        scale = ttk.Scale(width_frame, from_=2, to=50, variable=self.notch_width, orient='horizontal')
        scale.pack(fill='x', pady=(4, 0))
        
        minmax = ttk.Frame(width_frame)
        minmax.pack(fill='x')
        ttk.Label(minmax, text="Narrow (2 Hz)", font=('Segoe UI', 8), foreground=COLORS['fg_muted']).pack(side='left')
        ttk.Label(minmax, text="Wide (50 Hz)", font=('Segoe UI', 8), foreground=COLORS['fg_muted']).pack(side='right')
        
        self.notch_width.trace_add('write', lambda *a: self.width_value_label.config(text=f"{self.notch_width.get():.0f} Hz"))
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=16)
        
        info_box = ttk.Frame(parent, style='Card.TFrame', padding=12)
        info_box.pack(fill='x')
        
        ttk.Label(info_box, text="📊 Expected Results (Demo)", font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        
        metrics = [
            ("SNR Gain", "+34 dB", COLORS['success']),
            ("60 Hz Attenuation", ">60 dB", COLORS['accent']),
            ("2.5 kHz Attenuation", ">60 dB", COLORS['accent']),
            ("Process Time", "< 3 ms", COLORS['fg_muted']),
        ]
        
        for label, value, color in metrics:
            row = ttk.Frame(info_box)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=label, font=('Segoe UI', 8), foreground=COLORS['fg_muted']).pack(side='left')
            ttk.Label(row, text=value, font=('Segoe UI', 8, 'bold'), foreground=color).pack(side='right')
            
    def build_viz_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        
        ttk.Label(parent, text="Active 3D View", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        self.view_var = tk.StringVar(value="spectrogram")
        views = [
            ("📈  Spectrogram 3D", "spectrogram"),
            ("📊  Spectrum 3D", "spectrum"),
            ("🌊  Waterfall", "waterfall"),
            ("🔄  Comparison", "comparison"),
        ]
        
        for label, value in views:
            rb = ttk.Radiobutton(parent, text=label, variable=self.view_var, value=value, command=self.switch_view)
            rb.pack(anchor='w', pady=3)
            
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=12)
        
        ttk.Label(parent, text="Camera Controls", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        cam_controls = [
            ("Elevation", 'elev', -90, 90, 30),
            ("Azimuth", 'azim', -180, 180, 45),
            ("Distance", 'dist', 5, 20, 10),
        ]
        
        for label, key, min_v, max_v, default in cam_controls:
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=6)
            row.columnconfigure(1, weight=1)
            
            ttk.Label(row, text=label, font=('Segoe UI', 8), width=10).pack(side='left')
            
            var = tk.DoubleVar(value=default)
            self.cam_vars[key] = var
            
            scale = ttk.Scale(row, from_=min_v, to=max_v, variable=var, orient='horizontal', command=lambda v, k=key: self.update_camera())
            scale.pack(side='left', fill='x', expand=True, padx=8)
            
            val_label = ttk.Label(row, text=f"{default:.0f}", font=('Segoe UI', 8), foreground=COLORS['accent'], width=4)
            val_label.pack(side='right')
            var.trace_add('write', lambda *a, l=val_label, v=var: l.config(text=f"{v.get():.0f}"))
            
        ModernButton(parent, variant='ghost', text="🔄  Reset View", command=self.reset_camera).pack(fill='x', pady=(12, 0))
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=12)
        
        # Animación
        self.anim_btn = ModernButton(parent, variant='outline', text="▶  Auto-Rotate", command=self.toggle_animation)
        self.anim_btn.pack(fill='x')
        
        speed_row = ttk.Frame(parent)
        speed_row.pack(fill='x', pady=(8, 0))
        ttk.Label(speed_row, text="Speed", font=('Segoe UI', 8)).pack(side='left')
        ttk.Scale(speed_row, from_=0.2, to=3.0, variable=self.anim_speed, orient='horizontal').pack(side='left', fill='x', expand=True, padx=8)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=12)
        
        # Comparison Mode
        ttk.Label(parent, text="Comparison Mode", font=('Segoe UI', 9), foreground=COLORS['fg_muted']).pack(anchor='w')
        
        compare_modes = [
            ("🔀  Side by Side", "side_by_side"),
            ("🔄  Overlay (Transparent)", "overlay"),
            ("📊  Original Only", "original"),
            ("✨  Filtered Only", "filtered"),
            ("📈  Difference (Removed)", "difference"),
        ]
        
        for label, value in compare_modes:
            rb = ttk.Radiobutton(parent, text=label, variable=self.compare_mode, value=value, command=self.refresh_plots)
            rb.pack(anchor='w', pady=2)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=12)
        
        # Exportar
        ModernButton(parent, variant='ghost', text="💾  Save 3D Image", command=self.save_image).pack(fill='x', pady=2)
        ModernButton(parent, variant='ghost', text="📋  Copy to Clipboard", command=self.copy_image).pack(fill='x', pady=2)
        
    def build_main_panel(self, parent):
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.grid(row=0, column=1, sticky='nsew')
        
        self.plot_tabs = {}
        self.figures = {}
        self.canvases = {}
        self.axes = {}
        
        plot_configs = [
            ("📈  Spectrogram 3D", "spectrogram", "Time (s)", "Frequency (Hz)", "Magnitude (dB)"),
            ("📊  Spectrum 3D", "spectrum", "Frequency (Hz)", "Channel", "Magnitude (dB)"),
            ("🌊  Waterfall", "waterfall", "Time (s)", "Frequency (Hz)", "Magnitude (dB)"),
            ("🔄  Comparison", "comparison", "Frequency (Hz)", "Signal", "Magnitude (dB)"),
        ]
        
        for title, key, xl, yl, zl in plot_configs:
            tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(tab, text=title)
            self.plot_tabs[key] = tab
            
            fig = Figure(figsize=(7, 5), dpi=100, facecolor=COLORS['bg_secondary'])
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor(COLORS['bg_secondary'])
            
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.set_pane_color(COLORS['bg_secondary'])
                axis._axinfo['grid']['color'] = COLORS['border']
                axis._axinfo['tick']['color'] = COLORS['fg_muted']
                
            ax.xaxis.label.set_color(COLORS['fg'])
            ax.yaxis.label.set_color(COLORS['fg'])
            ax.zaxis.label.set_color(COLORS['fg'])
            ax.title.set_color(COLORS['fg'])
            
            canvas = FigureCanvasTkAgg(fig, tab)
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)
            
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.config(background=COLORS['bg_tertiary'])
            toolbar.pack(side='bottom', fill='x', padx=8, pady=(0, 8))
            
            self.figures[key] = fig
            self.canvases[key] = canvas
            self.axes[key] = ax
            
            ax.text2D(0.5, 0.5, "👈 Load audio file or run Demo", transform=ax.transAxes, ha='center', va='center', fontsize=14, color=COLORS['fg_muted'])
            ax.set_xlabel(xl, labelpad=10, color=COLORS['fg'])
            ax.set_ylabel(yl, labelpad=10, color=COLORS['fg'])
            ax.set_zlabel(zl, labelpad=10, color=COLORS['fg'])
            
        self.plot_notebook.bind('<<NotebookTabChanged>>', self.on_plot_tab_change)
        
    def build_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(12, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready • Drag & drop a WAV file or click Open")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Segoe UI', 9), foreground=COLORS['fg_muted'])
        status_label.grid(row=0, column=0, sticky='w')
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='indeterminate', length=200)
        self.progress_bar.grid(row=0, column=1, sticky='e', padx=(16, 0))
        self.progress_bar.grid_remove()
        
    def animate_entrance(self):
        self.root.attributes('-alpha', 0.0)
        def fade_in(alpha=0.0):
            alpha += 0.05
            self.root.attributes('-alpha', alpha)
            if alpha < 1.0:
                self.root.after(16, lambda: fade_in(alpha))
        fade_in()
        
    def browse_input(self):
        filename = filedialog.askopenfilename(title="Select Noisy Audio File", filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")])
        if filename:
            self.input_file.set(filename)
            base, ext = os.path.splitext(filename)
            self.output_file.set(f"{base}_clean{ext}")
            self.load_file_info(filename)
            self.status_var.set(f"Loaded: {os.path.basename(filename)}")
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(title="Save Clean Audio As", defaultextension=".wav", filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")])
        if filename:
            self.output_file.set(filename)
            
    def load_file_info(self, filepath):
        try:
            fs, data = wavfile.read(filepath)
            duration = len(data) / fs
            channels = data.shape[1] if len(data.shape) > 1 else 1
            self.file_info_labels['dur'].config(text=f"{duration:.1f} s")
            self.file_info_labels['sr'].config(text=f"{fs} Hz")
            self.file_info_labels['ch'].config(text="Stereo" if channels > 1 else "Mono")
        except:
            pass
            
    def load_preview(self):
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showwarning("No File", "Please select an input file first")
            return
            
        self.status_var.set("Loading audio & generating 3D views...")
        self.show_progress(True)
        threading.Thread(target=self._load_preview_thread, daemon=True).start()
        
    def _load_preview_thread(self):
        try:
            fs, data = wavfile.read(self.input_file.get())
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
                
            self.audio_data = data
            self.fs = fs
            self.filtered_data = None
            self.freqs_notch = [float(f.strip()) for f in self.frequencies.get().split(',') if f.strip()]
            
            self.root.after(0, lambda: self.status_var.set(f"Preview ready • {len(data)/fs:.1f}s @ {fs}Hz"))
            self.root.after(0, self.generate_all_plots)
            self.root.after(0, lambda: self.show_progress(False))
            self.root.after(0, lambda: self.update_playback_buttons(None))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load: {e}"))
            self.root.after(0, lambda: self.show_progress(False))
            
    def process_file(self):
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showwarning("No Input", "Select an input file first")
            return
        if not self.output_file.get():
            messagebox.showwarning("No Output", "Specify output file path")
            return
            
        try:
            freqs = [float(f.strip()) for f in self.frequencies.get().split(',') if f.strip()]
            if not freqs:
                raise ValueError
            self.freqs_notch = freqs
        except:
            messagebox.showwarning("Invalid Frequencies", "Use format: 60, 120, 2500")
            return
            
        self.is_processing = True
        self.process_btn.config(state='disabled', text="⏳  Processing...")
        self.show_progress(True)
        self.status_var.set("Applying FFT notch filter...")
        threading.Thread(target=self._process_thread, daemon=True).start()
        
    def _process_thread(self):
        try:
            fs, data = wavfile.read(self.input_file.get())
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
                
            self.audio_data = data
            self.fs = fs
            
            self.root.after(0, lambda: self.status_var.set(f"Filtering: {self.freqs_notch} Hz..."))
            
            y, Y, X, H = filtrar_espectral(data, fs, self.freqs_notch, self.notch_width.get())
            self.filtered_data = y
            
            y_clip = np.clip(y, -1.0, 1.0)
            y_int16 = (y_clip * 32767).astype(np.int16)
            wavfile.write(self.output_file.get(), fs, y_int16)
            
            snr_out = calcular_snr(data, y)
            
            self.root.after(0, lambda: self.status_var.set(f"✨ Done! SNR: {snr_out:.1f} dB • Saved: {os.path.basename(self.output_file.get())}"))
            self.root.after(0, self.generate_all_plots)
            self.root.after(0, lambda: self.show_progress(False))
            self.root.after(0, lambda: self.update_playback_buttons(None))
            self.root.after(0, lambda: messagebox.showinfo("Success!", f"Audio restored successfully!\n\nOutput: {self.output_file.get()}\nSNR: {snr_out:.1f} dB"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
            self.root.after(0, lambda: self.show_progress(False))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state='normal', text="✨  Process & Visualize"))
            
    def generate_all_plots(self):
        if self.audio_data is None:
            return
        self.status_var.set("Rendering 3D visualizations...")
        self.plot_spectrogram_3d()
        self.plot_spectrum_3d()
        self.plot_waterfall_3d()
        self.plot_comparison_3d()
        self.status_var.set("Ready • Interact with 3D views using mouse")
        
    # ============================================================
    # 3D PLOTTING FUNCTIONS WITH COMPARISON MODES
    # ============================================================
    
    def plot_spectrogram_3d(self):
        ax = self.axes['spectrogram']
        ax.clear()
        ax.set_facecolor(COLORS['bg_secondary'])
        
        mode = self.compare_mode.get()
        has_filtered = self.filtered_data is not None
        
        nperseg = min(1024, len(self.audio_data) // 8)
        f, t, Sxx_orig = spectrogram(self.audio_data, self.fs, nperseg=nperseg, noverlap=nperseg//2)
        mask = f <= 5000
        f = f[mask]
        Sxx_orig = Sxx_orig[mask, :]
        Sxx_orig_db = 10 * np.log10(Sxx_orig + 1e-10)
        
        if has_filtered:
            _, _, Sxx_filt = spectrogram(self.filtered_data, self.fs, nperseg=nperseg, noverlap=nperseg//2)
            Sxx_filt = Sxx_filt[mask, :]
            Sxx_filt_db = 10 * np.log10(Sxx_filt + 1e-10)
        
        stride_t = max(1, len(t) // 120)
        stride_f = max(1, len(f) // 80)
        T, F = np.meshgrid(t[::stride_t], f[::stride_f])
        
        if mode == "original" or not has_filtered:
            Z = Sxx_orig_db[::stride_f, ::stride_t]
            surf = ax.plot_surface(T, F, Z, cmap=CUSTOM_CMAP, linewidth=0, antialiased=True, alpha=0.95, rstride=1, cstride=1, shade=True, edgecolor='none')
            title = "Spectrogram 3D • Original (Noisy)"
            
        elif mode == "filtered":
            Z = Sxx_filt_db[::stride_f, ::stride_t]
            surf = ax.plot_surface(T, F, Z, cmap=CUSTOM_CMAP, linewidth=0, antialiased=True, alpha=0.95, rstride=1, cstride=1, shade=True, edgecolor='none')
            title = "Spectrogram 3D • Filtered (Clean)"
            
        elif mode == "difference":
            Z = Sxx_orig_db[::stride_f, ::stride_t] - Sxx_filt_db[::stride_f, ::stride_t]
            surf = ax.plot_surface(T, F, Z, cmap='RdBu_r', linewidth=0, antialiased=True, alpha=0.95, rstride=1, cstride=1, shade=True, edgecolor='none')
            title = "Spectrogram 3D • Difference (Removed)"
            
        elif mode == "overlay":
            Z_orig = Sxx_orig_db[::stride_f, ::stride_t]
            Z_filt = Sxx_filt_db[::stride_f, ::stride_t]
            surf1 = ax.plot_surface(T, F, Z_orig, cmap=CUSTOM_CMAP, linewidth=0, antialiased=True, alpha=0.6, rstride=1, cstride=1, shade=True, edgecolor='none')
            surf2 = ax.plot_surface(T, F, Z_filt, cmap='Greens', linewidth=0, antialiased=True, alpha=0.6, rstride=1, cstride=1, shade=True, edgecolor='none')
            title = "Spectrogram 3D • Overlay (Original + Filtered)"
            
        else:  # side_by_side
            t_max = t[-1]
            T_orig = T
            T_filt = T + t_max * 1.1
            Z_orig = Sxx_orig_db[::stride_f, ::stride_t]
            surf1 = ax.plot_surface(T_orig, F, Z_orig, cmap=CUSTOM_CMAP, linewidth=0, antialiased=True, alpha=0.9, rstride=1, cstride=1, shade=True, edgecolor='none')
            if has_filtered:
                Z_filt = Sxx_filt_db[::stride_f, ::stride_t]
                surf2 = ax.plot_surface(T_filt, F, Z_filt, cmap='Greens', linewidth=0, antialiased=True, alpha=0.9, rstride=1, cstride=1, shade=True, edgecolor='none')
                ax.text(t_max * 0.5, f[-1]*0.9, Z_orig.max()*0.9, "ORIGINAL", color=COLORS['accent'], fontsize=10, fontweight='bold', ha='center')
                ax.text(t_max * 1.6, f[-1]*0.9, Z_filt.max()*0.9, "FILTERED", color=COLORS['success'], fontsize=10, fontweight='bold', ha='center')
            title = "Spectrogram 3D • Side by Side"
        
        for freq in self.freqs_notch:
            if freq <= f[-1]:
                ax.axhline(y=freq, color=COLORS['danger'], linestyle='--', alpha=0.6, linewidth=1.5)
                if mode != "side_by_side":
                    ax.text(t[-1]*0.95, freq, ax.get_zlim()[1]*0.9, f"  🎯 {freq}Hz", color=COLORS['danger'], fontsize=9, fontweight='bold')
        
        ax.set_xlabel("Time (s)", labelpad=12, color=COLORS['fg'])
        ax.set_ylabel("Frequency (Hz)", labelpad=12, color=COLORS['fg'])
        ax.set_zlabel("Magnitude (dB)", labelpad=12, color=COLORS['fg'])
        ax.set_title(title, fontsize=12, color=COLORS['fg'], pad=15, fontweight='bold')
        self.apply_camera(ax)
        self.canvases['spectrogram'].draw()
        
    def plot_spectrum_3d(self):
        ax = self.axes['spectrum']
        ax.clear()
        ax.set_facecolor(COLORS['bg_secondary'])
        
        mode = self.compare_mode.get()
        has_filtered = self.filtered_data is not None
        
        N = len(self.audio_data)
        X = fftshift(fft(self.audio_data))
        freqs = fftshift(fftfreq(N, 1/self.fs))
        mag = fftshift(np.abs(X))
        pos = (freqs >= 0) & (freqs <= 5000)
        f = freqs[pos]
        m_orig = 20 * np.log10(mag[pos] + 1e-10)
        
        if has_filtered:
            Y = fftshift(fft(self.filtered_data))
            mag_y = fftshift(np.abs(Y))[pos]
            m_filt = 20 * np.log10(mag_y + 1e-10)
        
        dx = (f[1] - f[0]) * 0.7 if len(f) > 1 else 10
        
        if mode == "original" or not has_filtered:
            colors = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, m_orig, color=colors, alpha=0.85, shade=True, edgecolor='none')
            title = "Spectrum 3D • Original"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Original"])
            
        elif mode == "filtered":
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, m_filt, color=COLORS['success'], alpha=0.85, shade=True, edgecolor='none')
            title = "Spectrum 3D • Filtered"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Filtered"])
            
        elif mode == "difference":
            diff = m_orig - m_filt
            colors = [COLORS['danger'] if d > 3 else COLORS['fg_muted'] for d in diff]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, diff, color=colors, alpha=0.85, shade=True, edgecolor='none')
            title = "Spectrum 3D • Attenuation (Removed)"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Attenuation"])
            
        elif mode == "overlay":
            colors_orig = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, m_orig, color=colors_orig, alpha=0.6, shade=True, edgecolor='none')
            ax.bar3d(f, np.zeros_like(f)+0.3, np.zeros_like(f), dx, 0.5, m_filt, color=COLORS['success'], alpha=0.5, shade=True, edgecolor='none')
            title = "Spectrum 3D • Overlay"
            ax.set_yticks([0.25, 0.55])
            ax.set_yticklabels(["Original", "Filtered"])
            
        else:  # side_by_side
            colors_orig = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.4, m_orig, color=colors_orig, alpha=0.85, shade=True, edgecolor='none')
            ax.bar3d(f, np.ones_like(f)*0.7, np.zeros_like(f), dx, 0.4, m_filt, color=COLORS['success'], alpha=0.85, shade=True, edgecolor='none')
            title = "Spectrum 3D • Side by Side"
            ax.set_yticks([0.2, 0.9])
            ax.set_yticklabels(["Original", "Filtered"])
        
        for freq in self.freqs_notch:
            if freq <= f[-1]:
                ax.axvline(x=freq, color=COLORS['danger'], linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel("Frequency (Hz)", labelpad=12, color=COLORS['fg'])
        ax.set_ylabel("Channel", labelpad=12, color=COLORS['fg'])
        ax.set_zlabel("Magnitude (dB)", labelpad=12, color=COLORS['fg'])
        ax.set_title(title, fontsize=12, color=COLORS['fg'], pad=15, fontweight='bold')
        self.apply_camera(ax)
        self.canvases['spectrum'].draw()
        
    def plot_waterfall_3d(self):
        ax = self.axes['waterfall']
        ax.clear()
        ax.set_facecolor(COLORS['bg_secondary'])
        
        mode = self.compare_mode.get()
        has_filtered = self.filtered_data is not None
        
        window = self.fs // 10
        hop = window // 2
        n_win = min(40, (len(self.audio_data) - window) // hop)
        
        if n_win < 2:
            ax.text2D(0.5, 0.5, "Audio too short for waterfall", transform=ax.transAxes, ha='center', color=COLORS['fg_muted'])
            self.canvases['waterfall'].draw()
            return
            
        def compute_spectra(data):
            spectra = []
            times = []
            for i in range(n_win):
                start = i * hop
                win_data = data[start:start+window] * np.hanning(window)
                spec = np.abs(fft(win_data))[:window//2]
                freqs = fftfreq(window, 1/self.fs)[:window//2]
                mask = freqs <= 5000
                spectra.append(spec[mask])
                times.append(start / self.fs)
            return np.array(spectra).T, freqs[mask], times
        
        spectra_orig, freqs_plot, times = compute_spectra(self.audio_data)
        spectra_orig_db = 20 * np.log10(spectra_orig + 1e-10)
        
        if has_filtered:
            spectra_filt, _, _ = compute_spectra(self.filtered_data)
            spectra_filt_db = 20 * np.log10(spectra_filt + 1e-10)
        
        T, F = np.meshgrid(times, freqs_plot)
        
        if mode == "original" or not has_filtered:
            stride = max(1, n_win // 30)
            surf = ax.plot_surface(T[::5, ::stride], F[::5, ::stride], spectra_orig_db[::5, ::stride], cmap=CUSTOM_CMAP, alpha=0.7, linewidth=0, antialiased=True, shade=True)
            for i in range(0, len(times), max(1, len(times)//15)):
                ax.plot(np.full_like(freqs_plot, times[i]), freqs_plot, spectra_orig_db[:, i], color=COLORS['accent'], alpha=0.4, linewidth=0.8)
            title = "Waterfall 3D • Original"
            
        elif mode == "filtered":
            stride = max(1, n_win // 30)
            surf = ax.plot_surface(T[::5, ::stride], F[::5, ::stride], spectra_filt_db[::5, ::stride], cmap='Greens', alpha=0.7, linewidth=0, antialiased=True, shade=True)
            title = "Waterfall 3D • Filtered"
            
        elif mode == "difference":
            diff = spectra_orig_db - spectra_filt_db
            stride = max(1, n_win // 30)
            surf = ax.plot_surface(T[::5, ::stride], F[::5, ::stride], diff[::5, ::stride], cmap='RdBu_r', alpha=0.7, linewidth=0, antialiased=True, shade=True)
            title = "Waterfall 3D • Difference (Removed)"
            
        elif mode == "overlay":
            stride = max(1, n_win // 30)
            surf1 = ax.plot_surface(T[::5, ::stride], F[::5, ::stride], spectra_orig_db[::5, ::stride], cmap=CUSTOM_CMAP, alpha=0.5, linewidth=0, antialiased=True, shade=True)
            surf2 = ax.plot_surface(T[::5, ::stride], F[::5, ::stride], spectra_filt_db[::5, ::stride], cmap='Greens', alpha=0.5, linewidth=0, antialiased=True, shade=True)
            title = "Waterfall 3D • Overlay"
            
        else:  # side_by_side
            # Desplazar en X
            t_max = times[-1] if times else 1
            T_orig = T
            T_filt = T + t_max * 1.2
            stride = max(1, n_win // 30)
            surf1 = ax.plot_surface(T_orig[::5, ::stride], F[::5, ::stride], spectra_orig_db[::5, ::stride], cmap=CUSTOM_CMAP, alpha=0.7, linewidth=0, antialiased=True, shade=True)
            surf2 = ax.plot_surface(T_filt[::5, ::stride], F[::5, ::stride], spectra_filt_db[::5, ::stride], cmap='Greens', alpha=0.7, linewidth=0, antialiased=True, shade=True)
            title = "Waterfall 3D • Side by Side"
        
        ax.set_xlabel("Time (s)", labelpad=12, color=COLORS['fg'])
        ax.set_ylabel("Frequency (Hz)", labelpad=12, color=COLORS['fg'])
        ax.set_zlabel("Magnitude (dB)", labelpad=12, color=COLORS['fg'])
        ax.set_title(title, fontsize=12, color=COLORS['fg'], pad=15, fontweight='bold')
        self.apply_camera(ax)
        self.canvases['waterfall'].draw()
        
    def plot_comparison_3d(self):
        ax = self.axes['comparison']
        ax.clear()
        ax.set_facecolor(COLORS['bg_secondary'])
        
        mode = self.compare_mode.get()
        has_filtered = self.filtered_data is not None
        
        N = len(self.audio_data)
        X = fftshift(fft(self.audio_data))
        freqs = fftshift(fftfreq(N, 1/self.fs))
        pos = (freqs >= 0) & (freqs <= 5000)
        f = freqs[pos]
        m_orig = 20 * np.log10(np.abs(X[pos]) + 1e-10)
        
        if has_filtered:
            Y = fftshift(fft(self.filtered_data))
            m_filt = 20 * np.log10(np.abs(Y[pos]) + 1e-10)
        
        dx = (f[1] - f[0]) * 0.7 if len(f) > 1 else 10
        
        if mode == "original" or not has_filtered:
            colors = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, m_orig, color=colors, alpha=0.85, shade=True, edgecolor='none')
            title = "Comparison 3D • Original"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Original"])
            
        elif mode == "filtered":
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, m_filt, color=COLORS['success'], alpha=0.85, shade=True, edgecolor='none')
            title = "Comparison 3D • Filtered"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Filtered"])
            
        elif mode == "difference":
            diff = m_orig - m_filt
            colors = [COLORS['danger'] if d > 3 else COLORS['fg_muted'] for d in diff]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.5, diff, color=colors, alpha=0.85, shade=True, edgecolor='none')
            # Curva de atenuación
            ax.plot(f, np.full_like(f, 0.25), diff, color=COLORS['danger'], linewidth=3, alpha=0.9)
            title = "Comparison 3D • Attenuation Curve"
            ax.set_yticks([0.25])
            ax.set_yticklabels(["Attenuation"])
            
        elif mode == "overlay":
            colors_orig = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.4, m_orig, color=colors_orig, alpha=0.6, shade=True, edgecolor='none')
            ax.bar3d(f, np.zeros_like(f)+0.5, np.zeros_like(f), dx, 0.4, m_filt, color=COLORS['success'], alpha=0.5, shade=True, edgecolor='none')
            # Curva atenuación
            ax.plot(f, np.full_like(f, 0.25), m_orig - m_filt, color=COLORS['danger'], linewidth=3, alpha=0.9)
            title = "Comparison 3D • Overlay + Attenuation"
            ax.set_yticks([0.2, 0.7])
            ax.set_yticklabels(["Original", "Filtered"])
            
        else:  # side_by_side
            colors_orig = [COLORS['danger'] if any(abs(freq - fn) < self.notch_width.get()/2 for fn in self.freqs_notch) else COLORS['accent'] for freq in f]
            ax.bar3d(f, np.zeros_like(f), np.zeros_like(f), dx, 0.4, m_orig, color=colors_orig, alpha=0.85, shade=True, edgecolor='none')
            ax.bar3d(f, np.ones_like(f)*0.8, np.zeros_like(f), dx, 0.4, m_filt, color=COLORS['success'], alpha=0.85, shade=True, edgecolor='none')
            # Curva atenuación
            diff = m_orig - m_filt
            ax.plot(f, np.full_like(f, 0.4), diff, color=COLORS['danger'], linewidth=3, alpha=0.9, label='Attenuation')
            title = "Comparison 3D • Side by Side + Attenuation"
            ax.set_yticks([0.2, 1.0])
            ax.set_yticklabels(["Original", "Filtered"])
            ax.legend(loc='upper right', fontsize=9)
        
        for freq in self.freqs_notch:
            if freq <= f[-1]:
                ax.axvline(x=freq, color=COLORS['warning'], linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel("Frequency (Hz)", labelpad=12, color=COLORS['fg'])
        ax.set_ylabel("Signal", labelpad=12, color=COLORS['fg'])
        ax.set_zlabel("Magnitude (dB)", labelpad=12, color=COLORS['fg'])
        ax.set_title(title, fontsize=12, color=COLORS['fg'], pad=15, fontweight='bold')
        self.apply_camera(ax)
        self.canvases['comparison'].draw()
        
    def apply_camera(self, ax):
        ax.view_init(elev=self.cam_vars['elev'].get(), azim=self.cam_vars['azim'].get())
        ax.dist = self.cam_vars['dist'].get()
        
    def update_camera(self):
        for ax in self.axes.values():
            self.apply_camera(ax)
        for canvas in self.canvases.values():
            canvas.draw_idle()
            
    def reset_camera(self):
        self.cam_vars['elev'].set(30)
        self.cam_vars['azim'].set(45)
        self.cam_vars['dist'].set(10)
        self.update_camera()
        
    def toggle_animation(self):
        self.anim_var.set(not self.anim_var.get())
        if self.anim_var.get():
            self.anim_btn.config(text="⏸  Stop Rotation")
            self.animate_rotation()
        else:
            self.anim_btn.config(text="▶  Auto-Rotate")
            
    def animate_rotation(self):
        if not self.anim_var.get():
            return
        current = self.cam_vars['azim'].get()
        speed = self.anim_speed.get()
        new = (current + speed) % 360
        if new > 180: new -= 360
        self.cam_vars['azim'].set(new)
        self.update_camera()
        self.root.after(30, self.animate_rotation)
        
    def switch_view(self):
        self.generate_all_plots()
        
    def on_plot_tab_change(self, event):
        tab_id = self.plot_notebook.select()
        tab_text = self.plot_notebook.tab(tab_id, "text")
        view_map = {
            "📈  Spectrogram 3D": "spectrogram",
            "📊  Spectrum 3D": "spectrum",
            "🌊  Waterfall": "waterfall",
            "🔄  Comparison": "comparison",
        }
        key = view_map.get(tab_text, "spectrogram")
        self.view_var.set(key)
        
    def refresh_plots(self):
        if self.audio_data is not None:
            self.generate_all_plots()
        
    def save_image(self):
        current_tab = self.plot_notebook.select()
        tab_text = self.plot_notebook.tab(current_tab, "text")
        view_map = {
            "📈  Spectrogram 3D": "spectrogram",
            "📊  Spectrum 3D": "spectrum",
            "🌊  Waterfall": "waterfall",
            "🔄  Comparison": "comparison",
        }
        key = view_map.get(tab_text, "spectrogram")
        filename = filedialog.asksaveasfilename(title="Save 3D Visualization", defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("PDF Vector", "*.pdf"), ("SVG Vector", "*.svg")])
        if filename:
            self.figures[key].savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['bg_secondary'])
            self.status_var.set(f"💾 Saved: {os.path.basename(filename)}")
            
    def copy_image(self):
        self.status_var.set("📋 Copy to clipboard - feature coming soon")
        
    # ============================================================
    # AUDIO PLAYBACK METHODS
    # ============================================================
    
    def play_audio(self, which):
        if not AUDIO_BACKEND:
            messagebox.showwarning("No Audio Backend", "Install: pip install sounddevice (or simpleaudio/pygame)")
            return
            
        if which == 'original' and self.audio_data is None:
            messagebox.showwarning("No Audio", "Load an audio file first")
            return
        if which == 'filtered' and self.filtered_data is None:
            messagebox.showwarning("No Filtered Audio", "Process the audio first")
            return
        if which == 'removed' and (self.audio_data is None or self.filtered_data is None):
            messagebox.showwarning("No Audio", "Need both original and filtered audio")
            return
            
        self.stop_audio()
        
        if which == 'original':
            data = self.audio_data.copy()
            label = "Original (Noisy)"
            normalize = True
        elif which == 'filtered':
            data = self.filtered_data.copy()
            label = "Filtered (Clean)"
            normalize = True
        else:  # removed
            min_len = min(len(self.audio_data), len(self.filtered_data))
            data = self.audio_data[:min_len] - self.filtered_data[:min_len]
            label = "Removed (Noise/Interference)"
            normalize = not self.removed_relative_volume.get()
            if self.removed_relative_volume.get():
                label += " [Relative Level]"
            else:
                label += " [Boosted]"
            
        self.current_playback = which
        self.playback_state = 'playing'
        self.stop_playback_flag = False
        
        self.update_playback_buttons(which)
        self.status_var.set(f"🔊 Playing: {label}")
        
        self.playback_thread = threading.Thread(target=self._playback_thread, args=(data, label, normalize), daemon=True)
        self.playback_thread.start()
        
    def _playback_thread(self, data, label, normalize=True):
        try:
            if normalize:
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val * 0.95 * self.volume_var.get()
            else:
                data = data * self.volume_var.get()
            
            data = np.clip(data, -1.0, 1.0)
            audio_int16 = (data * 32767).astype(np.int16)
            
            if AUDIO_BACKEND == 'sounddevice':
                self._play_sounddevice(audio_int16)
            elif AUDIO_BACKEND == 'simpleaudio':
                self._play_simpleaudio(audio_int16)
            elif AUDIO_BACKEND == 'pygame':
                self._play_pygame(audio_int16)
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Playback error: {e}"))
        finally:
            self.root.after(0, self._playback_finished)
            
    def _play_sounddevice(self, audio_int16):
        import sounddevice as sd
        duration = len(audio_int16) / self.fs
        start_time = time.time()
        
        try:
            def callback(outdata, frames, time_info, status):
                if self.stop_playback_flag:
                    raise sd.CallbackStop()
                elapsed = time.time() - start_time
                progress = min(100, (elapsed / duration) * 100)
                self.root.after(0, lambda p=progress: self.playback_progress.config(value=p))
            
            with sd.OutputStream(samplerate=self.fs, channels=1, dtype='int16', callback=callback) as stream:
                sd.play(audio_int16, samplerate=self.fs, blocking=False)
                while not self.stop_playback_flag:
                    if not sd.get_stream().active:
                        break
                    time.sleep(0.05)
                sd.stop()
        except Exception:
            sd.play(audio_int16, samplerate=self.fs, blocking=False)
            start_time = time.time()
            while not self.stop_playback_flag:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
                progress = min(100, (elapsed / duration) * 100)
                self.root.after(0, lambda p=progress: self.playback_progress.config(value=p))
                time.sleep(0.05)
            sd.stop()
        
    def _play_simpleaudio(self, audio_int16):
        import simpleaudio as sa
        audio_bytes = audio_int16.tobytes()
        play_obj = sa.play_buffer(audio_bytes, num_channels=1, bytes_per_sample=2, sample_rate=self.fs)
        duration = len(audio_int16) / self.fs
        start_time = time.time()
        while play_obj.is_playing() and not self.stop_playback_flag:
            elapsed = time.time() - start_time
            progress = min(100, (elapsed / duration) * 100)
            self.root.after(0, lambda p=progress: self.playback_progress.config(value=p))
            time.sleep(0.05)
        play_obj.stop()
        
    def _play_pygame(self, audio_int16):
        import pygame
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(audio_int16.tobytes())
        
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        duration = len(audio_int16) / self.fs
        start_time = time.time()
        while pygame.mixer.music.get_busy() and not self.stop_playback_flag:
            elapsed = time.time() - start_time
            progress = min(100, (elapsed / duration) * 100)
            self.root.after(0, lambda p=progress: self.playback_progress.config(value=p))
            time.sleep(0.05)
        pygame.mixer.music.stop()
        try:
            os.unlink(tmp_path)
        except:
            pass
        
    def stop_audio(self):
        self.stop_playback_flag = True
        self.playback_state = 'stopped'
        
        if AUDIO_BACKEND == 'sounddevice':
            try:
                import sounddevice as sd
                sd.stop()
            except:
                pass
        elif AUDIO_BACKEND == 'simpleaudio':
            try:
                import simpleaudio as sa
                sa.stop_all()
            except:
                pass
        elif AUDIO_BACKEND == 'pygame':
            try:
                import pygame
                pygame.mixer.music.stop()
            except:
                pass
                
        self._playback_finished()
        
    def _playback_finished(self):
        self.playback_state = 'stopped'
        self.current_playback = None
        self.playback_progress['value'] = 0
        self.update_playback_buttons(None)
        self.status_var.set("Ready")
        
    def update_playback_buttons(self, playing):
        if playing == 'original':
            self.play_orig_btn.config(text="⏸ Original")
        else:
            self.play_orig_btn.config(text="🎵 Original")
        self.play_orig_btn.config(state='normal' if self.audio_data is not None else 'disabled')
        
        if playing == 'filtered':
            self.play_filt_btn.config(text="⏸ Filtered")
        else:
            self.play_filt_btn.config(text="✨ Filtered")
        self.play_filt_btn.config(state='normal' if self.filtered_data is not None else 'disabled')
        
        if playing == 'removed':
            self.play_removed_btn.config(text="⏸ Removed")
        else:
            self.play_removed_btn.config(text="🗑️ Removed")
        self.play_removed_btn.config(state='normal' if (self.audio_data is not None and self.filtered_data is not None) else 'disabled')
        
        self.stop_btn.config(state='normal' if playing else 'disabled')
        
        if playing:
            self.process_btn.config(state='disabled')
        elif not self.is_processing:
            self.process_btn.config(state='normal')
            
    def run_demo(self):
        self.is_processing = True
        self.show_progress(True)
        self.status_var.set("Running synthetic demo...")
        threading.Thread(target=self._demo_thread, daemon=True).start()
        
    def _demo_thread(self):
        try:
            fs = 44100
            dur = 3.0
            t, s = generar_senal_base(fs, dur)
            x = contaminar_senal(s, t, fs)
            y, Y, X, H = filtrar_espectral(x, fs, [60, 2500], 12)
            
            snr_i = calcular_snr(s, x)
            snr_f = calcular_snr(s, y)
            
            self.audio_data = x
            self.filtered_data = y
            self.fs = fs
            self.freqs_notch = [60, 2500]
            self.frequencies.set("60, 2500")
            
            self.root.after(0, lambda: self.status_var.set(f"✨ Demo complete! SNR: {snr_i:.1f} → {snr_f:.1f} dB (+{snr_f-snr_i:.1f} dB)"))
            self.root.after(0, self.generate_all_plots)
            self.root.after(0, lambda: self.show_progress(False))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Demo Error", str(e)))
            self.root.after(0, lambda: self.show_progress(False))
        finally:
            self.is_processing = False
            
    def generate_test_files(self):
        threading.Thread(target=self._gen_thread, daemon=True).start()
        
    def _gen_thread(self):
        try:
            from generate_test_audio import main as gen_main
            import io, sys
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                gen_main()
            self.root.after(0, lambda: self.status_var.set("✅ Test files generated in ./data/"))
            self.root.after(0, lambda: messagebox.showinfo("Done", "Test files created:\n• data/original.wav (noisy)\n• data/filtrado.wav (clean)\n• data/limpio_referencia.wav (reference)"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
    def show_progress(self, show):
        if show:
            self.progress_bar.grid()
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ModernGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()