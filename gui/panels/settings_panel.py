"""
Settings Panel - Complete Implementation
Feature #7: Adjust Recognition Threshold
Feature #4: Duplicate Detection Settings
"""

import customtkinter as ctk
from tkinter import filedialog
import json
from pathlib import Path

from config import Config
from utils.toast import ToastManager


class SettingsPanel(ctk.CTkScrollableFrame):
    """Complete settings panel with all configuration options"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        # Settings file path
        self.settings_file = Path("gui_settings.json")

        # Load saved settings
        self.load_settings()

        self.create_widgets()

    def load_settings(self):
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    self.saved_settings = json.load(f)
            else:
                self.saved_settings = {
                    'recognition_threshold': Config.RECOGNITION_THRESHOLD,
                    'duplicate_threshold': Config.DUPLICATE_THRESHOLD,
                    'top_k_default': 3,
                    'gallery_columns': 3
                }
        except Exception as e:
            print(f"Error loading settings: {e}")
            self.saved_settings = {
                'recognition_threshold': Config.RECOGNITION_THRESHOLD,
                'duplicate_threshold': Config.DUPLICATE_THRESHOLD,
                'top_k_default': 3,
                'gallery_columns': 3
            }

    def save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.saved_settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="⚙️ System Settings",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        subtitle = ctk.CTkLabel(
            self,
            text="Configure face recognition system parameters",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        subtitle.pack(pady=(0, Config.PADDING_LARGE))

        # ==================== RECOGNITION THRESHOLD ====================
        threshold_frame = ctk.CTkFrame(self, corner_radius=15)
        threshold_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            threshold_frame,
            text="🎯 Recognition Threshold",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_TINY))

        ctk.CTkLabel(
            threshold_frame,
            text="Minimum similarity score to consider a face match",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Current value display
        self.threshold_display = ctk.CTkFrame(threshold_frame, fg_color=("gray85", "gray20"))
        self.threshold_display.pack(pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            self.threshold_display,
            text="Current Threshold:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        ).pack(side="left", padx=(Config.PADDING_MEDIUM, Config.PADDING_TINY))

        self.threshold_value_label = ctk.CTkLabel(
            self.threshold_display,
            text=f"{self.app.recognizer.threshold:.2f}",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=Config.COLOR_SUCCESS_TEXT
        )
        self.threshold_value_label.pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkLabel(
            self.threshold_display,
            text=f"({self.app.recognizer.threshold*100:.0f}%)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(side="left", padx=(0, Config.PADDING_MEDIUM))

        # Slider
        slider_frame = ctk.CTkFrame(threshold_frame, fg_color="transparent")
        slider_frame.pack(pady=Config.PADDING_MEDIUM, padx=Config.PADDING_LARGE, fill="x")

        self.threshold_slider = ctk.CTkSlider(
            slider_frame,
            from_=0.30,
            to=0.90,
            number_of_steps=60,
            command=self.update_threshold
        )
        self.threshold_slider.set(self.app.recognizer.threshold)
        self.threshold_slider.pack(fill="x", pady=Config.PADDING_SMALL)

        # Scale labels
        scale_frame = ctk.CTkFrame(slider_frame, fg_color="transparent")
        scale_frame.pack(fill="x")

        labels = [
            ("0.30", "More Permissive"),
            ("0.50", "Balanced"),
            ("0.70", "Recommended"),
            ("0.90", "Very Strict")
        ]

        for val, desc in labels:
            label_container = ctk.CTkFrame(scale_frame, fg_color="transparent")
            label_container.pack(side="left", expand=True)

            ctk.CTkLabel(
                label_container,
                text=val,
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1, weight="bold")
            ).pack()

            ctk.CTkLabel(
                label_container,
                text=desc,
                font=ctk.CTkFont(size=9),
                text_color=Config.COLOR_INFO_TEXT
            ).pack()

        # Quick presets
        preset_frame = ctk.CTkFrame(threshold_frame, fg_color="transparent")
        preset_frame.pack(pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            preset_frame,
            text="Quick Presets:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold")
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        presets = [
            ("Permissive", 0.45, Config.COLOR_WARNING),
            ("Balanced", 0.60, "blue"),
            ("Strict", 0.75, Config.COLOR_ERROR)
        ]

        for name, value, color in presets:
            ctk.CTkButton(
                preset_frame,
                text=name,
                command=lambda v=value: self.set_threshold_preset(v),
                width=100,
                height=30,
                fg_color=color
            ).pack(side="left", padx=Config.PADDING_TINY)

        # Info box
        info_box = ctk.CTkFrame(threshold_frame, corner_radius=10)
        info_box.pack(fill="x", padx=Config.PADDING_LARGE,
                     pady=(Config.PADDING_SMALL, Config.PADDING_LARGE))

        info_text = """
ℹ️ Threshold Guide:

• 0.30-0.50 (Permissive): Accepts more matches
  Risk: May include false positives
  Use when: You want to catch all possible matches
  
• 0.50-0.70 (Balanced): Good trade-off ✅ Recommended
  Risk: Balanced false positives/negatives
  Use when: General face recognition tasks
  
• 0.70-0.90 (Strict): Only very confident matches
  Risk: May reject valid matches
  Use when: High security or accuracy is critical
        """

        info_label = ctk.CTkTextbox(info_box, height=180, wrap="word")
        info_label.insert("1.0", info_text.strip())
        info_label.configure(state="disabled")
        info_label.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_MEDIUM)

        # ==================== DUPLICATE DETECTION ====================
        dup_frame = ctk.CTkFrame(self, corner_radius=15)
        dup_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            dup_frame,
            text="🔍 Duplicate Detection Threshold",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_TINY))

        ctk.CTkLabel(
            dup_frame,
            text="Similarity threshold to warn about potential duplicates when adding new person",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Current value
        self.dup_value_label = ctk.CTkLabel(
            dup_frame,
            text=f"Current: {self.saved_settings.get('duplicate_threshold', Config.DUPLICATE_THRESHOLD):.2f}",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.dup_value_label.pack(pady=Config.PADDING_SMALL)

        # Slider
        self.dup_slider = ctk.CTkSlider(
            dup_frame,
            from_=0.70,
            to=0.95,
            number_of_steps=50,
            width=500,
            command=self.update_dup_threshold
        )
        self.dup_slider.set(self.saved_settings.get('duplicate_threshold', Config.DUPLICATE_THRESHOLD))
        self.dup_slider.pack(pady=Config.PADDING_SMALL)

        # Scale
        dup_scale_frame = ctk.CTkFrame(dup_frame, fg_color="transparent")
        dup_scale_frame.pack()

        ctk.CTkLabel(
            dup_scale_frame,
            text="0.70 (Low - More warnings)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(side="left", padx=Config.PADDING_LARGE)

        ctk.CTkLabel(
            dup_scale_frame,
            text="0.95 (High - Fewer warnings)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(side="right", padx=Config.PADDING_LARGE)

        ctk.CTkLabel(
            dup_frame,
            text="Higher threshold = Only warn about very similar faces",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(Config.PADDING_TINY, Config.PADDING_LARGE))

        # ==================== TOP-K DEFAULT ====================
        topk_frame = ctk.CTkFrame(self, corner_radius=15)
        topk_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            topk_frame,
            text="🔢 Top-K Default Value",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_TINY))

        ctk.CTkLabel(
            topk_frame,
            text="Default number of similar faces to show in recognition mode",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        self.topk_value_label = ctk.CTkLabel(
            topk_frame,
            text=f"K = {self.saved_settings.get('top_k_default', 3)}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        self.topk_value_label.pack(pady=Config.PADDING_SMALL)

        self.topk_slider = ctk.CTkSlider(
            topk_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            width=400,
            command=self.update_topk
        )
        self.topk_slider.set(self.saved_settings.get('top_k_default', 3))
        self.topk_slider.pack(pady=(Config.PADDING_SMALL, Config.PADDING_LARGE))

        # ==================== GALLERY SETTINGS ====================
        gallery_frame = ctk.CTkFrame(self, corner_radius=15)
        gallery_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            gallery_frame,
            text="🖼️ Gallery Display Settings",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_TINY))

        ctk.CTkLabel(
            gallery_frame,
            text="Number of columns in gallery grid view",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Column selector
        col_frame = ctk.CTkFrame(gallery_frame, fg_color="transparent")
        col_frame.pack(pady=Config.PADDING_MEDIUM)

        ctk.CTkLabel(
            col_frame,
            text="Columns:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(side="left", padx=(0, Config.PADDING_MEDIUM))

        self.gallery_col_var = ctk.StringVar(
            value=str(self.saved_settings.get('gallery_columns', 3))
        )

        for i in range(2, 7):
            ctk.CTkRadioButton(
                col_frame,
                text=str(i),
                variable=self.gallery_col_var,
                value=str(i),
                command=self.update_gallery_columns
            ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkLabel(
            gallery_frame,
            text="More columns = Smaller cards, Less columns = Larger cards",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_LARGE))

        # ==================== SYSTEM INFO ====================
        sys_frame = ctk.CTkFrame(self, corner_radius=15)
        sys_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            sys_frame,
            text="💻 System Information",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        sys_info = f"""
📦 Model Architecture:
   • GoogleNet + ResNet-18 Hybrid Fusion
   • Embedding Dimension: {Config.EMBEDDING_DIM}D
   • Dropout Rate: {Config.DROPOUT_RATE}

🖥️  Hardware:
   • Device: {Config.DEVICE}
   • Mixed Precision: {Config.USE_MIXED_PRECISION}

🗄️  Database:
   • Path: {Config.DATABASE_PATH}
   • Type: SQLite

👁️  Face Detection:
   • Engine: MTCNN
   • Mode: {Config.FACE_CROP_MODE}
   • Margin: {Config.FACE_CROP_MARGIN * 100}%
   • Min Confidence: {Config.FACE_MIN_CONFIDENCE * 100}%

🎨 Image Processing:
   • Input Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}
   • Normalization: ImageNet Stats
        """

        info_box = ctk.CTkTextbox(sys_frame, height=300, wrap="word")
        info_box.insert("1.0", sys_info.strip())
        info_box.configure(state="disabled")
        info_box.pack(fill="x", padx=Config.PADDING_LARGE,
                     pady=(0, Config.PADDING_LARGE))

        # ==================== ACTIONS ====================
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(pady=Config.PADDING_LARGE)

        ctk.CTkButton(
            action_frame,
            text="💾 Save All Settings",
            command=self.save_all_settings,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkButton(
            action_frame,
            text="🔄 Reset to Defaults",
            command=self.reset_to_defaults,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color=Config.COLOR_WARNING,
            hover_color="darkorange"
        ).pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkButton(
            action_frame,
            text="📤 Export Settings",
            command=self.export_settings,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(side="left", padx=Config.PADDING_SMALL)

    def update_threshold(self, value):
        """Update recognition threshold"""
        threshold = float(value)
        self.app.recognizer.threshold = threshold
        self.threshold_value_label.configure(text=f"{threshold:.2f}")
        self.saved_settings['recognition_threshold'] = threshold

        # Update color based on value
        if threshold < 0.5:
            color = Config.COLOR_WARNING_TEXT
        elif threshold > 0.75:
            color = Config.COLOR_ERROR_TEXT
        else:
            color = Config.COLOR_SUCCESS_TEXT

        self.threshold_value_label.configure(text_color=color)

    def set_threshold_preset(self, value):
        """Set threshold to preset value"""
        self.threshold_slider.set(value)
        self.update_threshold(value)

        self.toast.show_success(
            f"Threshold set to {value:.2f}",
            duration=Config.TOAST_DURATION_SUCCESS
        )

    def update_dup_threshold(self, value):
        """Update duplicate detection threshold"""
        threshold = float(value)
        self.dup_value_label.configure(text=f"Current: {threshold:.2f}")
        self.saved_settings['duplicate_threshold'] = threshold

    def update_topk(self, value):
        """Update top-k default value"""
        k = int(float(value))
        self.topk_value_label.configure(text=f"K = {k}")
        self.saved_settings['top_k_default'] = k

    def update_gallery_columns(self):
        """Update gallery columns setting"""
        cols = int(self.gallery_col_var.get())
        self.saved_settings['gallery_columns'] = cols

    def save_all_settings(self):
        """Save all settings to file"""
        success = self.save_settings()

        if success:
            self.toast.show_success(
                f"All settings saved successfully!",
                duration=Config.TOAST_DURATION_SUCCESS
            )
        else:
            self.toast.show_error(
                "Failed to save settings to file",
                duration=Config.TOAST_DURATION_ERROR
            )

    def reset_to_defaults(self):
        """Reset all settings to defaults with confirmation dialog"""
        # Create custom confirmation dialog
        confirm_dialog = ctk.CTkToplevel(self)
        confirm_dialog.title("⚠️ Reset Settings")
        confirm_dialog.geometry("450x350")
        confirm_dialog.transient(self)
        confirm_dialog.grab_set()

        # Center the dialog
        confirm_dialog.update_idletasks()
        x = (confirm_dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (confirm_dialog.winfo_screenheight() // 2) - (350 // 2)
        confirm_dialog.geometry(f"450x350+{x}+{y}")

        # Warning icon and title
        ctk.CTkLabel(
            confirm_dialog,
            text="⚠️",
            font=ctk.CTkFont(size=60)
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        ctk.CTkLabel(
            confirm_dialog,
            text="RESET SETTINGS",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            text_color=Config.COLOR_WARNING_TEXT
        ).pack(pady=(0, Config.PADDING_MEDIUM))

        # Warning message
        warning_frame = ctk.CTkFrame(confirm_dialog, corner_radius=10)
        warning_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        warning_text = """Are you sure you want to reset all settings?

This will reset:
• Recognition Threshold ➡️ 0.60
• Duplicate Threshold ➡️ 0.85
• Top-K Default ➡️ 3
• Gallery Columns ➡️ 3"""

        ctk.CTkLabel(
            warning_frame,
            text=warning_text,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            justify="center"
        ).pack(padx=Config.PADDING_LARGE, pady=Config.PADDING_LARGE)

        # Button frame
        btn_frame = ctk.CTkFrame(confirm_dialog, fg_color="transparent")
        btn_frame.pack(pady=Config.PADDING_LARGE)

        def on_confirm():
            confirm_dialog.destroy()
            # Reset values
            self.threshold_slider.set(0.6)
            self.update_threshold(0.6)

            self.dup_slider.set(0.85)
            self.update_dup_threshold(0.85)

            self.topk_slider.set(3)
            self.update_topk(3)

            self.gallery_col_var.set("3")
            self.update_gallery_columns()

            self.toast.show_success(
                "All settings reset to defaults",
                duration=Config.TOAST_DURATION_SUCCESS
            )

        def on_cancel():
            confirm_dialog.destroy()
            self.toast.show_info(
                "Reset cancelled",
                duration=Config.TOAST_DURATION_INFO
            )

        ctk.CTkButton(
            btn_frame,
            text="✅ Yes, Reset",
            command=on_confirm,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            fg_color=Config.COLOR_WARNING,
            hover_color="darkorange",
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            btn_frame,
            text="❌ Cancel",
            command=on_cancel,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            fg_color=Config.COLOR_INFO,
            hover_color=Config.COLOR_INFO_HOVER,
            font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=Config.PADDING_TINY)

    def export_settings(self):
        """Export settings to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="face_recognition_settings.json"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    export_data = {
                        'settings': self.saved_settings,
                        'system_info': {
                            'device': str(Config.DEVICE),
                            'embedding_dim': Config.EMBEDDING_DIM,
                            'face_crop_mode': Config.FACE_CROP_MODE
                        }
                    }
                    json.dump(export_data, f, indent=4)

                self.toast.show_success(
                    f"Settings exported successfully!",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            except Exception as e:
                self.toast.show_error(
                    f"Failed to export: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )