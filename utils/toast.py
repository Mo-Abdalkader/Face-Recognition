"""
Reusable Toast Notification System
Shared across the entire application
"""

import customtkinter as ctk
from config import Config


class ToastManager:
    """Manager for showing temporary toast notifications"""

    def __init__(self, parent):
        """
        Initialize ToastManager

        Args:
            parent: The parent widget where toasts will be displayed
        """
        self.parent = parent
        self.toast = None
        self.fade_job = None

    def show(self, message, level="info", duration=None):
        """
        Show a toast notification

        Args:
            message: The message to display
            level: Type of notification ("info", "success", "warning", "error")
            duration: How long to show the toast in milliseconds (uses config defaults if None)
        """
        # Clear any existing toast
        if self.toast:
            self.toast.destroy()
            self.toast = None

        # Cancel any pending fade job
        if self.fade_job:
            self.parent.after_cancel(self.fade_job)
            self.fade_job = None

        # Color mapping with config values
        colors = {
            "info": "#2a7fff",
            "success": Config.COLOR_SUCCESS,
            "warning": Config.COLOR_WARNING,
            "error": Config.COLOR_ERROR
        }

        # Icon mapping
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }

        # Get duration from config if not specified
        if duration is None:
            duration_map = {
                "info": Config.TOAST_DURATION_INFO,
                "success": Config.TOAST_DURATION_SUCCESS,
                "warning": Config.TOAST_DURATION_WARNING,
                "error": Config.TOAST_DURATION_ERROR
            }
            duration = duration_map.get(level, Config.TOAST_DURATION_INFO)

        # Create toast frame
        self.toast = ctk.CTkFrame(
            self.parent,
            fg_color=colors.get(level, "#333"),
            corner_radius=10,
            border_width=2,
            border_color="white"
        )
        self.toast.place(relx=0.5, rely=0.97, anchor="s")

        # Create message with icon
        icon = icons.get(level, "ℹ️")
        full_message = f"{icon} {message}"

        label = ctk.CTkLabel(
            self.toast,
            text=full_message,
            text_color="white",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1, weight="bold"),
            wraplength=600  # Wrap long messages
        )
        label.pack(padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        # Schedule toast removal
        self.fade_job = self.parent.after(duration, self._destroy_toast)

    def show_info(self, message, duration=None):
        """Show an info toast"""
        self.show(message, level="info", duration=duration)

    def show_success(self, message, duration=None):
        """Show a success toast"""
        self.show(message, level="success", duration=duration)

    def show_warning(self, message, duration=None):
        """Show a warning toast"""
        self.show(message, level="warning", duration=duration)

    def show_error(self, message, duration=None):
        """Show an error toast"""
        self.show(message, level="error", duration=duration)

    def _destroy_toast(self):
        """Internal method to safely destroy the toast"""
        if self.toast:
            try:
                self.toast.destroy()
            except:
                pass
            finally:
                self.toast = None

        if self.fade_job:
            self.fade_job = None

    def hide(self):
        """Manually hide the current toast"""
        if self.fade_job:
            self.parent.after_cancel(self.fade_job)
            self.fade_job = None
        self._destroy_toast()