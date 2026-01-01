"""
Optimized Home Panel - Unicode-Safe Version
Fixed for Windows locale encoding issues
"""

import customtkinter as ctk
from datetime import datetime
import threading
import logging

from config import Config
from utils.toast import ToastManager

logger = logging.getLogger(__name__)


class HomePanel(ctk.CTkScrollableFrame):
    """
    Optimized home panel with animated stats.
    Unicode-safe for Windows systems.
    """

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        # Create widgets
        self.create_widgets()

        # Load stats asynchronously
        self.load_stats_async()

    def create_widgets(self):
        """Create all widgets with fixed layout"""
        # Hero section (using ASCII-safe icons)
        hero = ctk.CTkFrame(self, fg_color="transparent")
        hero.pack(fill="x", pady=(Config.PADDING_LARGE, 30), padx=Config.PADDING_LARGE)

        # Use simple text instead of emoji for encoding safety
        ctk.CTkLabel(
            hero,
            text="●",  # Simple bullet point
            font=ctk.CTkFont(size=64, weight="bold"),
            text_color="#3b82f6"  # Blue color
        ).pack()

        ctk.CTkLabel(
            hero,
            text="Face Recognition System",
            font=ctk.CTkFont(size=36, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        ctk.CTkLabel(
            hero,
            text="Powered by GoogleNet + ResNet-18 Hybrid Architecture",
            font=ctk.CTkFont(size=Config.PADDING_MEDIUM),
            text_color=Config.COLOR_INFO_TEXT
        ).pack()

        # Stats cards container - FIXED grid
        stats_container = ctk.CTkFrame(self, fg_color="transparent")
        stats_container.pack(fill="x", pady=Config.PADDING_LARGE,
                           padx=Config.PADDING_LARGE)

        # Configure grid for 3 columns
        for i in range(3):
            stats_container.grid_columnconfigure(i, weight=1, uniform="stats")

        # Quick stats cards (using simple icons)
        self.people_card = self.create_stat_card(
            stats_container, 0, "People", "...", "Registered People", "blue"
        )

        self.emb_card = self.create_stat_card(
            stats_container, 1, "Images", "...", "Face Embeddings", "green"
        )

        self.avg_card = self.create_stat_card(
            stats_container, 2, "Average", "...", "Avg per Person", "purple"
        )

        # Quick actions
        actions = ctk.CTkFrame(self, corner_radius=15)
        actions.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_LARGE)

        ctk.CTkLabel(
            actions,
            text="Quick Actions",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_MEDIUM))

        # Button grid - FIXED 2x2 layout
        btn_container = ctk.CTkFrame(actions, fg_color="transparent")
        btn_container.pack(pady=(0, Config.PADDING_LARGE), padx=Config.PADDING_LARGE)

        # Configure grid
        btn_container.grid_columnconfigure(0, weight=1)
        btn_container.grid_columnconfigure(1, weight=1)

        # Buttons with simple text (encoding-safe)
        buttons = [
            ("+ Add Person", "add", Config.COLOR_SUCCESS, "Register new person"),
            (">> Recognize", "recognize", "blue", "Identify face from image"),
            ("[] Batch Process", "batch", Config.COLOR_WARNING, "Process multiple images"),
            ("## Gallery", "gallery", "violet", "View all registered people"),
        ]

        for idx, (text, panel, color, desc) in enumerate(buttons):
            row = idx // 2
            col = idx % 2

            btn_frame = ctk.CTkFrame(btn_container, fg_color="transparent")
            btn_frame.grid(row=row, column=col, padx=Config.PADDING_MEDIUM,
                          pady=Config.PADDING_SMALL, sticky="nsew")

            ctk.CTkButton(
                btn_frame,
                text=text,
                command=lambda p=panel: self.app.show_panel(p),
                width=250,
                height=55,
                font=ctk.CTkFont(size=Config.PADDING_MEDIUM, weight="bold"),
                fg_color=color,
                hover_color=f"dark{color}"
            ).pack()

            ctk.CTkLabel(
                btn_frame,
                text=desc,
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
                text_color=Config.COLOR_INFO_TEXT
            ).pack(pady=(3, 0))

        # System info
        info = ctk.CTkFrame(self, corner_radius=15)
        info.pack(fill="x", padx=Config.PADDING_LARGE, pady=(0, Config.PADDING_LARGE))

        ctk.CTkLabel(
            info,
            text="System Status",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Info grid - FIXED 3x2 layout
        info_grid = ctk.CTkFrame(info, fg_color="transparent")
        info_grid.pack(fill="x", padx=Config.PADDING_LARGE,
                      pady=(0, Config.PADDING_MEDIUM))

        # Configure grid
        for i in range(3):
            info_grid.grid_columnconfigure(i, weight=1, uniform="info")

        # Get config values safely (encoding-safe labels)
        try:
            info_items = [
                ("Model", "Hybrid GoogleNet-ResNet18"),
                ("Embedding", f"{Config.EMBEDDING_DIM}D"),
                ("Device", str(Config.DEVICE)),
                ("Threshold", f"{self.app.recognizer.threshold:.2f}" if self.app.recognizer else f"{Config.RECOGNITION_THRESHOLD:.2f}"),
                ("Database", "SQLite"),
                ("Detector", "MTCNN"),
            ]
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            info_items = [
                ("Model", "Hybrid GoogleNet-ResNet18"),
                ("Embedding", "512D"),
                ("Device", "CPU/GPU"),
                ("Threshold", "0.60"),
                ("Database", "SQLite"),
                ("Detector", "MTCNN"),
            ]

        for idx, (label, value) in enumerate(info_items):
            row = idx // 3
            col = idx % 3

            item = ctk.CTkFrame(info_grid, fg_color=("gray90", "gray20"), corner_radius=8)
            item.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

            ctk.CTkLabel(
                item,
                text=label,
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1, weight="bold")
            ).pack(pady=(8, 2))

            ctk.CTkLabel(
                item,
                text=value,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1),
                text_color=Config.COLOR_INFO_TEXT
            ).pack(pady=(0, 8))

        # Time display (encoding-safe)
        self.time_label = ctk.CTkLabel(
            info,
            text=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.time_label.pack(pady=(Config.PADDING_TINY, Config.PADDING_MEDIUM))

        # Start time updates
        self.update_time()

    def create_stat_card(self, parent, column, icon_text, value, label, color):
        """Create stat card in grid layout with simple text icons"""
        card = ctk.CTkFrame(parent, corner_radius=15, fg_color=color, height=180)
        card.grid(row=0, column=column, padx=Config.PADDING_SMALL,
                 pady=Config.PADDING_SMALL, sticky="nsew")
        card.grid_propagate(False)  # Prevent resizing

        # Use simple text instead of emoji
        ctk.CTkLabel(
            card,
            text=icon_text,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold"),
            text_color="white"
        ).pack(pady=(25, Config.PADDING_SMALL))

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=40, weight="bold"),
            text_color="white"
        )
        value_label.pack()

        ctk.CTkLabel(
            card,
            text=label,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1),
            text_color="white"
        ).pack(pady=(Config.PADDING_TINY, 25))

        # Store reference for updating
        card.value_label = value_label
        return card

    def load_stats_async(self):
        """Load stats in background thread"""
        def load():
            try:
                # Get stats from database
                stats = self.app.db.get_stats()

                # Animate values on main thread
                self.after(0, lambda: self.animate_value(
                    self.people_card.value_label,
                    stats['total_people'],
                    is_float=False
                ))

                self.after(0, lambda: self.animate_value(
                    self.emb_card.value_label,
                    stats['total_embeddings'],
                    is_float=False
                ))

                self.after(0, lambda: self.animate_value(
                    self.avg_card.value_label,
                    stats['avg_embeddings_per_person'],
                    is_float=True
                ))

                # Show success toast
                self.after(100, lambda: self.toast.show_success(
                    "Dashboard loaded successfully",
                    duration=Config.TOAST_DURATION_SUCCESS
                ))

            except Exception as e:
                logger.error(f"Stats load error: {e}")
                # Set default values on error
                self.after(0, lambda: self.people_card.value_label.configure(text="0"))
                self.after(0, lambda: self.emb_card.value_label.configure(text="0"))
                self.after(0, lambda: self.avg_card.value_label.configure(text="0.0"))

                # Show error toast
                self.after(100, lambda: self.toast.show_error(
                    "Failed to load dashboard statistics",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def animate_value(self, label, target, is_float=False):
        """Animate counter from 0 to target value"""
        steps = 20
        delay = 20  # ms

        def update(current_step):
            try:
                if current_step <= steps:
                    if is_float:
                        val = (target / steps) * current_step
                        label.configure(text=f"{val:.1f}")
                    else:
                        val = int((target / steps) * current_step)
                        label.configure(text=str(val))

                    self.after(delay, lambda: update(current_step + 1))
            except Exception as e:
                logger.error(f"Animation error: {e}")

        update(0)

    def update_time(self):
        """Update time display every second (encoding-safe)"""
        try:
            # Use simple format without special characters
            self.time_label.configure(
                text=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.after(1000, self.update_time)
        except Exception as e:
            logger.error(f"Time update error: {e}")
            # Try again in 5 seconds
            self.after(5000, self.update_time)

    def refresh(self):
        """Refresh the dashboard stats (can be called externally)"""
        self.load_stats_async()


# Export for import
__all__ = ['HomePanel']