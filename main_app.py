"""
Face Recognition System - High-Performance Main GUI Application
BACKWARD COMPATIBLE - All panel imports will work without changes

Performance Features:
- Async initialization with progress feedback
- Lazy panel loading with caching
- Model warm-up for consistent latency
- Efficient memory management
- Fixed layout issues (no screen splitting)
"""

# ==================================================================================
# CRITICAL: Fix for OpenMP library conflict (MUST BE FIRST!)
# ==================================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# ==================================================================================

import sys
from pathlib import Path
from typing import Optional, Dict
import time
import threading
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import customtkinter as ctk
from tkinter import messagebox
import torch
from PIL import Image

# Project imports
from config import Config
from database.face_database import FaceDatabase
from inference.embedding_generator import FaceEmbeddingGenerator
from inference.face_recognizer import FaceRecognizer
from data.face_cropper import FaceCropper
from models.hybrid_encoder import HybridFaceEncoder
from data.transforms import DataTransforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config (for backward compatibility)
CFG = Config


class FaceRecognitionApp:
    """
    High-performance face recognition GUI application.
    BACKWARD COMPATIBLE with all existing panel code.
    """

    def __init__(self):
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("🎯 Face Recognition System - Optimized")
        self.root.geometry("1400x850")
        self.root.minsize(1280, 720)

        # Center window
        self._center_window()

        # BACKWARD COMPATIBILITY: Keep original attribute names
        self.CFG = CFG  # Required by panels

        # Initialize attributes (for backward compatibility)
        self.model = None
        self.face_cropper = None
        self.transform = None
        self.embedding_generator = None
        self.db = None
        self.recognizer = None

        # State management
        self.is_initialized = False
        self.current_panel = None
        self.current_panel_id = None
        self.panel_cache = {}
        self.nav_buttons = {}

        # Performance tracking
        self.init_start_time = time.time()

        # Create UI structure (FIXED layout)
        self._create_layout_fixed()
        self._create_sidebar()

        # Show loading screen
        self._show_loading_screen()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Initialize system asynchronously
        self._initialize_system_async()

    def _center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 1400) // 2
        y = (screen_height - 850) // 2
        self.root.geometry(f"1400x850+{x}+{y}")

    def _create_layout_fixed(self):
        """Create fixed layout that prevents screen splitting"""
        # Main container with pack (more stable than grid for this use case)
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True)

        # Sidebar frame - FIXED width, no propagation
        self.sidebar_frame = ctk.CTkFrame(
            self.main_container,
            width=250,
            corner_radius=0
        )
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)  # CRITICAL: Prevents resizing

        # Content frame - fills remaining space
        self.content_frame = ctk.CTkFrame(
            self.main_container,
            corner_radius=0,
            fg_color="transparent"
        )
        self.content_frame.pack(side="left", fill="both", expand=True,
                               padx=Config.PADDING_SMALL, pady=Config.PADDING_SMALL)

    def _create_sidebar(self):
        """Create navigation sidebar with status indicator"""
        # Header
        header_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        header_frame.pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL),
                         padx=Config.PADDING_LARGE)

        ctk.CTkLabel(
            header_frame,
            text="🎯",
            font=ctk.CTkFont(size=42)
        ).pack()

        ctk.CTkLabel(
            header_frame,
            text="Face Recognition",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_TINY, 2))

        ctk.CTkLabel(
            header_frame,
            text="AI-Powered System",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        ).pack()

        # Status indicator
        self.status_frame = ctk.CTkFrame(
            self.sidebar_frame,
            fg_color=(Config.COLOR_WARNING, "darkorange"),
            corner_radius=10
        )
        self.status_frame.pack(pady=Config.PADDING_SMALL, padx=Config.PADDING_LARGE, fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="⏳ Initializing...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1, weight="bold"),
            text_color="white"
        )
        self.status_label.pack(pady=8)

        # Navigation buttons container - SCROLLABLE for many panels
        nav_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        nav_container.pack(fill="both", expand=True, padx=Config.PADDING_MEDIUM,
                          pady=Config.PADDING_SMALL)

        nav_items = [
            ("🏠 Home", "home"),
            ("➕ Add Person", "add"),
            ("🔍 Recognize", "recognize"),
            ("🖼️ Gallery", "gallery"),
            ("⚖️ Compare", "compare"),
            ("📁 Batch", "batch"),
            ("📜 History", "history"),
            ("📊 Statistics", "stats"),
            ("⚙️ Settings", "settings"),
        ]

        for text, panel_id in nav_items:
            btn = ctk.CTkButton(
                nav_container,
                text=text,
                command=lambda p=panel_id: self.show_panel(p),
                height=42,
                anchor="w",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray75", "gray25"),
                state="disabled"  # Disabled until initialization
            )
            btn.pack(pady=4, fill="x")
            self.nav_buttons[panel_id] = btn

        # Stats footer - FIXED at bottom
        self.stats_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.stats_frame.pack(side="bottom", pady=Config.PADDING_MEDIUM,
                             padx=Config.PADDING_LARGE)

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Loading...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT,
            justify="center"
        )
        self.stats_label.pack()

    def _show_loading_screen(self):
        """Display loading screen during initialization"""
        loading_frame = ctk.CTkFrame(self.content_frame)
        loading_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            loading_frame,
            text="🚀",
            font=ctk.CTkFont(size=80)
        ).pack(pady=(30, Config.PADDING_MEDIUM))

        ctk.CTkLabel(
            loading_frame,
            text="Initializing Face Recognition System",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=Config.PADDING_TINY)

        self.loading_detail = ctk.CTkLabel(
            loading_frame,
            text="Starting up...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.loading_detail.pack(pady=Config.PADDING_SMALL)

        self.loading_progress = ctk.CTkProgressBar(loading_frame, width=400)
        self.loading_progress.pack(pady=Config.PADDING_MEDIUM)
        self.loading_progress.set(0)

        ctk.CTkLabel(
            loading_frame,
            text="⚡ This may take 3-5 seconds on first launch",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(Config.PADDING_TINY, 30))

    def _initialize_system_async(self):
        """Initialize system components asynchronously"""
        def init_worker():
            try:
                steps = [
                    (0.1, "Validating model...", self._validate_model),
                    (0.2, "Loading neural network...", self._load_model),
                    (0.4, "Initializing face detector...", self._init_face_cropper),
                    (0.5, "Setting up transforms...", self._init_transforms),
                    (0.6, "Creating embedding generator...", self._init_embedding_gen),
                    (0.7, "Connecting to database...", self._init_database),
                    (0.85, "Initializing recognizer...", self._init_recognizer),
                    (0.95, "Warming up model...", self._warmup_model),
                ]

                for progress, message, func in steps:
                    self._update_loading(message, progress)
                    func()
                    time.sleep(0.05)

                self.is_initialized = True
                init_time = time.time() - self.init_start_time

                self.root.after(0, lambda: self._on_init_complete(init_time))

            except Exception as e:
                logger.error(f"Initialization failed: {e}", exc_info=True)
                self.root.after(0, lambda: self._on_init_failed(str(e)))

        init_thread = threading.Thread(target=init_worker, daemon=True)
        init_thread.start()

    def _validate_model(self):
        """Validate model file exists"""
        model_path = Path(Config.SAVED_FILES_DIR) / Config.BEST_MODEL_NAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please train the model first!"
            )

        logger.info(f"Model validated: {model_path}")

    def _load_model(self):
        """Load neural network model"""
        model_path = Path(Config.SAVED_FILES_DIR) / Config.BEST_MODEL_NAME

        logger.info(f"Loading model from: {model_path}")

        self.model = HybridFaceEncoder(
            embedding_dim=Config.EMBEDDING_DIM,
            dropout=Config.DROPOUT_RATE
        )

        checkpoint = torch.load(
            model_path,
            map_location=Config.DEVICE,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.DEVICE)
        self.model.eval()

        # Disable gradients for inference
        torch.set_grad_enabled(False)

        # Enable CUDNN benchmark
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        logger.info("Model loaded successfully")

    def _init_face_cropper(self):
        """Initialize MTCNN face detector"""
        logger.info("Initializing MTCNN face detector")
        self.face_cropper = FaceCropper(
            mode=Config.FACE_CROP_MODE,
            margin=Config.FACE_CROP_MARGIN,
            min_confidence=Config.FACE_MIN_CONFIDENCE
        )
        logger.info("Face detector initialized")

    def _init_transforms(self):
        """Initialize image transforms"""
        logger.info("Setting up image transforms")
        self.transform = DataTransforms.get_val_transforms(Config)
        logger.info("Transforms ready")

    def _init_embedding_gen(self):
        """Initialize embedding generator"""
        logger.info("Creating embedding generator")
        self.embedding_generator = FaceEmbeddingGenerator(
            model=self.model,
            transform=self.transform,
            face_cropper=self.face_cropper,
            device=Config.DEVICE
        )
        logger.info("Embedding generator ready")

    def _init_database(self):
        """Initialize database connection"""
        logger.info(f"Connecting to database: {Config.DATABASE_PATH}")
        self.db = FaceDatabase(Config.DATABASE_PATH)
        logger.info("Database connected")

    def _init_recognizer(self):
        """Initialize face recognizer"""
        logger.info("Initializing face recognizer")
        self.recognizer = FaceRecognizer(
            db_path=Config.DATABASE_PATH,
            embedding_generator=self.embedding_generator,
            threshold=Config.RECOGNITION_THRESHOLD
        )
        logger.info("Recognizer ready")

    def _warmup_model(self):
        """Warm up model with dummy inference"""
        logger.info("Warming up model (critical for performance)")

        try:
            dummy_img = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color='gray')
            dummy_path = Path('temp_warmup.jpg')
            dummy_img.save(dummy_path)

            _ = self.embedding_generator.generate_embedding(str(dummy_path))

            dummy_path.unlink(missing_ok=True)

            logger.info("Model warmed up - ready for fast predictions")

        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def _update_loading(self, message: str, progress: float):
        """Update loading screen"""
        def update():
            self.loading_detail.configure(text=message)
            self.loading_progress.set(progress)

        self.root.after(0, update)

    def _on_init_complete(self, init_time: float):
        """Handle successful initialization"""
        logger.info(f"System initialized in {init_time:.2f}s")

        # Update status
        self.status_frame.configure(fg_color=(Config.COLOR_SUCCESS, "darkgreen"))
        self.status_label.configure(text=f"✅ Ready ({init_time:.1f}s)")

        # Enable navigation
        for btn in self.nav_buttons.values():
            btn.configure(state="normal")

        # Update stats
        self._update_stats_display()

        # Show home panel
        self.show_panel("home")

        # Show notification if slow (only if > 5 seconds)
        if init_time > 5:
            self.root.after(500, lambda: messagebox.showinfo(
                "System Ready",
                f"✅ Face Recognition System initialized!\n\n"
                f"Initialization time: {init_time:.1f}s\n"
                f"Model cached for fast predictions."
            ))

    def _on_init_failed(self, error: str):
        """Handle initialization failure"""
        logger.error(f"Initialization failed: {error}")

        self.status_frame.configure(fg_color=(Config.COLOR_ERROR, "darkred"))
        self.status_label.configure(text="❌ Failed")

        messagebox.showerror(
            "Initialization Error",
            f"Failed to initialize system:\n\n{error}\n\n"
            f"Please check:\n"
            f"• Model file exists\n"
            f"• Database is accessible\n"
            f"• All dependencies installed"
        )

        self.root.quit()

    def show_panel(self, panel_id: str):
        """
        Show specified panel with lazy loading.
        BACKWARD COMPATIBLE with all existing panel code.
        """
        if not self.is_initialized and panel_id != "home":
            messagebox.showwarning(
                "Not Ready",
                "System is still initializing. Please wait..."
            )
            return

        # Update navigation buttons
        for btn_id, btn in self.nav_buttons.items():
            if btn_id == panel_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

        # FIXED: Properly destroy current panel before showing new one
        if self.current_panel:
            self.current_panel.pack_forget()  # Hide instead of destroy for cache

        # Check cache
        if panel_id in self.panel_cache:
            logger.debug(f"Loading panel '{panel_id}' from cache")
            self.current_panel = self.panel_cache[panel_id]
        else:
            # Create new panel
            logger.debug(f"Creating panel '{panel_id}'")
            self.current_panel = self._create_panel(panel_id)

            # Cache panel (max 5)
            if len(self.panel_cache) >= 5:
                oldest = next(iter(self.panel_cache))
                logger.debug(f"Removing panel '{oldest}' from cache")
                self.panel_cache[oldest].destroy()
                del self.panel_cache[oldest]

            self.panel_cache[panel_id] = self.current_panel

        # FIXED: Show panel with proper packing
        self.current_panel.pack(fill="both", expand=True)
        self.current_panel_id = panel_id

        logger.info(f"Showing panel: {panel_id}")

    def _create_panel(self, panel_id: str):
        """Create panel instance (lazy loading)"""
        from gui.panels import (
            home_panel, add_person_panel, recognition_panel,
            compare_panel, batch_panel, gallery_panel,
            stats_panel, history_panel, settings_panel
        )

        panel_classes = {
            "home": home_panel.HomePanel,
            "add": add_person_panel.AddPersonPanel,
            "recognize": recognition_panel.RecognitionPanel,
            "gallery": gallery_panel.GalleryPanel,
            "compare": compare_panel.ComparePanel,
            "batch": batch_panel.BatchPanel,
            "history": history_panel.HistoryPanel,
            "stats": stats_panel.StatsPanel,
            "settings": settings_panel.SettingsPanel,
        }

        panel_class = panel_classes.get(panel_id)
        if not panel_class:
            raise ValueError(f"Unknown panel: {panel_id}")

        return panel_class(self.content_frame, self)

    def refresh_stats(self):
        """
        Refresh database statistics.
        BACKWARD COMPATIBLE - called by panels after add/delete.
        """
        logger.info("Refreshing statistics")

        # Update stats display
        self._update_stats_display()

        # Reload recognizer
        if self.recognizer:
            self.recognizer.reload_database()

        # Clear stat-dependent cached panels
        panels_to_refresh = ["home", "stats", "gallery"]
        for panel_id in panels_to_refresh:
            if panel_id in self.panel_cache:
                logger.debug(f"Clearing cached panel: {panel_id}")
                self.panel_cache[panel_id].destroy()
                del self.panel_cache[panel_id]

        # Reload if current panel was cleared
        if self.current_panel_id in panels_to_refresh:
            self.show_panel(self.current_panel_id)

    def _update_stats_display(self):
        """Update statistics in sidebar"""
        if self.db:
            stats = self.db.get_stats()
            stats_text = (
                f"👥 {stats['total_people']} People\n"
                f"📸 {stats['total_embeddings']} Embeddings\n"
                f"📊 Avg: {stats['avg_embeddings_per_person']:.1f}"
            )
            self.stats_label.configure(text=stats_text)

    def _on_closing(self):
        """Handle application close"""
        logger.info("Shutting down application")

        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            # Cleanup
            if self.db:
                self.db.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.root.destroy()
            logger.info("Application closed")

    def run(self):
        """Start the application"""
        logger.info("Starting application")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted")
            self._on_closing()
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Unexpected error:\n\n{e}")
        finally:
            if self.db:
                self.db.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    """Main entry point"""
    try:
        app = FaceRecognitionApp()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        messagebox.showerror(
            "Fatal Error",
            f"Application failed to start:\n\n{e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()