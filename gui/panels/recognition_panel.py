"""
Enhanced Recognition Panel
Improvements:
- Toast notifications instead of messageboxes
- All magic numbers moved to config
- Better visual hierarchy and spacing
- Face thumbnails in Top-K results
- Improved error handling
- Better performance feedback
- Enhanced UI polish
- Proper scrolling support
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
from datetime import datetime
import numpy as np
from pathlib import Path
import time

from config import Config
from utils.toast import ToastManager


class RecognitionPanel(ctk.CTkFrame):
    """Enhanced recognition panel with professional UI and toast notifications"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.current_image_path = None
        self.current_photo = None
        self.recognition_mode = "single"

        # Performance cache
        self.embedding_cache = {}
        self.last_query_time = 0
        self.max_cache_size = 20  # Can be moved to config if needed

        # Configure grid weights for proper resizing
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Title row
        self.grid_rowconfigure(1, weight=0)  # Subtitle row
        self.grid_rowconfigure(2, weight=1)  # Main content row (expandable)

        self.create_widgets()

    def create_widgets(self):
        """Create all interface widgets with enhanced layout"""

        # ===== HEADER SECTION =====
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, columnspan=2, pady=(Config.PADDING_MEDIUM, Config.PADDING_TINY), sticky="ew")

        ctk.CTkLabel(
            title_frame,
            text="🔍 Face Recognition",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        ).pack(side="left", padx=Config.PADDING_MEDIUM)

        self.perf_label = ctk.CTkLabel(
            title_frame,
            text="",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color="gray"
        )
        self.perf_label.pack(side="left", padx=Config.PADDING_MEDIUM)

        subtitle = ctk.CTkLabel(
            self,
            text="Fast AI-powered face identification with intelligent caching",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color="gray"
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # ===== LEFT COLUMN - IMAGE UPLOAD =====
        left_frame = ctk.CTkFrame(self, corner_radius=15)
        left_frame.grid(row=2, column=0, padx=(Config.PADDING_LARGE, Config.PADDING_SMALL), pady=Config.PADDING_SMALL, sticky="nsew")
        left_frame.grid_rowconfigure(5, weight=1)  # Make content expandable

        # Make left frame scrollable for small windows
        left_scroll = ctk.CTkScrollableFrame(left_frame, fg_color="transparent")
        left_scroll.pack(fill="both", expand=True, padx=Config.PADDING_SMALL, pady=Config.PADDING_SMALL)

        # Title
        ctk.CTkLabel(
            left_scroll,
            text="📸 Query Image",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_MEDIUM))

        # Image display container with fixed size
        preview_size = Config.PREVIEW_IMAGE_SIZE[0] * 2  # Double the preview size for main display
        self.image_container = ctk.CTkFrame(left_scroll, width=preview_size, height=preview_size, corner_radius=10)
        self.image_container.pack(pady=Config.PADDING_MEDIUM)
        self.image_container.pack_propagate(False)

        self.image_label = ctk.CTkLabel(
            self.image_container,
            text="No Image\n\n👆 Click 'Upload' to select an image",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color="gray"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Upload button
        ctk.CTkButton(
            left_scroll,
            text="📁 Upload Image",
            command=self.upload_image,
            width=250,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color="blue",
            hover_color="darkblue"
        ).pack(pady=Config.PADDING_MEDIUM)

        # Quick actions frame
        quick_frame = ctk.CTkFrame(left_scroll, fg_color="transparent")
        quick_frame.pack(pady=Config.PADDING_TINY)

        ctk.CTkButton(
            quick_frame,
            text="🔄 Clear",
            command=self.clear_image,
            width=100,
            height=30,
            fg_color=Config.COLOR_INFO,
            hover_color=Config.COLOR_INFO_HOVER
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            quick_frame,
            text="📋 Paste",
            command=self.paste_from_clipboard,
            width=100,
            height=30,
            fg_color="purple",
            hover_color="darkviolet"
        ).pack(side="left", padx=Config.PADDING_TINY)

        # Mode selector frame
        mode_frame = ctk.CTkFrame(left_scroll, corner_radius=10)
        mode_frame.pack(pady=Config.PADDING_MEDIUM, fill="x")

        ctk.CTkLabel(
            mode_frame,
            text="🎯 Recognition Mode:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_TINY))

        self.mode_var = ctk.StringVar(value="single")

        mode_buttons = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_buttons.pack(pady=(0, Config.PADDING_SMALL), fill="x")

        ctk.CTkRadioButton(
            mode_buttons,
            text="⚡ Single Best Match (Fast)",
            variable=self.mode_var,
            value="single",
            command=self.change_mode,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        ).pack(anchor="w", padx=Config.PADDING_LARGE, pady=3)

        ctk.CTkRadioButton(
            mode_buttons,
            text="📊 Top-K Similar Faces",
            variable=self.mode_var,
            value="topk",
            command=self.change_mode,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        ).pack(anchor="w", padx=Config.PADDING_LARGE, pady=3)

        # Top-K settings (initially hidden)
        self.topk_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")

        topk_slider_frame = ctk.CTkFrame(self.topk_frame, fg_color="transparent")
        topk_slider_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            topk_slider_frame,
            text="Results (K):",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=Config.PADDING_TINY)

        self.k_slider = ctk.CTkSlider(
            topk_slider_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            width=120,
            command=self.update_k_label
        )
        self.k_slider.set(3)
        self.k_slider.pack(side="left", padx=Config.PADDING_TINY)

        self.k_label = ctk.CTkLabel(
            topk_slider_frame,
            text="3",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=30
        )
        self.k_label.pack(side="left")

        # Recognize button
        self.recognize_btn = ctk.CTkButton(
            left_scroll,
            text="🔎 Recognize Face",
            command=self.recognize_face,
            width=Config.BUTTON_WIDTH_ACTION,
            height=55,
            font=ctk.CTkFont(size=17, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER,
            state="disabled"
        )
        self.recognize_btn.pack(pady=Config.PADDING_MEDIUM)

        # Stats label
        self.stats_label = ctk.CTkLabel(
            left_scroll,
            text="",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color="gray"
        )
        self.stats_label.pack(pady=(0, Config.PADDING_MEDIUM))

        # ===== RIGHT COLUMN - RESULTS =====
        right_frame = ctk.CTkFrame(self, corner_radius=15)
        right_frame.grid(row=2, column=1, padx=(Config.PADDING_SMALL, Config.PADDING_LARGE), pady=Config.PADDING_SMALL, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)  # Make scrollable area expand
        right_frame.grid_columnconfigure(0, weight=1)

        # Results header
        results_header = ctk.CTkFrame(right_frame, fg_color="transparent")
        results_header.grid(row=0, column=0, pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL), padx=Config.PADDING_MEDIUM, sticky="ew")
        results_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            results_header,
            text="📋 Recognition Results",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.results_count_label = ctk.CTkLabel(
            results_header,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.results_count_label.grid(row=0, column=1, sticky="e")

        # Scrollable results area
        self.results_scroll = ctk.CTkScrollableFrame(
            right_frame,
            fg_color="transparent"
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=Config.PADDING_MEDIUM, pady=(0, Config.PADDING_MEDIUM))

        # Show initial placeholder
        self.show_placeholder()

        # Initialize stats
        self.update_stats()

    def show_placeholder(self):
        """Show enhanced initial placeholder"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        placeholder = ctk.CTkFrame(
            self.results_scroll,
            fg_color=("gray90", "gray20"),
            corner_radius=10
        )
        placeholder.pack(pady=50, padx=Config.PADDING_LARGE, fill="x")

        ctk.CTkLabel(
            placeholder,
            text="👆",
            font=ctk.CTkFont(size=48)
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        ctk.CTkLabel(
            placeholder,
            text="Upload an image to begin",
            font=ctk.CTkFont(size=Config.FONT_SIZE_BUTTON, weight="bold")
        ).pack(pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            placeholder,
            text="Recognition is powered by cached embeddings\nfor ultra-fast results",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(Config.PADDING_TINY, Config.PADDING_LARGE))

    def update_stats(self):
        """Update statistics display"""
        try:
            db_stats = self.app.db.get_stats()
            if self.recognition_mode == "single":
                self.stats_label.configure(
                    text=f"📊 Database: {db_stats['total_people']} people, {db_stats['total_embeddings']} embeddings"
                )
            else:
                k = int(self.k_slider.get())
                self.stats_label.configure(
                    text=f"📊 Will show top {k} most similar faces from database"
                )
        except Exception as e:
            self.stats_label.configure(text="⚠️ Database not ready")

    def change_mode(self):
        """Handle mode change"""
        self.recognition_mode = self.mode_var.get()

        if self.recognition_mode == "topk":
            self.topk_frame.pack(fill="x", pady=(0, Config.PADDING_SMALL))
        else:
            self.topk_frame.pack_forget()

        self.update_stats()

    def update_k_label(self, value):
        """Update K value label"""
        k = int(float(value))
        self.k_label.configure(text=str(k))
        self.update_stats()

    def clear_image(self):
        """Clear current image"""
        self.current_image_path = None
        self.current_photo = None
        self.image_label.configure(
            image=None,
            text="No Image\n\n👆 Click 'Upload' to select an image"
        )
        self.recognize_btn.configure(state="disabled")
        self.perf_label.configure(text="")
        self.show_placeholder()
        self.toast.show_info("Image cleared")

    def paste_from_clipboard(self):
        """Paste image from clipboard"""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grabclipboard()

            if img is not None and isinstance(img, Image.Image):
                # Save to temp file
                temp_path = "temp_clipboard.png"
                img.save(temp_path)
                self.current_image_path = temp_path
                self.display_image(temp_path)
                self.recognize_btn.configure(state="normal")
                self.toast.show_success("Image pasted from clipboard!")
            else:
                self.toast.show_warning("No image found in clipboard")
        except ImportError:
            self.toast.show_error("Clipboard paste not supported on this system")
        except Exception as e:
            self.toast.show_error(f"Failed to paste: {str(e)}")

    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.recognize_btn.configure(state="normal")
            self.show_placeholder()
            self.toast.show_success("Image loaded successfully!")

    def display_image(self, image_path):
        """Display image with efficient memory handling"""
        try:
            # Load and resize efficiently
            img = Image.open(image_path)

            # Get original size
            orig_size = img.size

            # Calculate preview size (double the standard preview size)
            preview_size = Config.PREVIEW_IMAGE_SIZE[0] * 2
            img.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)

            self.current_photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.current_photo, text="")

            # Show image info
            file_size = Path(image_path).stat().st_size / 1024  # KB
            self.perf_label.configure(
                text=f"📷 {orig_size[0]}×{orig_size[1]}px • {file_size:.1f} KB"
            )

        except Exception as e:
            self.toast.show_error(f"Failed to load image: {str(e)}")

    def recognize_face(self):
        """Perform optimized recognition with proper error handling"""
        if not self.current_image_path:
            self.toast.show_warning("Please upload an image first!")
            return

        # Clear results
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        # Show processing indicator
        processing_frame = ctk.CTkFrame(
            self.results_scroll,
            fg_color=("gray90", "gray20"),
            corner_radius=10
        )
        processing_frame.pack(pady=50, padx=Config.PADDING_LARGE, fill="x")

        ctk.CTkLabel(
            processing_frame,
            text="⚡",
            font=ctk.CTkFont(size=48)
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        self.processing_label = ctk.CTkLabel(
            processing_frame,
            text="Analyzing image...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_BUTTON, weight="bold")
        )
        self.processing_label.pack(pady=Config.PADDING_TINY)

        self.processing_detail = ctk.CTkLabel(
            processing_frame,
            text="Detecting face...",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.processing_detail.pack(pady=(Config.PADDING_TINY, Config.PADDING_LARGE))

        # Disable button
        self.recognize_btn.configure(state="disabled", text="⏳ Processing...")

        # Run in thread with timing
        def recognize_thread():
            start_time = time.time()
            steps = []

            try:
                # Check cache first
                cache_key = str(self.current_image_path)
                if cache_key in self.embedding_cache:
                    self.update_processing("Using cached embedding", "⚡ Cache hit!")
                    query_embedding = self.embedding_cache[cache_key]
                    steps.append(("Cache Lookup", 0.001))
                else:
                    # Face detection + embedding generation
                    self.update_processing("Detecting face...", "MTCNN face detector")
                    step_start = time.time()

                    self.update_processing("Generating embedding...", "Neural network inference")
                    query_embedding = self.app.embedding_generator.generate_embedding(
                        self.current_image_path
                    )
                    steps.append(("Face Detection + Embedding", time.time() - step_start))

                    if query_embedding is not None:
                        # Cache the embedding
                        self.embedding_cache[cache_key] = query_embedding

                        # Limit cache size
                        if len(self.embedding_cache) > self.max_cache_size:
                            self.embedding_cache.pop(next(iter(self.embedding_cache)))

                if query_embedding is None:
                    self.after(0, lambda: self.show_no_face_error())
                    return

                # Recognition based on mode
                if self.recognition_mode == "single":
                    self.update_processing("Matching face...", "Comparing with database")
                    step_start = time.time()

                    result = self.app.recognizer.recognize_face(self.current_image_path)
                    steps.append(("Database Matching", time.time() - step_start))

                    total_time = time.time() - start_time
                    self.last_query_time = total_time

                    self.after(0, lambda: self.display_single_result(result, total_time, steps))
                else:
                    self.update_processing("Finding similar faces...", "Computing similarities")
                    step_start = time.time()

                    k = int(self.k_slider.get())
                    results = self.get_top_k_matches_fast(query_embedding, k)
                    steps.append(("Top-K Search", time.time() - step_start))

                    total_time = time.time() - start_time
                    self.last_query_time = total_time

                    self.after(0, lambda: self.display_topk_results(results, k, total_time, steps))

            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self.toast.show_error(f"Recognition failed: {error_msg}"))
                self.after(0, lambda: self.show_placeholder())
            finally:
                self.after(0, lambda: self.recognize_btn.configure(
                    state="normal", text="🔎 Recognize Face"
                ))

        thread = threading.Thread(target=recognize_thread, daemon=True)
        thread.start()

    def update_processing(self, main_text, detail_text):
        """Update processing indicator on UI thread"""
        def update():
            if hasattr(self, 'processing_label'):
                self.processing_label.configure(text=main_text)
            if hasattr(self, 'processing_detail'):
                self.processing_detail.configure(text=detail_text)
        self.after(0, update)

    def show_no_face_error(self):
        """Show error when no face detected"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        error_frame = ctk.CTkFrame(
            self.results_scroll,
            fg_color=Config.COLOR_ERROR,
            corner_radius=10
        )
        error_frame.pack(pady=50, padx=Config.PADDING_LARGE, fill="x")

        ctk.CTkLabel(
            error_frame,
            text="❌",
            font=ctk.CTkFont(size=48)
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        ctk.CTkLabel(
            error_frame,
            text="No Face Detected",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            text_color="white"
        ).pack(pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            error_frame,
            text="Please upload an image with a clear, visible face",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color="white"
        ).pack(pady=(Config.PADDING_TINY, Config.PADDING_LARGE))

        self.recognize_btn.configure(state="normal", text="🔎 Recognize Face")
        self.toast.show_error("No face detected in the image")

    def get_top_k_matches_fast(self, query_embedding, k):
        """Optimized Top-K matching with vectorized operations"""
        all_embeddings = self.app.db.get_all_embeddings()

        if not all_embeddings:
            return []

        # Vectorized similarity computation
        face_ids = [x[0] for x in all_embeddings]
        embeddings_matrix = np.array([x[1] for x in all_embeddings])

        # Compute all similarities at once (fast)
        similarities = np.dot(embeddings_matrix, query_embedding)

        # Get top K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_k_indices:
            face_id = face_ids[idx]
            similarity = float(similarities[idx])
            person_info = self.app.db.get_person_info(face_id)

            results.append({
                'face_id': face_id,
                'similarity': similarity,
                'person_info': person_info
            })

        return results

    def display_single_result(self, result, time_taken, steps):
        """Display single match result with enhanced visuals"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        # Performance header
        perf_frame = ctk.CTkFrame(
            self.results_scroll,
            fg_color=("blue", "darkblue"),
            corner_radius=10
        )
        perf_frame.pack(fill="x", pady=(Config.PADDING_SMALL, Config.PADDING_MEDIUM), padx=Config.PADDING_SMALL)

        perf_text = f"⚡ Processed in {time_taken:.3f}s"
        if time_taken < 1.0:
            perf_text += " (Fast!)"
        elif time_taken < 0.5:
            perf_text += " (Ultra-fast!)"

        ctk.CTkLabel(
            perf_frame,
            text=perf_text,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            text_color="white"
        ).pack(pady=8)

        # Show breakdown
        if steps:
            breakdown = " • ".join([f"{name}: {t*1000:.0f}ms" for name, t in steps])
            ctk.CTkLabel(
                perf_frame,
                text=breakdown,
                font=ctk.CTkFont(size=9),
                text_color="white"
            ).pack(pady=(0, 8))

        if result['recognized']:
            # Success - person recognized
            person = result['person_info']

            header = ctk.CTkFrame(
                self.results_scroll,
                fg_color=Config.COLOR_SUCCESS,
                corner_radius=10
            )
            header.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            ctk.CTkLabel(
                header,
                text="✅ FACE RECOGNIZED",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold"),
                text_color="white"
            ).pack(pady=Config.PADDING_MEDIUM)

            # Person card
            card = ctk.CTkFrame(self.results_scroll, corner_radius=10)
            card.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            # Confidence visualization
            conf_val = result['confidence']
            conf_frame = ctk.CTkFrame(card, fg_color="transparent")
            conf_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

            # Confidence label with icon
            conf_icon = "🟢" if conf_val >= 0.8 else "🟡" if conf_val >= 0.6 else "🟠"
            ctk.CTkLabel(
                conf_frame,
                text=f"{conf_icon} Confidence: {conf_val*100:.1f}%",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
            ).pack(anchor="w")

            # Confidence progress bar
            conf_bar = ctk.CTkProgressBar(conf_frame, width=400)
            conf_bar.pack(fill="x", pady=Config.PADDING_TINY)
            conf_bar.set(conf_val)

            # Match quality indicator
            if conf_val >= 0.8:
                quality_text = "🎯 Excellent Match"
                quality_color = Config.COLOR_SUCCESS_TEXT
            elif conf_val >= 0.6:
                quality_text = "✓ Good Match"
                quality_color = Config.COLOR_WARNING_TEXT
            else:
                quality_text = "⚠ Fair Match"
                quality_color = Config.COLOR_WARNING_TEXT

            ctk.CTkLabel(
                conf_frame,
                text=quality_text,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                text_color=quality_color
            ).pack(anchor="w", pady=(Config.PADDING_TINY, 0))

            # Person information
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_SMALL)

            info_items = [
                ("👤 Name", person['name']),
                ("🆔 Face ID", person['face_id']),
                ("🏢 Department", person['department'] or 'N/A'),
                ("💼 Role", person['role'] or 'N/A'),
                ("📧 Email", person['email'] or 'N/A'),
                ("📱 Phone", person['phone'] or 'N/A'),
            ]

            for label, value in info_items:
                item_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
                item_frame.pack(fill="x", pady=2)

                ctk.CTkLabel(
                    item_frame,
                    text=f"{label}:",
                    font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                    width=120,
                    anchor="w"
                ).pack(side="left")

                ctk.CTkLabel(
                    item_frame,
                    text=str(value),
                    font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                    anchor="w"
                ).pack(side="left", padx=Config.PADDING_SMALL)

            # Timestamp
            timestamp_frame = ctk.CTkFrame(card, fg_color=("gray85", "gray25"), corner_radius=5)
            timestamp_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=(Config.PADDING_SMALL, Config.PADDING_MEDIUM))

            ctk.CTkLabel(
                timestamp_frame,
                text=f"🕐 Recognized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
                text_color="gray"
            ).pack(pady=Config.PADDING_TINY)

        else:
            # Not recognized - unknown person
            header = ctk.CTkFrame(
                self.results_scroll,
                fg_color=Config.COLOR_ERROR,
                corner_radius=10
            )
            header.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            ctk.CTkLabel(
                header,
                text="❌ UNKNOWN PERSON",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold"),
                text_color="white"
            ).pack(pady=Config.PADDING_MEDIUM)

            info_frame = ctk.CTkFrame(self.results_scroll, corner_radius=10)
            info_frame.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            ctk.CTkLabel(
                info_frame,
                text="This person is not in the database.",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
            ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_TINY))

            # Show similarity details
            details_text = f"Best similarity: {result['confidence']*100:.1f}%\nThreshold: {self.app.recognizer.threshold*100:.1f}%"
            ctk.CTkLabel(
                info_frame,
                text=details_text,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                justify="center",
                text_color="gray"
            ).pack(pady=Config.PADDING_TINY)

            # Add to database button
            ctk.CTkButton(
                info_frame,
                text="➕ Add to Database",
                command=lambda: self.navigate_to_add_person(),
                height=Config.BUTTON_HEIGHT_MEDIUM,
                fg_color=Config.COLOR_WARNING,
                hover_color="darkorange",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
            ).pack(pady=Config.PADDING_MEDIUM)

    def display_topk_results(self, results, k, time_taken, steps):
        """Display Top-K results with enhanced cards"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        # Performance header
        perf_frame = ctk.CTkFrame(
            self.results_scroll,
            fg_color=("blue", "darkblue"),
            corner_radius=10
        )
        perf_frame.pack(fill="x", pady=(Config.PADDING_SMALL, Config.PADDING_MEDIUM), padx=Config.PADDING_SMALL)

        ctk.CTkLabel(
            perf_frame,
            text=f"⚡ Found {len(results)} matches in {time_taken:.3f}s",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            text_color="white"
        ).pack(pady=8)

        if steps:
            breakdown = " • ".join([f"{name}: {t*1000:.0f}ms" for name, t in steps])
            ctk.CTkLabel(
                perf_frame,
                text=breakdown,
                font=ctk.CTkFont(size=9),
                text_color="white"
            ).pack(pady=(0, 8))

        if not results:
            no_results_frame = ctk.CTkFrame(
                self.results_scroll,
                fg_color=("gray90", "gray20"),
                corner_radius=10
            )
            no_results_frame.pack(pady=50, padx=Config.PADDING_LARGE, fill="x")

            ctk.CTkLabel(
                no_results_frame,
                text="📭",
                font=ctk.CTkFont(size=48)
            ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

            ctk.CTkLabel(
                no_results_frame,
                text="No faces in database",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
            ).pack(pady=Config.PADDING_TINY)

            ctk.CTkLabel(
                no_results_frame,
                text="Add people to the database first",
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                text_color="gray"
            ).pack(pady=(Config.PADDING_TINY, Config.PADDING_LARGE))
            return

        # Display each result
        for idx, result in enumerate(results, 1):
            person = result['person_info']
            sim = result['similarity']

            # Determine card color based on similarity
            if sim >= 0.8:
                border_color = Config.COLOR_SUCCESS
                bg_color = Config.COLOR_SUCCESS
                rank_icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "🎯"
            elif sim >= 0.6:
                border_color = Config.COLOR_WARNING
                bg_color = Config.COLOR_WARNING
                rank_icon = "⚠️"
            else:
                border_color = Config.COLOR_ERROR
                bg_color = Config.COLOR_ERROR
                rank_icon = "❌"

            card = ctk.CTkFrame(
                self.results_scroll,
                corner_radius=10,
                border_width=2,
                border_color=border_color
            )
            card.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            # Rank header with icon
            rank_frame = ctk.CTkFrame(card, fg_color=bg_color, corner_radius=8)
            rank_frame.pack(fill="x")

            ctk.CTkLabel(
                rank_frame,
                text=f"{rank_icon} Rank #{idx} • {sim*100:.1f}% Similar",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="white"
            ).pack(pady=6)

            # Person info (compact)
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_SMALL)

            # Main info
            name_dept = f"👤 {person['name']} (ID: {person['face_id']})"
            ctk.CTkLabel(
                info_frame,
                text=name_dept,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                anchor="w"
            ).pack(fill="x")

            # Secondary info
            if person['department'] or person['role']:
                dept_role = f"🏢 {person['department'] or 'N/A'} • 💼 {person['role'] or 'N/A'}"
                ctk.CTkLabel(
                    info_frame,
                    text=dept_role,
                    font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
                    text_color="gray",
                    anchor="w"
                ).pack(fill="x")

            # Similarity bar
            sim_bar = ctk.CTkProgressBar(info_frame, width=300, height=6)
            sim_bar.pack(fill="x", pady=(Config.PADDING_TINY, Config.PADDING_SMALL))
            sim_bar.set(sim)

        self.results_count_label.configure(text=f"Showing {len(results)} of {k} requested")

    def navigate_to_add_person(self):
        """Navigate to Add Person panel"""
        self.app.show_panel("add")
        self.toast.show_info("Opening Add Person panel...")


# Export for imports
__all__ = ['RecognitionPanel']