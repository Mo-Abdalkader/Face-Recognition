"""
Compare Panel - Face Similarity Comparison
Feature #13: Compare two face images and calculate similarity
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import numpy as np
from datetime import datetime

from config import Config
from utils.toast import ToastManager


class ComparePanel(ctk.CTkScrollableFrame):
    """Panel for comparing similarity between two face images"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.image_a_path = None
        self.image_b_path = None
        self.photo_a = None
        self.photo_b = None

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="⚖️ Face Similarity Comparison",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        subtitle = ctk.CTkLabel(
            self,
            text="Upload two face images to compare their similarity",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        subtitle.pack(pady=(0, Config.PADDING_LARGE))

        # Container for side-by-side image frames
        images_container = ctk.CTkFrame(self, fg_color="transparent")
        images_container.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                             pady=Config.PADDING_SMALL)

        # Configure grid for responsive layout
        images_container.grid_columnconfigure(0, weight=1)
        images_container.grid_columnconfigure(1, weight=1)

        # Image A Frame
        frame_a = ctk.CTkFrame(images_container, corner_radius=15)
        frame_a.grid(row=0, column=0, padx=(0, Config.PADDING_SMALL),
                    pady=Config.PADDING_SMALL, sticky="nsew")

        ctk.CTkLabel(
            frame_a,
            text="📸 Image A",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Image A display
        self.image_a_frame = ctk.CTkFrame(frame_a, width=350, height=350)
        self.image_a_frame.pack(pady=Config.PADDING_SMALL, padx=Config.PADDING_LARGE)
        self.image_a_frame.pack_propagate(False)

        self.image_a_label = ctk.CTkLabel(
            self.image_a_frame,
            text="No Image",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.image_a_label.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkButton(
            frame_a,
            text="📁 Upload Image A",
            command=lambda: self.upload_image("A"),
            width=200,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=Config.PADDING_SMALL)

        self.status_a = ctk.CTkLabel(
            frame_a,
            text="⚪ Not Uploaded",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.status_a.pack(pady=(0, Config.PADDING_MEDIUM))

        # Image B Frame
        frame_b = ctk.CTkFrame(images_container, corner_radius=15)
        frame_b.grid(row=0, column=1, padx=(Config.PADDING_SMALL, 0),
                    pady=Config.PADDING_SMALL, sticky="nsew")

        ctk.CTkLabel(
            frame_b,
            text="📸 Image B",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Image B display
        self.image_b_frame = ctk.CTkFrame(frame_b, width=350, height=350)
        self.image_b_frame.pack(pady=Config.PADDING_SMALL, padx=Config.PADDING_LARGE)
        self.image_b_frame.pack_propagate(False)

        self.image_b_label = ctk.CTkLabel(
            self.image_b_frame,
            text="No Image",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.image_b_label.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkButton(
            frame_b,
            text="📁 Upload Image B",
            command=lambda: self.upload_image("B"),
            width=200,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=Config.PADDING_SMALL)

        self.status_b = ctk.CTkLabel(
            frame_b,
            text="⚪ Not Uploaded",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.status_b.pack(pady=(0, Config.PADDING_MEDIUM))

        # Compare Button
        self.compare_btn = ctk.CTkButton(
            self,
            text="⚖️ Compare Similarity",
            command=self.compare_faces,
            width=300,
            height=60,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            fg_color="purple",
            hover_color="darkviolet",
            state="disabled"
        )
        self.compare_btn.pack(pady=Config.PADDING_LARGE)

        # Results Frame
        self.results_frame = ctk.CTkFrame(self, corner_radius=15)
        self.results_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                               pady=(0, Config.PADDING_LARGE))

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Upload both images to compare",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.results_label.pack(pady=30)

    def upload_image(self, which):
        """Upload image A or B"""
        file_path = filedialog.askopenfilename(
            title=f"Select Face Image {which}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            if which == "A":
                self.image_a_path = file_path
                self.display_image(file_path, "A")
                self.status_a.configure(
                    text="✅ Uploaded",
                    text_color=Config.COLOR_SUCCESS_TEXT
                )
                self.toast.show_success(
                    "Image A uploaded successfully",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            else:
                self.image_b_path = file_path
                self.display_image(file_path, "B")
                self.status_b.configure(
                    text="✅ Uploaded",
                    text_color=Config.COLOR_SUCCESS_TEXT
                )
                self.toast.show_success(
                    "Image B uploaded successfully",
                    duration=Config.TOAST_DURATION_SUCCESS
                )

            # Enable compare button if both uploaded
            if self.image_a_path and self.image_b_path:
                self.compare_btn.configure(state="normal")

    def display_image(self, image_path, which):
        """Display uploaded image"""
        try:
            img = Image.open(image_path)
            img.thumbnail((330, 330), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            if which == "A":
                self.photo_a = photo
                self.image_a_label.configure(image=photo, text="")
            else:
                self.photo_b = photo
                self.image_b_label.configure(image=photo, text="")

        except Exception as e:
            self.toast.show_error(
                f"Failed to load image: {str(e)}",
                duration=Config.TOAST_DURATION_ERROR
            )

    def compare_faces(self):
        """Compare the two uploaded faces"""
        if not self.image_a_path or not self.image_b_path:
            self.toast.show_warning(
                "Please upload both images!",
                duration=Config.TOAST_DURATION_WARNING
            )
            return

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Show loading
        loading = ctk.CTkLabel(
            self.results_frame,
            text="🔄 Calculating similarity...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        )
        loading.pack(pady=30)

        self.toast.show_info(
            "Calculating similarity...",
            duration=Config.TOAST_DURATION_INFO
        )

        # Run comparison in thread
        def compare_thread():
            try:
                result = self.calculate_similarity()
                self.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.after(0, lambda err=str(e): self.toast.show_error(
                    f"Comparison failed: {err}",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=compare_thread, daemon=True)
        thread.start()

    def calculate_similarity(self):
        """Calculate similarity between two faces"""
        # Generate embeddings
        emb_a = self.app.embedding_generator.generate_embedding(self.image_a_path)
        emb_b = self.app.embedding_generator.generate_embedding(self.image_b_path)

        if emb_a is None or emb_b is None:
            return {'error': 'Face detection failed in one or both images'}

        # Cosine similarity (embeddings are L2-normalized)
        cosine_sim = float(np.dot(emb_a, emb_b))

        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(emb_a - emb_b))

        # Interpretation
        if cosine_sim > 0.8:
            interpretation = "Same person (High confidence)"
            color = Config.COLOR_SUCCESS_TEXT
            emoji = "✅"
        elif cosine_sim > 0.6:
            interpretation = "Likely same person (Moderate)"
            color = "yellow"
            emoji = "🟡"
        elif cosine_sim > 0.4:
            interpretation = "Uncertain (Low similarity)"
            color = Config.COLOR_WARNING_TEXT
            emoji = "🟠"
        else:
            interpretation = "Different people (High confidence)"
            color = Config.COLOR_ERROR_TEXT
            emoji = "❌"

        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'interpretation': interpretation,
            'color': color,
            'emoji': emoji
        }

    def display_results(self, result):
        """Display comparison results"""
        # Clear loading
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if 'error' in result:
            ctk.CTkLabel(
                self.results_frame,
                text=f"❌ {result['error']}",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
                text_color=Config.COLOR_ERROR_TEXT
            ).pack(pady=30)

            self.toast.show_error(
                result['error'],
                duration=Config.TOAST_DURATION_ERROR
            )
            return

        # Show success toast
        self.toast.show_success(
            f"Comparison complete: {result['interpretation']}",
            duration=Config.TOAST_DURATION_SUCCESS
        )

        # Result header
        header = ctk.CTkFrame(
            self.results_frame,
            fg_color=result['color'],
            corner_radius=10
        )
        header.pack(fill="x", padx=Config.PADDING_MEDIUM,
                   pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        ctk.CTkLabel(
            header,
            text=f"{result['emoji']} {result['interpretation'].upper()}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold"),
            text_color="white"
        ).pack(pady=Config.PADDING_MEDIUM)

        # Metrics container
        metrics_container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        metrics_container.pack(pady=Config.PADDING_SMALL)

        # Configure grid for metrics
        metrics_container.grid_columnconfigure(0, weight=1)
        metrics_container.grid_columnconfigure(1, weight=1)

        # Cosine Similarity
        sim_frame = ctk.CTkFrame(metrics_container, corner_radius=10)
        sim_frame.grid(row=0, column=0, padx=Config.PADDING_SMALL,
                      pady=Config.PADDING_TINY, sticky="nsew")

        ctk.CTkLabel(
            sim_frame,
            text="📊 Cosine Similarity",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        ctk.CTkLabel(
            sim_frame,
            text=f"{result['cosine_similarity'] * 100:.2f}%",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=result['color']
        ).pack()

        ctk.CTkLabel(
            sim_frame,
            text="(0% = Different, 100% = Identical)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Euclidean Distance
        dist_frame = ctk.CTkFrame(metrics_container, corner_radius=10)
        dist_frame.grid(row=0, column=1, padx=Config.PADDING_SMALL,
                       pady=Config.PADDING_TINY, sticky="nsew")

        ctk.CTkLabel(
            dist_frame,
            text="📏 Euclidean Distance",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        ctk.CTkLabel(
            dist_frame,
            text=f"{result['euclidean_distance']:.4f}",
            font=ctk.CTkFont(size=32, weight="bold")
        ).pack()

        ctk.CTkLabel(
            dist_frame,
            text="(Lower = More Similar)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Interpretation guide
        guide_frame = ctk.CTkFrame(self.results_frame, corner_radius=10)
        guide_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            guide_frame,
            text="📖 Similarity Scale Guide",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        guide_text = """
🟢 > 80%: Same person (High confidence)
🟡 60-80%: Likely same person (Moderate)
🟠 40-60%: Uncertain (Low similarity)
🔴 < 40%: Different people (High confidence)
        """

        ctk.CTkLabel(
            guide_frame,
            text=guide_text.strip(),
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            justify="left"
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Timestamp
        time_label = ctk.CTkLabel(
            self.results_frame,
            text=f"🕒 Compared at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        )
        time_label.pack(pady=(Config.PADDING_TINY, Config.PADDING_MEDIUM))

        # Action buttons
        action_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        action_frame.pack(pady=(0, Config.PADDING_MEDIUM))

        ctk.CTkButton(
            action_frame,
            text="🔄 Compare Again",
            command=self.reset_comparison,
            width=150,
            height=35
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            action_frame,
            text="💾 Save Result",
            command=lambda: self.save_result(result),
            width=150,
            height=35,
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_TINY)

    def reset_comparison(self):
        """Reset for new comparison"""
        self.image_a_path = None
        self.image_b_path = None
        self.photo_a = None
        self.photo_b = None

        self.image_a_label.configure(image=None, text="No Image")
        self.image_b_label.configure(image=None, text="No Image")

        self.status_a.configure(
            text="⚪ Not Uploaded",
            text_color=Config.COLOR_INFO_TEXT
        )
        self.status_b.configure(
            text="⚪ Not Uploaded",
            text_color=Config.COLOR_INFO_TEXT
        )

        self.compare_btn.configure(state="disabled")

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="Upload both images to compare",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.results_label.pack(pady=30)

        self.toast.show_info(
            "Ready for new comparison",
            duration=Config.TOAST_DURATION_INFO
        )

    def save_result(self, result):
        """Save comparison result to file"""
        try:
            filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(filename, 'w') as f:
                f.write("FACE SIMILARITY COMPARISON RESULT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Image A: {self.image_a_path}\n")
                f.write(f"Image B: {self.image_b_path}\n\n")
                f.write(f"Cosine Similarity: {result['cosine_similarity'] * 100:.2f}%\n")
                f.write(f"Euclidean Distance: {result['euclidean_distance']:.4f}\n")
                f.write(f"Interpretation: {result['interpretation']}\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            self.toast.show_success(
                f"Result saved to: {filename}",
                duration=Config.TOAST_DURATION_SUCCESS
            )

        except Exception as e:
            self.toast.show_error(
                f"Failed to save: {str(e)}",
                duration=Config.TOAST_DURATION_ERROR
            )