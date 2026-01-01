"""
Add Person Panel - Register new person with face preview and duplicate detection
Features: #1 (Add Person), #4 (Duplicate Detection), #11 (Face Preview)
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import numpy as np
from pathlib import Path
import cv2

from config import Config
from utils.toast import ToastManager


class AddPersonPanel(ctk.CTkScrollableFrame):
    """Panel for adding new person to database"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.selected_images = []
        self.image_previews = []
        self.face_data = []  # [(original_img, cropped_face, detected, img_path)]

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="➕ Add New Person to Database",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_LARGE))

        # Form Frame
        form_frame = ctk.CTkFrame(self, corner_radius=15)
        form_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        # Face ID
        ctk.CTkLabel(
            form_frame,
            text="Face ID: *",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(Config.PADDING_LARGE, Config.PADDING_TINY))

        self.face_id_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., EMP001, STUDENT_001",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.face_id_entry.grid(row=1, column=0, padx=Config.PADDING_LARGE,
                                pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Name
        ctk.CTkLabel(
            form_frame,
            text="Full Name: *",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=2, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.name_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., Ahmed Hassan",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.name_entry.grid(row=3, column=0, padx=Config.PADDING_LARGE,
                             pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Department
        ctk.CTkLabel(
            form_frame,
            text="Department:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=4, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.dept_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., Engineering, Data Science",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.dept_entry.grid(row=5, column=0, padx=Config.PADDING_LARGE,
                             pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Role
        ctk.CTkLabel(
            form_frame,
            text="Role:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=6, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.role_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., Software Engineer, Student",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.role_entry.grid(row=7, column=0, padx=Config.PADDING_LARGE,
                             pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Email
        ctk.CTkLabel(
            form_frame,
            text="Email:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=8, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.email_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., name@company.com",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.email_entry.grid(row=9, column=0, padx=Config.PADDING_LARGE,
                              pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Phone
        ctk.CTkLabel(
            form_frame,
            text="Phone:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=10, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.phone_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="e.g., +20-123-456-7890",
            width=Config.BUTTON_WIDTH_FORM
        )
        self.phone_entry.grid(row=11, column=0, padx=Config.PADDING_LARGE,
                              pady=(0, Config.PADDING_MEDIUM), sticky="ew")

        # Notes
        ctk.CTkLabel(
            form_frame,
            text="Notes:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).grid(row=12, column=0, sticky="w", padx=Config.PADDING_LARGE,
               pady=(0, Config.PADDING_TINY))

        self.notes_entry = ctk.CTkTextbox(
            form_frame,
            height=Config.TEXTBOX_HEIGHT_NOTES,
            width=Config.BUTTON_WIDTH_FORM
        )
        self.notes_entry.grid(row=13, column=0, padx=Config.PADDING_LARGE,
                              pady=(0, Config.PADDING_LARGE), sticky="ew")

        form_frame.grid_columnconfigure(0, weight=1)

        # Image Upload Section
        upload_frame = ctk.CTkFrame(self, corner_radius=15)
        upload_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                          pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            upload_frame,
            text="📸 Upload Face Images (3-5 angles recommended)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        ctk.CTkButton(
            upload_frame,
            text="📁 Select Images",
            command=self.select_images,
            width=Config.BUTTON_WIDTH_FORM,
            height=Config.BUTTON_HEIGHT_MEDIUM
        ).pack(pady=Config.PADDING_SMALL)

        # Preview frame (scrollable area for images)
        self.preview_frame = ctk.CTkFrame(upload_frame, fg_color="transparent")
        self.preview_frame.pack(fill="both", expand=True,
                                padx=Config.PADDING_LARGE,
                                pady=(Config.PADDING_SMALL, Config.PADDING_LARGE))

        # Action Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=Config.PADDING_LARGE)

        ctk.CTkButton(
            button_frame,
            text="💾 Save to Database",
            command=self.save_person,
            width=Config.BUTTON_WIDTH_ACTION,
            height=Config.BUTTON_HEIGHT_LARGE,
            font=ctk.CTkFont(size=Config.FONT_SIZE_BUTTON, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkButton(
            button_frame,
            text="🔄 Clear Form",
            command=self.clear_form,
            width=Config.BUTTON_WIDTH_SECONDARY,
            height=Config.BUTTON_HEIGHT_LARGE,
            font=ctk.CTkFont(size=Config.FONT_SIZE_BUTTON, weight="bold"),
            fg_color=Config.COLOR_INFO,
            hover_color=Config.COLOR_INFO_HOVER
        ).pack(side="left", padx=Config.PADDING_SMALL)

    def select_images(self):
        """Open file dialog to select multiple images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Face Images (multiple angles recommended)",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_paths:
            self.selected_images = list(file_paths)
            self.process_images()

    def process_images(self):
        """Process images with MTCNN face detection and show previews"""
        # Clear previous previews
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        self.face_data = []

        if not self.selected_images:
            return

        # Show loading indicator
        loading_label = ctk.CTkLabel(
            self.preview_frame,
            text="🔄 Processing images with face detection...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        )
        loading_label.pack(pady=Config.PADDING_LARGE)

        # Process in background thread
        def process_thread():
            for img_path in self.selected_images:
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Detect and crop face
                    cropped_face = self.app.face_cropper.detect_and_crop(img_rgb)

                    if cropped_face is not None:
                        self.face_data.append((img_rgb, cropped_face, True, img_path))
                    else:
                        self.face_data.append((img_rgb, None, False, img_path))

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    self.face_data.append((None, None, False, img_path))

            # Update UI in main thread
            self.after(0, self.display_previews)

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def display_previews(self):
        """Display side-by-side previews of original and cropped faces"""
        # Clear loading message
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        if not self.face_data:
            return

        # Grid layout for previews
        for idx, (original, cropped, detected, img_path) in enumerate(self.face_data):
            # Card for each image
            card = ctk.CTkFrame(self.preview_frame, corner_radius=10)
            card.grid(row=idx, column=0, padx=Config.PADDING_SMALL,
                      pady=Config.PADDING_SMALL, sticky="ew")

            # Original image preview
            orig_frame = ctk.CTkFrame(card)
            orig_frame.grid(row=0, column=0, padx=Config.PADDING_SMALL,
                            pady=Config.PADDING_SMALL)

            ctk.CTkLabel(
                orig_frame,
                text="Original Image",
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold")
            ).pack()

            if original is not None:
                # Resize for display using configured size
                orig_pil = Image.fromarray(original)
                orig_pil.thumbnail(Config.PREVIEW_IMAGE_SIZE, Image.Resampling.LANCZOS)
                orig_tk = ImageTk.PhotoImage(orig_pil)

                img_label = ctk.CTkLabel(orig_frame, image=orig_tk, text="")
                img_label.image = orig_tk  # Keep reference
                img_label.pack(pady=Config.PADDING_TINY)

            # Professional arrow
            ctk.CTkLabel(
                card,
                text="➡️",
                font=ctk.CTkFont(size=24)
            ).grid(row=0, column=1, padx=Config.PADDING_TINY)

            # Cropped face preview
            crop_frame = ctk.CTkFrame(card)
            crop_frame.grid(row=0, column=2, padx=Config.PADDING_SMALL,
                            pady=Config.PADDING_SMALL)

            if detected and cropped is not None:
                ctk.CTkLabel(
                    crop_frame,
                    text="✅ Face Detected",
                    font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                    text_color=Config.COLOR_SUCCESS_TEXT
                ).pack()

                crop_pil = Image.fromarray(cropped)
                crop_pil.thumbnail(Config.PREVIEW_IMAGE_SIZE, Image.Resampling.LANCZOS)
                crop_tk = ImageTk.PhotoImage(crop_pil)

                crop_label = ctk.CTkLabel(crop_frame, image=crop_tk, text="")
                crop_label.image = crop_tk
                crop_label.pack(pady=Config.PADDING_TINY)
            else:
                ctk.CTkLabel(
                    crop_frame,
                    text="❌ No Face Detected",
                    font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                    text_color=Config.COLOR_ERROR_TEXT
                ).pack()

                ctk.CTkLabel(
                    crop_frame,
                    text="This image will be skipped",
                    font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
                    text_color=Config.COLOR_INFO_TEXT
                ).pack()

            # File name
            ctk.CTkLabel(
                card,
                text=f"File: {Path(img_path).name}",
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
                text_color=Config.COLOR_INFO_TEXT
            ).grid(row=1, column=0, columnspan=3, pady=(0, Config.PADDING_SMALL))

        self.preview_frame.grid_columnconfigure(0, weight=1)

    def save_person(self):
        """Save person to database with duplicate detection"""
        # Validate inputs
        face_id = self.face_id_entry.get().strip()
        name = self.name_entry.get().strip()

        if not face_id or not name:
            self.toast.show_error(
                "Face ID and Name are required!",
                duration=Config.TOAST_DURATION_ERROR
            )
            return

        # Count valid faces
        valid_faces = [(orig, crop, det, path) for orig, crop, det, path
                       in self.face_data if det and crop is not None]

        if not valid_faces:
            self.toast.show_error(
                "Please upload at least one image with a detectable face!",
                duration=Config.TOAST_DURATION_ERROR
            )
            return

        # Check if face_id already exists
        existing_ids = self.app.db.get_all_face_ids()
        if face_id in existing_ids:
            self.toast.show_error(
                f"Face ID '{face_id}' already exists in database!",
                duration=Config.TOAST_DURATION_ERROR
            )
            return

        # Generate embeddings and check for duplicates
        self.toast.show_info(
            "Processing: Generating embeddings and checking for duplicates...",
            duration=Config.TOAST_DURATION_INFO
        )

        def save_thread():
            try:
                # Generate embeddings
                embeddings = []
                for _, cropped, _, _ in valid_faces:
                    # Save cropped face temporarily
                    temp_path = "temp_face.jpg"
                    Image.fromarray(cropped).save(temp_path)

                    embedding = self.app.embedding_generator.generate_embedding(temp_path)
                    if embedding is not None:
                        embeddings.append(embedding)

                if not embeddings:
                    self.after(0, lambda: self.toast.show_error(
                        "Failed to generate embeddings from images!",
                        duration=Config.TOAST_DURATION_ERROR
                    ))
                    return

                # Check for duplicates
                duplicates = self.check_duplicates(embeddings)

                if duplicates:
                    # Show duplicate warning
                    self.after(0, lambda: self.show_duplicate_warning(
                        duplicates, face_id, name, embeddings, valid_faces
                    ))
                else:
                    # No duplicates, proceed to save
                    self.after(0, lambda: self.finalize_save(
                        face_id, name, embeddings, valid_faces
                    ))

            except Exception as e:
                self.after(0, lambda err=str(e): self.toast.show_error(
                    f"Failed to process: {err}",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=save_thread, daemon=True)
        thread.start()

    def check_duplicates(self, new_embeddings):
        """Check if embeddings match existing people in database"""
        all_embeddings = self.app.db.get_all_embeddings()

        duplicates = []
        for new_emb in new_embeddings:
            for face_id, db_emb, img_path in all_embeddings:
                similarity = float(np.dot(new_emb, db_emb))

                if similarity >= Config.DUPLICATE_THRESHOLD:
                    person_info = self.app.db.get_person_info(face_id)
                    duplicates.append({
                        'face_id': face_id,
                        'person_info': person_info,
                        'similarity': similarity
                    })

        # Return unique people sorted by similarity
        unique_duplicates = {}
        for dup in duplicates:
            fid = dup['face_id']
            if fid not in unique_duplicates or dup['similarity'] > unique_duplicates[fid]['similarity']:
                unique_duplicates[fid] = dup

        return sorted(unique_duplicates.values(),
                      key=lambda x: x['similarity'],
                      reverse=True)[:Config.MAX_DUPLICATES_SHOWN]

    def show_duplicate_warning(self, duplicates, face_id, name, embeddings, valid_faces):
        """Show warning dialog about potential duplicates"""
        # Create a custom dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title("⚠️ Duplicate Detection")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")

        # Warning frame
        warning_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        warning_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                           pady=Config.PADDING_LARGE)

        # Title
        ctk.CTkLabel(
            warning_frame,
            text="⚠️ POTENTIAL DUPLICATE DETECTED",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            text_color=Config.COLOR_WARNING_TEXT
        ).pack(pady=(0, Config.PADDING_MEDIUM))

        # Message
        ctk.CTkLabel(
            warning_frame,
            text="The uploaded face is highly similar to:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        ).pack(pady=(0, Config.PADDING_SMALL))

        # Scrollable list of duplicates
        dup_frame = ctk.CTkScrollableFrame(warning_frame, height=150)
        dup_frame.pack(fill="both", expand=True, pady=Config.PADDING_SMALL)

        for dup in duplicates:
            person = dup['person_info']
            card = ctk.CTkFrame(dup_frame, corner_radius=10)
            card.pack(fill="x", pady=Config.PADDING_TINY)

            info_text = f"• {person['name']} ({person['face_id']})\n"
            info_text += f"  Similarity: {dup['similarity'] * 100:.1f}%\n"
            info_text += f"  Department: {person['department'] or 'N/A'}"

            ctk.CTkLabel(
                card,
                text=info_text,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                justify="left"
            ).pack(padx=Config.PADDING_SMALL, pady=Config.PADDING_SMALL, anchor="w")

        # Question
        ctk.CTkLabel(
            warning_frame,
            text="Do you want to add this person anyway?",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Button frame
        btn_frame = ctk.CTkFrame(warning_frame, fg_color="transparent")
        btn_frame.pack(pady=Config.PADDING_SMALL)

        def on_yes():
            dialog.destroy()
            self.finalize_save(face_id, name, embeddings, valid_faces)

        def on_no():
            dialog.destroy()
            self.toast.show_info(
                "Operation cancelled. Person was not added.",
                duration=Config.TOAST_DURATION_INFO
            )

        ctk.CTkButton(
            btn_frame,
            text="✅ Yes, Add Anyway",
            command=on_yes,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            btn_frame,
            text="❌ No, Cancel",
            command=on_no,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            fg_color=Config.COLOR_ERROR,
            hover_color="darkred"
        ).pack(side="left", padx=Config.PADDING_TINY)

    def finalize_save(self, face_id, name, embeddings, valid_faces):
        """Actually save person to database"""
        try:
            # Add person
            success = self.app.db.add_person(
                face_id=face_id,
                name=name,
                department=self.dept_entry.get().strip() or None,
                role=self.role_entry.get().strip() or None,
                email=self.email_entry.get().strip() or None,
                phone=self.phone_entry.get().strip() or None,
                notes=self.notes_entry.get("1.0", "end").strip() or None
            )

            if not success:
                self.toast.show_error(
                    "Failed to add person to database!",
                    duration=Config.TOAST_DURATION_ERROR
                )
                return

            # Add embeddings
            for idx, (embedding, (_, _, _, img_path)) in enumerate(zip(embeddings, valid_faces)):
                self.app.db.add_embedding(face_id, embedding, img_path)

            # Reload recognizer database
            self.app.recognizer.reload_database()

            # Show success message
            self.toast.show_success(
                f"✓ Added {name} (ID: {face_id})\n✓ Generated {len(embeddings)} embeddings",
                duration=Config.TOAST_DURATION_SUCCESS
            )

            # Clear form
            self.clear_form()

            # Refresh stats if available
            if hasattr(self.app, 'refresh_stats'):
                self.app.refresh_stats()

        except Exception as e:
            self.toast.show_error(
                f"Failed to save: {str(e)}",
                duration=Config.TOAST_DURATION_ERROR
            )

    def clear_form(self):
        """Clear all form fields and previews"""
        self.face_id_entry.delete(0, "end")
        self.name_entry.delete(0, "end")
        self.dept_entry.delete(0, "end")
        self.role_entry.delete(0, "end")
        self.email_entry.delete(0, "end")
        self.phone_entry.delete(0, "end")
        self.notes_entry.delete("1.0", "end")

        self.selected_images = []
        self.face_data = []

        for widget in self.preview_frame.winfo_children():
            widget.destroy()
