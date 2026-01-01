"""
Gallery Panel - View all registered faces with search/filter
Features: #8 (View Person Details), #9 (Gallery), #2 (Update), #3 (Delete)
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
from pathlib import Path

from config import Config
from utils.toast import ToastManager


class GalleryPanel(ctk.CTkFrame):
    """Panel for viewing all registered people in grid layout"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.all_people = []
        self.filtered_people = []
        self.selected_person = None

        self.create_widgets()
        self.load_people()

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="🖼️ Face Gallery",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        subtitle = ctk.CTkLabel(
            self,
            text="View and manage all registered people",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        subtitle.pack(pady=(0, Config.PADDING_MEDIUM))

        # Search and filter frame
        filter_frame = ctk.CTkFrame(self, corner_radius=15)
        filter_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        # Search bar
        search_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        search_frame.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_MEDIUM)

        ctk.CTkLabel(
            search_frame,
            text="🔍 Search:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search by name or ID...",
            width=300
        )
        self.search_entry.pack(side="left", padx=Config.PADDING_TINY)
        self.search_entry.bind("<KeyRelease>", lambda e: self.apply_filters())

        ctk.CTkButton(
            search_frame,
            text="🔄 Refresh",
            command=self.load_people,
            width=100,
            height=35
        ).pack(side="right", padx=Config.PADDING_TINY)

        # Stats
        self.stats_label = ctk.CTkLabel(
            filter_frame,
            text="Loading...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.stats_label.pack(pady=(0, Config.PADDING_SMALL))

        # Gallery grid (scrollable)
        self.gallery_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.gallery_scroll.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                                 pady=Config.PADDING_SMALL)

    def load_people(self):
        """Load all registered people from database"""
        # Show loading
        for widget in self.gallery_scroll.winfo_children():
            widget.destroy()

        loading = ctk.CTkLabel(
            self.gallery_scroll,
            text="🔄 Loading gallery...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        )
        loading.pack(pady=50)

        def load_thread():
            try:
                # Get all face IDs
                face_ids = self.app.db.get_all_face_ids()
                self.all_people = []

                for face_id in face_ids:
                    person_info = self.app.db.get_person_info(face_id)
                    embeddings = self.app.db.get_embeddings_by_face_id(face_id)

                    if person_info:
                        person_info['embedding_count'] = len(embeddings)
                        self.all_people.append(person_info)

                self.filtered_people = self.all_people.copy()

                # Update UI
                self.after(0, self.display_gallery)
                self.after(0, lambda: self.toast.show_success(
                    f"Loaded {len(self.all_people)} people from gallery",
                    duration=Config.TOAST_DURATION_SUCCESS
                ))

            except Exception as e:
                self.after(0, lambda err=str(e): self.toast.show_error(
                    f"Failed to load gallery: {err}",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def apply_filters(self):
        """Apply search filter"""
        search_term = self.search_entry.get().strip().lower()

        if not search_term:
            self.filtered_people = self.all_people.copy()
        else:
            self.filtered_people = [
                p for p in self.all_people
                if search_term in p['name'].lower() or
                   search_term in p['face_id'].lower()
            ]

        self.display_gallery()

    def display_gallery(self):
        """Display people in grid layout"""
        # Clear gallery
        for widget in self.gallery_scroll.winfo_children():
            widget.destroy()

        if not self.filtered_people:
            ctk.CTkLabel(
                self.gallery_scroll,
                text="No people found",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
                text_color=Config.COLOR_INFO_TEXT
            ).pack(pady=50)
            return

        # Update stats
        total = len(self.all_people)
        shown = len(self.filtered_people)
        self.stats_label.configure(
            text=f"Showing {shown} of {total} people"
        )

        # Create grid container
        grid_frame = ctk.CTkFrame(self.gallery_scroll, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=Config.PADDING_SMALL,
                       pady=Config.PADDING_SMALL)

        # Display in 3 columns
        columns = 3
        for idx, person in enumerate(self.filtered_people):
            row = idx // columns
            col = idx % columns

            card = self.create_person_card(grid_frame, person)
            card.grid(row=row, column=col, padx=Config.PADDING_SMALL,
                     pady=Config.PADDING_SMALL, sticky="nsew")

        # Configure grid weights
        for i in range(columns):
            grid_frame.grid_columnconfigure(i, weight=1)

    def create_person_card(self, parent, person):
        """Create card for person"""
        card = ctk.CTkFrame(parent, corner_radius=10, width=250, height=350)
        card.grid_propagate(False)

        # Placeholder image (no actual photo stored)
        icon_label = ctk.CTkLabel(
            card,
            text="👤",
            font=ctk.CTkFont(size=80)
        )
        icon_label.pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        # Name
        name_label = ctk.CTkLabel(
            card,
            text=person['name'],
            font=ctk.CTkFont(size=Config.FONT_SIZE_BUTTON, weight="bold"),
            wraplength=220
        )
        name_label.pack(pady=Config.PADDING_TINY)

        # Face ID
        id_label = ctk.CTkLabel(
            card,
            text=f"ID: {person['face_id']}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        )
        id_label.pack()

        # Department
        if person['department']:
            dept_label = ctk.CTkLabel(
                card,
                text=f"🏢 {person['department']}",
                font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1)
            )
            dept_label.pack(pady=2)

        # Embeddings count
        emb_label = ctk.CTkLabel(
            card,
            text=f"📸 {person['embedding_count']} embeddings",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        )
        emb_label.pack(pady=Config.PADDING_TINY)

        # Buttons
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(pady=Config.PADDING_SMALL)

        ctk.CTkButton(
            btn_frame,
            text="📄 Details",
            command=lambda p=person: self.show_details(p),
            width=90,
            height=30,
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1)
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            btn_frame,
            text="✏️ Edit",
            command=lambda p=person: self.edit_person(p),
            width=70,
            height=30,
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            fg_color=Config.COLOR_WARNING,
            hover_color="darkorange"
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            btn_frame,
            text="🗑️",
            command=lambda p=person: self.delete_person(p),
            width=40,
            height=30,
            fg_color=Config.COLOR_ERROR,
            hover_color="darkred"
        ).pack(side="left", padx=3)

        return card

    def show_details(self, person):
        """Show detailed person information"""
        detail_window = ctk.CTkToplevel(self)
        detail_window.title(f"Person Details - {person['name']}")
        detail_window.geometry("500x600")
        detail_window.transient(self)
        detail_window.grab_set()

        # Center the window
        detail_window.update_idletasks()
        x = (detail_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (detail_window.winfo_screenheight() // 2) - (600 // 2)
        detail_window.geometry(f"500x600+{x}+{y}")

        # Title
        ctk.CTkLabel(
            detail_window,
            text=f"👤 {person['name']}",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        # Details frame (scrollable)
        details_frame = ctk.CTkScrollableFrame(detail_window, corner_radius=15)
        details_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                          pady=Config.PADDING_SMALL)

        # Create info text with better formatting
        info_items = [
            ("🆔 Face ID", person['face_id']),
            ("👤 Name", person['name']),
            ("🏢 Department", person['department'] or 'N/A'),
            ("💼 Role", person['role'] or 'N/A'),
            ("📧 Email", person['email'] or 'N/A'),
            ("📱 Phone", person['phone'] or 'N/A'),
            ("📸 Embeddings", str(person['embedding_count'])),
            ("📅 Registered", person['registered_at']),
        ]

        for label, value in info_items:
            item_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
            item_frame.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

            ctk.CTkLabel(
                item_frame,
                text=label,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
                anchor="w",
                width=120
            ).pack(side="left")

            ctk.CTkLabel(
                item_frame,
                text=value,
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                anchor="w"
            ).pack(side="left", fill="x", expand=True)

        # Notes section
        if person['notes']:
            notes_frame = ctk.CTkFrame(details_frame, corner_radius=10)
            notes_frame.pack(fill="x", pady=Config.PADDING_SMALL, padx=Config.PADDING_SMALL)

            ctk.CTkLabel(
                notes_frame,
                text="📝 Notes:",
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold")
            ).pack(anchor="w", padx=Config.PADDING_SMALL, pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

            ctk.CTkLabel(
                notes_frame,
                text=person['notes'],
                font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
                justify="left",
                anchor="w"
            ).pack(fill="x", padx=Config.PADDING_SMALL, pady=(0, Config.PADDING_SMALL))

        # Close button
        ctk.CTkButton(
            detail_window,
            text="Close",
            command=detail_window.destroy,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM
        ).pack(pady=Config.PADDING_LARGE)

    def edit_person(self, person):
        """Open edit dialog for person"""
        edit_window = ctk.CTkToplevel(self)
        edit_window.title(f"Edit Person - {person['name']}")
        edit_window.geometry("500x700")
        edit_window.transient(self)
        edit_window.grab_set()

        # Center the window
        edit_window.update_idletasks()
        x = (edit_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (edit_window.winfo_screenheight() // 2) - (700 // 2)
        edit_window.geometry(f"500x700+{x}+{y}")

        # Title
        ctk.CTkLabel(
            edit_window,
            text="✏️ Edit Person Information",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_MEDIUM))

        # Form (scrollable)
        form_frame = ctk.CTkScrollableFrame(edit_window)
        form_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                       pady=Config.PADDING_SMALL)

        # Face ID (read-only)
        ctk.CTkLabel(
            form_frame,
            text=f"Face ID: {person['face_id']} (cannot be changed)",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(pady=Config.PADDING_TINY)

        # Name
        ctk.CTkLabel(
            form_frame,
            text="Name:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        name_entry = ctk.CTkEntry(form_frame, width=400)
        name_entry.insert(0, person['name'])
        name_entry.pack()

        # Department
        ctk.CTkLabel(
            form_frame,
            text="Department:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        dept_entry = ctk.CTkEntry(form_frame, width=400)
        if person['department']:
            dept_entry.insert(0, person['department'])
        dept_entry.pack()

        # Role
        ctk.CTkLabel(
            form_frame,
            text="Role:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        role_entry = ctk.CTkEntry(form_frame, width=400)
        if person['role']:
            role_entry.insert(0, person['role'])
        role_entry.pack()

        # Email
        ctk.CTkLabel(
            form_frame,
            text="Email:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        email_entry = ctk.CTkEntry(form_frame, width=400)
        if person['email']:
            email_entry.insert(0, person['email'])
        email_entry.pack()

        # Phone
        ctk.CTkLabel(
            form_frame,
            text="Phone:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        phone_entry = ctk.CTkEntry(form_frame, width=400)
        if person['phone']:
            phone_entry.insert(0, person['phone'])
        phone_entry.pack()

        # Notes
        ctk.CTkLabel(
            form_frame,
            text="Notes:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(anchor="w", pady=(Config.PADDING_SMALL, 2))
        notes_entry = ctk.CTkTextbox(form_frame, width=400, height=Config.TEXTBOX_HEIGHT_NOTES)
        if person['notes']:
            notes_entry.insert("1.0", person['notes'])
        notes_entry.pack()

        # Save button
        def save_changes():
            try:
                # Note: This requires a proper update_person method in database
                # For now, show a toast notification
                self.toast.show_warning(
                    "Update functionality requires database update method implementation",
                    duration=Config.TOAST_DURATION_WARNING
                )
                edit_window.destroy()
                # Uncomment when update method is available:
                # self.load_people()

            except Exception as e:
                self.toast.show_error(
                    f"Failed to update: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

        ctk.CTkButton(
            edit_window,
            text="💾 Save Changes",
            command=save_changes,
            width=200,
            height=Config.BUTTON_HEIGHT_LARGE - 5,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(pady=Config.PADDING_MEDIUM)

    def delete_person(self, person):
        """Delete person from database with confirmation dialog"""
        # Create custom confirmation dialog
        confirm_dialog = ctk.CTkToplevel(self)
        confirm_dialog.title("⚠️ Delete Confirmation")
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
            text="DELETE CONFIRMATION",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            text_color=Config.COLOR_ERROR_TEXT
        ).pack(pady=(0, Config.PADDING_MEDIUM))

        # Warning message
        warning_frame = ctk.CTkFrame(confirm_dialog, corner_radius=10)
        warning_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        warning_text = f"""Are you sure you want to delete?

Person: {person['name']}
Face ID: {person['face_id']}
Embeddings: {person['embedding_count']}

⚠️ This action cannot be undone!"""

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
            try:
                success = self.app.db.delete_person(person['face_id'])

                if success:
                    self.app.recognizer.reload_database()
                    if hasattr(self.app, 'refresh_stats'):
                        self.app.refresh_stats()

                    self.toast.show_success(
                        f"Successfully deleted {person['name']} and {person['embedding_count']} embeddings",
                        duration=Config.TOAST_DURATION_SUCCESS
                    )
                    self.load_people()
                else:
                    self.toast.show_error(
                        "Failed to delete person",
                        duration=Config.TOAST_DURATION_ERROR
                    )

            except Exception as e:
                self.toast.show_error(
                    f"Failed to delete: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

        def on_cancel():
            confirm_dialog.destroy()
            self.toast.show_info(
                "Delete operation cancelled",
                duration=Config.TOAST_DURATION_INFO
            )

        ctk.CTkButton(
            btn_frame,
            text="🗑️ Yes, Delete",
            command=on_confirm,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            fg_color=Config.COLOR_ERROR,
            hover_color="darkred",
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