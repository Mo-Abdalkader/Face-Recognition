"""
History Panel - Complete Implementation with Recognition Logging
Feature #12: Recognition History Log with Filters and Export
"""

import customtkinter as ctk
from tkinter import filedialog
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import threading
from pathlib import Path

from config import Config
from utils.toast import ToastManager


class HistoryPanel(ctk.CTkScrollableFrame):
    """Complete history panel with recognition logging and filtering"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.history_data = []
        self.filtered_data = []

        # Initialize history table
        self.init_history_table()

        self.create_widgets()
        self.load_history()

    def init_history_table(self):
        """Create history table if it doesn't exist"""
        try:
            conn = sqlite3.connect(Config.DATABASE_PATH)
            cursor = conn.cursor()

            # Create history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_image_path TEXT,
                    result_type TEXT,
                    matched_face_id TEXT,
                    matched_name TEXT,
                    confidence REAL,
                    threshold_used REAL,
                    FOREIGN KEY (matched_face_id) REFERENCES people(face_id)
                )
            ''')

            # Create indices for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                ON recognition_history(timestamp)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_history_result_type 
                ON recognition_history(result_type)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_history_face_id 
                ON recognition_history(matched_face_id)
            ''')

            conn.commit()
            conn.close()

            print("✓ Recognition history table initialized")

        except Exception as e:
            print(f"Error initializing history table: {e}")

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="📜 Recognition History",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        subtitle = ctk.CTkLabel(
            self,
            text="View and analyze past face recognition attempts",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        subtitle.pack(pady=(0, Config.PADDING_MEDIUM))

        # ==================== FILTERS ====================
        filter_frame = ctk.CTkFrame(self, corner_radius=15)
        filter_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            filter_frame,
            text="🔍 Filters",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Filter controls container
        filter_controls = ctk.CTkFrame(filter_frame, fg_color="transparent")
        filter_controls.pack(fill="x", padx=Config.PADDING_MEDIUM,
                           pady=(0, Config.PADDING_MEDIUM))

        # Date range filter
        date_frame = ctk.CTkFrame(filter_controls, fg_color="transparent")
        date_frame.pack(fill="x", pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            date_frame,
            text="Date Range:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=100
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        # Date range options
        self.date_range_var = ctk.StringVar(value="all")

        date_options = [
            ("All Time", "all"),
            ("Today", "today"),
            ("Last 7 Days", "week"),
            ("Last 30 Days", "month")
        ]

        for text, value in date_options:
            ctk.CTkRadioButton(
                date_frame,
                text=text,
                variable=self.date_range_var,
                value=value,
                command=self.apply_filters
            ).pack(side="left", padx=Config.PADDING_TINY)

        # Result type filter
        result_frame = ctk.CTkFrame(filter_controls, fg_color="transparent")
        result_frame.pack(fill="x", pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            result_frame,
            text="Result Type:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=100
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        self.result_type_var = ctk.StringVar(value="all")

        result_options = [
            ("All", "all"),
            ("Recognized", "recognized"),
            ("Unknown", "unknown")
        ]

        for text, value in result_options:
            ctk.CTkRadioButton(
                result_frame,
                text=text,
                variable=self.result_type_var,
                value=value,
                command=self.apply_filters
            ).pack(side="left", padx=Config.PADDING_TINY)

        # Person filter
        person_frame = ctk.CTkFrame(filter_controls, fg_color="transparent")
        person_frame.pack(fill="x", pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            person_frame,
            text="Person:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=100
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        self.person_search = ctk.CTkEntry(
            person_frame,
            placeholder_text="Search by name or ID...",
            width=300
        )
        self.person_search.pack(side="left", padx=Config.PADDING_TINY)
        self.person_search.bind("<KeyRelease>", lambda e: self.apply_filters())

        ctk.CTkButton(
            person_frame,
            text="🔄 Refresh",
            command=self.load_history,
            width=100,
            height=30
        ).pack(side="right", padx=Config.PADDING_TINY)

        # Confidence range
        conf_frame = ctk.CTkFrame(filter_controls, fg_color="transparent")
        conf_frame.pack(fill="x", pady=Config.PADDING_TINY)

        ctk.CTkLabel(
            conf_frame,
            text="Min Confidence:",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=100
        ).pack(side="left", padx=(0, Config.PADDING_SMALL))

        self.min_conf_slider = ctk.CTkSlider(
            conf_frame,
            from_=0.0,
            to=1.0,
            width=300,
            command=self.update_conf_label
        )
        self.min_conf_slider.set(0.0)
        self.min_conf_slider.pack(side="left", padx=Config.PADDING_TINY)

        self.conf_label = ctk.CTkLabel(
            conf_frame,
            text="0%",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL, weight="bold"),
            width=50
        )
        self.conf_label.pack(side="left", padx=Config.PADDING_TINY)

        # ==================== STATISTICS ====================
        stats_frame = ctk.CTkFrame(self, corner_radius=15)
        stats_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            stats_frame,
            text="📊 Quick Statistics",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        self.stats_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
        self.stats_container.pack(fill="x", padx=Config.PADDING_MEDIUM,
                                 pady=(0, Config.PADDING_MEDIUM))

        # Stats labels
        self.total_label = ctk.CTkLabel(
            self.stats_container,
            text="Total: 0",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        )
        self.total_label.grid(row=0, column=0, padx=Config.PADDING_LARGE)

        self.recognized_label = ctk.CTkLabel(
            self.stats_container,
            text="Recognized: 0",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        )
        self.recognized_label.grid(row=0, column=1, padx=Config.PADDING_LARGE)

        self.unknown_label = ctk.CTkLabel(
            self.stats_container,
            text="Unknown: 0",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        )
        self.unknown_label.grid(row=0, column=2, padx=Config.PADDING_LARGE)

        self.avg_conf_label = ctk.CTkLabel(
            self.stats_container,
            text="Avg Confidence: 0%",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL)
        )
        self.avg_conf_label.grid(row=0, column=3, padx=Config.PADDING_LARGE)

        for i in range(4):
            self.stats_container.grid_columnconfigure(i, weight=1)

        # ==================== HISTORY TABLE ====================
        table_frame = ctk.CTkFrame(self, corner_radius=15)
        table_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                        pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            table_frame,
            text="📋 Recognition Log",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Scrollable table
        self.table_scroll = ctk.CTkScrollableFrame(table_frame)
        self.table_scroll.pack(fill="both", expand=True, padx=Config.PADDING_MEDIUM,
                              pady=(0, Config.PADDING_MEDIUM))

        # Placeholder
        self.table_placeholder = ctk.CTkLabel(
            self.table_scroll,
            text="Loading history...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.table_placeholder.pack(pady=50)

        # ==================== ACTIONS ====================
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(pady=Config.PADDING_MEDIUM)

        ctk.CTkButton(
            action_frame,
            text="📤 Export to CSV",
            command=self.export_csv,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1, weight="bold")
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            action_frame,
            text="📊 Export to Excel",
            command=self.export_excel,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_TINY)

        ctk.CTkButton(
            action_frame,
            text="🗑️ Clear Old Records",
            command=self.clear_old_records,
            width=150,
            height=Config.BUTTON_HEIGHT_MEDIUM,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1, weight="bold"),
            fg_color=Config.COLOR_ERROR,
            hover_color="darkred"
        ).pack(side="left", padx=Config.PADDING_TINY)

    def update_conf_label(self, value):
        """Update confidence label"""
        conf = float(value)
        self.conf_label.configure(text=f"{conf * 100:.0f}%")
        self.apply_filters()

    def load_history(self):
        """Load history from database"""
        # Show loading
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        loading = ctk.CTkLabel(
            self.table_scroll,
            text="🔄 Loading history...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL)
        )
        loading.pack(pady=50)

        def load_thread():
            try:
                conn = sqlite3.connect(Config.DATABASE_PATH)
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT history_id, timestamp, query_image_path, result_type,
                           matched_face_id, matched_name, confidence, threshold_used
                    FROM recognition_history
                    ORDER BY timestamp DESC
                ''')

                self.history_data = []
                for row in cursor.fetchall():
                    self.history_data.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'image_path': row[2],
                        'result_type': row[3],
                        'face_id': row[4],
                        'name': row[5],
                        'confidence': row[6],
                        'threshold': row[7]
                    })

                conn.close()

                self.filtered_data = self.history_data.copy()

                # Update UI
                self.after(0, self.display_history)
                self.after(0, lambda: self.toast.show_success(
                    f"Loaded {len(self.history_data)} history records",
                    duration=Config.TOAST_DURATION_SUCCESS
                ))

            except Exception as e:
                self.after(0, lambda err=str(e): self.toast.show_error(
                    f"Failed to load history: {err}",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def apply_filters(self):
        """Apply filters to history data"""
        if not self.history_data:
            return

        filtered = self.history_data.copy()

        # Date range filter
        date_range = self.date_range_var.get()
        if date_range != "all":
            now = datetime.now()

            if date_range == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_range == "week":
                cutoff = now - timedelta(days=7)
            elif date_range == "month":
                cutoff = now - timedelta(days=30)

            filtered = [
                h for h in filtered
                if datetime.fromisoformat(h['timestamp']) >= cutoff
            ]

        # Result type filter
        result_type = self.result_type_var.get()
        if result_type != "all":
            filtered = [h for h in filtered if h['result_type'] == result_type]

        # Person search
        search_term = self.person_search.get().strip().lower()
        if search_term:
            filtered = [
                h for h in filtered
                if (h['name'] and search_term in h['name'].lower()) or
                   (h['face_id'] and search_term in h['face_id'].lower())
            ]

        # Confidence filter
        min_conf = self.min_conf_slider.get()
        filtered = [h for h in filtered if h['confidence'] >= min_conf]

        self.filtered_data = filtered
        self.display_history()

    def display_history(self):
        """Display filtered history in table"""
        # Clear table
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        if not self.filtered_data:
            ctk.CTkLabel(
                self.table_scroll,
                text="No history records found",
                font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
                text_color=Config.COLOR_INFO_TEXT
            ).pack(pady=50)

            # Update stats
            self.update_stats()
            return

        # Update stats
        self.update_stats()

        # Display records
        for idx, record in enumerate(self.filtered_data):
            self.create_history_row(record, idx)

    def create_history_row(self, record, idx):
        """Create a row for history record"""
        # Determine color based on result
        if record['result_type'] == 'recognized':
            bg_color = (Config.HISTORY_ROW_RECOGNIZED_LIGHT,
                       Config.HISTORY_ROW_RECOGNIZED_DARK)
            result_color = Config.COLOR_SUCCESS_TEXT
            result_icon = "✅"
        else:
            bg_color = (Config.HISTORY_ROW_UNKNOWN_LIGHT,
                       Config.HISTORY_ROW_UNKNOWN_DARK)
            result_color = Config.COLOR_ERROR_TEXT
            result_icon = "❌"

        # Row frame
        row = ctk.CTkFrame(self.table_scroll, corner_radius=10, fg_color=bg_color)
        row.pack(fill="x", pady=Config.PADDING_TINY, padx=Config.PADDING_SMALL)

        # Content frame
        content = ctk.CTkFrame(row, fg_color="transparent")
        content.pack(fill="x", padx=Config.PADDING_MEDIUM, pady=Config.PADDING_SMALL)

        # Left side - Info
        info_frame = ctk.CTkFrame(content, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True)

        # Timestamp
        timestamp = datetime.fromisoformat(record['timestamp'])
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        ctk.CTkLabel(
            info_frame,
            text=f"🕒 {time_str}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY + 1),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(anchor="w")

        # Result
        if record['result_type'] == 'recognized':
            result_text = f"{result_icon} {record['name']} ({record['face_id']})"
        else:
            result_text = f"{result_icon} Unknown Person"

        ctk.CTkLabel(
            info_frame,
            text=result_text,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL + 1, weight="bold"),
            text_color=result_color
        ).pack(anchor="w", pady=2)

        # Image path (truncated)
        img_name = Path(record['image_path']).name if record['image_path'] else "N/A"
        ctk.CTkLabel(
            info_frame,
            text=f"📸 {img_name}",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack(anchor="w")

        # Right side - Metrics
        metrics_frame = ctk.CTkFrame(content, fg_color="transparent")
        metrics_frame.pack(side="right")

        # Confidence with color coding
        conf_pct = record['confidence'] * 100

        if conf_pct >= Config.HISTORY_CONFIDENCE_HIGH * 100:
            conf_color = Config.COLOR_SUCCESS_TEXT
        elif conf_pct >= Config.HISTORY_CONFIDENCE_MEDIUM * 100:
            conf_color = Config.COLOR_WARNING_TEXT
        else:
            conf_color = Config.COLOR_ERROR_TEXT

        conf_label = ctk.CTkLabel(
            metrics_frame,
            text=f"{conf_pct:.1f}%",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=conf_color
        )
        conf_label.pack()

        ctk.CTkLabel(
            metrics_frame,
            text="Confidence",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TINY),
            text_color=Config.COLOR_INFO_TEXT
        ).pack()

    def update_stats(self):
        """Update statistics labels"""
        if not self.filtered_data:
            self.total_label.configure(text="Total: 0")
            self.recognized_label.configure(text="Recognized: 0")
            self.unknown_label.configure(text="Unknown: 0")
            self.avg_conf_label.configure(text="Avg Confidence: 0%")
            return

        total = len(self.filtered_data)
        recognized = sum(1 for r in self.filtered_data if r['result_type'] == 'recognized')
        unknown = total - recognized

        confidences = [r['confidence'] for r in self.filtered_data]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        self.total_label.configure(text=f"Total: {total}")
        self.recognized_label.configure(text=f"Recognized: {recognized}")
        self.unknown_label.configure(text=f"Unknown: {unknown}")
        self.avg_conf_label.configure(text=f"Avg Confidence: {avg_conf * 100:.1f}%")

    def export_csv(self):
        """Export history to CSV"""
        if not self.filtered_data:
            self.toast.show_warning(
                "No history records to export!",
                duration=Config.TOAST_DURATION_WARNING
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Export History to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"recognition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if filename:
            try:
                df = pd.DataFrame(self.filtered_data)
                df.to_csv(filename, index=False)

                self.toast.show_success(
                    f"History exported to CSV successfully!",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            except Exception as e:
                self.toast.show_error(
                    f"Failed to export: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

    def export_excel(self):
        """Export history to Excel"""
        if not self.filtered_data:
            self.toast.show_warning(
                "No history records to export!",
                duration=Config.TOAST_DURATION_WARNING
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Export History to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"recognition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        if filename:
            try:
                df = pd.DataFrame(self.filtered_data)
                df.to_excel(filename, index=False, engine='openpyxl')

                self.toast.show_success(
                    "History exported to Excel successfully!",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            except Exception as e:
                self.toast.show_error(
                    f"Failed to export: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

    def clear_old_records(self):
        """Clear old history records with confirmation dialog"""
        # Create custom confirmation dialog
        confirm_dialog = ctk.CTkToplevel(self)
        confirm_dialog.title("⚠️ Clear Old Records")
        confirm_dialog.geometry("450x300")
        confirm_dialog.transient(self)
        confirm_dialog.grab_set()

        # Center the dialog
        confirm_dialog.update_idletasks()
        x = (confirm_dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (confirm_dialog.winfo_screenheight() // 2) - (300 // 2)
        confirm_dialog.geometry(f"450x300+{x}+{y}")

        # Warning icon and title
        ctk.CTkLabel(
            confirm_dialog,
            text="⚠️",
            font=ctk.CTkFont(size=60)
        ).pack(pady=(Config.PADDING_LARGE, Config.PADDING_SMALL))

        ctk.CTkLabel(
            confirm_dialog,
            text="CLEAR OLD RECORDS",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SECTION, weight="bold"),
            text_color=Config.COLOR_WARNING_TEXT
        ).pack(pady=(0, Config.PADDING_MEDIUM))

        # Warning message
        warning_frame = ctk.CTkFrame(confirm_dialog, corner_radius=10)
        warning_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        warning_text = f"""Delete all history records older than 
{Config.HISTORY_OLD_RECORDS_DAYS} days?

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
                cutoff = datetime.now() - timedelta(days=Config.HISTORY_OLD_RECORDS_DAYS)

                conn = sqlite3.connect(Config.DATABASE_PATH)
                cursor = conn.cursor()

                cursor.execute('''
                    DELETE FROM recognition_history
                    WHERE timestamp < ?
                ''', (cutoff.isoformat(),))

                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()

                self.toast.show_success(
                    f"Deleted {deleted_count} old records successfully",
                    duration=Config.TOAST_DURATION_SUCCESS
                )

                self.load_history()

            except Exception as e:
                self.toast.show_error(
                    f"Failed to clear records: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

        def on_cancel():
            confirm_dialog.destroy()
            self.toast.show_info(
                "Clear operation cancelled",
                duration=Config.TOAST_DURATION_INFO
            )

        ctk.CTkButton(
            btn_frame,
            text="🗑️ Yes, Clear",
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

    @staticmethod
    def log_recognition(db_path, image_path, result):
        """
        Static method to log a recognition attempt
        Call this from recognition_panel.py after recognition

        Args:
            db_path: Path to database
            image_path: Path to query image
            result: Recognition result dict
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO recognition_history 
                (query_image_path, result_type, matched_face_id, matched_name, 
                 confidence, threshold_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                image_path,
                'recognized' if result['recognized'] else 'unknown',
                result.get('face_id'),
                result['person_info']['name'] if result.get('person_info') else None,
                result['confidence'],
                result.get('threshold', Config.RECOGNITION_THRESHOLD)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error logging recognition: {e}")