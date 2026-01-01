"""
Statistics Panel - Complete Implementation with Visualizations
Feature #11: Database Statistics Dashboard
"""

import customtkinter as ctk
from tkinter import filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from collections import Counter
import threading

from config import Config
from utils.toast import ToastManager


class StatsPanel(ctk.CTkScrollableFrame):
    """Complete statistics panel with charts and detailed analytics"""

    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.toast = ToastManager(self)

        self.stats_data = None
        self.create_widgets()
        self.load_stats()

    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="📊 Database Statistics & Analytics",
            font=ctk.CTkFont(size=Config.FONT_SIZE_TITLE, weight="bold")
        )
        title.pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        subtitle = ctk.CTkLabel(
            self,
            text="Comprehensive overview of your face recognition database",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        subtitle.pack(pady=(0, Config.PADDING_LARGE))

        # ==================== KEY METRICS CARDS ====================
        metrics_container = ctk.CTkFrame(self, fg_color="transparent")
        metrics_container.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        # People count card
        self.people_card = self.create_metric_card(
            metrics_container,
            "👥",
            "0",
            "Total People",
            Config.COLOR_SUCCESS
        )
        self.people_card.grid(row=0, column=0, padx=Config.PADDING_SMALL,
                             pady=Config.PADDING_TINY, sticky="ew")

        # Embeddings count card
        self.emb_card = self.create_metric_card(
            metrics_container,
            "📸",
            "0",
            "Total Embeddings",
            "blue"
        )
        self.emb_card.grid(row=0, column=1, padx=Config.PADDING_SMALL,
                          pady=Config.PADDING_TINY, sticky="ew")

        # Average embeddings card
        self.avg_card = self.create_metric_card(
            metrics_container,
            "📊",
            "0.0",
            "Avg per Person",
            "purple"
        )
        self.avg_card.grid(row=0, column=2, padx=Config.PADDING_SMALL,
                          pady=Config.PADDING_TINY, sticky="ew")

        # Departments count card
        self.dept_card = self.create_metric_card(
            metrics_container,
            "🏢",
            "0",
            "Departments",
            Config.COLOR_WARNING
        )
        self.dept_card.grid(row=0, column=3, padx=Config.PADDING_SMALL,
                           pady=Config.PADDING_TINY, sticky="ew")

        # Configure grid weights
        for i in range(4):
            metrics_container.grid_columnconfigure(i, weight=1)

        # ==================== CHARTS ====================
        charts_frame = ctk.CTkFrame(self, corner_radius=15)
        charts_frame.pack(fill="both", expand=True, padx=Config.PADDING_LARGE,
                         pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            charts_frame,
            text="📈 Visual Analytics",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Chart container
        self.chart_container = ctk.CTkFrame(charts_frame, fg_color="transparent")
        self.chart_container.pack(fill="both", expand=True, padx=Config.PADDING_MEDIUM,
                                 pady=(0, Config.PADDING_MEDIUM))

        # Placeholder
        self.chart_placeholder = ctk.CTkLabel(
            self.chart_container,
            text="Loading charts...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL),
            text_color=Config.COLOR_INFO_TEXT
        )
        self.chart_placeholder.pack(pady=50)

        # ==================== DETAILED STATISTICS ====================
        details_frame = ctk.CTkFrame(self, corner_radius=15)
        details_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            details_frame,
            text="📋 Detailed Statistics",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        self.details_text = ctk.CTkTextbox(
            details_frame,
            height=400,
            font=ctk.CTkFont(family="Courier", size=Config.FONT_SIZE_TINY + 1),
            wrap="none"
        )
        self.details_text.pack(fill="both", expand=True, padx=Config.PADDING_MEDIUM,
                              pady=(0, Config.PADDING_MEDIUM))

        # ==================== TOP PERFORMERS ====================
        top_frame = ctk.CTkFrame(self, corner_radius=15)
        top_frame.pack(fill="x", padx=Config.PADDING_LARGE, pady=Config.PADDING_SMALL)

        ctk.CTkLabel(
            top_frame,
            text="🏆 Top Records",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LARGE, weight="bold")
        ).pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_SMALL))

        # Top people with most embeddings
        top_container = ctk.CTkFrame(top_frame, fg_color="transparent")
        top_container.pack(fill="x", padx=Config.PADDING_MEDIUM,
                          pady=(0, Config.PADDING_MEDIUM))

        # Most embeddings
        most_emb_frame = ctk.CTkFrame(top_container, corner_radius=10)
        most_emb_frame.grid(row=0, column=0, padx=Config.PADDING_SMALL,
                           pady=Config.PADDING_TINY, sticky="nsew")

        ctk.CTkLabel(
            most_emb_frame,
            text="📸 Most Embeddings",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        self.most_emb_label = ctk.CTkLabel(
            most_emb_frame,
            text="Loading...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            justify="left"
        )
        self.most_emb_label.pack(pady=(0, Config.PADDING_SMALL), padx=Config.PADDING_SMALL)

        # Most common department
        common_dept_frame = ctk.CTkFrame(top_container, corner_radius=10)
        common_dept_frame.grid(row=0, column=1, padx=Config.PADDING_SMALL,
                              pady=Config.PADDING_TINY, sticky="nsew")

        ctk.CTkLabel(
            common_dept_frame,
            text="🏢 Largest Department",
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(pady=(Config.PADDING_SMALL, Config.PADDING_TINY))

        self.common_dept_label = ctk.CTkLabel(
            common_dept_frame,
            text="Loading...",
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            justify="left"
        )
        self.common_dept_label.pack(pady=(0, Config.PADDING_SMALL), padx=Config.PADDING_SMALL)

        top_container.grid_columnconfigure(0, weight=1)
        top_container.grid_columnconfigure(1, weight=1)

        # ==================== ACTIONS ====================
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(pady=Config.PADDING_LARGE)

        ctk.CTkButton(
            action_frame,
            text="🔄 Refresh Statistics",
            command=self.load_stats,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color=Config.COLOR_SUCCESS,
            hover_color=Config.COLOR_SUCCESS_HOVER
        ).pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkButton(
            action_frame,
            text="📊 Export Report",
            command=self.export_report,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold")
        ).pack(side="left", padx=Config.PADDING_SMALL)

        ctk.CTkButton(
            action_frame,
            text="📸 Save Charts",
            command=self.save_charts,
            width=180,
            height=45,
            font=ctk.CTkFont(size=Config.FONT_SIZE_LABEL, weight="bold"),
            fg_color="purple",
            hover_color="darkviolet"
        ).pack(side="left", padx=Config.PADDING_SMALL)

    def create_metric_card(self, parent, icon, value, label, color):
        """Create a metric display card"""
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=color, height=140)
        card.grid_propagate(False)

        # Icon
        icon_label = ctk.CTkLabel(
            card,
            text=icon,
            font=ctk.CTkFont(size=36)
        )
        icon_label.pack(pady=(Config.PADDING_MEDIUM, Config.PADDING_TINY))

        # Value
        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="white"
        )
        value_label.pack()

        # Label
        text_label = ctk.CTkLabel(
            card,
            text=label,
            font=ctk.CTkFont(size=Config.FONT_SIZE_SMALL),
            text_color="white"
        )
        text_label.pack(pady=(0, Config.PADDING_MEDIUM))

        # Store references
        card.value_label = value_label

        return card

    def load_stats(self):
        """Load statistics from database"""
        # Show loading
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", "🔄 Loading statistics...")

        self.toast.show_info(
            "Loading statistics...",
            duration=Config.TOAST_DURATION_INFO
        )

        def load_thread():
            try:
                # Get basic stats
                basic_stats = self.app.db.get_stats()

                # Get all people for detailed analysis
                face_ids = self.app.db.get_all_face_ids()

                people_data = []
                emb_counts = []
                dept_list = []

                for face_id in face_ids:
                    person = self.app.db.get_person_info(face_id)
                    embeddings = self.app.db.get_embeddings_by_face_id(face_id)

                    emb_count = len(embeddings)
                    emb_counts.append(emb_count)

                    dept = person.get('department', 'Unknown')
                    if dept:
                        dept_list.append(dept)

                    people_data.append({
                        'face_id': face_id,
                        'name': person['name'],
                        'department': dept,
                        'embedding_count': emb_count
                    })

                # Prepare stats data
                self.stats_data = {
                    'basic': basic_stats,
                    'people': people_data,
                    'emb_counts': emb_counts,
                    'dept_list': dept_list
                }

                # Update UI
                self.after(0, self.display_stats)
                self.after(0, lambda: self.toast.show_success(
                    "Statistics loaded successfully",
                    duration=Config.TOAST_DURATION_SUCCESS
                ))

            except Exception as e:
                self.after(0, lambda err=str(e): self.toast.show_error(
                    f"Failed to load statistics: {err}",
                    duration=Config.TOAST_DURATION_ERROR
                ))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def display_stats(self):
        """Display loaded statistics"""
        if not self.stats_data:
            return

        basic = self.stats_data['basic']
        people = self.stats_data['people']
        emb_counts = self.stats_data['emb_counts']
        dept_list = self.stats_data['dept_list']

        # Update metric cards
        self.people_card.value_label.configure(text=str(basic['total_people']))
        self.emb_card.value_label.configure(text=str(basic['total_embeddings']))
        self.avg_card.value_label.configure(text=f"{basic['avg_embeddings_per_person']:.1f}")

        # Count unique departments
        unique_depts = len(set(dept_list)) if dept_list else 0
        self.dept_card.value_label.configure(text=str(unique_depts))

        # Create charts
        self.create_charts()

        # Update detailed text
        self.update_detailed_text()

        # Update top performers
        self.update_top_performers()

    def create_charts(self):
        """Create visualization charts"""
        # Clear placeholder
        if hasattr(self, 'chart_placeholder'):
            self.chart_placeholder.destroy()

        # Clear existing charts
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        if not self.stats_data:
            return

        emb_counts = self.stats_data['emb_counts']
        dept_list = self.stats_data['dept_list']

        # Create matplotlib figure with 2 subplots
        fig = Figure(figsize=(12, 5), facecolor='#2b2b2b')

        # ========== Chart 1: Embeddings Distribution ==========
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor('#2b2b2b')

        if emb_counts:
            # Histogram
            ax1.hist(emb_counts, bins=20, color='#3b82f6', edgecolor='white', alpha=0.7)
            ax1.set_xlabel('Number of Embeddings', color='white', fontsize=10)
            ax1.set_ylabel('Number of People', color='white', fontsize=10)
            ax1.set_title('Embeddings Distribution per Person', color='white', fontsize=12, pad=10)
            ax1.tick_params(colors='white', labelsize=8)
            ax1.grid(True, alpha=0.2, color='white')

            # Add statistics text
            stats_text = f'Min: {min(emb_counts)}\nMax: {max(emb_counts)}\nAvg: {np.mean(emb_counts):.1f}'
            ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='#3b82f6', alpha=0.5),
                     color='white', fontsize=9)

        # ========== Chart 2: Department Distribution ==========
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor('#2b2b2b')

        if dept_list:
            # Count departments
            dept_counts = Counter(dept_list)

            # Get top 8 departments
            top_depts = dept_counts.most_common(8)
            labels = [d[0] if d[0] else 'Unknown' for d in top_depts]
            sizes = [d[1] for d in top_depts]

            # Pie chart
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444',
                      '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'color': 'white', 'fontsize': 9}
            )

            ax2.set_title('People by Department (Top 8)', color='white', fontsize=12, pad=10)

        fig.tight_layout(pad=2)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Store figure for saving
        self.current_figure = fig

    def update_detailed_text(self):
        """Update detailed statistics text"""
        if not self.stats_data:
            return

        basic = self.stats_data['basic']
        people = self.stats_data['people']
        emb_counts = self.stats_data['emb_counts']
        dept_list = self.stats_data['dept_list']

        # Build report
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    DATABASE STATISTICS REPORT                        ║
╚══════════════════════════════════════════════════════════════════════╝

📊 OVERVIEW
════════════════════════════════════════════════════════════════════════
Total Registered People:        {basic['total_people']:>6}
Total Face Embeddings:          {basic['total_embeddings']:>6}
Average Embeddings per Person:  {basic['avg_embeddings_per_person']:>6.2f}
Unique Departments:             {len(set(dept_list)) if dept_list else 0:>6}

📈 EMBEDDING STATISTICS
════════════════════════════════════════════════════════════════════════
"""

        if emb_counts:
            report += f"""Minimum Embeddings:             {min(emb_counts):>6}
Maximum Embeddings:             {max(emb_counts):>6}
Median Embeddings:              {np.median(emb_counts):>6.1f}
Standard Deviation:             {np.std(emb_counts):>6.2f}

Embedding Distribution:
"""
            # Distribution ranges
            ranges = [
                (1, 2, "1-2 embeddings"),
                (3, 5, "3-5 embeddings"),
                (6, 10, "6-10 embeddings"),
                (11, float('inf'), "11+ embeddings")
            ]

            for low, high, label in ranges:
                count = sum(1 for e in emb_counts if low <= e < high)
                pct = (count / len(emb_counts) * 100) if emb_counts else 0
                bar = "█" * int(pct / 2)
                report += f"  {label:20s}: {count:>4} ({pct:>5.1f}%) {bar}\n"

        report += f"""
🏢 DEPARTMENT BREAKDOWN
════════════════════════════════════════════════════════════════════════
"""

        if dept_list:
            dept_counts = Counter(dept_list)
            for dept, count in sorted(dept_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(dept_list) * 100) if dept_list else 0
                dept_name = dept if dept else "Unknown"
                report += f"{dept_name:30s}: {count:>4} ({pct:>5.1f}%)\n"
        else:
            report += "No department data available\n"

        report += f"""
⚙️ SYSTEM CONFIGURATION
════════════════════════════════════════════════════════════════════════
Recognition Threshold:          {self.app.recognizer.threshold:>6.2f}
Device:                         {str(Config.DEVICE):>15}
Model Embedding Dimension:      {Config.EMBEDDING_DIM:>6}D
Face Detection Mode:            {Config.FACE_CROP_MODE:>15}
Database Path:                  {Config.DATABASE_PATH}

🕒 REPORT METADATA
════════════════════════════════════════════════════════════════════════
Generated:                      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Version:                 1.0
"""

        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", report.strip())

    def update_top_performers(self):
        """Update top performers section"""
        if not self.stats_data:
            return

        people = self.stats_data['people']
        dept_list = self.stats_data['dept_list']

        # Most embeddings
        if people:
            top_people = sorted(people, key=lambda x: x['embedding_count'], reverse=True)[:5]

            text = ""
            for i, person in enumerate(top_people, 1):
                text += f"{i}. {person['name']}\n"
                text += f"   {person['embedding_count']} embeddings\n"
                if i < len(top_people):
                    text += "\n"

            self.most_emb_label.configure(text=text.strip() if text else "No data")

        # Most common department
        if dept_list:
            dept_counts = Counter(dept_list)
            top_dept = dept_counts.most_common(1)[0]

            total = len(dept_list)
            pct = (top_dept[1] / total * 100) if total > 0 else 0

            text = f"{top_dept[0] if top_dept[0] else 'Unknown'}\n"
            text += f"{top_dept[1]} people ({pct:.1f}%)"

            self.common_dept_label.configure(text=text)
        else:
            self.common_dept_label.configure(text="No data")

    def export_report(self):
        """Export statistics report to file"""
        if not self.stats_data:
            self.toast.show_warning(
                "No statistics to export!",
                duration=Config.TOAST_DURATION_WARNING
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Export Statistics Report",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile=f"statistics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        if filename:
            try:
                report_text = self.details_text.get("1.0", "end")

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_text)

                self.toast.show_success(
                    "Statistics report exported successfully!",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            except Exception as e:
                self.toast.show_error(
                    f"Failed to export: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )

    def save_charts(self):
        """Save charts as image"""
        if not hasattr(self, 'current_figure'):
            self.toast.show_warning(
                "No charts to save!",
                duration=Config.TOAST_DURATION_WARNING
            )
            return

        filename = filedialog.asksaveasfilename(
            title="Save Charts",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ],
            initialfile=f"statistics_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')

                self.toast.show_success(
                    "Charts saved successfully!",
                    duration=Config.TOAST_DURATION_SUCCESS
                )
            except Exception as e:
                self.toast.show_error(
                    f"Failed to save: {str(e)}",
                    duration=Config.TOAST_DURATION_ERROR
                )