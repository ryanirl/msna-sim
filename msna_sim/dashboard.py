from datetime import datetime
import pandas as pd
import numpy as np
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import RangeTool
from bokeh.models import Range1d
from bokeh.models import Toggle
from bokeh.models import Select
from bokeh.models import Slider
from bokeh.models import NumericInput
from bokeh.models import Button
from bokeh.models import Div
from bokeh.models import CheckboxGroup
from bokeh.models import TextInput
from bokeh.models.tools import HoverTool
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from tornado.ioloop import IOLoop

from msna_sim.simulator import Simulation
from msna_sim.config import SignalConfig
from msna_sim.config import create_preset_config

# Global plot parameters
MAIN_PLOT_WIDTH = 1200
MAIN_PLOT_HEIGHT = 400
RANGE_PLOT_WIDTH = 1200
RANGE_PLOT_HEIGHT = 150
WIDGET_WIDTH = 280
ADVANCED_WIDGET_WIDTH = 180
Y_RANGE_START = -1.0
Y_RANGE_END = 4.0
TRUE_BURST_Y_POS = 3.5


class MSNASimDashboard:
    def __init__(self):
        self.current_results = None
        self.current_config = None
        self.current_signal_config = SignalConfig()  # Advanced signal parameters
        self.preset_configs = self._load_preset_configs()
        
        # Initialize with default preset
        self.current_config = create_preset_config("normal_adult")
        self.duration = 120.0  # Default duration in seconds
        self.sampling_rate = 250  # Default sampling rate
        self.previous_duration = self.duration  # Track duration changes
        
        # Export history tracking
        self.export_history = []  # Store last few exports
        
        # Advanced features state
        self.advanced_features_enabled = False
        
        # Store default advanced settings for reset
        self.default_signal_config = SignalConfig()
        
        # Generate initial simulation
        self._generate_simulation()
    
    def _load_preset_configs(self):
        """Load all available preset configurations"""
        preset_names = [
            "normal_adult", "young_healthy", "athlete", "elderly_healthy",
            "hypertensive", "heart_failure", "bradycardia", "tachycardia",
            "diabetes", "obesity", "sleep_apnea", "copd", "anxiety", 
            "post_exercise", "pristine_lab", "noisy_clinical"
        ]
        
        configs = {}
        for name in preset_names:
            try:
                configs[name] = create_preset_config(name)
            except ValueError:
                continue
        return configs
    
    def _generate_simulation(self):
        """Generate MSNA simulation with current parameters"""
        simulation = Simulation(self.current_config, self.current_signal_config)
        self.current_results = simulation.simulate(
            duration = self.duration, 
            sampling_rate = int(self.sampling_rate),
            seed = 42  # Fixed seed for reproducibility during parameter exploration
        )
    
    def _get_safe_filename(self, filename):
        """Get a safe filename that doesn't overwrite existing files"""
        if not filename.endswith(".csv"):
            filename += ".csv"
        
        # If file doesn't exist, use original name
        if not os.path.exists(filename):
            return filename
        
        # Extract base name and extension
        base_name = filename[:-4]  # Remove .csv
        counter = 1
        
        # Keep incrementing until we find an available filename
        while True:
            new_filename = f"{base_name}_{counter}.csv"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1
    
    def _add_to_export_history(self, filename):
        """Add exported filename to history and maintain queue of 3"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {"filename": filename, "time": timestamp}
        
        # Add to beginning of list
        self.export_history.insert(0, entry)
        
        # Keep only last 3 exports
        if len(self.export_history) > 3:
            self.export_history = self.export_history[:3]
    
    def _get_export_history_text(self):
        """Generate HTML text for export history"""
        if not self.export_history:
            return ""
        
        history_text = "<br><b>Recent exports:</b><br>"
        for i, entry in enumerate(self.export_history):
            time_ago = "just now" if i == 0 else f"at {entry['time']}"
            history_text += f"• {entry['filename']} ({time_ago})<br>"
        
        return history_text
    
    def _create_data_sources(self):
        """Create Bokeh data sources from simulation results"""
        # Main data source for time series
        data_source = ColumnDataSource(data = dict(
            time = self.current_results.time,
            clean_msna = self.current_results.clean_msna,
            noisy_msna = self.current_results.noisy_msna,
            respiratory = self.current_results.respiratory_signal
        ))
        
        # Burst markers
        burst_times = self.current_results.get_burst_times()
        burst_y_pos = np.full(len(burst_times), TRUE_BURST_Y_POS)
        
        burst_source = ColumnDataSource(data = dict(
            time = burst_times, 
            y_pos = burst_y_pos
        ))
        
        return data_source, burst_source
    
    def _create_plots(self, data_source, burst_source):
        """Create the main and range plots"""
        # Main plot with vertical padding
        main_plot = figure(
            width = MAIN_PLOT_WIDTH, 
            height = MAIN_PLOT_HEIGHT,
            title = "MSNA Simulation - Interactive Parameter Explorer",
            x_axis_label = "Time (seconds)",
            tools = "pan,box_zoom,wheel_zoom,reset,save",
            output_backend = "webgl",
            margin = (25, 5, 5, 5)  # top, right, bottom, left padding
        )
        
        # Set y-range
        main_plot.y_range = Range1d(start = Y_RANGE_START, end = Y_RANGE_END)
        main_plot.yaxis.visible = False
        
        # Set initial x-range to show first 10 seconds
        max_time = self.current_results.time[-1]
        initial_end = min(60.0, max_time)
        main_plot.x_range = Range1d(start = 0, end = initial_end)
        
        # Plot signals
        clean_line = main_plot.line(
            "time", "clean_msna", 
            source = data_source,
            line_width = 2, 
            color = Category10[10][0], 
            alpha = 0.9,
            legend_label = "Clean MSNA"
        )
        
        noisy_line = main_plot.line(
            "time", "noisy_msna", 
            source = data_source,
            line_width = 1, 
            color = Category10[10][1], 
            alpha = 0.7,
            legend_label = "Noisy MSNA"
        )
        
        respiratory_line = main_plot.line(
            "time", "respiratory", 
            source = data_source,
            line_width = 1, 
            color = Category10[10][4], 
            alpha = 0.6,
            legend_label = "Respiratory"
        )
        
        # Burst markers
        burst_markers = main_plot.scatter(
            "time", "y_pos",
            source = burst_source,
            size = 8,
            marker = "circle",
            color = Category10[10][2],
            alpha = 0.8,
            legend_label = "Bursts"
        )
        
        # Add hover tools
        main_plot.add_tools(HoverTool(
            tooltips = [("Time", "@time{0.00}s"), ("Clean MSNA", "@clean_msna{0.000}")],
            renderers = [clean_line]
        ))
        
        # Legend
        main_plot.legend.click_policy = "hide"
        main_plot.legend.location = "top_left"
        
        # Range tool plot - disable pan/scroll interactions
        range_plot = figure(
            width = RANGE_PLOT_WIDTH, 
            height = RANGE_PLOT_HEIGHT,
            x_axis_label = "Time (seconds)",
            title = "Navigation Tool",
            tools = "reset"  # Only reset tool, no pan
        )
        
        range_plot.line("time", "noisy_msna", source = data_source, 
                       line_width = 1, color = Category10[10][1], alpha = 0.6)
        
        range_plot.x_range.start = 0
        range_plot.x_range.end = max_time
        range_plot.y_range = Range1d(start = Y_RANGE_START, end = Y_RANGE_END)
        range_plot.yaxis.visible = False
        
        # Range tool
        range_tool = RangeTool(x_range = main_plot.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2
        range_plot.add_tools(range_tool)
        
        return main_plot, range_plot
    
    def _create_widgets(self):
        """Create control widgets"""
        # Preset selection
        preset_select = Select(
            title = "Preset Configuration:", 
            value = "normal_adult",
            options = list(self.preset_configs.keys()),
            width = WIDGET_WIDTH
        )
        
        # Duration and sampling rate controls (side by side)
        duration_input = NumericInput(
            title = "Duration (seconds)", 
            value = self.duration,
            low = 5, 
            high = 600,
            width = WIDGET_WIDTH // 2 - 5
        )
        
        sampling_rate_input = NumericInput(
            title = "Sampling Rate (Hz)", 
            value = self.sampling_rate,
            low = 100, 
            high = 5000,
            width = WIDGET_WIDTH // 2 - 5
        )
        
        # Key physiological parameters
        heart_rate_input = NumericInput(
            title = "Heart Rate (bpm)", 
            value = int(self.current_config.heart_rate),
            low = 40, 
            high = 120, 
            width = WIDGET_WIDTH
        )
        
        burst_incidence_input = NumericInput(
            title = "Burst Incidence (%)", 
            value = int(self.current_config.burst_incidence),
            low = 0, 
            high = 100,
            width = WIDGET_WIDTH
        )
        
        resp_rate_input = NumericInput(
            title = "Respiratory Rate (breaths/min)", 
            value = int(self.current_config.resp_rate),
            low = 8, 
            high = 30,
            width = WIDGET_WIDTH
        )
        
        # Sliders for decimal values
        noise_floor_slider = Slider(
            title = "Noise Floor", 
            start = 0, 
            end = 1.0, 
            value = self.current_config.noise_floor,
            step = 0.01,
            width = WIDGET_WIDTH
        )
        
        signal_amplitude_slider = Slider(
            title = "Signal Amplitude", 
            start = 0, 
            end = 5.0, 
            value = self.current_config.signal_amplitude,
            step = 0.1,
            width = WIDGET_WIDTH
        )
        
        # File export controls
        filename_input = TextInput(
            title = "Export Filename:",
            value = "msna_simulation_export.csv",
            width = WIDGET_WIDTH
        )
        
        # Action buttons
        regenerate_button = Button(
            label = "Regenerate Signal", 
            button_type = "primary",
            width = WIDGET_WIDTH
        )
        
        export_button = Button(
            label = "Export to CSV", 
            button_type = "success",
            width = WIDGET_WIDTH
        )
        
        # Display options - respiratory off by default
        display_options = CheckboxGroup(
            labels = ["Show Clean MSNA", "Show Noisy MSNA", "Show Respiratory", "Show Bursts"],
            active = [1, 3],  # Noisy and Bursts (no respiratory or clean by default)
            width = WIDGET_WIDTH
        )
        
        # Status display
        status_div = Div(
            text = self._get_status_text(),
            width = WIDGET_WIDTH
        )
        
        # Export status with history
        export_status_div = Div(
            text = "<i>Ready to export...</i>",
            width = WIDGET_WIDTH
        )
        
        return {
            "preset_select": preset_select,
            "duration_input": duration_input,
            "sampling_rate_input": sampling_rate_input,
            "heart_rate_input": heart_rate_input,
            "burst_incidence_input": burst_incidence_input,
            "resp_rate_input": resp_rate_input,
            "noise_floor_slider": noise_floor_slider,
            "signal_amplitude_slider": signal_amplitude_slider,
            "filename_input": filename_input,
            "regenerate_button": regenerate_button,
            "export_button": export_button,
            "display_options": display_options,
            "status_div": status_div,
            "export_status_div": export_status_div
        }
    
    def _create_advanced_widgets(self):
        """Create advanced parameter widgets"""
        # Advanced features toggle
        advanced_toggle = Toggle(
            label = "Advanced Features", 
            active = False,
            button_type = "default",
            width = 150
        )
        
        # Advanced Patient Config parameters
        hrv_std_slider = Slider(
            title = "Heart Rate Variability (s)", 
            start = 0.01, 
            end = 0.15, 
            value = self.current_config.hrv_std,
            step = 0.005,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        resp_modulation_slider = Slider(
            title = "Respiratory Modulation", 
            start = 0, 
            end = 1.0, 
            value = self.current_config.resp_modulation_strength,
            step = 0.05,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        burst_delay_mean_slider = Slider(
            title = "Burst Delay Mean (s)", 
            start = 0.8, 
            end = 2.0, 
            value = self.current_config.burst_delay_mean,
            step = 0.05,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        burst_delay_std_slider = Slider(
            title = "Burst Delay Std (s)", 
            start = 0.05, 
            end = 0.3, 
            value = self.current_config.burst_delay_std,
            step = 0.01,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        burst_duration_mean_slider = Slider(
            title = "Burst Duration Mean (s)", 
            start = 0.3, 
            end = 0.8, 
            value = self.current_config.burst_duration_mean,
            step = 0.05,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        burst_duration_std_slider = Slider(
            title = "Burst Duration Std (s)", 
            start = 0.05, 
            end = 0.2, 
            value = self.current_config.burst_duration_std,
            step = 0.01,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        # Advanced Signal Config parameters
        integration_smoothing_slider = Slider(
            title = "Integration Smoothing", 
            start = 0.8, 
            end = 0.99, 
            value = self.current_signal_config.integration_smoothing,
            step = 0.01,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        phase_noise_slider = Slider(
            title = "Phase Noise Amplitude", 
            start = 0, 
            end = 0.5, 
            value = self.current_signal_config.phase_noise_amplitude,
            step = 0.01,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        breathing_irregularity_slider = Slider(
            title = "Breathing Irregularity", 
            start = 0, 
            end = 0.01, 
            value = self.current_signal_config.breathing_irregularity,
            step = 0.0005,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        burst_gaussian_sigma_slider = Slider(
            title = "Burst Shape (σ)", 
            start = 0.3, 
            end = 2.0, 
            value = self.current_signal_config.burst_gaussian_sigma,
            step = 0.1,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        pink_noise_slider = Slider(
            title = "Pink Noise Amplitude", 
            start = 0, 
            end = 1.0, 
            value = self.current_signal_config.pink_noise_amplitude,
            step = 0.05,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        spike_amplitude_slider = Slider(
            title = "Spike Artifact Amplitude", 
            start = 0, 
            end = 1.0, 
            value = self.current_signal_config.spike_artifact_amplitude,
            step = 0.05,
            width = ADVANCED_WIDGET_WIDTH
        )
        
        # Reset button for advanced settings
        reset_advanced_button = Button(
            label = "Reset Advanced Settings", 
            button_type = "light",
            width = 200
        )
        
        return {
            "advanced_toggle": advanced_toggle,
            "hrv_std_slider": hrv_std_slider,
            "resp_modulation_slider": resp_modulation_slider,
            "burst_delay_mean_slider": burst_delay_mean_slider,
            "burst_delay_std_slider": burst_delay_std_slider,
            "burst_duration_mean_slider": burst_duration_mean_slider,
            "burst_duration_std_slider": burst_duration_std_slider,
            "integration_smoothing_slider": integration_smoothing_slider,
            "phase_noise_slider": phase_noise_slider,
            "breathing_irregularity_slider": breathing_irregularity_slider,
            "burst_gaussian_sigma_slider": burst_gaussian_sigma_slider,
            "pink_noise_slider": pink_noise_slider,
            "spike_amplitude_slider": spike_amplitude_slider,
            "reset_advanced_button": reset_advanced_button
        }
    
    def _get_status_text(self):
        """Generate status text with simulation info"""
        if self.current_results is None:
            return "<b>Status:</b> No simulation loaded"
        
        return f"""
        <b>Simulation Status:</b><br>
        Duration: {self.current_results.duration:.1f}s<br>
        Bursts: {self.current_results.n_bursts}<br>
        Burst Rate: {self.current_results.burst_rate:.1f}/min<br>
        Burst Incidence: {self.current_results.actual_burst_incidence:.1f}%<br>
        Heart Rate: {self.current_results.mean_heart_rate:.1f} bpm
        """
    
    def _setup_callbacks(self, widgets, advanced_widgets, data_source, burst_source, main_plot, range_plot, advanced_section):
        """Setup widget callbacks"""
        
        def update_from_preset():
            preset_name = widgets["preset_select"].value
            self.current_config = create_preset_config(preset_name)
            
            # Update basic widget values
            widgets["heart_rate_input"].value = int(self.current_config.heart_rate)
            widgets["burst_incidence_input"].value = int(self.current_config.burst_incidence)
            widgets["resp_rate_input"].value = int(self.current_config.resp_rate)
            widgets["noise_floor_slider"].value = self.current_config.noise_floor
            widgets["signal_amplitude_slider"].value = self.current_config.signal_amplitude
            
            # Update advanced widget values
            advanced_widgets["hrv_std_slider"].value = self.current_config.hrv_std
            advanced_widgets["resp_modulation_slider"].value = self.current_config.resp_modulation_strength
            advanced_widgets["burst_delay_mean_slider"].value = self.current_config.burst_delay_mean
            advanced_widgets["burst_delay_std_slider"].value = self.current_config.burst_delay_std
            advanced_widgets["burst_duration_mean_slider"].value = self.current_config.burst_duration_mean
            advanced_widgets["burst_duration_std_slider"].value = self.current_config.burst_duration_std
            
            regenerate_simulation()
        
        def update_config_from_widgets():
            # Basic parameters
            self.current_config.heart_rate = float(widgets["heart_rate_input"].value)
            self.current_config.burst_incidence = float(widgets["burst_incidence_input"].value)
            self.current_config.resp_rate = float(widgets["resp_rate_input"].value)
            self.current_config.noise_floor = float(widgets["noise_floor_slider"].value)
            self.current_config.signal_amplitude = float(widgets["signal_amplitude_slider"].value)
            self.duration = float(widgets["duration_input"].value)
            self.sampling_rate = float(widgets["sampling_rate_input"].value)
            
            # Advanced patient parameters
            self.current_config.hrv_std = float(advanced_widgets["hrv_std_slider"].value)
            self.current_config.resp_modulation_strength = float(advanced_widgets["resp_modulation_slider"].value)
            self.current_config.burst_delay_mean = float(advanced_widgets["burst_delay_mean_slider"].value)
            self.current_config.burst_delay_std = float(advanced_widgets["burst_delay_std_slider"].value)
            self.current_config.burst_duration_mean = float(advanced_widgets["burst_duration_mean_slider"].value)
            self.current_config.burst_duration_std = float(advanced_widgets["burst_duration_std_slider"].value)
            
            # Advanced signal parameters
            self.current_signal_config.integration_smoothing = float(advanced_widgets["integration_smoothing_slider"].value)
            self.current_signal_config.phase_noise_amplitude = float(advanced_widgets["phase_noise_slider"].value)
            self.current_signal_config.breathing_irregularity = float(advanced_widgets["breathing_irregularity_slider"].value)
            self.current_signal_config.burst_gaussian_sigma = float(advanced_widgets["burst_gaussian_sigma_slider"].value)
            self.current_signal_config.pink_noise_amplitude = float(advanced_widgets["pink_noise_slider"].value)
            self.current_signal_config.spike_artifact_amplitude = float(advanced_widgets["spike_amplitude_slider"].value)
        
        def regenerate_simulation():
            try:
                update_config_from_widgets()
                
                # Check if duration changed
                duration_changed = abs(self.duration - self.previous_duration) > 0.1
                
                self._generate_simulation()
                
                # Update data sources
                new_data_source, new_burst_source = self._create_data_sources()
                data_source.data.update(new_data_source.data)
                burst_source.data.update(new_burst_source.data)
                
                # Only update plot range if duration actually changed
                if duration_changed:
                    max_time = self.current_results.time[-1]
                    range_plot.x_range.end = max_time
                    self.previous_duration = self.duration
                
                # Update status
                widgets["status_div"].text = self._get_status_text()
                widgets["export_status_div"].text = "<i>Ready to export...</i>" + self._get_export_history_text()
                
            except Exception as e:
                widgets["export_status_div"].text = f"<b style='color:red'>Error:</b> {str(e)}"
        
        def toggle_advanced_features(_ = None):
            self.advanced_features_enabled = advanced_widgets["advanced_toggle"].active
            advanced_section.visible = self.advanced_features_enabled
        
        def reset_advanced_settings(_ = None):
            """Reset all advanced settings to defaults"""
            # Reset signal config to defaults
            self.current_signal_config = SignalConfig()
            
            # Update advanced widgets to default values
            advanced_widgets["integration_smoothing_slider"].value = self.default_signal_config.integration_smoothing
            advanced_widgets["phase_noise_slider"].value = self.default_signal_config.phase_noise_amplitude
            advanced_widgets["breathing_irregularity_slider"].value = self.default_signal_config.breathing_irregularity
            advanced_widgets["burst_gaussian_sigma_slider"].value = self.default_signal_config.burst_gaussian_sigma
            advanced_widgets["pink_noise_slider"].value = self.default_signal_config.pink_noise_amplitude
            advanced_widgets["spike_amplitude_slider"].value = self.default_signal_config.spike_artifact_amplitude
            
            # Reset patient config advanced parameters to current preset defaults
            preset_config = create_preset_config(widgets["preset_select"].value)
            advanced_widgets["hrv_std_slider"].value = preset_config.hrv_std
            advanced_widgets["resp_modulation_slider"].value = preset_config.resp_modulation_strength
            advanced_widgets["burst_delay_mean_slider"].value = preset_config.burst_delay_mean
            advanced_widgets["burst_delay_std_slider"].value = preset_config.burst_delay_std
            advanced_widgets["burst_duration_mean_slider"].value = preset_config.burst_duration_mean
            advanced_widgets["burst_duration_std_slider"].value = preset_config.burst_duration_std
            
            # Regenerate with reset values
            regenerate_simulation()
        
        def export_data(_ = None):
            if self.current_results is None:
                widgets["export_status_div"].text = "<b style='color:red'>Error:</b> No simulation data to export"
                return
                
            try:
                # Get safe filename that won't overwrite
                requested_filename = widgets["filename_input"].value
                safe_filename = self._get_safe_filename(requested_filename)
                
                # Create DataFrame with specified column names
                df = pd.DataFrame({
                    "time": self.current_results.time,
                    "Integrated MSNA": self.current_results.noisy_msna,
                    "Clean MSNA": self.current_results.clean_msna
                })
                
                # Add burst information
                burst_times = self.current_results.get_burst_times()
                df["Burst"] = 0
                for burst_time in burst_times:
                    # Find closest time index
                    idx = np.argmin(np.abs(self.current_results.time - burst_time))
                    if idx < len(df):
                        df.loc[idx, "Burst"] = 1
                
                # Save to CSV
                df.to_csv(safe_filename, index = False)
                
                # Add to export history
                self._add_to_export_history(safe_filename)
                
                # Update status with success message and history
                success_msg = f"<b style='color:green'>Success:</b> Data exported to {safe_filename}"
                if safe_filename != requested_filename:
                    success_msg += f"<br><i>(Auto-renamed to prevent overwrite)</i>"
                
                widgets["export_status_div"].text = success_msg + self._get_export_history_text()
                
            except Exception as e:
                widgets["export_status_div"].text = f"<b style='color:red'>Error:</b> Failed to export - {str(e)}"
        
        def update_display():
            active_displays = widgets["display_options"].active
            
            # Toggle line visibility based on checkboxes
            main_plot.renderers[0].visible = 0 in active_displays  # Clean MSNA
            main_plot.renderers[1].visible = 1 in active_displays  # Noisy MSNA  
            main_plot.renderers[2].visible = 2 in active_displays  # Respiratory
            main_plot.renderers[3].visible = 3 in active_displays  # Bursts
        
        # Connect callbacks
        widgets["preset_select"].on_change("value", lambda attr, old, new: update_from_preset())
        widgets["regenerate_button"].on_click(regenerate_simulation)
        widgets["export_button"].on_click(export_data)
        widgets["display_options"].on_change("active", lambda attr, old, new: update_display())
        
        # Advanced feature callbacks
        advanced_widgets["advanced_toggle"].on_click(toggle_advanced_features)
        advanced_widgets["reset_advanced_button"].on_click(reset_advanced_settings)
        
        # Apply initial display settings to sync with checkbox state
        update_display()
    
    def create_dashboard(self, doc):
        """Create the complete dashboard"""
        # Generate initial data sources
        data_source, burst_source = self._create_data_sources()
        
        # Create plots
        main_plot, range_plot = self._create_plots(data_source, burst_source)
        
        # Create widgets
        widgets = self._create_widgets()
        advanced_widgets = self._create_advanced_widgets()
        
        # Create advanced features section (initially hidden)
        advanced_section = column(
            Div(text = "<h4>Advanced Parameters</h4>"),
            row(
                column(
                    Div(text = "<b>Timing & Variability</b>"),
                    advanced_widgets["hrv_std_slider"],
                    advanced_widgets["resp_modulation_slider"],
                    advanced_widgets["burst_delay_mean_slider"],
                    width = ADVANCED_WIDGET_WIDTH + 20
                ),
                column(
                    Div(text = "<b>Burst Characteristics</b>"),
                    advanced_widgets["burst_delay_std_slider"],
                    advanced_widgets["burst_duration_mean_slider"],
                    advanced_widgets["burst_duration_std_slider"],
                    width = ADVANCED_WIDGET_WIDTH + 20
                ),
                column(
                    Div(text = "<b>Signal Processing</b>"),
                    advanced_widgets["integration_smoothing_slider"],
                    advanced_widgets["phase_noise_slider"],
                    advanced_widgets["breathing_irregularity_slider"],
                    width = ADVANCED_WIDGET_WIDTH + 20
                ),
                column(
                    Div(text = "<b>Signal Shape & Noise</b>"),
                    advanced_widgets["burst_gaussian_sigma_slider"],
                    advanced_widgets["pink_noise_slider"],
                    advanced_widgets["spike_amplitude_slider"],
                    width = ADVANCED_WIDGET_WIDTH + 20
                )
            ),
            row(advanced_widgets["reset_advanced_button"]),
            visible = False,  # Initially hidden
            width = MAIN_PLOT_WIDTH
        )
        
        # Setup callbacks
        self._setup_callbacks(widgets, advanced_widgets, data_source, burst_source, main_plot, range_plot, advanced_section)
        
        # Create layout
        controls = column(
            Div(text = "<h3>MSNA Simulation Controls</h3>"),
            widgets["preset_select"],
            Div(text = "<h4>Recording Parameters</h4>"),
            row(widgets["duration_input"], widgets["sampling_rate_input"]),
            Div(text = "<h4>Physiological Parameters</h4>"),
            widgets["heart_rate_input"],
            widgets["burst_incidence_input"],
            widgets["resp_rate_input"],
            widgets["noise_floor_slider"],
            widgets["signal_amplitude_slider"],
            Div(text = "<h4>Actions</h4>"),
            widgets["regenerate_button"],
            Div(text = "<h4>Export Data</h4>"),
            widgets["filename_input"],
            widgets["export_button"],
            widgets["export_status_div"],
            Div(text = "<h4>Display Options</h4>"),
            widgets["display_options"],
            Div(text = "<h4>Simulation Info</h4>"),
            widgets["status_div"],
            width = WIDGET_WIDTH + 50
        )
        
        plots_and_advanced = column(
            main_plot, 
            range_plot,
            row(advanced_widgets["advanced_toggle"]),
            advanced_section
        )
        
        dashboard_layout = row(controls, plots_and_advanced)
        
        # Add to document
        doc.add_root(dashboard_layout)
        doc.title = "MSNA Simulation Dashboard"


def main():
    """Main function to start the dashboard"""
    dashboard = MSNASimDashboard()
    
    def modify_doc(doc):
        dashboard.create_dashboard(doc)
    
    bokeh_app = Application(FunctionHandler(modify_doc))
    server = Server(
        applications = {"/": bokeh_app},
        io_loop = IOLoop(),
        allow_websocket_origin = ["localhost:5006"]
    )
    server.start()
    
    print("MSNA Simulation Dashboard running at: http://localhost:5006/")
    print("Controls:")
    print("- Select presets or adjust parameters manually")
    print("- Click 'Regenerate Signal' to apply changes")
    print("- Toggle 'Advanced Features' for detailed parameter control")
    print("- Use 'Reset Advanced Settings' to restore defaults")
    print("- Specify filename and use 'Export to CSV' to save current simulation")
    print("- Toggle display options to show/hide different signals")
    print("- Files are auto-renamed to prevent overwrites (file.csv → file_1.csv)")
    
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def main_cli():
    """CLI entry point for the MSNA simulation dashboard."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("Please check that all dependencies are installed correctly.")
        print("Try: pip install --upgrade msna-sim")
        return 1
    return 0


if __name__ == "__main__":
    main_cli()


