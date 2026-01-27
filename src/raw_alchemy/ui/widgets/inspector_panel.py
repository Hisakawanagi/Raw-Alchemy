import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from qfluentwidgets import (
    ScrollArea, SimpleCardWidget, StrongBodyLabel, BodyLabel, 
    SwitchButton, ComboBox, Slider, LineEdit, ToolButton, 
    PushButton, InfoBar, FluentIcon as FIF
)
from raw_alchemy import config, lensfun_wrapper
from raw_alchemy.i18n import tr
from raw_alchemy.ui.widgets.histogram import HistogramWidget
from raw_alchemy.ui.widgets.waveform import WaveformWidget

class InspectorPanel(ScrollArea):
    """Right side control panel"""
    param_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget()
        self.view.setObjectName("view")
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        self.view.setStyleSheet("#view { background-color: transparent; }")
        
        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setSpacing(20)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        
        # 保存的基准参数
        self.saved_baseline_params = None
        
        # 保存各模式的EV值
        self.manual_ev_value = 0.0  # 手动模式的EV
        self.auto_ev_value = 0.0    # 自动模式计算的EV（只读）
        
        # --- Histogram / Waveform with Switch ---
        self.hist_widget = HistogramWidget()
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.hide()  # Initially hidden
        
        # Create a container for the display mode switch
        display_mode_card = SimpleCardWidget()
        display_mode_layout = QVBoxLayout(display_mode_card)
        display_mode_layout.setSpacing(5)
        
        # Switch button for histogram/waveform
        self.display_mode_switch = SwitchButton()
        self.display_mode_switch.setChecked(False)  # False = Histogram, True = Waveform
        self.display_mode_switch.checkedChanged.connect(self._on_display_mode_changed)
        self._update_display_mode_switch_text()
        
        display_mode_layout.addWidget(self.display_mode_switch)
        
        self.add_section(tr('histogram_waveform'), display_mode_card)
        self.v_layout.addWidget(self.hist_widget)
        self.v_layout.addWidget(self.waveform_widget)

        # --- Exposure ---
        self.exp_card = SimpleCardWidget()
        exp_layout = QVBoxLayout(self.exp_card)
        
        self.auto_exp_radio = SwitchButton()
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.auto_exp_radio.checkedChanged.connect(self._on_exposure_mode_changed)
        self._update_exposure_switch_text()
        
        self.metering_lbl = BodyLabel(tr('metering_mode'))
        self.metering_combo = ComboBox()
        # Store metering mode mapping: display text -> internal key
        self.metering_mode_map = {
            tr('matrix'): 'matrix',
            tr('average'): 'average',
            tr('center_weighted'): 'center-weighted',
            tr('highlight_safe'): 'highlight-safe',
            tr('hybrid'): 'hybrid'
        }
        # Reverse mapping: internal key -> display text
        self.metering_mode_reverse_map = {v: k for k, v in self.metering_mode_map.items()}
        
        self.metering_combo.addItems([tr('matrix'), tr('average'), tr('center_weighted'), tr('highlight_safe'), tr('hybrid')])
        self.metering_combo.setCurrentText(tr('matrix'))
        self.metering_combo.currentTextChanged.connect(self._on_param_change)
        
        self.exp_slider = Slider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-100, 100) # -10.0 to 10.0
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        
        # Add exposure value label
        self.exp_value_label = BodyLabel(tr('exposure_ev') + ": 0.0")
        
        def update_exp_label(val):
            """Update label and trigger debounced parameter change"""
            real_val = val / 10.0
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {real_val:+.1f}")
            # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
            self._on_param_change()
        
        # 保存回调函数引用，以便后续临时断开连接
        self.exp_slider_callback = update_exp_label
        self.exp_slider.valueChanged.connect(update_exp_label)
        
        exp_layout.addWidget(self.auto_exp_radio)
        exp_layout.addWidget(self.metering_lbl)
        exp_layout.addWidget(self.metering_combo)
        exp_layout.addWidget(self.exp_value_label)
        exp_layout.addWidget(self.exp_slider)
        
        self._update_exposure_ui_state()
        
        self.add_section(tr('exposure'), self.exp_card)
        
        # --- Color / Log ---
        self.color_card = SimpleCardWidget()
        color_layout = QVBoxLayout(self.color_card)
        
        # Log Space
        color_layout.addWidget(BodyLabel(tr('log_space')))
        self.log_combo = ComboBox()
        # Store log space mapping: display text -> internal key
        self.log_space_map = {tr('none'): 'None'}
        # Map actual log space names to themselves
        for log_name in config.LOG_TO_WORKING_SPACE.keys():
            self.log_space_map[log_name] = log_name
        # Reverse mapping: internal key -> display text
        self.log_space_reverse_map = {v: k for k, v in self.log_space_map.items()}
        
        log_items = [tr('none')] + list(config.LOG_TO_WORKING_SPACE.keys())
        self.log_combo.addItems(log_items)
        self.log_combo.setCurrentText(tr('none'))
        self.log_combo.currentTextChanged.connect(self._on_param_change)
        color_layout.addWidget(self.log_combo)
        
        # LUT
        color_layout.addWidget(BodyLabel(tr('lut')))
        lut_layout = QHBoxLayout()
        self.lut_combo = ComboBox()
        self.lut_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lut_combo.addItem(tr('none'))
        self.lut_combo.currentTextChanged.connect(self._on_param_change)
        
        self.lut_btn = ToolButton(FIF.FOLDER)
        self.lut_btn.clicked.connect(self._browse_lut_folder)
        
        lut_layout.addWidget(self.lut_combo, 1)
        lut_layout.addWidget(self.lut_btn)
        color_layout.addLayout(lut_layout)
        
        self.add_section(tr('color_management'), self.color_card)
        
        # --- Lens Correction ---
        self.lens_card = SimpleCardWidget()
        lens_layout = QVBoxLayout(self.lens_card)
        
        self.lens_correct_switch = SwitchButton(text=tr('enable_lens_correction'))
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.lens_correct_switch.checkedChanged.connect(self._on_param_change)
        lens_layout.addWidget(self.lens_correct_switch)
        
        # Custom Lensfun DB
        lens_layout.addWidget(BodyLabel(tr('custom_lensfun_db')))
        db_layout = QHBoxLayout()
        self.db_path_edit = LineEdit()
        self.db_path_edit.setPlaceholderText(tr('optional_db_path'))
        self.db_path_edit.setReadOnly(True)
        self.db_path_edit.textChanged.connect(self._on_param_change)
        
        self.db_browse_btn = ToolButton(FIF.FOLDER)
        self.db_browse_btn.clicked.connect(self._browse_lensfun_db)
        
        self.db_clear_btn = ToolButton(FIF.CLOSE)
        self.db_clear_btn.clicked.connect(self._clear_lensfun_db)
        
        db_layout.addWidget(self.db_path_edit, 1)
        db_layout.addWidget(self.db_browse_btn)
        db_layout.addWidget(self.db_clear_btn)
        lens_layout.addLayout(db_layout)
        
        self.add_section(tr('lens_correction'), self.lens_card)
        
        # --- Adjustments ---
        self.adj_card = SimpleCardWidget()
        adj_layout = QVBoxLayout(self.adj_card)
        
        self.sliders = {}
        self.slider_labels = {}  # 存储标签引用以便更新
        
        def add_slider(key, name, min_v, max_v, default_v, scale=1.0):
            layout = QVBoxLayout()
            lbl = BodyLabel(f"{name}: {default_v}")
            slider = Slider(Qt.Orientation.Horizontal)
            # slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(int(min_v*scale), int(max_v*scale))
            slider.setValue(int(default_v*scale))
            
            def update_lbl(val):
                """Update label and trigger debounced parameter change"""
                real_val = val / scale
                lbl.setText(f"{name}: {real_val:.2f}")
                # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
                self._on_param_change()
            
            slider.valueChanged.connect(update_lbl)
                
            layout.addWidget(lbl)
            layout.addWidget(slider)
            adj_layout.addLayout(layout)
            self.sliders[key] = (slider, scale, default_v, name)  # 添加 name 到元组
            self.slider_labels[key] = lbl  # 存储标签引用

        add_slider('wb_temp', tr('temp'), -100, 100, 0, 1)
        add_slider('wb_tint', tr('tint'), -100, 100, 0, 1)
        add_slider('saturation', tr('saturation'), 0, 3, 1.25, 100)
        add_slider('contrast', tr('contrast'), 0, 3, 1.1, 100)
        add_slider('highlight', tr('highlights'), -100, 100, 0, 1)
        add_slider('shadow', tr('shadows'), -100, 100, 0, 1)
        
        # 按钮布局：保存参数和Reset ALL并排
        btn_layout = QHBoxLayout()
        self.save_baseline_btn = PushButton(tr('save_baseline'))
        self.save_baseline_btn.clicked.connect(self.save_baseline_params)
        self.reset_btn = PushButton(tr('reset_all'))
        self.reset_btn.setToolTip(tr('long_press_reset'))
        # 使用事件过滤器实现长按检测
        self._reset_press_timer = QTimer(self)
        self._reset_press_timer.setSingleShot(True)
        self._reset_press_timer.timeout.connect(self._on_long_press_reset)
        self._reset_long_pressed = False
        self.reset_btn.installEventFilter(self)
        btn_layout.addWidget(self.save_baseline_btn)
        btn_layout.addWidget(self.reset_btn)
        adj_layout.addLayout(btn_layout)
        
        self.add_section(tr('adjustments'), self.adj_card)
        
        # Filler
        self.v_layout.addStretch()
        
        self.lut_folder = None

    def set_params(self, params):
        """Update UI controls from params dict"""
        if not params: return
        
        self.blockSignals(True) # Pause signals to avoid triggering processing loops
        
        # Exposure
        if 'exposure_mode' in params:
            self.auto_exp_radio.setChecked(params['exposure_mode'] == 'Auto')
        
        if 'metering_mode' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.metering_mode_reverse_map.get(params['metering_mode'], tr('matrix'))
            self.metering_combo.setCurrentText(display_text)

        self._update_exposure_ui_state()

        if 'exposure' in params:
            exp_val = params['exposure']
            self.exp_slider.setValue(int(exp_val * 10))
            self.exp_slider.update()
            # Update the exposure value label
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {exp_val:+.1f}")
            
        # Color
        if 'log_space' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.log_space_reverse_map.get(params['log_space'], tr('none'))
            self.log_combo.setCurrentText(display_text)
        
        # LUT (Path reconstruction logic needed if we only store path)
        # Assuming lut_path is full path
        if 'lut_path' in params and params['lut_path']:
            lut_name = os.path.basename(params['lut_path'])
            idx = self.lut_combo.findText(lut_name)
            if idx >= 0:
                self.lut_combo.setCurrentIndex(idx)
            else:
                 # Maybe LUT folder changed? For now set to None or handle gracefully
                 pass
        else:
            self.lut_combo.setCurrentIndex(0)
        
        # Lens Correction
        if 'lens_correct' in params:
            self.lens_correct_switch.setChecked(params['lens_correct'])
        
        if 'custom_db_path' in params and params['custom_db_path']:
            self.db_path_edit.setText(params['custom_db_path'])
        else:
            self.db_path_edit.clear()
            
        # Sliders
        for key, (slider, scale, _, name) in self.sliders.items():
            if key in params:
                slider.setValue(int(params[key] * scale))
                # 更新标签文本
                if key in self.slider_labels:
                    real_val = params[key]
                    self.slider_labels[key].setText(f"{name}: {real_val:.2f}")
                
        self.blockSignals(False)

    def add_section(self, title, widget):
        self.v_layout.addWidget(StrongBodyLabel(title))
        self.v_layout.addWidget(widget)

    def _browse_lut_folder(self):
        # Get the main window to access last_lut_folder_path
        main_window = self.window()
        
        # Use last LUT folder path, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lut_folder_path') and main_window.last_lut_folder_path and os.path.exists(main_window.last_lut_folder_path):
            start_dir = main_window.last_lut_folder_path
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        folder = QFileDialog.getExistingDirectory(self, tr('select_lut_folder'), start_dir)
        if folder:
            self.lut_folder = folder
            # Remember this path in main window
            if hasattr(main_window, 'last_lut_folder_path'):
                main_window.last_lut_folder_path = folder
            self.refresh_lut_list()

    def refresh_lut_list(self):
        if not self.lut_folder: return
        self.lut_combo.clear()
        self.lut_combo.addItem(tr('none'))
        files = sorted([f for f in os.listdir(self.lut_folder) if f.lower().endswith('.cube')])
        self.lut_combo.addItems(files)
    
    def _browse_lensfun_db(self):
        # Get the main window to access last_lensfun_db_path
        main_window = self.window()
        
        # Use last Lensfun DB path's directory, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lensfun_db_path') and main_window.last_lensfun_db_path and os.path.exists(main_window.last_lensfun_db_path):
            start_dir = os.path.dirname(main_window.last_lensfun_db_path)
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr('select_lensfun_db'),
            start_dir,
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.db_path_edit.setText(file_path)
            # Remember this path in main window
            if hasattr(main_window, 'last_lensfun_db_path'):
                main_window.last_lensfun_db_path = file_path
            # 重新加载lensfun数据库
            try:
                lensfun_wrapper.reload_lensfun_database(custom_db_path=file_path)
                InfoBar.success(tr('db_loaded'), tr('using_custom_db', name=os.path.basename(file_path)), parent=self)
            except Exception as e:
                InfoBar.error(tr('db_load_failed'), tr('failed_to_load_db', error=str(e)), parent=self)
                self.db_path_edit.clear()
    
    def _clear_lensfun_db(self):
        self.db_path_edit.clear()
        # 重新加载默认lensfun数据库
        try:
            lensfun_wrapper.reload_lensfun_database(custom_db_path=None)
            InfoBar.info(tr('db_cleared'), tr('using_default_db'), parent=self)
        except Exception as e:
            InfoBar.warning(tr('db_cleared'), f"Warning: {str(e)}", parent=self)

    def _update_display_mode_switch_text(self):
        """Update the display mode switch button text based on its state"""
        if self.display_mode_switch.isChecked():
            self.display_mode_switch.setText(tr('waveform'))
        else:
            self.display_mode_switch.setText(tr('histogram'))
    
    def _on_display_mode_changed(self):
        """Handle display mode switch between histogram and waveform"""
        is_waveform = self.display_mode_switch.isChecked()
        
        if is_waveform:
            # Switch to waveform
            self.hist_widget.hide()
            self.waveform_widget.show()
        else:
            # Switch to histogram
            self.waveform_widget.hide()
            self.hist_widget.show()
        
        self._update_display_mode_switch_text()

    def _update_exposure_switch_text(self):
        """Update the switch button text based on its state"""
        if self.auto_exp_radio.isChecked():
            self.auto_exp_radio.setText(tr('auto_exposure'))
        else:
            self.auto_exp_radio.setText(tr('manual_exposure'))
    
    def _update_exposure_ui_state(self):
        is_auto = self.auto_exp_radio.isChecked()
        self.metering_combo.setEnabled(is_auto)
        self.metering_lbl.setEnabled(is_auto)
        self.exp_slider.setEnabled(not is_auto)
        self._update_exposure_switch_text()

    def _on_exposure_mode_changed(self):
        is_auto = self.auto_exp_radio.isChecked()
        
        # Update UI state FIRST (enable/disable controls)
        self._update_exposure_ui_state()
        
        if is_auto:
            # Switching to auto mode: save current manual value
            self.manual_ev_value = self.exp_slider.value() / 10.0
            # Display auto mode EV
            self.exp_slider.setValue(int(self.auto_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.auto_ev_value:+.1f}")
        else:
            # Switching to manual mode: save current auto value, restore manual value
            self.auto_ev_value = self.exp_slider.value() / 10.0
            self.exp_slider.setValue(int(self.manual_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.manual_ev_value:+.1f}")
        
        self._on_param_change()

    def _on_param_change(self):
        self.param_changed.emit(self.get_params())
        
    def get_params(self):
        """Get parameters from UI to send to processor"""
        is_auto = self.auto_exp_radio.isChecked()
        
        params = {
            'exposure_mode': 'Auto' if is_auto else 'Manual',
            'metering_mode': self.metering_mode_map.get(self.metering_combo.currentText(), 'matrix'),
            # Always pass the slider value as 'exposure' - in auto mode this is visualized but respected if passed?
            # Actually, in auto mode, the processor calculates gain. The slider shows it.
            # But if we send 'exposure', it might override?
            # Looking at ImageProcessor:
            # if params.get('exposure_mode') == 'Manual': use 'exposure'
            # else: use auto exposure
            'exposure': self.exp_slider.value() / 10.0,
            
            'log_space': self.log_space_map.get(self.log_combo.currentText(), 'None'),
            'lut_path': os.path.join(self.lut_folder, self.lut_combo.currentText()) if self.lut_folder and self.lut_combo.currentText() != tr('none') else None,
            
            'lens_correct': self.lens_correct_switch.isChecked(),
            'custom_db_path': self.db_path_edit.text() if self.db_path_edit.text() else None
        }
        
        # Add sliders
        for key, (slider, scale, _, _) in self.sliders.items():
            params[key] = slider.value() / scale
            
        return params

    def save_baseline_params(self):
        """保存当前参数作为基准点"""
        self.saved_baseline_params = self.get_params().copy()
        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)

    def reset_adjustments(self):
        """重置到保存的基准点，如果没有保存则重置到默认值"""
        if self.saved_baseline_params:
            # 重置到保存的基准点
            self.set_params(self.saved_baseline_params)
            self._on_param_change()
        else:
            # 重置到默认值
            for key, (slider, scale, default, name) in self.sliders.items():
                slider.setValue(int(default * scale))
            self._on_param_change()

    def reset_params(self):
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.metering_combo.setCurrentText(tr('matrix'))
        self._update_exposure_ui_state()
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        self.exp_value_label.setText(tr('exposure_ev') + ": 0.0")
        self.log_combo.setCurrentText(tr('none'))
        self.lut_combo.setCurrentIndex(0)
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.db_path_edit.clear()
        
        # Reset sliders
        for key, (slider, scale, default, name) in self.sliders.items():
            slider.setValue(int(default * scale))
            # 更新标签
            if key in self.slider_labels:
                self.slider_labels[key].setText(f"{name}: {default:.2f}")
        
        # Clear saved baseline
        self.saved_baseline_params = None
        # Trigger parameter change
        self._on_param_change()
    
    def eventFilter(self, obj, event):
        """Event filter for long-press detection on reset button"""
        # Guard against calls during initialization before reset_btn exists
        if hasattr(self, 'reset_btn') and obj == self.reset_btn:
            if event.type() == QEvent.MouseButtonPress:
                self._reset_long_pressed = False
                self._reset_press_timer.start(1000)  # 1 second for long press
                return False  # Don't consume the event
            elif event.type() == QEvent.MouseButtonRelease:
                self._reset_press_timer.stop()
                if not self._reset_long_pressed:
                    # Short press: reset to baseline
                    self.reset_adjustments()
                return True  # Consume the event to prevent clicked signal
        return super().eventFilter(obj, event)
    
    def _on_long_press_reset(self):
        """Handle long press on reset button - reset to initial defaults"""
        self._reset_long_pressed = True
        self.reset_params()
        InfoBar.success(tr('reset_to_default'), tr('reset_to_default_message'), parent=self)
