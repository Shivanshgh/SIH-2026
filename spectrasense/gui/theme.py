"""
SpectraSense GUI Theme & Styling
Clean, serious, intelligence-grade dark theme for PyQt6 desktop application.
Adheres to strict anti-slop rules: dark matte backgrounds, crisp typography,
refined borders, and tactical status indicators (cyan, amber, emerald, ruby).
"""

DARK_STYLESHEET = """
QMainWindow {
    background-color: #0f141c;
    color: #e2e8f0;
}

QWidget {
    background-color: #0f141c;
    color: #e2e8f0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

/* Menu Bar */
QMenuBar {
    background-color: #161d28;
    color: #cbd5e1;
    border-bottom: 1px solid #232d3d;
    padding: 2px 4px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background: #232d3d;
    color: #38bdf8;
}
QMenu {
    background-color: #161d28;
    color: #cbd5e1;
    border: 1px solid #2d3748;
    padding: 4px 0px;
}
QMenu::item {
    padding: 6px 24px;
}
QMenu::item:selected {
    background-color: #232d3d;
    color: #38bdf8;
}

/* Sidebar & Panels */
QFrame#sidebarPanel {
    background-color: #141b24;
    border-right: 1px solid #232d3d;
}
QFrame#headerPanel {
    background-color: #161d28;
    border-bottom: 1px solid #232d3d;
}
QFrame#cardPanel {
    background-color: #161e2a;
    border: 1px solid #232d3d;
    border-radius: 6px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #232d3d;
    background-color: #121822;
}
QTabBar::tab {
    background-color: #141b24;
    color: #94a3b8;
    padding: 8px 18px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #232d3d;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1a2332;
    color: #38bdf8;
    font-weight: bold;
    border-top: 2px solid #00d2ff;
}
QTabBar::tab:hover:!selected {
    background-color: #1a2230;
    color: #cbd5e1;
}

/* Push Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #273549;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton#primaryButton {
    background-color: #0369a1;
    border-color: #0284c7;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #0284c7;
}
QPushButton#accentButton {
    background-color: #065f46;
    border-color: #059669;
    color: #ffffff;
}
QPushButton#accentButton:hover {
    background-color: #059669;
}
QPushButton#refineButton {
    background-color: #92400e;
    border-color: #b45309;
    color: #ffffff;
}
QPushButton#refineButton:hover {
    background-color: #b45309;
}

/* Tables & Lists */
QTableWidget, QTreeWidget, QListWidget {
    background-color: #121822;
    color: #e2e8f0;
    border: 1px solid #232d3d;
    gridline-color: #1e293b;
    selection-background-color: #1e293b;
    selection-color: #38bdf8;
}
QHeaderView::section {
    background-color: #161e2a;
    color: #94a3b8;
    padding: 6px;
    border: 1px solid #232d3d;
    font-weight: 600;
}

/* Labels & Status */
QLabel#sectionHeader {
    font-size: 11px;
    font-weight: bold;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
}
QLabel#metricValue {
    font-family: "Consolas", "SF Mono", monospace;
    font-size: 16px;
    font-weight: bold;
    color: #38bdf8;
}
QLabel#statusBadgeValidated {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
}
QLabel#statusBadgeRefine {
    background-color: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
}
QLabel#statusBadgeUnresolved {
    background-color: #450a0a;
    color: #f87171;
    border: 1px solid #dc2626;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
}

/* Status Bar */
QStatusBar {
    background-color: #0a0d13;
    color: #64748b;
    border-top: 1px solid #1e293b;
    font-size: 11px;
}

/* Sliders and Inputs */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #161e2a;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox::drop-down {
    border: none;
}
"""
