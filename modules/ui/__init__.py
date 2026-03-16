from .components import UiComponents, styled_alert

section_divider = UiComponents.section_divider
active_file_scan_progress_bar = UiComponents.active_file_scan_progress_bar
page_header = UiComponents.page_header
overview_header = UiComponents.overview_header
workspace_status = UiComponents.workspace_status
file_inventory = UiComponents.file_inventory
preview_panel = UiComponents.preview_panel
feature_navigation = UiComponents.feature_navigation
sidebar_branding = UiComponents.sidebar_branding
footer = UiComponents.footer
audit_metric = UiComponents.audit_metric   # backward-compat alias → metric_card
metric_card  = UiComponents.metric_card
scan_animation = UiComponents.scan_animation
login_form = UiComponents.login_form
sidebar_user_info = UiComponents.sidebar_user_info
comparison_charts = UiComponents.comparison_charts
method_selectbox = UiComponents.method_selectbox
sidebar_ai_chat = UiComponents.sidebar_ai_chat
render_scrubber_tab = UiComponents.render_scrubber_tab
render_missing_and_dupes_tab = UiComponents.render_missing_and_dupes_tab
render_outlier_tab = UiComponents.render_outlier_tab
pipeline_card = UiComponents.pipeline_card
pipeline_done_banner = UiComponents.pipeline_done_banner
detail_analysis_header = UiComponents.detail_analysis_header

from .styles import apply_style
from .dialogs import upload_dialog, profile_dialog
