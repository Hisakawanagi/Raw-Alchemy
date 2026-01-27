from PySide6.QtCore import Signal, QThread
import requests
from packaging import version

class VersionCheckWorker(QThread):
    """
    Check for new version from GitHub releases
    """
    version_checked = Signal(bool, str, str)  # has_update, latest_version, download_url

    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version

    def run(self):
        try:
            # GitHub API endpoint for latest release
            url = "https://api.github.com/repos/Alchemist-X/Raw-Alchemy/releases/latest"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                latest_tag = data.get('tag_name', '').lstrip('v')
                
                # Compare versions
                if version.parse(latest_tag) > version.parse(self.current_version):
                    html_url = data.get('html_url', '')
                    self.version_checked.emit(True, latest_tag, html_url)
                else:
                    self.version_checked.emit(False, latest_tag, "")
            else:
                self.version_checked.emit(False, "", "")
                
        except Exception:
            # Network error or other issues - silently fail
            self.version_checked.emit(False, "", "")
