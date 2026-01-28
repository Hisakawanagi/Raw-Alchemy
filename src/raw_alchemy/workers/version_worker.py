from PySide6.QtCore import Signal, QThread
import requests
from packaging import version
from loguru import logger

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
            url = "https://api.github.com/repos/shenmintao/Raw-Alchemy/releases/latest"
            logger.info(f"Checking for updates from {url}")
            logger.info(f"Current version: {self.current_version}")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                latest_tag = data.get('tag_name', '')
                logger.info(f"Latest version from GitHub: {latest_tag}")
                
                if not latest_tag:
                    logger.warning("No tag_name found in GitHub response")
                    self.version_checked.emit(False, "", "")
                    return
                
                # Remove 'v' prefix and 'studio-' prefix if present
                latest_version = latest_tag.lstrip('v').replace('studio-v', '').replace('studio-', '')
                logger.info(f"Parsed latest version: {latest_version}")
                
                # Compare versions
                try:
                    current_ver = version.parse(self.current_version)
                    latest_ver = version.parse(latest_version)
                    
                    if latest_ver > current_ver:
                        html_url = data.get('html_url', '')
                        logger.info(f"New version available: {latest_version}, URL: {html_url}")
                        self.version_checked.emit(True, latest_version, html_url)
                    else:
                        logger.info("Already running the latest version")
                        self.version_checked.emit(False, latest_version, "")
                except Exception as e:
                    logger.error(f"Version comparison failed: {e}")
                    self.version_checked.emit(False, "", "")
            else:
                logger.warning(f"GitHub API returned status code: {response.status_code}")
                self.version_checked.emit(False, "", "")
                
        except requests.exceptions.Timeout:
            logger.error("Version check timed out")
            self.version_checked.emit(False, "", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during version check: {e}")
            self.version_checked.emit(False, "", "")
        except Exception as e:
            logger.error(f"Unexpected error during version check: {e}")
            self.version_checked.emit(False, "", "")
