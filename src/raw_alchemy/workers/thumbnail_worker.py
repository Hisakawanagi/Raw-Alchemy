from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QImage, QTransform
import os
import concurrent.futures
from loguru import logger
import rawpy

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS

class ThumbnailWorker(QThread):
    """
    Scan folder and generate thumbnails - 优化版本使用线程池
    """
    # Define signals
    thumbnail_ready = Signal(str, QImage)
    progress_update = Signal(int, int)
    finished_scanning = Signal()

    def __init__(self, folder_path, max_workers=4):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False
        self.max_workers = max_workers

    @staticmethod
    def extract_thumbnail(full_path):
        """
        静态方法用于线程池并行处理
        """
        try:
            # 1. 快速检查文件扩展名
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in SUPPORTED_RAW_EXTENSIONS:
                return None

            image = None
            orientation = 0

            # 2. 尝试使用 rawpy 提取缩略图 (最快)
            try:
                with rawpy.imread(full_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        # Get orientation
                        # 0: no rotation
                        # 3: 180 deg
                        # 5: 90 deg CCW (270 deg CW) - standard portrait
                        # 6: 90 deg CW
                        orientation = raw.sizes.flip
                    except rawpy.LibRawNoThumbnailError:
                        thumb = None
                    
                    if thumb:
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            image = QImage.fromData(thumb.data)
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            # 处理未压缩的缩略图
                            image = QImage(thumb.data, thumb.width, thumb.height, QImage.Format_RGB888)
                    else:
                        # Fallback: calculate thumbnail from raw data if extraction fails
                        try:
                            # use_camera_wb=True: 使用相机白平衡
                            # half_size=True: 半尺寸解码 (速度快)
                            # user_flip=None: 自动根据元数据旋转
                            thumb_array = raw.postprocess(
                                use_camera_wb=True,
                                bright=1.0,
                                user_sat=None,
                                no_auto_bright=False,
                                half_size=True,
                                user_flip=None
                            )
                            
                            if thumb_array is not None:
                                height, width, channel = thumb_array.shape
                                # Create QImage from data. Must use copy() as data is from local numpy array
                                image = QImage(
                                    thumb_array.data, 
                                    width, 
                                    height, 
                                    3 * width, 
                                    QImage.Format_RGB888
                                ).copy()
                                # Ensure orientation is 0 since postprocess(user_flip=None) already handles rotation
                                orientation = 0
                        except Exception:
                            pass
            except Exception:
                pass

            # 3. 缩略图处理失败，不做任何操作，直接返回None            
            # 4. 统一缩放 & 旋转
            if image and not image.isNull():
                # Apply rotation based on orientation
                if orientation == 3:
                   image = image.transformed(QTransform().rotate(180))
                elif orientation == 5:
                   image = image.transformed(QTransform().rotate(-90))
                elif orientation == 6:
                   image = image.transformed(QTransform().rotate(90))

                # 统一缩放为 300px 高度，保持比例
                return image.scaledToHeight(300)
            
            return None

        except Exception as e:
            # logger.error(f"Error extracting thumbnail for {full_path}: {e}")
            return None

    def run(self):
        # 1. Scan folder
        valid_extensions = SUPPORTED_RAW_EXTENSIONS
        files = []
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if self.stopped:
                        return
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_extensions:
                        files.append(entry.path)
        except Exception as e:
            logger.error(f"Failed to scan directory: {e}")
            self.finished_scanning.emit()
            return

        total = len(files)
        # self.progress_update.emit(0, total) # Optional startup signal

        # 2. Parallel processing using ThreadPoolExecutor
        # 注意: QThread.run 是在独立线程中，我们可以开启一个在这里wait的线程池
        # 或者直接使用 max_workers 限制并发
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.extract_thumbnail, f): f for f in files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                if self.stopped:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

                f_path = future_to_file[future]
                try:
                    qimg = future.result()
                    if qimg:
                        self.thumbnail_ready.emit(f_path, qimg)
                except Exception as e:
                    logger.error(f"Worker exception for {f_path}: {e}")
                
                processed_count += 1
                if processed_count % 5 == 0 or processed_count == total:
                     self.progress_update.emit(processed_count, total)

        self.finished_scanning.emit()

    def stop(self):
        self.stopped = True
