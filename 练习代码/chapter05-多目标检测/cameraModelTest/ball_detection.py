import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional


class DetectionResult:
    def __init__(self):
        self.position = (0, 0)  # (u, v) - 检测框底部中点坐标
        self.confidence = 0.0
        self.bbox = (0, 0, 0, 0)  # (x, y, width, height) - 完整边界框信息


class SimpleFootballDetector:
    def __init__(self, model_path: str, confidence_threshold=0.25, nms_threshold=0.4):
        self.model_path = model_path
        self.confidence_ = confidence_threshold
        self.nms_threshold_ = nms_threshold

        print(f"正在加载ONNX模型: {model_path}")

        # 初始化ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")

    def inference(self, img: np.ndarray) -> Optional[DetectionResult]:
        # 预处理图像
        input_img = self.preprocess(img)

        # 运行推理
        print("正在进行推理...")
        outputs = self.session.run([self.output_name], {self.input_name: input_img})
        print(f"推理完成! 输出形状: {outputs[0].shape}")

        # 后处理 - 只返回足球检测结果
        detection = self.postprocess(outputs[0], img.shape)
        return detection

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # 调整大小并归一化
        input_size = (640, 640)  # YOLOv8默认输入尺寸
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0

        # 转换通道顺序: HWC to NCHW
        input_img = np.transpose(normalized, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        return input_img

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> Optional[DetectionResult]:
        orig_h, orig_w = orig_shape[:2]
        input_size = 640
        output_data = outputs[0]

        football_detections = []

        # 遍历所有检测框，只收集足球检测结果（类别0）
        for i in range(output_data.shape[1]):
            detection = output_data[:, i]
            conf = detection[4]  # 置信度

            if conf > self.confidence_:
                x_center, y_center, w, h = detection[0:4]
                class_probs = detection[5:]
                class_id = np.argmax(class_probs)

                # 只处理足球类别（类别0）
                if class_id == 0:
                    # 坐标转换
                    x_center_orig = (x_center / input_size) * orig_w
                    y_center_orig = (y_center / input_size) * orig_h
                    w_orig = (w / input_size) * orig_w
                    h_orig = (h / input_size) * orig_h

                    # 计算边界框的左上角坐标
                    x1 = int(max(0, x_center_orig - w_orig / 2))
                    y1 = int(max(0, y_center_orig - h_orig / 2))
                    w_int = int(w_orig)
                    h_int = int(h_orig)

                    # 计算底部中点坐标 (u, v)
                    u = int(x_center_orig)  # 水平中心
                    v = int(y1 + h_int)  # 底部

                    football_detections.append({
                        'position': (u, v),
                        'confidence': conf,
                        'bbox': (x1, y1, w_int, h_int)
                    })

        # 如果没有检测到足球，返回None
        if len(football_detections) == 0:
            print("未检测到足球")
            return None

        # 选择置信度最高的足球检测结果
        best_detection = max(football_detections, key=lambda x: x['confidence'])

        result = DetectionResult()
        result.position = best_detection['position']
        result.confidence = best_detection['confidence']
        result.bbox = best_detection['bbox']

        print(f"足球检测结果: 位置{result.position}, 置信度{result.confidence:.3f}")
        return result

    def draw_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        result_image = image.copy()

        if detection is None:
            print("没有检测结果可绘制")
            return result_image

        # 绘制检测框
        x, y, w, h = detection.bbox
        cv2.rectangle(result_image,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)  # 绿色框

        # 绘制底部中点
        u, v = detection.position
        cv2.circle(result_image, (u, v), 5, (0, 0, 255), -1)  # 红色圆点

        # 构建显示文本
        display_text = f"Football: {detection.confidence:.2f} at ({u},{v})"

        # 绘制文本
        font_scale = 0.6
        text_color = (0, 0, 255)
        thickness = 2

        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        # 文本位置
        text_org = (x, y - 10 if y - 10 > 10 else y + 20)

        cv2.putText(result_image, display_text, text_org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        print(f"绘制检测结果: 位置({u},{v}), 置信度{detection.confidence:.3f}")

        return result_image


def test_football_detection(image_path: str, model_path: str, confidence_threshold=0.1):
    print("=== 足球检测测试 ===")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功! 尺寸: {image.shape[1]} × {image.shape[0]}")

    try:
        # 初始化检测器
        detector = SimpleFootballDetector(model_path, confidence_threshold=confidence_threshold)

        # 进行检测
        detection = detector.inference(image)

        if detection is not None:
            u, v = detection.position
            print(f"成功检测到足球!")
            print(f"位置: ({u}, {v})")
            print(f"置信度: {detection.confidence:.3f}")
            print(f"边界框: {detection.bbox}")
        else:
            print("未检测到足球")

        # 绘制检测结果
        result_image = detector.draw_detection(image, detection)

        # 显示结果
        cv2.imshow("Football Detection", result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = "football_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请替换为实际的文件路径
    image_path = "football_image.jpg"
    model_path = "model/best.onnx"

    # 只进行一次检测，使用适当的置信度阈值
    test_football_detection(image_path, model_path, confidence_threshold=0.1)
